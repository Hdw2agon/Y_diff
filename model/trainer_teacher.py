import os
import copy
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import lpips
import clip
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from diffae.templates_latent import ffhq256_autoenc_latent
from diffae.experiment import LitModel

from model.SPNet import SPNet, VGGStructureLoss

from utils.network_ncsn import D_NLayersMulti
from utils.GAN_loss import GANLoss

import wandb

class TrainDataset(Dataset):
    def __init__(self, img_dirA, img_dirB, is_train=True, pickup_length=None):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        dir_path = 'train' if is_train else 'test'

        self.imgsA = [p for p in glob.glob(os.path.join(img_dirA, dir_path, '*')) if p.lower().endswith(exts)]
        self.imgsB = [p for p in glob.glob(os.path.join(img_dirB, dir_path, '*')) if p.lower().endswith(exts)]

        if pickup_length is not None:
            if pickup_length < min(len(self.imgsA), len(self.imgsB)):
                self.imgsA = self.imgsA[:pickup_length]
                self.imgsB = self.imgsB[:pickup_length]
            else:
                print(f'[Dataset] Pickup_length-{pickup_length} exceeds dataset size {min(len(self.imgsA), len(self.imgsB))}, the full dataset will be loaded!')
        
        if len(self.imgsA) == 0:
            raise RuntimeError(f'Cannot find image: {os.path.join(img_dirA, dir_path)}')
        if len(self.imgsB) == 0:
            raise RuntimeError(f'Cannot find image: {os.path.join(img_dirB, dir_path)}')
        
        self.transform_img = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
        ])      

    def __len__(self):
        return min(len(self.imgsA), len(self.imgsB))

    def __getitem__(self, idx):
        img_A = Image.open(self.imgsA[idx]).convert('RGB')
        img_B = Image.open(self.imgsB[idx]).convert('RGB')
        img_A = self.transform_img(img_A)
        img_B = self.transform_img(img_B)
        filename = os.path.basename(self.imgsA[idx]).split(".")[0]

        return {"A": img_A, "B": img_B, "A_paths": self.imgsA[idx], "B_paths": self.imgsB[idx], "filename": filename}


class DiffFSTrainer:
    def __init__(self, args):
        self.args = args

        # set seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # load model
        self.model_clip, model_diffae, self.conf = self.load_model(diffae_ckpt=self.args.diffae_ckpt)

        # prepare train
        self.diffae_A, self.diffae_B, self.conds_std, self.conds_mean = self.prepare_train(model_diffae)
        self.diffae_ghost = copy.deepcopy(self.diffae_B)

        if self.args.enable_spnet:
            self.model_map = SPNet().to(args.device)        

        self.netD = D_NLayersMulti(input_nc=3, ndf=64, n_layers=3, num_D=1).to(self.args.device)
        self.optim_D = optim.Adam(self.netD.parameters(), lr=self.args.lr * 0.2, betas=(0.5, 0.999))
        self.gan_loss_weight = 1
        self.criterionGAN = GANLoss('lsgan').to(self.args.device) 

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.diffae_B = nn.DataParallel(self.diffae_B)
            if self.args.enable_spnet:
                self.model_map = nn.DataParallel(self.model_map)

            self.diffae_ghost = nn.DataParallel(self.diffae_ghost)
            self.netD = nn.DataParallel(self.netD) 

        self.optim_diffae_B = optim.Adam(self.diffae_B.parameters(), lr=self.args.lr)
        self.optim_diffae_ghost = optim.Adam(self.diffae_ghost.parameters(), lr=self.args.lr)
        if self.args.enable_spnet:
            self.optim_map = optim.Adam(self.model_map.parameters(), lr=1e-4)
         
        step_size = getattr(self.args, 'lr_step', 50)
        gamma = getattr(self.args, 'lr_gamma', 0.5)
        self.scheduler_diffae_B = optim.lr_scheduler.StepLR(self.optim_diffae_B, step_size=step_size, gamma=gamma)

        self.train_samp_for = self.conf._make_diffusion_conf(self.args.T_train_for).make_sampler()      # T_train_for=50 
        self.train_samp_back = self.conf._make_diffusion_conf(self.args.T_train_back).make_sampler()    # T_train_back=20

        self.l1_loss = nn.L1Loss()
        self.percept_loss = lpips.LPIPS(net='alex').to(self.args.device)
        self.cosine_loss = nn.CosineSimilarity()

        if 'ccd' in self.args.dataset_path.lower():
            Dataset = TrainDataset(f'{self.args.dataset_path}/HE', f'{self.args.dataset_path}/pCLE', is_train=True, pickup_length=self.args.pickup_length)
        elif 'er-004' in self.args.dataset_path.lower():
            Dataset = TrainDataset(f'{self.args.dataset_path}/HE', f'{self.args.dataset_path}/IHC', is_train=True, pickup_length=self.args.pickup_length)

        self.dataloader = DataLoader(Dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    num_workers=self.args.num_workers, pin_memory=True)
         
        if self.args.start_epoch > 1:
            self.load_checkpoint_for_continue(self.args.start_epoch)

       # ======================= VGG-based structure loss --begin =======================
        self.structure_loss_fn = VGGStructureLoss(self.args.device, content_layers=[21])
        self.content_decay_epochs = [10, 20, 30, 40, 50]
        if hasattr(self.args, 'start_epoch') and self.args.start_epoch > 1:
            decay_epochs = self.content_decay_epochs
            decay_count = sum([self.args.start_epoch - 1 >= e for e in decay_epochs])
            if decay_count > 0:
                self.content_weight = self.content_weight * (0.5 ** decay_count)
                print(f"[Init] Adjusted content_weight to {self.content_weight} for start_epoch={self.args.start_epoch} (decay {decay_count} times)")
        # ======================= VGG-based structure loss --end =======================

        if self.args.use_wandb:
            print("Weights & Biases logging is enabled.")

            # 准备 wandb.init 的参数
            wandb_kwargs = {
                'project': "Teacher_model_training",
                'config': self.args,
            }

            if self.args.wandb_resume_id:
                print(f"Resuming wandb run with ID: {self.args.wandb_resume_id}")
                wandb_kwargs['id'] = self.args.wandb_resume_id  # lwyno1gr
                wandb_kwargs['resume'] = "allow"
            else:
                print("Starting a new wandb run.")
                wandb_kwargs['name'] = self.args.work_dir.split('/')[-1]

            wandb.init(**wandb_kwargs)

            wandb.watch(self.diffae_B, log="all", log_freq=100)
            wandb.watch(self.model_map, log="all", log_freq=100)

    # todo. load continue loading
    def load_checkpoint_for_continue(self, start_epoch):
        load_epoch = start_epoch - 1
        ckpt_path = os.path.join(self.args.work_dir, 'ckpt', f'epoch_{load_epoch}.pt')

        print(f"Loading checkpoint from {ckpt_path} to continue training from epoch {start_epoch}...")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}.")

        ckpt = torch.load(ckpt_path, map_location=self.args.device)

        def load_state_dict_flexible(model, state_dict):
            is_data_parallel = isinstance(model, nn.DataParallel)
            is_ckpt_data_parallel = any(k.startswith('module.') for k in state_dict)

            if not is_data_parallel and is_ckpt_data_parallel:
                new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            elif is_data_parallel and not is_ckpt_data_parallel:
                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)

        load_state_dict_flexible(self.diffae_B, ckpt['diffae_B'])
        if self.args.map_net and 'model_map' in ckpt:
            load_state_dict_flexible(self.model_map, ckpt['model_map'])
        if 'netD' in ckpt:
            load_state_dict_flexible(self.netD, ckpt['netD'])

        if 'optim_diffae_B' in ckpt:
            self.optim_diffae_B.load_state_dict(ckpt['optim_diffae_B'])
        if self.args.map_net and 'optim_map' in ckpt:
            self.optim_map.load_state_dict(ckpt['optim_map'])
        if 'optim_D' in ckpt:
            self.optim_D.load_state_dict(ckpt['optim_D'])

        print("Checkpoint loaded successfully.")
   
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def load_model(self,
                   clip_encoder='ViT-B/32',
                   diffae_ckpt='diffae/checkpoints'):

        model_clip, _ = clip.load(clip_encoder, device=self.args.device)

        conf = ffhq256_autoenc_latent()
        conf.pretrain.path = os.path.join(diffae_ckpt, 'ffhq256_autoenc/last.ckpt')
        conf.latent_infer_path = os.path.join(diffae_ckpt, 'ffhq256_autoenc/latent.pkl')

        model_diffae = LitModel(conf)
        state = torch.load(conf.pretrain.path, map_location='cpu', weights_only=False)
        model_diffae.load_state_dict(state['state_dict'], strict=False)

        return model_clip, model_diffae, conf

    def prepare_train(self, model_diffae):
        diffae_A = copy.deepcopy(model_diffae.ema_model)
        diffae_A = diffae_A.to(self.args.device)
        diffae_A.eval()
        diffae_A.requires_grad_(False)

        diffae_B = copy.deepcopy(model_diffae.ema_model)
        diffae_B = diffae_B.to(self.args.device)
        diffae_B.train()
        diffae_B.requires_grad_(True)

        conds_std = model_diffae.conds_std
        conds_mean = model_diffae.conds_mean

        return diffae_A, diffae_B, conds_std, conds_mean

    def compute_loss(self, img_A, img_B, x0_pred_style_B):

        loss = 0
        loss_dict = {}

        recon_clip_loss = self.compute_recon_clip_loss(img_B, x0_pred_style_B)
        loss += recon_clip_loss
        loss_dict['recon_clip'] = recon_clip_loss.item()

        # ref image lpips loss.  no clip used in.
        recon_lpips_loss = self.compute_recon_lpips_loss(img_B, x0_pred_style_B)
        loss += recon_lpips_loss
        loss_dict['recon_lpips'] = recon_lpips_loss.item()

        l_content = self.structure_loss_fn(x0_pred_style_B,img_A)
        loss_content = l_content * self.content_weight
        loss += loss_content
        loss_dict['structure_loss'] = loss_content.item()

        return loss, loss_dict

    # compute style image clip embedding reconstruction loss
    def compute_recon_clip_loss(self, img_B, x0_pred_style_B):
        x0_pred_style_clip = self.model_clip.encode_image(F.interpolate(x0_pred_style_B, (224,224)))
        # x0_pred_style_clip /= x0_pred_style_clip.clone().norm(dim=-1, keepdim=True)
        img_style_B_clip = self.model_clip.encode_image(F.interpolate(img_B, (224,224)))
        # img_style_B_clip /= img_style_B_clip.clone().norm(dim=-1, keepdim=True)

        recon_clip_loss = (1- self.cosine_loss(x0_pred_style_clip, img_style_B_clip)).mean()
        recon_clip_loss = recon_clip_loss*self.args.recon_clip # recon_clip=30.0
        return recon_clip_loss

    def compute_recon_l1_loss(self, img_B, x0_pred_style_B):
        return self.l1_loss(x0_pred_style_B, img_B)*self.args.recon_l1 # recon_l1=10.0

    def compute_recon_lpips_loss(self, img_B, x0_pred_style_B):
        return self.percept_loss(x0_pred_style_B, img_B).mean()*self.args.recon_lpips # recon_lpips=10.0
    
    def print_loss(self, i, loss, loss_dict):
        print('Iter {} | Loss {:5f} '.format(i+1, loss.item()), end='')
        for k in loss_dict:
            print(f'| {k} {loss_dict[k]:5f} ', end='')
        print('')

    def save_image(self, i, epoch, img_A, img_B, A_pred_B):
        imgs = torch.cat([img_A, img_B, A_pred_B], dim=0)
        img_dir = os.path.join(self.args.work_dir, 'imgs_train')
        os.makedirs(img_dir, exist_ok=True)
        save_image(imgs/2+0.5, os.path.join(img_dir, f'epoch_{epoch+1}_iter_{i+1}.png'))

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.work_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        if self.args.enable_spnet:
            content = {
                'iter': epoch+1,
                'diffae_B': self.diffae_B.state_dict(),
                'model_map': self.model_map.state_dict(),
                'optim_diffae_B': self.optim_diffae_B.state_dict(),
                'optim_map': self.optim_map.state_dict(),
                'netD': self.netD.state_dict(),
                'optim_D': self.optim_D.state_dict(),
            }
        else:
            content = {
                'iter': epoch+1,
                'diffae_B': self.diffae_B.state_dict(),
                'optim_diffae_B': self.optim_diffae_B.state_dict(),
                'netD': self.netD.state_dict(),
                'optim_D': self.optim_D.state_dict(),
            }

        torch.save(content, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt'))

    def train(self):

        print(f"Training will start from epoch {self.args.start_epoch}.")
        start_epoch = self.args.start_epoch - 1 if self.args.start_epoch > 1 else 0

        for epoch in range(start_epoch, self.args.n_iter):
            print(f"Starting epoch {epoch+1}/{self.args.n_iter}...")

            if (epoch) in [10, 20, 30, 40, 50]:
                print(f"Adjusting gram content weight at epoch {epoch+1}.")
                self.content_weight = self.content_weight * 0.5

            for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                img_A = data['A'].to(self.args.device)
                img_B = data['B'].to(self.args.device)
                filename = data['filename'][0]

                z_sem_A = self.diffae_A.encode(img_A)    
                z_sem_A = z_sem_A.detach().clone()
                z_sem_B = self.diffae_A.encode(img_B)    
                z_sem_B = z_sem_B.detach().clone()

                xt_A = img_A.clone()

                with torch.no_grad():
                    forward_indices = list(range(self.args.T_train_for))[:int(self.args.T_train_for*self.args.t0_ratio)]
                    for j in forward_indices:

                        t = torch.tensor([j]*len(img_A), device=self.args.device)
                        out_style = self.train_samp_for.ddim_reverse_sample(self.diffae_A,
                                                                            xt_A,
                                                                            t,
                                                                            model_kwargs={'cond': z_sem_A})
                        xt_A = out_style['sample']
                        
                xt_A = xt_A.detach().clone()
                
                ################################################################################
                # backward ddim
                backward_indices = list(range(self.args.T_train_back))[::-1][int(self.args.T_train_back*(1-self.args.t0_ratio)):] 
                #todo. process img A to style B backward ddim 
                for j in backward_indices:

                    t = torch.tensor([j]*len(img_A), device=self.args.device) 
                 
                    if self.args.enable_spnet:
                        map_style = self.model_map(img_A, t)
                        map_style = torch.sigmoid(map_style)
                        xt_A = xt_A + 0.1*map_style
                        # xt_A = xt_A + self.args.lambda_map*map_style

                    out_style = self.train_samp_back.ddim_sample(self.diffae_B,
                                                                xt_A,
                                                                t,
                                                                model_kwargs={'cond': z_sem_B})
                    x0_pred_B = out_style['pred_xstart'] 

                    total_loss, loss_dict = self.compute_loss(img_A, img_B, x0_pred_B)

                    #================Discriminator Losses Added Begin================#
                    self.set_requires_grad(self.netD, True)
                    self.optim_D.zero_grad()

                    pred_real = self.netD(img_B)
                    loss_D_real = self.criterionGAN(pred_real, True).mean()
                    pred_fake = self.netD(x0_pred_B.detach()) 
                    loss_D_fake = self.criterionGAN(pred_fake, False).mean()

                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    loss_D.backward()
                    self.optim_D.step()
                    # -----------------------------------------------------------
                    self.set_requires_grad(self.netD, False)
                    
                    pred_fake_G = self.netD(x0_pred_B)
                    loss_G_GAN = self.criterionGAN(pred_fake_G, True).mean()

                    total_loss = total_loss + loss_G_GAN * self.gan_loss_weight
                    
                    loss_dict['D_real'] = loss_D_real.item()
                    loss_dict['D_fake'] = loss_D_fake.item()
                    loss_dict['G_gan'] = loss_G_GAN.item()
                    #================Discriminator Losses Added End==================#

                    self.optim_diffae_B.zero_grad()
                    self.optim_map.zero_grad()

                    total_loss.backward()

                    self.optim_diffae_B.step()
                    self.optim_map.step()

                    xt_A = out_style['sample'].detach().clone()

                # print loss
                if (i+1)%self.args.print_freq==0:
                    self.print_loss(i, total_loss, loss_dict)

                # save images
                if (i+1)%10==0:
                    self.save_image(i, epoch, img_A, img_B, x0_pred_B)
                    
                # ##########################################################################
                if self.args.use_wandb:
                    log_data = {
                        'epoch': epoch + 1,
                        'iter': i + 1,
                        'total_loss': total_loss.item(),
                        'lr_diffae_B': self.scheduler_diffae_B.get_last_lr()[0]
                    }
                    
                    for k, v in loss_dict.items():
                        log_data[f'loss/{k}'] = v

                    wandb.log(log_data)

            # save model
            if (epoch+1)%self.args.ckpt_freq==0:
                self.save_model(epoch)
