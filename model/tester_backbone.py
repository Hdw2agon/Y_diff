import os
import glob
import copy
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from diffae.templates_latent import ffhq256_autoenc_latent
from diffae.experiment import LitModel

from model.SPNet import SPNet
from model.FlowMatching import FlowMatchingModule

from utils.Metrics import Metrics

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.imgs = glob.glob(os.path.join(img_dir, '*'))
        self.transform_img = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
        ])  

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_cont_A = Image.open(self.imgs[idx]).convert('RGB')
        img_cont_A = self.transform_img(img_cont_A)
        return {"A": img_cont_A, "A_paths": self.imgs[idx], "filename": Path(self.imgs[idx]).name}

class TestDataset1(Dataset):
    def __init__(self, img_dirA, img_dirB, is_train=False, pickup_length=None):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        dir_path = 'train' if is_train else 'test'

        self.imgsA = [p for p in glob.glob(os.path.join(img_dirA, dir_path, '*')) if p.lower().endswith(exts)]
        self.imgsB = [p for p in glob.glob(os.path.join(img_dirB, dir_path, '*')) if p.lower().endswith(exts)]

        if pickup_length is not None:
            if pickup_length < min(len(self.imgsA), len(self.imgsB)):
                self.imgsA = self.imgsA[:pickup_length]
                self.imgsB = self.imgsB[:pickup_length]
            else:
                raise ValueError(f'[Dataset] Pickup_length-{pickup_length} exceeds dataset size {min(len(self.imgsA), len(self.imgsB))}')
        
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
        img_A = self.transform_img(img_A)
        img_B = Image.open(self.imgsB[idx]).convert('RGB')
        img_B = self.transform_img(img_B)
        filename = Path(self.imgsA[idx]).name
        return {"A": img_A, "B": img_B, "A_paths": self.imgsA[idx], "B_paths": self.imgsB[idx], "filename": filename}



class DiffFSTester:
    def __init__(self, args):
        self.args = args

        # set seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # load diffae
        conf = ffhq256_autoenc_latent()
        conf.pretrain.path = os.path.join(self.args.diffae_ckpt, 'ffhq256_autoenc/last.ckpt')
        conf.latent_infer_path = os.path.join(self.args.diffae_ckpt, 'ffhq256_autoenc/latent.pkl')

        model_diffae = LitModel(conf)
        state = torch.load(conf.pretrain.path, map_location='cpu',weights_only=False)
        model_diffae.load_state_dict(state['state_dict'], strict=False)

        # make diffae for domainA (photo) / freeze
        self.diffae_A = copy.deepcopy(model_diffae.ema_model)
        self.diffae_A = self.diffae_A.to(self.args.device)
        self.diffae_A.eval()
        self.diffae_A.requires_grad_(False)

        # make diffae for domainB (style) / train 
        self.diffae_B = copy.deepcopy(model_diffae.ema_model)
        self.diffae_B = self.diffae_B.to(self.args.device)
        self.diffae_B.eval()
        self.diffae_B.requires_grad_(False)

        self.metrics = Metrics(res_save_path=self.args.work_dir)

        if self.args.enable_spnet:
            self.model_map = SPNet().to(args.device)
            self.model_map.eval()
            self.model_map.requires_grad_(False)

        if self.args.z_sem_flowmatch:
            self.flowmatch_z_sem = FlowMatchingModule(dim=512, hidden_dim=2048, num_layers=8, use_ot=True).to(self.args.device)
            self.flowmatch_z_sem.eval()
            self.flowmatch_z_sem.requires_grad_(False)

        self.infer_samp_for = conf._make_diffusion_conf(self.args.T_infer_for).make_sampler()
        self.infer_samp_back = conf._make_diffusion_conf(self.args.T_infer_back).make_sampler()
        
        if 'ccd' in self.args.dataset_path.lower():
            Dataset = TestDataset1(f'{self.args.dataset_path}/HE', f'{self.args.dataset_path}/pCLE', is_train=False, pickup_length=self.args.pickup_length)
            self.dataloader = DataLoader(Dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=self.args.num_workers, pin_memory=True)
        elif 'er-004' in self.args.dataset_path.lower():
            Dataset = TestDataset1(f'{self.args.dataset_path}/HE', f'{self.args.dataset_path}/IHC', is_train=False, pickup_length=self.args.pickup_length)  
            self.dataloader = DataLoader(Dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=self.args.num_workers, pin_memory=True)
        elif 'infer' in self.args.dataset_path.lower():
            self.infer_only = True
            Dataset = TestDataset(f'{self.args.dataset_path}')
            self.dataloader = DataLoader(Dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=self.args.num_workers, pin_memory=True)
  

    #TODO overwrite infer_image function
    def infer_image(self):
        for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            img_A = data['A'].to(self.args.device)
            img_B = data['B'].to(self.args.device)
            filename = data['filename']

            z_sem_A = self.diffae_A.encode(img_A)    
            z_sem_B = self.diffae_A.encode(img_B)

            z_sem_A = z_sem_A.detach().clone()
            xt_A = img_A.clone()

            with torch.no_grad():
                forwad_indices = list(range(self.args.T_infer_for))[:int(self.args.T_infer_for*self.args.t0_ratio)]
                for j in forwad_indices:
                    t = torch.tensor([j]*len(img_A), device=self.args.device)
                    out = self.infer_samp_for.ddim_reverse_sample(self.diffae_A,
                                                                  xt_A,
                                                                  t,
                                                                  model_kwargs={'cond': z_sem_A})
                    xt_A = out['sample']

                xt_B = xt_A.detach().clone()

                reverse_indices = list(range(self.args.T_infer_back))[::-1][int(self.args.T_infer_back*(1-self.args.t0_ratio)):]
                for j in reverse_indices:
                    t = torch.tensor([j]*len(img_A), device=self.args.device)
                    if self.args.enable_spnet:
                        map_cont = self.model_map(img_A, t)
                        map_cont = torch.sigmoid(map_cont)
                        # xt_B = xt_B + self.args.lambda_map*map_cont
                        xt_B = xt_B + map_cont * 0.1

                    out = self.infer_samp_back.ddim_sample(self.diffae_B,
                                                            xt_B,
                                                            t,
                                                            model_kwargs={
                                                                'cond': z_sem_A
                                                            })
                    x0_pred_B = out['pred_xstart']
                    xt_B = out['sample'].detach().clone()
                
                x0_pred_B = x0_pred_B.detach().clone()
                
                save_dir = os.path.join(self.args.work_dir, 'imgs_test')
                os.makedirs(save_dir, exist_ok=True)
                # save_image(xt_cont_B/2+0.5, os.path.join(save_dir, Path(input_path).name))

                for i, img in enumerate(x0_pred_B):
                    save_image(img/2+0.5, os.path.join(save_dir, filename[i]))

    def infer_image_all(self):
        ckpt_path = os.path.join(self.args.work_dir, 'ckpt', f'epoch_{self.args.n_iter}.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        print("begin load state dict diffae_B")
        self.diffae_B.load_state_dict(ckpt['diffae_B'])
        self.diffae_B = self.diffae_B.to(self.args.device)
        print("end load state dict diffae_B")
        
        print("begin load state dict map_net")
        if self.args.enable_spnet:
            self.model_map.load_state_dict(ckpt['model_map'])
            self.model_map = self.model_map.to(self.args.device)
        print("end load state dict map_net")

        if self.args.z_sem_flowmatch:
            print("begin load state dict flowmatch z_sem")
            ckpt_best = torch.load(f'fm/{self.args.fm_work_dir}/best_fm.pt',map_location='cpu')
            self.flowmatch_z_sem.load_state_dict(ckpt_best['fm_module'])
            if 'stats_mean' in ckpt_best and 'stats_std' in ckpt_best:
                self.flowmatch_z_sem.set_normalization_stats(ckpt_best['stats_mean'], ckpt_best['stats_std'])
            else:
                self.flowmatch_z_sem.stats_initialized = True
            self.flowmatch_z_sem = self.flowmatch_z_sem.to(self.args.device)
            print("end load state dict flowmatch z_sem")

        self.infer_image()

    def infer_image_all_multiGPU(self):
        ckpt_path = os.path.join(self.args.work_dir, 'ckpt', f'epoch_{self.args.n_iter}.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        print("begin load state dict diffae_B")
        new_diffae_b_state_dict = {}
        original_diffae_b_state_dict = ckpt['diffae_B']
        for k, v in original_diffae_b_state_dict.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
                new_diffae_b_state_dict[new_key] = v
            else:
                new_diffae_b_state_dict[k] = v

        self.diffae_B.load_state_dict(new_diffae_b_state_dict)
        self.diffae_B = self.diffae_B.to(self.args.device)
        print("end load state dict diffae_B")
        
        if self.args.enable_spnet:
            print("begin load state dict map_net")
            new_map_state_dict = {}
            for k, v in ckpt['model_map'].items():
                if k.startswith('module.'):
                    new_map_state_dict[k[len('module.'):]] = v
                else:
                    new_map_state_dict[k] = v

            self.model_map.load_state_dict(new_map_state_dict)
            self.model_map = self.model_map.to(self.args.device)
            print("end load state dict map_net")

        if self.args.z_sem_flowmatch:
            print("begin load state dict flowmatch z_sem")
            fm_ckpt_path = os.path.join('fm', self.args.fm_work_dir, 'best_fm.pt')
            ckpt_best = torch.load(fm_ckpt_path,map_location='cpu')
            new_fm_zsem_state_dict = {}
            for k, v in ckpt_best['fm_module'].items():
                if k.startswith('module.'):
                    new_fm_zsem_state_dict[k[len('module.'):]] = v
                else:
                    new_fm_zsem_state_dict[k] = v
            
            self.flowmatch_z_sem.load_state_dict(new_fm_zsem_state_dict,strict=False)
            if 'stats_mean' in ckpt_best and 'stats_std' in ckpt_best:
                self.flowmatch_z_sem.set_normalization_stats(ckpt_best['stats_mean'], ckpt_best['stats_std'])
            else:
                self.flowmatch_z_sem.stats_initialized = True
            self.flowmatch_z_sem = self.flowmatch_z_sem.to(self.args.device)
            print("end load state dict flowmatch z_sem")
            
        self.infer_image()

        if 'ccd' in self.args.dataset_path.lower():
            self.metrics.metrics_loop(f'{self.args.dataset_path}/HE/test', f'{self.args.dataset_path}/pCLE/test', os.path.join(self.args.work_dir, 'imgs_test'))
        else:
            self.metrics.metrics_loop(f'{self.args.dataset_path}/HE/test', f'{self.args.dataset_path}/IHC/test', os.path.join(self.args.work_dir, 'imgs_test'))

