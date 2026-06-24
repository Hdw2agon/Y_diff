import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffae'))
import argparse
import glob
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # TensorBoard
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffae.templates_latent import ffhq256_autoenc_latent
from diffae.experiment import LitModel

from utils.FlowMatching import FlowMatchingModule


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
        
        self.transform_img = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  
            ])      

    def __len__(self):
        return min(len(self.imgsA), len(self.imgsB))

    def __getitem__(self, idx):
        try:
            img_A = Image.open(self.imgsA[idx]).convert('RGB')
            img_B = Image.open(self.imgsB[idx]).convert('RGB')
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self)-1))

        return {
            "A": self.transform_img(img_A), 
            "B": self.transform_img(img_B)
        }

def load_diffae_encoder(ckpt_path, device):
    print(f"[DiffAE] Loading Encoder from {ckpt_path}...")
    conf = ffhq256_autoenc_latent()
    conf.pretrain.path = os.path.join(ckpt_path, 'ffhq256_autoenc/last.ckpt')
    conf.latent_infer_path = os.path.join(ckpt_path, 'ffhq256_autoenc/latent.pkl')
    model_diffae = LitModel(conf)
    state = torch.load(conf.pretrain.path, map_location='cpu', weights_only=False)
    model_diffae.load_state_dict(state['state_dict'], strict=False)
    
    model = model_diffae.ema_model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

def print_loss(self, i, loss, loss_dict):
    print('Iter {} | Loss {:5f} '.format(i+1, loss), end='')
    for k in loss_dict:
        print(f'| {k} {loss_dict[k]:5f} ', end='')
    print('')

def main():
    parser = argparse.ArgumentParser(description="Standalone Training for Flow Matching")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--diffae_ckpt', type=str, default='diffae/checkpoints')
    parser.add_argument('--work_dir', type=str, default='./fm', help='Directory to save logs and models')
    parser.add_argument('--batch_size', type=int, default=32, help='Increase this if memory allows')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--num_layers', type=int, default=8, help='Depth of ResFlowNet')
    parser.add_argument('--exp_name', type=str, default='default')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device: {device}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.work_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard] Logging to {log_dir}")

    dataset = TrainDataset(f'{args.dataset_path}/HE', f'{args.dataset_path}/pCLE', is_train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    encoder = load_diffae_encoder(args.diffae_ckpt, device)
    
    fm_module = FlowMatchingModule(
        dim=512, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers,
        lr=args.lr, 
        use_ot=True, 
        device=device
    )
    
    if os.path.exists(os.path.join(args.work_dir, 'stats.pt')):
        print("[Stats] Found existing stats, loading...")
        stats = torch.load(os.path.join(args.work_dir, 'stats.pt'))
        fm_module.set_normalization_stats(stats['mean'], stats['std'])
    else:
        print("[Stats] Calculating Latent Statistics for Normalization...")
        z_list = []
        for i, data in enumerate(tqdm(dataloader, desc="Collecting Stats")):
            if i >= 50: break
            img_A = data['A'].to(device)
            with torch.no_grad():
                z = encoder.encode(img_A)
                z_list.append(z.cpu())
        z_all = torch.cat(z_list, dim=0)
        data_mean = z_all.mean(dim=0)
        data_std = z_all.std(dim=0)
        fm_module.set_normalization_stats(data_mean, data_std)

        torch.save({'mean': data_mean, 'std': data_std}, os.path.join(args.work_dir, 'stats.pt'))


    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.n_epochs):
        fm_module.net.train()
        epoch_loss_sum = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.n_epochs}")
        for i, data in enumerate(pbar):
            img_A = data['A'].to(device)
            img_B = data['B'].to(device)
            
            with torch.no_grad():
                z_sem_A = encoder.encode(img_A).detach()
                z_sem_B = encoder.encode(img_B).detach()
            
            loss_dict = fm_module.train_step(z_sem_A, z_sem_B)
            loss = loss_dict['loss_fm']
            
            epoch_loss_sum += loss
            global_step += 1
        
        avg_loss = epoch_loss_sum / len(dataloader)
        print_loss(fm_module, epoch, avg_loss, loss_dict)
        
        if (epoch + 1) % 5 == 0:
            fm_module.net.eval()
            with torch.no_grad():

                z_pred = fm_module.transfer(z_sem_A)
                
                cos_sim = F.cosine_similarity(z_pred, z_sem_B).mean().item()
                l2_dist = F.pairwise_distance(z_pred, z_sem_B).mean().item()
                cos_source = F.cosine_similarity(z_pred, z_sem_A).mean().item()
                
                print(f"\n[Val] Epoch {epoch+1}: CosSim(Target)={cos_sim:.4f}, L2={l2_dist:.4f}")
                
                writer.add_scalar('Val/CosSim_Target', cos_sim, epoch)
                writer.add_scalar('Val/L2_Dist', l2_dist, epoch)
                writer.add_scalar('Val/CosSim_Source', cos_source, epoch)

        save_dict = {
            'epoch': epoch + 1,
            'fm_module': fm_module.state_dict(),
            'stats_mean': fm_module.data_mean, 
            'stats_std': fm_module.data_std
        }

        torch.save(save_dict, os.path.join(log_dir, 'latest_fm.pt'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_dict, os.path.join(log_dir, 'best_fm.pt'))
            
    print("Training Finished.")
    writer.close()

if __name__ == "__main__":
    main()