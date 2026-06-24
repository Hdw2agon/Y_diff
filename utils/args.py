import os
import json
import argparse
from pathlib import Path

def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainer_version', type=str, default='fm_student', 
                        help='Alias for the DiffFSTrainer version to use.')
    parser.add_argument('--tester_version', type=str, default='y_diff', 
                        help='Alias for the DiffFSTester version to use.')
    
    parser.add_argument('--dataset_path', type=str, default='datasets/BCI_dataset', help='path to dataset root directory')
    parser.add_argument('--pickup_length', type=int, default=None, help='length of picked up images during training')
    parser.add_argument('--gpus', type=str, default='0,1', help='gpu ids to use, split by comma') 
    parser.add_argument('--start_epoch', type=int, default=1, 
                        help='Epoch to start training from.')
    
    parser.add_argument('--use_wandb', action='store_true', help='Enable logging with Weights & Biases')
    parser.add_argument('--wandb_resume_id', type=str, default=None, help='Resume ID for Weights & Biases')

    parser.add_argument("--load_size", type=int, default=256, help="scale images to this size")
    parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--diffae_ckpt', default='diffae/checkpoints', type=str, help='diffusion autoencoder checkpoints')
    parser.add_argument('--work_dir', default='exp', type=str, help='experiment working directory')

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--ckpt_freq', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    parser.add_argument('--n_iter', default=200, type=int, help='training iteration')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')

    parser.add_argument('--T_train_for', default=50, type=int, help='forward timesteps during training')
    parser.add_argument('--T_train_back', default=50, type=int, help='backward timesteps during training')
    parser.add_argument('--T_infer_for', default=50, type=int, help='forward timesteps during inference')
    parser.add_argument('--T_infer_back', default=50, type=int, help='backward timesteps during inference')
    parser.add_argument('--t0_ratio', default=0.5, type=float, help='return step ratio')

    parser.add_argument('--recon_clip', default=30, type=float, help='clip reconstruction loss')
    parser.add_argument('--recon_l1', default=10, type=float, help='l1 reconstruction')
    parser.add_argument('--recon_lpips', default=10, type=float, help='lpips reconstruction')
    parser.add_argument('--struct_loss_w', type=float, default=0.1, 
                    help='Weight for gram content loss.')

    parser.add_argument('--enable_spnet', action='store_true', help='using mappingnet')
    parser.add_argument('--lambda_map', default=0.1, type=float, help='weight of mapping net output')

    parser.add_argument('--z_sem_flowmatch', action='store_true', help='use flow matching on z_sem')
    parser.add_argument('--fm_work_dir', type=str, default='2048_8', help='work dir for flow matching module')

    args = parser.parse_args()

    args.device = 'cuda'

    if args.train:
        os.makedirs(args.work_dir, exist_ok=True)
        with open(os.path.join(args.work_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return args

