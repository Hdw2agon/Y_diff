import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from diffae.model.nn import timestep_embedding


class SPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('gray_weights', torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))
        
        self.in_layer = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1),
        )

        self.block1 = SPNetBaseBlock(
            in_ch = 128,
            out_ch = 256,
            t_emb_dim = 512
        )

        self.block2 = SPNetBaseBlock(
            in_ch = 256,
            out_ch = 128,
            t_emb_dim = 512
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1)
        )
      
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, t): # [B, 3, H, W], [B]

        t_emb = timestep_embedding(t, 128) # [B, 128]
        t_emb = self.time_embed(t_emb) # [B, 512]

        x_gray = (x * self.gray_weights).sum(dim=1, keepdim=True) # [B, 1, H, W]

        x = self.in_layer(x_gray)    # [B, 1, H, W] -> [B, 128, H, W]

        x = self.block1(x, t_emb) # [B, 128, H, W] -> [B, 256, H, W]
        x = self.block2(x, t_emb) # [B, 256, H, W] -> [B, 128, H, W]

        output = self.out_layer(x)   # [B, 128, H, W] -> [B, 3, H, W]

        return output # [B, 3, H, W]


class SPNetBaseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()

        self.pre_layer = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GroupNorm(32, out_ch),
        )

        self.post_layer = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch*2)
        )
    
    def forward(self, x, t_emb, scale_bias:float=1):

        t_emb = self.emb_layer(t_emb) # [B, out_ch*2] 512, 256
        
        # match shape 
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]

        scale, shift = torch.chunk(t_emb, 2, dim=1)

        x = self.pre_layer(x)
        x = x * (scale_bias + scale)
        x = x + shift
        x = self.post_layer(x)

        return x
    

class VGGStructureLoss(nn.Module):
    def __init__(self, device, content_layers=[21]):
        super().__init__()
        self.content_layers = content_layers
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1) / 2
        return (x01 - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        feats = {}
        x = self._preprocess(x)
        max_idx = max(self.content_layers)
        
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.content_layers:
                feats[i] = h
            if i >= max_idx:
                break
        return feats

    def forward(self, input_img, target_img):
        with torch.no_grad():
            target_feats = self._extract(target_img)
        input_feats = self._extract(input_img)
        
        loss = 0.0
        for k in self.content_layers:
            loss = loss + F.mse_loss(input_feats[k], target_feats[k])
            
        return loss
