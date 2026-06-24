import torch
import torch.nn as nn
import numpy as np
from torchdyn.core import NeuralODE
from scipy.optimize import linear_sum_assignment


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.register_buffer('W', torch.randn(embed_dim // 2) * scale)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim, t_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        self.t_proj = nn.Linear(t_dim, dim)
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x, t_embed):

        h = self.norm(x)
        
        t_feat = self.t_proj(self.act(t_embed))
        h = h + t_feat
        
        h = self.linear1(self.act(h))
        h = self.dropout(h)
        h = self.linear2(self.act(h))
        
        return x + h

class DeepResFlowNet(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, num_layers=8, dropout=0.0):
        super().__init__()
        
        self.t_embed_dim = 256
        self.t_encoder = GaussianFourierProjection(self.t_embed_dim)
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ResBlock(hidden_dim, self.t_embed_dim, dropout=dropout) 
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_dim)
     
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        feat = x[..., :-1]
        t = x[..., -1]
        
        t_embed = self.t_encoder(t) # [B, 256]
        
        h = self.input_proj(feat)

        for layer in self.layers:
            h = layer(h, t_embed)
      
        out = self.output_proj(self.output_norm(h))
        return out


class FlowMatchingModule(nn.Module):
    def __init__(self, dim=512, hidden_dim=2048, num_layers=8, lr=1e-4, use_ot=True, device='cuda'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.sigma = 0.001 
        self.use_ot = use_ot
        
        self.net = DeepResFlowNet(in_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999))

        self.node = NeuralODE(
            self._TorchWrapper(self.net), 
            solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5
        )

        self.register_buffer('data_mean', torch.zeros(1, dim))
        self.register_buffer('data_std', torch.ones(1, dim))
        self.stats_initialized = False

    def set_normalization_stats(self, mean_vec, std_vec):
        self.data_mean = mean_vec.reshape(1, -1).to(self.device).detach()
        self.data_std = std_vec.reshape(1, -1).to(self.device).detach()
        self.data_std = torch.clamp(self.data_std, min=1e-5)
        self.stats_initialized = True
        print(f"[FlowMatching] Stats Initialized. Mean range: [{self.data_mean.min():.2f}, {self.data_mean.max():.2f}]")

    def _normalize(self, z):
        if not self.stats_initialized: return z
        return (z - self.data_mean) / self.data_std

    def _denormalize(self, z_norm):
        if not self.stats_initialized: return z_norm
        return z_norm * self.data_std + self.data_mean

    class _TorchWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, t, x):
            n_batch = x.shape[0]
            if isinstance(t, torch.Tensor):
                t_tensor = t.reshape(-1)[0].expand(n_batch, 1) if t.numel() != n_batch else t.view(n_batch, 1)
            else:
                t_tensor = torch.tensor(t).type_as(x).expand(n_batch, 1)
            return self.model(torch.cat([x, t_tensor], dim=-1))

    def compute_ot_batch(self, x0, x1):
        cost_matrix = torch.cdist(x0, x1, p=2).pow(2) 
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_matrix_np)
        
        x1_aligned = x1[col_idx]
        return x0, x1_aligned



    def train_step(self, z_sem_A, z_sem_B, lambda_adv=0.01, train_generator=True):

        real_A = self._normalize(z_sem_A.to(self.device).detach())
        real_B = self._normalize(z_sem_B.to(self.device).detach())
        
        if train_generator:

            self.optimizer.zero_grad()
            if self.use_ot:
                x0, x1 = self.compute_ot_batch(real_A, real_B)
            else:
                x0, x1 = real_A, real_B
                
            t = torch.rand(x0.shape[0]).type_as(x0)
            t_expand = t.reshape(-1, *([1] * (x0.dim() - 1)))
            mu_t = t_expand * x1 + (1 - t_expand) * x0
            xt = mu_t + self.sigma * torch.randn_like(x0)
            ut = x1 - x0
            vt = self.net(torch.cat([xt, t[:, None]], dim=-1))
            loss_fm = torch.mean((vt - ut) ** 2)

            loss_total = loss_fm
            
            loss_total.backward()
            self.optimizer.step()
            
            loss_fm_val = loss_fm.item()

        return {"loss_fm": loss_fm_val}


    def transfer(self, z_sem_A):
        self.net.eval()
        with torch.no_grad():
            x0 = self._normalize(z_sem_A.to(self.device))
            traj = self.node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 2).type_as(x0)
            )
            return self._denormalize(traj[-1])