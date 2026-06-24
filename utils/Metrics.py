import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from sewar import vifp
from skimage import io as skio
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
from lpips import LPIPS
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

class Metrics:
    def __init__(self, res_save_path: str = None):
        # self.result_keys = ['SS', 'LC', 'VIF', 'Hist', 'lpips', 'fid' ]
        self.result_keys = ['SS', 'LC', 'Hist', 'lpips', 'fid', 'SSIM', 'PSNR', 'src_SSIM', 'src_PSNR']
        self.metrics = {k: 0.0 for k in self.result_keys}
        self.metrics['count'] = 0
        self.numpics = 20000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = LPIPS(net='alex').to(self.device).eval()
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device).eval()

        self.res_save_path = res_save_path

    @staticmethod
    def _to_uint8(x_hwc01: np.ndarray) -> np.ndarray:
        x = np.clip(x_hwc01, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    @staticmethod
    def _to_tensor_01(img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.ndim == 3 and img.shape[-1] == 3:
            # 转为 float32 [0,1]
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return t
        raise ValueError(f"Unsupported image shape: {img.shape}")


    @staticmethod
    def _to_numpy_hwc01(x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            if x.dim() == 4:
                x = x[0]
            if x.dim() == 3:
                x = x.detach().cpu().float()
                if x.min() < 0.0:
                    x = (x + 1.0) * 0.5
                x = x.clamp(0.0, 1.0)
                return x.permute(1, 2, 0).numpy()
        raise ValueError(f"Unsupported tensor shape for numpy conversion: {x.shape}")

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x
    
    @staticmethod
    def _to_gray01(x: torch.Tensor) -> np.ndarray:
        """NCHW/CHW tensor -> HxW numpy in [0,1] (BT.601)"""
        if x.dim() == 4: x = x[0]
        x = x.detach().cpu().float()
        if x.min() < 0: x = (x + 1) * 0.5
        x = x.clamp(0,1)
        r,g,b = x[0], x[1], x[2]
        y = 0.2989*r + 0.5870*g + 0.1140*b
        return y.numpy()
    
    @staticmethod
    def _z_norm01(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu, std = float(y.mean()), float(y.std())
        z = (y - mu) / (std + eps)
        z = (z - z.min()) / (z.max() - z.min() + eps)
        return z

    @staticmethod
    def _shift2d(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
        h, w = img.shape
        out = np.zeros_like(img)
        y0 = max(0, dy); y1 = min(h, h + dy)
        x0 = max(0, dx); x1 = min(w, w + dx)
        out[y0:y1, x0:x1] = img[y0-dy:y1-dy, x0-dx:x1-dx]
        return out

    @staticmethod
    def _lin_align(y: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        my, mx = float(y.mean()), float(x.mean())
        vy = float(((y - my) ** 2).mean())
        if vy < eps:
            a = 1.0
            b = mx - my
        else:
            cov = float(((y - my) * (x - mx)).mean())
            a = cov / (vy + eps)
            b = mx - a * my
        y_aligned = np.clip(a * y + b, 0.0, 1.0)
        return y_aligned

    def _lpips_calculation(self, output: torch.Tensor, target: torch.Tensor) -> float:
        out = output.clone().float()
        tgt = target.clone().float()
        tgt = self._match_size(tgt, out)

        out = out.to(self.device)
        tgt = tgt.to(self.device)
        if out.min() >= 0 and out.max() <= 1: out = out * 2 - 1
        if tgt.min() >= 0 and tgt.max() <= 1: tgt = tgt * 2 - 1
        with torch.no_grad():
            return self.lpips(out, tgt).mean().item()

    def _ssim_calculation(self, output: torch.Tensor, target: torch.Tensor) -> float:
        x = self._to_numpy_hwc01(output)
        y = self._to_numpy_hwc01(self._match_size(target, output))
        return float(ssim(x, y, channel_axis=-1, data_range=1.0))
    def _ssim_gray_calc(self, output: torch.Tensor, target: torch.Tensor) -> float:
        out_gray = self._to_gray01(output)
        tgt_gray = self._to_gray01(self._match_size(target, output))
        return float(ssim(out_gray, tgt_gray, data_range=1.0))

    def _psnr_gray_calc(self, output: torch.Tensor, target: torch.Tensor) -> float:
        out_gray = self._to_gray01(output)
        tgt_gray = self._to_gray01(self._match_size(target, output))
        return float(psnr(tgt_gray, out_gray, data_range=1.0))

    def _psnr_calculation(self, output: torch.Tensor, target: torch.Tensor) -> float:
        x = self._to_numpy_hwc01(output)
        y = self._to_numpy_hwc01(self._match_size(target, output))
        return float(psnr(x, y, data_range=1.0))

    def update_metrics(self, output_ten: torch.Tensor, target_ten: torch.Tensor, source_ten: torch.Tensor, output_np: np.ndarray, target_np: np.ndarray, source_np: np.ndarray):
        self.metrics['lpips'] += self._lpips_calculation(output_ten, target_ten)

        h, w = output_np.shape[:2]

        if source_np.shape[:2] != (h, w):
            source_np = cv.resize(source_np, (w, h), interpolation=cv.INTER_LINEAR)

        if target_np.shape[:2] != (h, w):
            target_np = cv.resize(target_np, (w, h), interpolation=cv.INTER_LINEAR)

        self.metrics['SS'] += self.Structure_Similarity(output_np, source_np)
        self.metrics['LC'] += self.Luminance_constrast(output_np, target_np)
        self.metrics['Hist'] += self.histograms_correlation(output_np, target_np)
        self.metrics['SSIM'] += self._ssim_calculation(output_ten, target_ten)
        self.metrics['PSNR'] += self._psnr_calculation(output_ten, target_ten)
        self.metrics['src_SSIM'] += self._ssim_gray_calc(output_ten, self._to_tensor_01(source_np))
        self.metrics['src_PSNR'] += self._psnr_gray_calc(output_ten, self._to_tensor_01(source_np))


    @staticmethod
    def _list_images(dir_path: str):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)
                if f.lower().endswith(exts)])

    @staticmethod
    def _index_by_stem(dir_path: str, length_limit: int = 10000):
        files = Metrics._list_images(dir_path)[:length_limit]
        idx = {}
        warned = False
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            if stem in idx and not warned:
                print(f"[Metrics][WARN] duplicate stem '{stem}' in {dir_path}, keeping first.")
                warned = True
            idx.setdefault(stem, p)
        return idx

    def _files_features(self, paths, batch_size: int = 32) -> np.ndarray:
        feats = []
        target_size = (256, 256)       
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch = []
                for f in paths[i:i+batch_size]:
                    img = skio.imread(f)
                    t = self._to_tensor_01(img)  # [1,3,H,W], [0,1]
                    if t.shape[-2:] != target_size:
                        t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
                    batch.append(t)
                if not batch:
                    continue
                x = torch.cat(batch, dim=0).to(self.device)
                act = self.inception(x)[0].view(x.size(0), -1)  # [B,2048]
                feats.append(act.cpu())
        return (torch.cat(feats, dim=0).numpy() if len(feats) > 0
                else np.zeros((0, 2048), dtype=np.float32))

    def _fid_between_lists(self, paths1, paths2) -> float:
        f1 = self._files_features(paths1)
        f2 = self._files_features(paths2)
        if f1.shape[0] == 0 or f2.shape[0] == 0:
            raise ValueError("Empty feature set for FID.")
        mu1, mu2 = f1.mean(0), f2.mean(0)
        sigma1 = np.cov(f1, rowvar=False)
        sigma2 = np.cov(f2, rowvar=False)
        return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

    def display_result(self) -> str:
        line = "\n" + "=" * 100 + "\n"
        for k in self.metrics.keys(): line += f"{k:>10} "
        line += "\n"
        for k, v in self.metrics.items():
            val = int(v) if k == 'count' else (v / max(1, self.metrics['count']) if k != 'fid' else v)
            line += f"{val:10.4f} " if k != 'count' else f"{val:10d} "
        line += "\n" + "=" * 100 + "\n"
        return line
   
    def Structure_Similarity(self, Generated_img, Source_img, c3 = 1e-6, use_gradient = True) -> float:
        def safe_gray(img):
            if img.ndim == 2: return img
            if img.shape[2] == 3: return cv.cvtColor(img, cv.COLOR_RGB2GRAY) 
            return img[:,:,0]
        if use_gradient:

            def gradient_map(img):
                gray = safe_gray(img)
                grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
                grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
                return np.sqrt(grad_x**2 + grad_y**2)
            
            S_x = gradient_map(Source_img)
            S_y = gradient_map(Generated_img)
        else:
            S_x = safe_gray(Source_img).astype(np.float64)
            S_y = safe_gray(Generated_img).astype(np.float64)
        
        cov = np.mean((S_x - S_x.mean()) * (S_y - S_y.mean()))
        return (cov+c3) / (S_x.std()*S_y.std() + c3)

    def Luminance_constrast(self,Generated_img, Target_img) -> float:
        L_gt = cv.cvtColor(Target_img, cv.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
        L_pred = cv.cvtColor(Generated_img, cv.COLOR_BGR2LAB)[:,:,0].astype(np.float32)

        mu_gt = L_gt.mean()
        mu_pred = L_pred.mean()
        sigma_gt = L_gt.std()
        sigma_pred = L_pred.std()

        L_max = 255.0
        c1 = (0.01 * L_max) ** 2
        c2 = (0.03 * L_max) ** 2

        luminance = (2 * mu_gt * mu_pred + c1) / (mu_gt**2 + mu_pred**2 + c1)
        contrast = (2 * sigma_gt * sigma_pred + c2) / (sigma_gt**2 + sigma_pred**2 + c2)


        return luminance * contrast        

    def histograms_correlation(self, Generated_img, Target_img) -> float:
        if Generated_img.ndim == 2:
            channels = [0]
            histSize = [256] 
            ranges = [0, 256]
        else:
            channels = [0, 1, 2]
            histSize = [8, 8, 8] 
            ranges = [0, 256, 0, 256, 0, 256]

        hist_pred = cv.calcHist([Generated_img], channels, None, histSize, ranges)
        hist_gt = cv.calcHist([Target_img], channels, None, histSize, ranges)

        cv.normalize(hist_pred, hist_pred)
        cv.normalize(hist_gt, hist_gt)

        correlation = cv.compareHist(hist_pred, hist_gt, cv.HISTCMP_CORREL)
        return float(correlation)

    def metrics_loop(self, A_Domine_dir: str, B_Domine_dir: str, fakeB_Domine_dir: str, numpics: int = 10000):
        A_idx = self._index_by_stem(A_Domine_dir, length_limit = numpics)
        B_idx = self._index_by_stem(B_Domine_dir, length_limit = numpics)
        F_idx = self._index_by_stem(fakeB_Domine_dir, length_limit = numpics)
        
        common_keys = sorted(set(A_idx.keys()) & set(B_idx.keys()) & set(F_idx.keys()))
        print(f"[Metrics] A={len(A_idx)}, B={len(B_idx)}, fakeB={len(F_idx)}, matched={len(common_keys)}")
        if len(common_keys) == 0:
            print("[Metrics] No matched filenames across A/B/fakeB. Check naming.")
            return
        vis_dir = os.path.join(fakeB_Domine_dir, "_vis")
        os.makedirs(vis_dir, exist_ok=True)

        matched_B_paths, matched_F_paths = [], []

        for i, k in tqdm(list(enumerate(common_keys)), desc="Calculating Metrics"):
            source_np = skio.imread(A_idx[k]) 
            target_np = skio.imread(B_idx[k])
            output_np = skio.imread(F_idx[k])
            source_ten = self._to_tensor_01(source_np)
            target_ten = self._to_tensor_01(target_np)
            output_ten = self._to_tensor_01(output_np)
            self.update_metrics(output_ten, target_ten, source_ten, output_np, target_np, source_np)
            self.metrics['count'] += 1

            matched_B_paths.append(B_idx[k])
            matched_F_paths.append(F_idx[k])

            if i % 50 == 0:
                A_vis = self._to_numpy_hwc01(self._match_size(source_ten, output_ten))
                F_vis = self._to_numpy_hwc01(output_ten)
                B_vis = self._to_numpy_hwc01(self._match_size(target_ten, output_ten))
                A_u8 = self._to_uint8(A_vis)
                F_u8 = self._to_uint8(F_vis)
                B_u8 = self._to_uint8(B_vis)
                row = np.concatenate([A_u8, F_u8, B_u8], axis=1)
                out_path = os.path.join(vis_dir, f"{i:05d}_{k}.png")
                try:
                    skio.imsave(out_path, row)
                except Exception:
                    from imageio import imwrite
                    imwrite(out_path, row)
                print(f"[Metrics][vis] saved: {out_path}")
        try:
            self.metrics['fid'] = self._fid_between_lists(matched_B_paths, matched_F_paths)
        except Exception as e:
            print(f"[WARN] FID calculation failed: {e}")

        res = self.display_result()
        print(res)
        if self.res_save_path is not None:
            txt_path = os.path.join(self.res_save_path, "metrics.txt")
            with open(txt_path, 'w') as f:
                f.write(res)
                f.write("\n")
            print(f"[Metrics] Results saved to: {txt_path}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import argparse

    parser = argparse.ArgumentParser(description="Calculate image translation metrics.")
    parser.add_argument('--num', type=int, default=1600, help='maximum number of pictures to evaluate')
    parser.add_argument("--A_dir", type=str, help="Directory of source domain A images.")
    parser.add_argument("--B_dir", type=str, help="Directory of target domain B images.")
    parser.add_argument("--pred_dir", type=str, help="Directory of generated images.")
    
    args = parser.parse_args()

    metrics = Metrics()
    metrics.metrics_loop(args.A_dir, args.B_dir, args.pred_dir, args.num)