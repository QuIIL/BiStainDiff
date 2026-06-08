#!/usr/bin/env python3
import os
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image

# Metrics-related imports
try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None

from dataset import CSVPairedDataset
from models.autoencoder import vae_models
from utils import preprocess_imgs_vae

# Local imports for LPIPS and Inception features (for FID)
from loss.lpips import LPIPS
from torchvision.models import inception_v3, Inception_V3_Weights
import torch_fidelity

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, images_list):
        self.images = images_list
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx]

def tensor_to_pil(t):
    """
    Converts a torch.Tensor [C, H, W] in range [0, 1] to a PIL Image.
    """
    t = t.clamp(0.0, 1.0)
    arr = t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(arr)

def calculate_numpy_ssim(gt, recon):
    """
    Computes SSIM between two [C, H, W] tensors in range [0,1]
    using skimage or falling back to simple channel-wise formula.
    """
    gt_np = gt.permute(1, 2, 0).cpu().numpy()
    rec_np = recon.permute(1, 2, 0).cpu().numpy()
    
    if ssim_fn is not None:
        # skimage ssim (multi_channel=True has been deprecated in newer versions, use channel_axis=-1)
        try:
            return ssim_fn(gt_np, rec_np, data_range=1.0, channel_axis=-1)
        except TypeError:
            return ssim_fn(gt_np, rec_np, data_range=1.0, multichannel=True)
    else:
        # Simple native fallback if skimage is missing
        # Compute dynamic range constant
        K1, K2 = 0.01, 0.03
        L = 1.0
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        mu_x = gt.mean(dim=[1,2])
        mu_y = recon.mean(dim=[1,2])
        
        var_x = gt.var(dim=[1,2])
        var_y = recon.var(dim=[1,2])
        
        cov_xy = torch.mean((gt - mu_x[:, None, None]) * (recon - mu_y[:, None, None]), dim=[1,2])
        
        ssim_val = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))
        return ssim_val.mean().item()

# Matrix square root for FID
def matrix_sqrt(matrix):
    """
    Computes matrix square root for symmetric positive-definite matrices.
    """
    # Use torch.linalg.eigh since covariance matrices are symmetric
    vals, vecs = torch.linalg.eigh(matrix)
    # Clamp small eigenvalues to avoid numerical instability
    vals = torch.clamp(vals, min=0.0)
    sqrt_vals = torch.sqrt(vals)
    return vecs @ torch.diag(sqrt_vals) @ vecs.T

def calculate_fid_from_features(mu1, cov1, mu2, cov2):
    """
    Calculates the Fréchet Inception Distance between two feature distributions:
    FID = ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2*sqrt(cov1_sqrt @ cov2 @ cov1_sqrt))
    """
    # Convert numpy-like arrays if necessary
    if not torch.is_tensor(mu1):
        mu1 = torch.tensor(mu1)
    else:
        mu1 = mu1.detach().clone()
    if not torch.is_tensor(mu2):
        mu2 = torch.tensor(mu2)
    else:
        mu2 = mu2.detach().clone()
    if not torch.is_tensor(cov1):
        cov1 = torch.tensor(cov1)
    else:
        cov1 = cov1.detach().clone()
    if not torch.is_tensor(cov2):
        cov2 = torch.tensor(cov2)
    else:
        cov2 = cov2.detach().clone()

    diff = mu1 - mu2
    offset_loss = torch.sum(diff ** 2)

    # Compute symmetric product: cov1_sqrt @ cov2 @ cov1_sqrt
    cov1_sqrt = matrix_sqrt(cov1)
    symmetric_prod = cov1_sqrt @ cov2 @ cov1_sqrt
    cov_sqrt = matrix_sqrt(symmetric_prod)
    
    # Trace term
    trace_loss = torch.trace(cov1 + cov2 - 2.0 * cov_sqrt)
    
    fid_val = offset_loss + trace_loss
    return fid_val.clamp(min=0.0).item()


def main(args):
    # Detect multiple CUDA devices if "cuda" or default is specified
    if args.device.startswith("cuda") and torch.cuda.device_count() >= 2:
        device_vae = torch.device("cuda:0")
        device_metrics = torch.device("cuda:1")
        print(f"[*] Multi-GPU environment detected. Splitting models:")
        print(f"    - VAE models: {device_vae}")
        print(f"    - Evaluation metrics (LPIPS, Inception, FID): {device_metrics}")
    else:
        device_vae = torch.device(args.device)
        device_metrics = torch.device(args.device)
        print(f"[*] Device environment: {device_vae}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("="*60)
    print(" [VAE Academic Comparison Tool] Starting Evaluation...")
    print("  Includes: MSE, L1, PSNR, SSIM, LPIPS, FID")
    print("="*60)

    # ---- 1. Load VAE Models ----
    vae_type = args.vae_type
    
    # 1a. Load Pre-trained VAE (Base)
    print(f"[*] Loading Pre-trained VAE from '{args.base_vae_ckpt}'...")
    vae_base = vae_models[vae_type]().to(device_vae)
    base_ckpt = torch.load(args.base_vae_ckpt, map_location=device_vae, weights_only=False)
    missing_base, unexpected_base = vae_base.load_state_dict(base_ckpt["vae"] if "vae" in base_ckpt else base_ckpt, strict=False)
    if missing_base or unexpected_base:
        print(f"    [!] Base VAE load warning -> Missing: {len(missing_base)} keys, Unexpected: {len(unexpected_base)} keys")
    vae_base.float()
    vae_base.eval()

    # 1b. Load Fine-tuned VAE
    print(f"[*] Loading Fine-tuned VAE from checkpoint '{args.ft_vae_ckpt}'...")
    vae_ft = vae_models[vae_type]().to(device_vae)
    ft_ckpt = torch.load(args.ft_vae_ckpt, map_location=device_vae, weights_only=False)
    missing_ft, unexpected_ft = vae_ft.load_state_dict(ft_ckpt["vae"] if "vae" in ft_ckpt else ft_ckpt, strict=False)
    if missing_ft or unexpected_ft:
        print(f"    [!] Fine-tuned VAE load warning -> Missing: {len(missing_ft)} keys, Unexpected: {len(unexpected_ft)} keys")
    vae_ft.float()
    vae_ft.eval()

    # ---- 2. Initialize Perceptual Metrics ----
    print("[*] Initializing LPIPS Model...")
    lpips_evaluator = LPIPS().to(device_metrics).eval()

    print("[*] Initializing Inception-v3 Model for FID calculation...")
    # Load Inception model for feature extraction
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device_metrics)
    inception.eval()
    
    # We want features from the pool3 layer (2048 dimensions)
    # To do this cleanly, register a forward hook or manipulate features:
    inception_features = []
    def hook_fn(module, input, output):
        inception_features.append(output.squeeze(-1).squeeze(-1))
    
    # InceptionV3's global average pooling is right before fc
    hook = inception.avgpool.register_forward_hook(hook_fn)

    # We want features from the Mixed_6b layer (768 dimensions) for sFID
    spatial_features = []
    def spatial_hook_fn(module, input, output):
        spatial_features.append(output.mean(dim=(2, 3)))
        
    spatial_hook = inception.Mixed_6b.register_forward_hook(spatial_hook_fn)

    print("[✓] VAE Models & Perceptual metrics initialized.")

    # ---- 3. Dataset Preparation ----
    dataset = CSVPairedDataset(
        csv_path=args.csv_path,
        image_size=args.resolution,
        no_crop=args.no_crop,
        center_crop=True
    )
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[✓] Loaded dataset. Total samples: {num_samples} (Batch Size: {args.batch_size})")
    print(f"    Evaluating target domain: {'HE (a)' if args.target_domain == 'a' else 'IHC (b)'}")

    # ---- 4. Evaluation Loop ----
    all_base_mse, all_base_l1, all_base_ssim, all_base_lpips = [], [], [], []
    all_ft_mse, all_ft_l1, all_ft_ssim, all_ft_lpips         = [], [], [], []

    # Arrays to accumulate Inception features for FID
    gt_features_list = []
    base_features_list = []
    ft_features_list = []

    # Arrays to accumulate spatial Inception features for sFID
    gt_spatial_list = []
    base_spatial_list = []
    ft_spatial_list = []

    # Arrays to accumulate raw uint8 images for Inception Score, Precision, and Recall
    gt_images_list = []
    base_images_list = []
    ft_images_list = []

    global_idx = 0
    pbar = tqdm(loader, desc="Evaluating Metrics")

    for batch in pbar:
        raw_a = batch[0].to(device_vae)  # Source HE [B, C, H, W]
        raw_b = batch[1].to(device_vae)  # Target IHC [B, C, H, W]
        B = raw_a.shape[0]

        # Preprocess to [-1, 1] range for VAE
        proc_a = preprocess_imgs_vae(raw_a)
        proc_b = preprocess_imgs_vae(raw_b)

        # Decide which domain we want to evaluate (usually a=HE, b=IHC)
        gt_imgs = (proc_a + 1) / 2.0 if args.target_domain == "a" else (proc_b + 1) / 2.0
        inputs  = proc_a if args.target_domain == "a" else proc_b

        with torch.no_grad():
            # Base VAE reconstruction (mode-based for noise-free evaluation)
            posterior_base = vae_base.encode(inputs)
            base_recon = vae_base.decode(posterior_base.mode()).sample
            base_recon_01 = (base_recon + 1) / 2.0

            # Fine-tuned VAE reconstruction (mode-based for noise-free evaluation)
            posterior_ft = vae_ft.encode(inputs)
            ft_recon = vae_ft.decode(posterior_ft.mode()).sample
            ft_recon_01 = (ft_recon + 1) / 2.0

            # Compute LPIPS (Evaluator expects [-1, 1] input range)
            base_lpips_batch = lpips_evaluator(base_recon.clamp(-1.0, 1.0).to(device_metrics), inputs.to(device_metrics))
            ft_lpips_batch   = lpips_evaluator(ft_recon.clamp(-1.0, 1.0).to(device_metrics), inputs.to(device_metrics))

            # --- Inception Feature Extraction for FID ---
            # Inception expects 299x299 resized inputs in range [0, 1]
            gt_299 = torch.nn.functional.interpolate(gt_imgs, size=(299, 299), mode="bilinear", align_corners=False).to(device_metrics)
            base_299 = torch.nn.functional.interpolate(base_recon_01, size=(299, 299), mode="bilinear", align_corners=False).to(device_metrics)
            ft_299 = torch.nn.functional.interpolate(ft_recon_01, size=(299, 299), mode="bilinear", align_corners=False).to(device_metrics)

            # Ground Truth Features
            inception_features.clear()
            spatial_features.clear()
            _ = inception(gt_299)
            gt_features_list.append(inception_features[0].cpu())
            gt_spatial_list.append(spatial_features[0].cpu())

            # Base VAE Features
            inception_features.clear()
            spatial_features.clear()
            _ = inception(base_299)
            base_features_list.append(inception_features[0].cpu())
            base_spatial_list.append(spatial_features[0].cpu())

            # Fine-tuned VAE Features
            inception_features.clear()
            spatial_features.clear()
            _ = inception(ft_299)
            ft_features_list.append(inception_features[0].cpu())
            ft_spatial_list.append(spatial_features[0].cpu())

        # Process sample-level metrics (MSE, L1, SSIM, LPIPS lists)
        for b_idx in range(B):
            gt_img = gt_imgs[b_idx]
            base_rec = base_recon_01[b_idx]
            ft_rec = ft_recon_01[b_idx]

            # Convert and save in uint8 [0, 255] for torch-fidelity
            gt_images_list.append((gt_img * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu())
            base_images_list.append((base_rec * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu())
            ft_images_list.append((ft_rec * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu())

            # MSE, L1
            b_mse = torch.mean((gt_img - base_rec)**2).item()
            b_l1  = torch.mean(torch.abs(gt_img - base_rec)).item()
            f_mse = torch.mean((gt_img - ft_rec)**2).item()
            f_l1  = torch.mean(torch.abs(gt_img - ft_rec)).item()

            all_base_mse.append(b_mse)
            all_base_l1.append(b_l1)
            all_ft_mse.append(f_mse)
            all_ft_l1.append(f_l1)

            # SSIM
            all_base_ssim.append(calculate_numpy_ssim(gt_img, base_rec))
            all_ft_ssim.append(calculate_numpy_ssim(gt_img, ft_rec))

            # LPIPS
            all_base_lpips.append(base_lpips_batch[b_idx].item())
            all_ft_lpips.append(ft_lpips_batch[b_idx].item())

            # Save visual comparison strips
            if str(args.save_images).lower() == 'true':
                if args.save_max <= 0 or global_idx < args.save_max:
                    comp_strip = torch.cat([gt_img, base_rec, ft_rec], dim=2)
                    comp_pil = tensor_to_pil(comp_strip.cpu())
                    save_path = os.path.join(args.out_dir, f"sample_{global_idx:04d}_gt_vs_base_vs_ft.png")
                    comp_pil.save(save_path)

            global_idx += 1

    # Remove the inception hooks
    hook.remove()
    spatial_hook.remove()

    # ---- 5. Calculate Final Averages ----
    avg_base_mse = np.mean(all_base_mse)
    avg_base_l1  = np.mean(all_base_l1)
    avg_base_ssim = np.mean(all_base_ssim)
    avg_base_lpips = np.mean(all_base_lpips)
    avg_base_psnr = 10 * np.log10(1.0 / avg_base_mse) if avg_base_mse > 0 else float("inf")

    avg_ft_mse = np.mean(all_ft_mse)
    avg_ft_l1  = np.mean(all_ft_l1)
    avg_ft_ssim = np.mean(all_ft_ssim)
    avg_ft_lpips = np.mean(all_ft_lpips)
    avg_ft_psnr = 10 * np.log10(1.0 / avg_ft_mse) if avg_ft_mse > 0 else float("inf")

    # ---- 6. Calculate gFID (Global Fréchet Inception Distance) ----
    print("[*] Computing gFID Scores (covariance analysis)...")
    gt_feats = torch.cat(gt_features_list, dim=0)     # [N, 2048]
    base_feats = torch.cat(base_features_list, dim=0) # [N, 2048]
    ft_feats = torch.cat(ft_features_list, dim=0)     # [N, 2048]

    # Ground Truth stats
    mu_gt = gt_feats.mean(dim=0)
    cov_gt = torch.from_numpy(np.cov(gt_feats.numpy(), rowvar=False)).to(torch.float32)

    # Base VAE stats
    mu_base = base_feats.mean(dim=0)
    cov_base = torch.from_numpy(np.cov(base_feats.numpy(), rowvar=False)).to(torch.float32)

    # Fine-tuned VAE stats
    mu_ft = ft_feats.mean(dim=0)
    cov_ft = torch.from_numpy(np.cov(ft_feats.numpy(), rowvar=False)).to(torch.float32)

    # Calculate final gFID values
    gfid_base = calculate_fid_from_features(mu_gt, cov_gt, mu_base, cov_base)
    gfid_ft   = calculate_fid_from_features(mu_gt, cov_gt, mu_ft, cov_ft)

    # ---- 7. Calculate sFID (Spatial Fréchet Inception Distance) ----
    print("[*] Computing sFID Scores (covariance analysis)...")
    gt_spatial_feats = torch.cat(gt_spatial_list, dim=0)     # [N, 768]
    base_spatial_feats = torch.cat(base_spatial_list, dim=0) # [N, 768]
    ft_spatial_feats = torch.cat(ft_spatial_list, dim=0)     # [N, 768]

    # Ground Truth stats
    mu_gt_spatial = gt_spatial_feats.mean(dim=0)
    cov_gt_spatial = torch.from_numpy(np.cov(gt_spatial_feats.numpy(), rowvar=False)).to(torch.float32)

    # Base VAE stats
    mu_base_spatial = base_spatial_feats.mean(dim=0)
    cov_base_spatial = torch.from_numpy(np.cov(base_spatial_feats.numpy(), rowvar=False)).to(torch.float32)

    # Fine-tuned VAE stats
    mu_ft_spatial = ft_spatial_feats.mean(dim=0)
    cov_ft_spatial = torch.from_numpy(np.cov(ft_spatial_feats.numpy(), rowvar=False)).to(torch.float32)

    # Calculate final sFID values
    sfid_base = calculate_fid_from_features(mu_gt_spatial, cov_gt_spatial, mu_base_spatial, cov_base_spatial)
    sfid_ft   = calculate_fid_from_features(mu_gt_spatial, cov_gt_spatial, mu_ft_spatial, cov_ft_spatial)

    # ---- 8. Calculate IS, Precision, and Recall ----
    print("[*] Computing Inception Score, Precision, and Recall via torch-fidelity...")
    gt_dataset = InMemoryDataset(gt_images_list)
    base_dataset = InMemoryDataset(base_images_list)
    ft_dataset = InMemoryDataset(ft_images_list)

    if device_metrics.type == "cuda":
        torch.cuda.set_device(device_metrics)

    # Reference Ground Truth Inception Score
    gt_fidelity = torch_fidelity.calculate_metrics(
        input1=gt_dataset,
        cuda=device_metrics.type == "cuda",
        isc=True,
        verbose=False
    )
    is_gt_mean = gt_fidelity['inception_score_mean']
    is_gt_std  = gt_fidelity['inception_score_std']

    # Base VAE
    base_fidelity = torch_fidelity.calculate_metrics(
        input1=base_dataset,
        input2=gt_dataset,
        cuda=device_metrics.type == "cuda",
        isc=True,
        prc=True,
        verbose=False
    )
    is_base_mean = base_fidelity['inception_score_mean']
    is_base_std  = base_fidelity['inception_score_std']
    prec_base    = base_fidelity['precision']
    rec_base     = base_fidelity['recall']

    # Fine-tuned VAE
    ft_fidelity = torch_fidelity.calculate_metrics(
        input1=ft_dataset,
        input2=gt_dataset,
        cuda=device_metrics.type == "cuda",
        isc=True,
        prc=True,
        verbose=False
    )
    is_ft_mean = ft_fidelity['inception_score_mean']
    is_ft_std  = ft_fidelity['inception_score_std']
    prec_ft    = ft_fidelity['precision']
    rec_ft     = ft_fidelity['recall']

    # ---- 9. Improvements percentage (safely computed to prevent ZeroDivisionError) ----
    def safe_pct(base, ft):
        if base is None or np.isnan(base) or np.isinf(base) or base == 0:
            return 0.0
        return ((base - ft) / base) * 100

    def safe_pct_ssim(base, ft):
        if base is None or np.isnan(base) or np.isinf(base) or base == 0:
            return 0.0
        return ((ft - base) / base) * 100

    pct_mse = safe_pct(avg_base_mse, avg_ft_mse)
    pct_l1  = safe_pct(avg_base_l1, avg_ft_l1)
    psnr_diff = avg_ft_psnr - avg_base_psnr
    pct_ssim = safe_pct_ssim(avg_base_ssim, avg_ft_ssim)
    pct_lpips = safe_pct(avg_base_lpips, avg_ft_lpips)
    pct_gfid = safe_pct(gfid_base, gfid_ft)
    pct_sfid = safe_pct(sfid_base, sfid_ft)
    pct_prec = safe_pct_ssim(prec_base, prec_ft)
    pct_rec = safe_pct_ssim(rec_base, rec_ft)
    is_diff = is_ft_mean - is_base_mean

    # Save summary report to txt file
    report_path = os.path.join(args.out_dir, args.report_name)
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write(" Comprehensive VAE Academic Comparison Report\n")
        f.write("="*60 + "\n")
        f.write(f"Evaluated Samples  : {num_samples}\n")
        f.write(f"Image Resolution   : {args.resolution}x{args.resolution}\n")
        f.write(f"Target Domain      : {'HE (a)' if args.target_domain == 'a' else 'IHC (b)'}\n\n")
        
        f.write("[1] Pre-trained VAE (Original Base)\n")
        f.write(f"  - Mean Squared Error (MSE)  : {avg_base_mse:.6f}\n")
        f.write(f"  - L1 Reconstruction Loss    : {avg_base_l1:.6f}\n")
        f.write(f"  - Peak Signal-to-Noise Ratio: {avg_base_psnr:.4f} dB\n")
        f.write(f"  - Structural Similarity(SSIM): {avg_base_ssim:.4f}\n")
        f.write(f"  - LPIPS Perceptual Distance : {avg_base_lpips:.4f}\n")
        f.write(f"  - Global FID (gFID) Score   : {gfid_base:.4f}\n")
        f.write(f"  - Spatial FID (sFID) Score  : {sfid_base:.4f}\n")
        f.write(f"  - Inception Score (IS)      : {is_base_mean:.4f} +/- {is_base_std:.4f}\n")
        f.write(f"  - Precision (Prec)          : {prec_base:.4f}\n")
        f.write(f"  - Recall (Rec)              : {rec_base:.4f}\n\n")
        
        f.write("[2] Fine-tuned VAE (Domain Aligned)\n")
        f.write(f"  - Mean Squared Error (MSE)  : {avg_ft_mse:.6f}\n")
        f.write(f"  - L1 Reconstruction Loss    : {avg_ft_l1:.6f}\n")
        f.write(f"  - Peak Signal-to-Noise Ratio: {avg_ft_psnr:.4f} dB\n")
        f.write(f"  - Structural Similarity(SSIM): {avg_ft_ssim:.4f}\n")
        f.write(f"  - LPIPS Perceptual Distance : {avg_ft_lpips:.4f}\n")
        f.write(f"  - Global FID (gFID) Score   : {gfid_ft:.4f}\n")
        f.write(f"  - Spatial FID (sFID) Score  : {sfid_ft:.4f}\n")
        f.write(f"  - Inception Score (IS)      : {is_ft_mean:.4f} +/- {is_ft_std:.4f}\n")
        f.write(f"  - Precision (Prec)          : {prec_ft:.4f}\n")
        f.write(f"  - Recall (Rec)              : {rec_ft:.4f}\n\n")

        f.write("[Reference] Ground Truth (GT)\n")
        f.write(f"  - Inception Score (IS)      : {is_gt_mean:.4f} +/- {is_gt_std:.4f}\n\n")
        
        f.write("[3] Quantitative Improvements by Fine-Tuning\n")
        f.write(f"  - MSE reduction             : {pct_mse:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - L1 Loss reduction         : {pct_l1:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - PSNR Quality gain         : {psnr_diff:+.4f} dB (Positive is improvement!)\n")
        f.write(f"  - SSIM Increase             : {pct_ssim:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - LPIPS reduction           : {pct_lpips:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - gFID Score reduction      : {pct_gfid:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - sFID Score reduction      : {pct_sfid:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - Inception Score Gain      : {is_diff:+.4f} (Positive is improvement!)\n")
        f.write(f"  - Precision Increase        : {pct_prec:+.2f}% (Positive is improvement!)\n")
        f.write(f"  - Recall Increase           : {pct_rec:+.2f}% (Positive is improvement!)\n")
        f.write("="*60 + "\n")

    # Print summary to console
    print("\n" + "="*60)
    print(" [✓] Academic Evaluation Complete! Summary Results:")
    print("="*60)
    print(f" ▶ Pre-trained VAE  | PSNR: {avg_base_psnr:.2f} dB | SSIM: {avg_base_ssim:.4f} | LPIPS: {avg_base_lpips:.4f} | gFID: {gfid_base:.2f} | sFID: {sfid_base:.2f}")
    print(f" ▶ Fine-tuned VAE   | PSNR: {avg_ft_psnr:.2f} dB  | SSIM: {avg_ft_ssim:.4f} | LPIPS: {avg_ft_lpips:.4f} | gFID: {gfid_ft:.2f} | sFID: {sfid_ft:.2f}")
    print(f" ▶ Pre-trained VAE  | IS: {is_base_mean:.2f} | Precision: {prec_base:.3f} | Recall: {rec_base:.3f}")
    print(f" ▶ Fine-tuned VAE   | IS: {is_ft_mean:.2f} | Precision: {prec_ft:.3f} | Recall: {rec_ft:.3f}")
    print("-"*60)
    print(f"  ▷ Fine-tuning Performance Leap (Positive is improvement):")
    print(f"      - PSNR quality gain : {psnr_diff:+.2f} dB  (Sharper textures)")
    print(f"      - SSIM Increase     : {pct_ssim:+.2f}%     (Better structural retention)")
    print(f"      - LPIPS reduction   : {pct_lpips:+.2f}%     (Closer to human perception)")
    print(f"      - gFID reduction    : {pct_gfid:+.2f}%     (Higher image realness)")
    print(f"      - sFID reduction    : {pct_sfid:+.2f}%     (Better spatial coherence)")
    print(f"      - Prec increase     : {pct_prec:+.2f}%     (Better sample quality)")
    print(f"      - Rec increase      : {pct_rec:+.2f}%     (Better sample diversity)")
    print("="*60)
    if str(args.save_images).lower() == 'true':
        print(f"\n[✓] Results & visual strips saved to: '{args.out_dir}'")
    else:
        print(f"\n[✓] Results saved to: '{args.out_dir}'")
    print(f"    Academic Report generated: '{report_path}'")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone VAE Academic Comparison Tool")
    
    parser.add_argument("--ft-vae-ckpt",    required=True, help="Path to fine-tuned SiTBB checkpoint (containing 'vae' state dict)")
    parser.add_argument("--csv-path",       required=True, help="Path to CSV dataset containing paired paths")
    parser.add_argument("--base-vae-ckpt",  default="./pretrained/sdvae-f8d4/sdvae-f8d4.pt", help="Path to pure pre-trained VAE weights")
    parser.add_argument("--vae-type",       default="f8d4", choices=["f8d4", "f16d32"])
    parser.add_argument("--resolution",     type=int, default=1024, help="Target image resolution")
    parser.add_argument("--target-domain",  default="b", choices=["a", "b"], help="'a' for HE, 'b' for IHC reconstruction")
    parser.add_argument("--no-crop",        action="store_true", default=False, help="Disable cropping/resizing and keep original image size")
    
    parser.add_argument("--num-samples",    type=int, default=0, help="Number of samples to evaluate (0 = entire dataset)")
    parser.add_argument("--batch-size",     type=int, default=4, help="Batch size for execution")
    parser.add_argument("--save-images",    type=str, default="False", help="Save visual comparison images ('True' or 'False')")
    parser.add_argument("--save-max",       type=int, default=0, help="Maximum number of comparison images to save visually (0 = all)")
    parser.add_argument("--out-dir",        default="outputs/vae_comparison_standalone", help="Output directory")
    parser.add_argument("--report-name",    default="vae_comparison_report.txt", help="Filename of the saved TXT report")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")

    args = parser.parse_args()
    main(args)
