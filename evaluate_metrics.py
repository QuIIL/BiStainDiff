import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from scipy.stats import pearsonr

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import lpips
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

import sys
# DISTS_pytorch local import
sys.path.insert(0, "/home/quiil/jiwoo/DISTS")
try:
    from DISTS_pytorch import DISTS
except ImportError:
    print("Warning: DISTS_pytorch not found at /home/quiil/jiwoo/DISTS. DISTS metric will be skipped.")
    DISTS = None

def get_od(img):
    """Convert an RGB image to Optical Density space."""
    img_float = img.astype(np.float32) / 255.0
    img_float = np.clip(img_float, 1e-6, 1.0)
    return -np.log10(img_float)

def evaluate_two_folders(gen_dir, gt_dir, label="Evaluation"):
    if not os.path.exists(gen_dir) or not os.path.exists(gt_dir):
        print(f"Error: gen_dir '{gen_dir}' or gt_dir '{gt_dir}' does not exist.")
        return None
        
    gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    common_files = sorted(list(set(gen_files).intersection(set(gt_files))))
    if not common_files:
        print(f"No common image files found between {gen_dir} and {gt_dir}")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    # KID subset size shouldn't exceed the number of samples
    subset_size = min(50, len(common_files))
    kid_metric = KernelInceptionDistance(subset_size=subset_size).to(device)
    
    dists_metric = DISTS().to(device) if DISTS else None
    
    psnr_vals = []
    ssim_vals = []
    msssim_vals = []
    lpips_vals = []
    dists_vals = []
    pearson_od_vals = []
    
    print(f"\nEvaluating {label} (Samples: {len(common_files)}) ...")
    for f in tqdm(common_files):
        gen_path = os.path.join(gen_dir, f)
        gt_path = os.path.join(gt_dir, f)
        
        gen_img_np = np.array(Image.open(gen_path).convert("RGB"))
        gt_img_np = np.array(Image.open(gt_path).convert("RGB"))
        
        if gen_img_np.shape != gt_img_np.shape:
            gen_img_np = np.array(Image.fromarray(gen_img_np).resize((gt_img_np.shape[1], gt_img_np.shape[0])))
            
        # PSNR & SSIM (skimage)
        score_psnr = compute_psnr(gt_img_np, gen_img_np, data_range=255)
        score_ssim = compute_ssim(gt_img_np, gen_img_np, channel_axis=-1, data_range=255)
        psnr_vals.append(score_psnr)
        ssim_vals.append(score_ssim)
        
        # Pearson-OD
        od_gen = get_od(gen_img_np)
        od_gt = get_od(gt_img_np)
        r_od_val, _ = pearsonr(od_gen.flatten(), od_gt.flatten())
        pearson_od_vals.append(r_od_val)
        
        # Tensors
        gen_t_uint8 = torch.from_numpy(gen_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t_uint8 = torch.from_numpy(gt_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        gen_t_float = gen_t_uint8.float() / 255.0
        gt_t_float = gt_t_uint8.float() / 255.0
        
        # Update FID and KID
        fid_metric.update(gt_t_uint8, real=True)
        fid_metric.update(gen_t_uint8, real=False)
        kid_metric.update(gt_t_uint8, real=True)
        kid_metric.update(gen_t_uint8, real=False)
        
        with torch.no_grad():
            msssim_vals.append(msssim_metric(gen_t_float, gt_t_float).item())
            
            score_lpips = lpips_metric(gen_t_float * 2.0 - 1.0, gt_t_float * 2.0 - 1.0)
            lpips_vals.append(score_lpips.item())
            
            if dists_metric:
                score_dists = dists_metric(gen_t_float, gt_t_float)
                dists_vals.append(score_dists.item())
                
    print(f"Computing FID and KID for {label} (this may take a moment)...")
    fid_score = fid_metric.compute().item()
    kid_mean, kid_std = kid_metric.compute()
    kid_score = kid_mean.item()
    
    res = {
        "Directory": label,
        "Samples": len(common_files),
        "PSNR": np.mean(psnr_vals),
        "SSIM": np.mean(ssim_vals),
        "S-SSIM": np.mean(msssim_vals),
        "Pearson-OD": np.mean(pearson_od_vals),
        "DISTS": np.mean(dists_vals) if dists_vals else np.nan,
        "LPIPS": np.mean(lpips_vals),
        "FID": fid_score,
        "KID": kid_score,
        "KID_std": kid_std.item()
    }
    
    print("=" * 60)
    print(f" Results for {label}")
    print("=" * 60)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)
    
    return res

def evaluate_folder(base_dir):
    gen_dir = os.path.join(base_dir, "generated")
    gt_dir = os.path.join(base_dir, "gt")
    
    if not os.path.exists(gen_dir) or not os.path.exists(gt_dir):
        print(f"Skipping {base_dir}: missing 'generated' or 'gt' directory")
        return None
        
    return evaluate_two_folders(gen_dir, gt_dir, label=base_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Image Translation Metrics")
    parser.add_argument("--dirs", nargs='+', default=None, 
                        help="Directories to evaluate (each must contain 'generated' and 'gt' subfolders)")
    parser.add_argument("--gen-dir", type=str, default=None, help="Direct path to generated/fake folder")
    parser.add_argument("--gt-dir", type=str, default=None, help="Direct path to ground-truth/real folder")
    parser.add_argument("--out_csv", type=str, default="evaluation_results.csv", help="Output CSV file name")
    args = parser.parse_args()
    
    results = []
    
    if args.gen_dir and args.gt_dir:
        label = f"custom_{os.path.basename(os.path.dirname(args.gen_dir))}_{os.path.basename(args.gen_dir)}"
        r = evaluate_two_folders(args.gen_dir, args.gt_dir, label=label)
        if r:
            results.append(r)
            
    dirs_to_evaluate = args.dirs
    if not args.gen_dir and not args.gt_dir and dirs_to_evaluate is None:
        dirs_to_evaluate = ["outputs/a2b_400k", "outputs/b2a_400k"]
        
    if dirs_to_evaluate:
        for d in dirs_to_evaluate:
            r = evaluate_folder(d)
            if r:
                results.append(r)
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ All results saved to {args.out_csv}")

if __name__ == "__main__":
    main()
