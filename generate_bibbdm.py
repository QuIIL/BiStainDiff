"""
BiBBDM Inference Script
=======================
Translates images from domain A→B (or B→A) using a trained BiBBDM + REPA-E checkpoint.

Usage (single GPU):
    python generate_bibbdm.py \
        --ckpt exps/REPA_E_bibbdm_he2ihc/checkpoints/0130000.pt \
        --csv-path /home/quiil/jiwoo/MINIM/train_src_tar_txt_updated_1.csv \
        --direction a2b \
        --out-dir outputs/he2ihc_130k \
        --num-samples 64 \
        --batch-size 8

Multi-GPU (torchrun):
    torchrun --nproc_per_node=2 generate_bibbdm.py \
        --ckpt exps/REPA_E_bibbdm_he2ihc/checkpoints/0130000.pt \
        --csv-path /home/quiil/jiwoo/MINIM/train_src_tar_txt_updated_1.csv \
        --direction a2b \
        --out-dir outputs/he2ihc_130k \
        --num-samples 64 \
        --batch-size 8
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import CSVPairedDataset
from models.autoencoder import vae_models
from models.sit import SiTBB_models
from samplers import bibbdm_sampler
from utils import preprocess_imgs_vae


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def is_dist():
    return dist.is_available() and dist.is_initialized()


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank  = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank, world = 0, 1
    return rank, world


def tensor_to_pil(t):
    """[C,H,W] float [0,1] → PIL Image"""
    arr = (t.clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    rank, world = setup_dist()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Read training config saved inside the checkpoint
    train_args = ckpt["args"]
    # train_args may be a Namespace object or a dict
    if isinstance(train_args, dict):
        from argparse import Namespace
        train_args = Namespace(**train_args)

    vae_type    = getattr(train_args, "vae",          "f8d4")
    model_name  = getattr(train_args, "model",         "SiTBB-XL/2")
    resolution  = getattr(train_args, "resolution",    256)
    enc_type    = getattr(train_args, "enc_type",      "uni-vit-l")
    num_classes = getattr(train_args, "num_classes",   1)
    enc_depth   = getattr(train_args, "encoder_depth", 8)
    qk_norm     = getattr(train_args, "qk_norm",       False)
    fused_attn  = getattr(train_args, "fused_attn",    True)
    bn_momentum = getattr(train_args, "bn_momentum",   0.1)
    use_text    = getattr(train_args, "use_text_ctx",  False)
    text_dim    = getattr(train_args, "text_dim",      1536)
    bb_kwargs = dict(
        num_timesteps = getattr(train_args, "bb_num_timesteps", 1000),
        mt_type       = getattr(train_args, "bb_mt_type",       "linear"),
        m0            = getattr(train_args, "bb_m0",            0.001),
        mT            = getattr(train_args, "bb_mT",            0.999),
        var_scale     = getattr(train_args, "bb_var_scale",     1.0),
        objective     = getattr(train_args, "bb_objective",     "dlns"),
        skip_sample   = getattr(train_args, "bb_skip_sample",   True),
        sample_step   = getattr(train_args, "bb_sample_step",   200),
        eta           = getattr(train_args, "bb_eta",           1.0),
    )

    if vae_type == "f8d4":
        latent_size = resolution // 8
        in_channels = 4
    elif vae_type == "f16d32":
        latent_size = resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError(vae_type)

    # ---- Build model ----
    from utils import load_encoders
    encoders, _, _ = load_encoders(enc_type, device, resolution)
    z_dims = [enc.embed_dim for enc in encoders]
    del encoders  # only needed for z_dims

    block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm}
    model = SiTBB_models[model_name](
        input_size   = latent_size,
        in_channels  = in_channels,
        num_classes  = num_classes,
        class_dropout_prob = 0.0,
        z_dims       = z_dims,
        encoder_depth= enc_depth,
        bn_momentum  = bn_momentum,
        bb_kwargs    = bb_kwargs,
        use_text_ctx = use_text,
        text_dim     = text_dim,
        **block_kwargs,
    ).to(device)

    # Load EMA weights (preferred) or model weights
    weight_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[weight_key], strict=False)
    model.eval()
    if rank == 0:
        print(f"[✓] Loaded model from '{args.ckpt}' (key='{weight_key}')")

    # ---- Load text encoder if enabled ----
    tokenizer = None
    text_encoder = None
    if use_text:
        from transformers import CLIPTextModel, CLIPTokenizer
        if rank == 0:
            print("[✓] Initializing CLIP text encoder for text conditioning...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        text_encoder.eval()

    # ---- Build VAE ----
    vae = vae_models[vae_type]().to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    if rank == 0:
        print(f"[✓] Loaded VAE from checkpoint")

    # ---- Load Latent Statistics for Normalization ----
    from utils import normalize_latents, denormalize_latents
    vae_ckpt_path = getattr(train_args, "vae_ckpt", "")
    if not vae_ckpt_path:
        # Fallback to reconstructing the stats filename based on standard locations
        stats_path = "ckpts/sd-vae-f8-d4-latents-stats.pt"
    else:
        stats_path = vae_ckpt_path.replace(".pt", "-latents-stats.pt")

    try:
        latents_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        latents_scale = latents_stats["latents_scale"].view(1, -1, 1, 1).to(device)
        latents_bias  = latents_stats["latents_bias"].view(1, -1, 1, 1).to(device)
        if rank == 0:
            print(f"[✓] Loaded Latent Stats from '{stats_path}'")
    except Exception as e:
        if rank == 0:
            print(f"[!] Warning: Failed to load latent stats from '{stats_path}'. Error: {e}")
            print(f"[!] Falling back to standard stats path...")
        stats_path = "ckpts/sd-vae-f8-d4-latents-stats.pt"
        latents_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        latents_scale = latents_stats["latents_scale"].view(1, -1, 1, 1).to(device)
        latents_bias  = latents_stats["latents_bias"].view(1, -1, 1, 1).to(device)
        if rank == 0:
            print(f"[✓] Loaded Latent Stats from '{stats_path}'")

    # ---- Dataset ----
    dataset = CSVPairedDataset(csv_path=args.csv_path, image_size=resolution, center_crop=True)
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    indices = list(range(num_samples))

    # Shard per GPU
    per_rank = math.ceil(num_samples / world)
    start    = rank * per_rank
    end      = min(start + per_rank, num_samples)
    sub_idx  = indices[start:end]
    subset   = Subset(dataset, sub_idx)

    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    # ---- Output dir ----
    direction = args.direction if args.direction else getattr(train_args, "direction", "a2b")
    os.makedirs(args.out_dir, exist_ok=True)
    tgt_dir = os.path.join(args.out_dir, "generated")
    os.makedirs(tgt_dir, exist_ok=True)
    input_dir = os.path.join(args.out_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    gt_dir = os.path.join(args.out_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    if rank == 0:
        print(f"Direction : {direction}")
        print(f"Samples   : {num_samples}")
        print(f"Output    : {args.out_dir}")

    # ---- Inference loop ----
    global_idx = start  # global image index for file naming
    pbar = tqdm(loader, desc=f"[rank {rank}] Translating", disable=(rank != 0))

    is_first_batch = True
    for batch in pbar:
        raw_a = batch[0].to(device)  # [B, C, H, W] uint8
        raw_b = batch[1].to(device)
        src_txt = batch[2] if len(batch) > 2 else None
        tar_txt = batch[3] if len(batch) > 3 else None
        B_img = raw_a.shape[0]

        # Compute text embeddings if text is available
        text_emb = None
        if use_text and src_txt is not None and tar_txt is not None:
            with torch.no_grad():
                src_tokens = tokenizer(list(src_txt), padding=True, truncation=True, return_tensors="pt").to(device)
                tar_tokens = tokenizer(list(tar_txt), padding=True, truncation=True, return_tensors="pt").to(device)
                src_pool = text_encoder(**src_tokens).pooler_output # [B, 768]
                tar_pool = text_encoder(**tar_tokens).pooler_output # [B, 768]
                text_emb = torch.cat([src_pool, tar_pool], dim=-1)   # [B, 1536]

        with torch.no_grad():
            proc_a = preprocess_imgs_vae(raw_a)
            proc_b = preprocess_imgs_vae(raw_b)
            _, a_lat, _ = vae(proc_a)
            _, b_lat, _ = vae(proc_b)

            # Pick source latent based on direction
            src_lat = a_lat.to(torch.float32) if direction == "a2b" else b_lat.to(torch.float32)
            dummy_y = torch.zeros(src_lat.shape[0], dtype=torch.long, device=device)

            # BiBBDM sampling
            translated_lat = bibbdm_sampler(
                model, src_lat, dummy_y, direction=direction, text_emb=text_emb
            ).to(torch.float32)

            trans_imgs = vae.decode(translated_lat).sample  # [-1, 1]
            trans_imgs = (trans_imgs + 1) / 2.0             # [0, 1]

        # Save one image per sample
        for i in range(B_img):
            tensor_to_pil(trans_imgs[i]).save(os.path.join(tgt_dir, f"{global_idx + i:06d}.png"))
            
            # Save input and gt images
            in_img = raw_a[i] if direction == "a2b" else raw_b[i]
            gt_img = raw_b[i] if direction == "a2b" else raw_a[i]
            
            Image.fromarray(in_img.permute(1, 2, 0).byte().cpu().numpy()).save(os.path.join(input_dir, f"{global_idx + i:06d}.png"))
            Image.fromarray(gt_img.permute(1, 2, 0).byte().cpu().numpy()).save(os.path.join(gt_dir, f"{global_idx + i:06d}.png"))

        global_idx += B_img

    if is_dist():
        dist.barrier()

    if rank == 0:
        print(f"\n[✓] Done! Saved {num_samples} images to '{args.out_dir}'")
        print(f"  translated → {tgt_dir}")

    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiBBDM Image Translation Inference")

    parser.add_argument("--ckpt",        required=True, help="Path to .pt checkpoint")
    parser.add_argument("--csv-path",    required=True, help="Path to paired CSV dataset")
    parser.add_argument("--out-dir",     default="outputs/bibbdm_samples", help="Output directory")
    parser.add_argument("--direction",   default=None, choices=["a2b", "b2a", None],
                        help="Translation direction (default: read from checkpoint args)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples to generate (0 = full dataset)")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--cfg-scale",   type=float, default=1.0,
                        help="Guidance scale (float >= 1.0, 1.0 disables CFG)")
    parser.add_argument("--compare-vae", action="store_true",
                        help="Enable to compare reconstruction error and visual sharpness of Pre-trained vs Fine-tuned VAE")

    args = parser.parse_args()
    main(args)
