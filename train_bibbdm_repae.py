"""
BiBBDM + REPA-E End-to-End Training Script
==========================================
Trains SiTBB (SiT + Brownian Bridge) with:
  - Paired HE <-> IHC images
  - REPA alignment loss (pathology FM: UNI / CONCH / GigaPath / DINOv2)
  - End-to-end VAE training (same as REPA-E)
  - Bidirectional translation support

Usage:
    torchrun --nproc_per_node=4 train_bibbdm_repae.py \
        --exp-name bibbdm_he2ihc \
        --data-dir /data/paired \
        --h5-name-a he --h5-name-b ihc \
        --enc-type uni-vit-l \
        --vae-ckpt pretrained/sdvae-f8d4/sdvae-f8d4.pt
"""

import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision.utils import make_grid
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import wandb

from dataset import CustomPairedH5Dataset, CSVPairedDataset
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models, DiagonalGaussianDistribution
from models.sit import SiTBB_models
from samplers import bibbdm_sampler
from utils import (
    load_encoders, preprocess_imgs_vae,
    normalize_latents, denormalize_latents, count_trainable_params
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buf in model_buffers.items():
        name = name.replace("module.", "")
        if buf.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_buffers[name].mul_(decay).add_(buf.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buf)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def preprocess_raw_image(x, enc_type):
    """Normalise raw uint8 images for a given encoder type."""
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from torchvision.transforms import Normalize

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    x = x / 255.0
    if "clip" in enc_type:
        x = Normalize(CLIP_MEAN, CLIP_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode="bicubic")
    elif enc_type in ("uni", "conch"):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # UNI and CONCH have patch size 16. To get 256 patch tokens,
        # we interpolate to 256x256.
        x = torch.nn.functional.interpolate(x, 256, mode="bicubic")
    elif enc_type in ("gigapath", "uni2") or "dinov2" in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # GigaPath, DINOv2 and UNI2 have patch size 14. To get 256 patch tokens,
        # we interpolate to 224x224.
        x = torch.nn.functional.interpolate(x, 224, mode="bicubic")
    else:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args):
    if args.data_dir is None and args.csv_path is None:
        raise ValueError("Either --data-dir (for H5 loading) or --csv-path (for CSV loading) must be provided.")

    # ---- Accelerator ----
    logging_dir = Path(args.output_dir, args.logging_dir)
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        logger = create_logger(save_dir)
        logger.info(f"Experiment: {save_dir}")

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # ---- VAE ----
    if args.vae == "f8d4":
        latent_size = args.resolution // 8
        in_channels = 4
    elif args.vae == "f16d32":
        latent_size = args.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError(args.vae)

    vae = vae_models[args.vae]().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_ckpt, strict=False)
    del vae_ckpt

    latents_stats = torch.load(args.vae_ckpt.replace(".pt", "-latents-stats.pt"))
    latents_scale = latents_stats["latents_scale"].squeeze().to(device)
    latents_bias  = latents_stats["latents_bias"].squeeze().to(device)

    # ---- Encoders (pathology FMs) ----
    encoders, encoder_types, architectures = load_encoders(
        args.enc_type, device, args.resolution
    )
    z_dims = [enc.embed_dim for enc in encoders]

    # ---- CLIP Text Encoder for text conditioning ----
    tokenizer = None
    text_encoder = None
    if args.use_text_ctx:
        from transformers import CLIPTextModel, CLIPTokenizer
        if accelerator.is_main_process:
            logger.info("Initializing CLIP text encoder for text conditioning...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        text_encoder.eval()
        requires_grad(text_encoder, False)

    # Automatically switch default objective based on direction to prevent sampling errors
    if args.direction == "b2a" and args.bb_objective == "gradb":
        if accelerator.is_main_process:
            logger.info("Detected --direction 'b2a' with default --bb-objective 'gradb'. Automatically switching objective to 'grada' for correct target reconstruction.")
        args.bb_objective = "grada"

    # ---- SiTBB model ----
    bb_kwargs = dict(
        num_timesteps=args.bb_num_timesteps,
        mt_type=args.bb_mt_type,
        m0=args.bb_m0,
        mT=args.bb_mT,
        var_scale=args.bb_var_scale,
        objective=args.bb_objective,
        skip_sample=args.bb_skip_sample,
        sample_step=args.bb_sample_step,
        eta=args.bb_eta,
    )
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiTBB_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,           # will be doubled inside SiTBB
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        bb_kwargs=bb_kwargs,
        weight_obj=args.bb_weight_obj,
        weight_a_recon=args.bb_weight_a_recon,
        weight_b_recon=args.bb_weight_b_recon,
        bb_loss_type=args.bb_loss_type,
        use_text_ctx=args.use_text_ctx,
        text_dim=args.text_dim,
        **block_kwargs,
    ).to(device)

    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)
    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---- Loss fn ----
    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    if args.disc_pretrained_ckpt is not None:
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vae_loss_fn.discriminator.load_state_dict(disc_ckpt)

    # ---- Optimizers ----
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    optimizer_vae = torch.optim.AdamW(
        list(vae.parameters()) + list(model.projectors_a.parameters()) + list(model.projectors_b.parameters()), lr=args.vae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    optimizer_disc = torch.optim.AdamW(
        vae_loss_fn.parameters(), lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    # ---- Dataset ----
    if args.csv_path is not None:
        train_dataset = CSVPairedDataset(
            csv_path=args.csv_path, image_size=args.resolution
        )
    else:
        if args.data_dir is None:
            raise ValueError("Either --data-dir must be provided for CustomPairedH5Dataset, or --csv-path for CSVPairedDataset.")
        train_dataset = CustomPairedH5Dataset(
            args.data_dir, h5_name_a=args.h5_name_a, h5_name_b=args.h5_name_b
        )
    local_bs = int(args.batch_size // accelerator.num_processes)
    train_loader = DataLoader(
        train_dataset, batch_size=local_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ---- Resume ----
    global_step = 0
    if args.resume_step > 0:
        ckpt_path = f"{args.cont_dir}/checkpoints/{args.resume_step:07d}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        ema.load_state_dict(ckpt["ema"], strict=False)
        vae.load_state_dict(ckpt["vae"])
        vae_loss_fn.discriminator.load_state_dict(ckpt["discriminator"])
        optimizer.load_state_dict(ckpt["opt"])
        optimizer_vae.load_state_dict(ckpt["opt_vae"])
        optimizer_disc.load_state_dict(ckpt["opt_disc"])
        global_step = ckpt["steps"]

    # ---- Compile ----
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")
        vae   = torch.compile(vae,   backend="inductor", mode="default")
        vae_loss_fn = torch.compile(vae_loss_fn, backend="inductor", mode="default")

    model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_disc, train_loader = \
        accelerator.prepare(model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_disc, train_loader)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="bibbdm-repae_aligned",
            config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=global_step,
        desc="Steps", disable=not accelerator.is_local_main_process,
    )

    # ---- Training loop ----
    loss_kwargs_align = dict(align_only=True)
    loss_kwargs_full  = dict(align_only=False)

    for epoch in range(args.epochs):
        model.train()

        for batch in train_loader:
            raw_a = batch[0].to(device)   # [B, C, H, W] uint8, domain A (HE)
            raw_b = batch[1].to(device)   # [B, C, H, W] uint8, domain B (IHC)
            src_txt = batch[2] if len(batch) > 2 else None
            tar_txt = batch[3] if len(batch) > 3 else None
            # Dummy labels created dynamically to match batch size
            dummy_y = torch.zeros(raw_a.shape[0], dtype=torch.long, device=device)

            # ---- Compute Text Embeddings (if enabled) ----
            text_emb = None
            if args.use_text_ctx and src_txt is not None and tar_txt is not None:
                with torch.no_grad():
                    src_tokens = tokenizer(list(src_txt), padding=True, truncation=True, return_tensors="pt").to(device)
                    tar_tokens = tokenizer(list(tar_txt), padding=True, truncation=True, return_tensors="pt").to(device)
                    
                    src_pool = text_encoder(**src_tokens).pooler_output # [B, 768]
                    tar_pool = text_encoder(**tar_tokens).pooler_output # [B, 768]
                    
                    text_emb = torch.cat([src_pool, tar_pool], dim=-1) # [B, 1536]

            # Get Brownian Bridge and sample timestep before extracting FM features to support mixed y alignment
            bb = accelerator.unwrap_model(model).bb
            B = raw_a.shape[0]
            t_int = torch.randint(0, bb.num_timesteps, (B,), device=device).long()

            # ---- Extract FM features (no grad) ----
            with torch.no_grad():
                zs_a = []
                zs_b = []
                with accelerator.autocast():
                    for encoder, enc_type, _ in zip(encoders, encoder_types, architectures):
                        # 1. Concat domain A (HE) and B (IHC) along batch axis to extract features in a single forward pass
                        raw_concat = torch.cat([raw_a, raw_b], dim=0) # [2*B, C, H, W]
                        x_for_enc = preprocess_raw_image(raw_concat.float(), enc_type)
                        z_concat = encoder.forward_features(x_for_enc)
                        
                        # Slice and clean patch tokens based on FM type
                        if "dinov2" in enc_type:
                            z_concat = z_concat["x_norm_patchtokens"]
                        elif "mocov3" in enc_type or enc_type in ("uni", "conch"):
                            z_concat = z_concat[:, 1:]
                            
                        # 2. Chunk back to domain A and domain B features
                        z_a, z_b = torch.chunk(z_concat, 2, dim=0) # [B, T_seq, D]
                        
                        zs_a.append(z_a)
                        zs_b.append(z_b)

            vae.train()
            model.train()

            with accelerator.accumulate([model, vae, vae_loss_fn]), accelerator.autocast():
                # ---- 1. VAE encode both domains ----
                proc_a = preprocess_imgs_vae(raw_a)  # [-1, 1]
                proc_b = preprocess_imgs_vae(raw_b)

                posterior_a, a_lat, recon_a = vae(proc_a)
                posterior_b, b_lat, recon_b = vae(proc_b)

                # ---- 2. Brownian Bridge forward sample ----
                x_t, objective = bb.q_sample(a_lat, b_lat, t_int)

                # ---- 3. VAE loss (A side) ----
                unwrapped_model = accelerator.unwrap_model(model)
                requires_grad(unwrapped_model, False)
                requires_grad(unwrapped_model.projectors_a, True)
                requires_grad(unwrapped_model.projectors_b, True)
                model.eval()
                unwrapped_model.projectors_a.train()
                unwrapped_model.projectors_b.train()

                proc_concat = torch.cat([proc_a, proc_b], dim=0)
                recon_concat = torch.cat([recon_a, recon_b], dim=0)
                posterior_concat = DiagonalGaussianDistribution(
                    torch.cat([posterior_a.parameters, posterior_b.parameters], dim=0)
                )

                vae_loss, vae_loss_dict = vae_loss_fn(
                    proc_concat, recon_concat, posterior_concat, global_step, "generator"
                )
                vae_loss = vae_loss.mean()

                vae_align_out = model(
                    x_t=x_t,
                    t_int=t_int, y=dummy_y, 
                    zs_a=zs_a, zs_b=zs_b,
                    loss_kwargs=loss_kwargs_align,
                    text_emb=text_emb,
                )
                vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_out["proj_loss"].mean()

                accelerator.backward(vae_loss)
                if accelerator.sync_gradients:
                    grad_norm_vae = accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer_vae.step()
                optimizer_vae.zero_grad(set_to_none=True)

                # Discriminator
                d_loss, d_dict = vae_loss_fn(proc_concat, recon_concat, None, global_step, "discriminator")
                d_loss = d_loss.mean()
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

                # ---- 4. SiTBB loss ----
                unwrapped_model = accelerator.unwrap_model(model)
                requires_grad(unwrapped_model, True)
                model.train()

                sit_out = model(
                    x_t=x_t.detach(),
                    t_int=t_int,
                    y=dummy_y,
                    zs_a=zs_a, zs_b=zs_b,
                    loss_kwargs=loss_kwargs_full,
                    objective=objective.detach(),
                    a_latent=a_lat.detach(),
                    b_latent=b_lat.detach(),
                    text_emb=text_emb,
                )

                bb_loss  = sit_out["bb_loss"].mean()
                proj_loss = sit_out["proj_loss"].mean()
                sit_loss = bb_loss + args.proj_coeff * proj_loss

                accelerator.backward(sit_loss)
                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    update_ema(ema, unwrapped._orig_mod if args.compile else unwrapped)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "sit_loss":    accelerator.gather(sit_loss).mean().item(),
                    "bb_loss":     accelerator.gather(bb_loss).mean().item(),
                    "proj_loss":   accelerator.gather(proj_loss).mean().item(),
                    "vae_loss":    accelerator.gather(vae_loss).mean().item(),
                    "d_loss":      accelerator.gather(d_loss).mean().item(),
                    "grad_norm_sit":  accelerator.gather(grad_norm_sit).mean().item(),
                    "grad_norm_vae":  accelerator.gather(grad_norm_vae).mean().item(),
                    "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().item(),
                    "epoch": epoch,
                }
                if "bb_obj_loss" in sit_out:
                    logs["bb_obj_loss"]   = accelerator.gather(sit_out["bb_obj_loss"]).mean().item()
                    logs["bb_a_rec_loss"] = accelerator.gather(sit_out["bb_a_rec_loss"]).mean().item()
                    logs["bb_b_rec_loss"] = accelerator.gather(sit_out["bb_b_rec_loss"]).mean().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            # ---- Checkpointing ----
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    uw_model = accelerator.unwrap_model(model)
                    uw_vae   = accelerator.unwrap_model(vae)
                    uw_loss  = accelerator.unwrap_model(vae_loss_fn)
                    orig_model = uw_model._orig_mod if args.compile else uw_model
                    orig_vae   = uw_vae._orig_mod   if args.compile else uw_vae
                    orig_disc  = uw_loss._orig_mod.discriminator if args.compile else uw_loss.discriminator
                    checkpoint = {
                        "model": orig_model.state_dict(),
                        "ema":   ema.state_dict(),
                        "vae":   orig_vae.state_dict(),
                        "discriminator": orig_disc.state_dict(),
                        "opt":      optimizer.state_dict(),
                        "opt_vae":  optimizer_vae.state_dict(),
                        "opt_disc": optimizer_disc.state_dict(),
                        "args":  args,
                        "steps": global_step,
                    }
                    ckpt_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint: {ckpt_path}")

            # ---- Sampling ----
            if global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0):
                model.eval()
                vae.eval()
                with torch.no_grad():
                    uw_model = accelerator.unwrap_model(model)
                    uw_vae   = accelerator.unwrap_model(vae)
                    # 1. HE -> IHC (a2b) translation
                    sample_src_a2b = a_lat[:8].to(torch.float32)
                    sample_y_a2b = torch.zeros(sample_src_a2b.shape[0], dtype=torch.long, device=device)
                    sample_text_emb_a2b = text_emb[:8] if text_emb is not None else None
                    translated_a2b = bibbdm_sampler(
                        uw_model, sample_src_a2b, sample_y_a2b, direction="a2b", text_emb=sample_text_emb_a2b
                    ).to(torch.float32)
                    recon_trans_a2b = uw_vae.decode(translated_a2b).sample
                    recon_trans_a2b = (recon_trans_a2b + 1) / 2.0

                    # 2. IHC -> HE (b2a) translation
                    sample_src_b2a = b_lat[:8].to(torch.float32)
                    sample_y_b2a = torch.zeros(sample_src_b2a.shape[0], dtype=torch.long, device=device)
                    sample_text_emb_b2a = text_emb[:8] if text_emb is not None else None
                    translated_b2a = bibbdm_sampler(
                        uw_model, sample_src_b2a, sample_y_b2a, direction="b2a", text_emb=sample_text_emb_b2a
                    ).to(torch.float32)
                    recon_trans_b2a = uw_vae.decode(translated_b2a).sample
                    recon_trans_b2a = (recon_trans_b2a + 1) / 2.0

                    # --- debug: log both VAE reconstructions to verify VAE works on both domains ---
                    recon_a = uw_vae.decode(a_lat[:8].to(torch.float32)).sample
                    recon_a = (recon_a + 1) / 2.0

                    recon_b = uw_vae.decode(b_lat[:8].to(torch.float32)).sample
                    recon_b = (recon_b + 1) / 2.0

                    # --- log ground truth images for visual sanity checks ---
                    sample_gt_a = raw_a[:8].float() / 255.0
                    sample_gt_b = raw_b[:8].float() / 255.0

                gathered_a2b     = accelerator.gather(recon_trans_a2b.to(torch.float32))
                gathered_b2a     = accelerator.gather(recon_trans_b2a.to(torch.float32))
                gathered_recon_a = accelerator.gather(recon_a.to(torch.float32))
                gathered_recon_b = accelerator.gather(recon_b.to(torch.float32))
                gathered_gt_a    = accelerator.gather(sample_gt_a.to(torch.float32))
                gathered_gt_b    = accelerator.gather(sample_gt_b.to(torch.float32))
                accelerator.log({
                    "a2b_samples":                     wandb.Image(array2grid(gathered_a2b)),
                    "b2a_samples":                     wandb.Image(array2grid(gathered_b2a)),
                    "recon_a":                         wandb.Image(array2grid(gathered_recon_a)),
                    "recon_b":                         wandb.Image(array2grid(gathered_recon_b)),
                    "gt_a":                            wandb.Image(array2grid(gathered_gt_a)),
                    "gt_b":                            wandb.Image(array2grid(gathered_gt_b)),
                })
                model.train()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BiBBDM + REPA-E Training")

    # logging
    p.add_argument("--output-dir", default="exps")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--logging-dir", default="logs")
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--sampling-steps", type=int, default=5000)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--cont-dir", default=None)
    p.add_argument("--direction", default="a2b", choices=["a2b", "b2a"], help="Training translation direction")

    # model
    p.add_argument("--model", default="SiTBB-XL/2", choices=["SiTBB-XL/2", "SiTBB-L/2", "SiTBB-B/2"])
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--use-text-ctx", action="store_true", default=True, help="Enable text conditioning context")
    p.add_argument("--text-dim", type=int, default=1536, help="Text embedding dimension (e.g. 1536 for dual CLIP)")
    p.add_argument("--encoder-depth", type=int, default=8)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bn-momentum", type=float, default=0.1)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    # data
    p.add_argument("--data-dir", default=None)
    p.add_argument("--csv-path", default=None, help="Path to CSV containing paired image paths")
    p.add_argument("--h5-name-a", default="he",  help="H5 basename for domain A")
    p.add_argument("--h5-name-b", default="ihc", help="H5 basename for domain B")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)

    # precision
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--max-train-steps", type=int, default=400000)
    p.add_argument("--checkpointing-steps", type=int, default=20000)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--vae-learning-rate", type=float, default=5e-5)
    p.add_argument("--disc-learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # REPA alignment
    p.add_argument("--enc-type", default="uni-vit-l",
                   help="Encoder(s): uni-vit-l, conch-vit-b, gigapath-vit-g, dinov2-vit-b, etc.")
    p.add_argument("--proj-coeff", type=float, default=0.5)
    p.add_argument("--vae-align-proj-coeff", type=float, default=1.5)
    p.add_argument("--align-mixed-y", action="store_true", default=True,
                   help="Mix domain A (HE) and B (IHC) images based on Brownian Bridge timestep interpolation coefficient m_t for representation alignment target extraction")
    p.add_argument("--cfg-prob", type=float, default=0.0)

    # VAE
    p.add_argument("--vae", default="f8d4", choices=["f8d4", "f16d32"])
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--loss-cfg-path", default="configs/l1_lpips_kl_gan.yaml")
    p.add_argument("--disc-pretrained-ckpt", default=None)

    # Brownian Bridge
    p.add_argument("--bb-num-timesteps", type=int, default=1000)
    p.add_argument("--bb-mt-type", default="linear", choices=["linear", "sin", "log"])
    p.add_argument("--bb-m0", type=float, default=0.001)
    p.add_argument("--bb-mT", type=float, default=0.999)
    p.add_argument("--bb-var-scale", type=float, default=1.0)
    p.add_argument("--bb-objective", default="dlns",
                   choices=["a", "b", "grada", "gradb", "noise", "bsuba", "dlns", "dlab", "dlgab"])
    p.add_argument("--bb-loss-type", default="l2", choices=["l1", "l2"])
    p.add_argument("--bb-weight-obj", type=float, default=1.0)
    p.add_argument("--bb-weight-a-recon", type=float, default=1.0)
    p.add_argument("--bb-weight-b-recon", type=float, default=1.0)
    p.add_argument("--bb-skip-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bb-sample-step", type=int, default=200)
    p.add_argument("--bb-eta", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
