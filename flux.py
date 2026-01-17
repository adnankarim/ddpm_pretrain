#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1 ControlNet + LoRA + Drug (Morgan 1024-bit) conditioning training script.

Dataset format (CSV):
  - image: path to TARGET image (treated)
  - conditioning_image: path to CONTROL image (e.g., DMSO / paired control)
  - prompt: (optional) text prompt per row; can be constant
  - smiles: (optional) SMILES string per row (Morgan FP computed)
    OR
  - fingerprint: (optional) precomputed 0/1 string length 1024 (e.g., "01001...")

Example run (single GPU):
  accelerate launch train_flux_controlnet_lora_drug.py \
    --pretrained_model black-forest-labs/FLUX.1-dev \
    --train_csv ./data/train.csv \
    --output_dir ./out_flux_cell \
    --resolution 96 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --max_train_steps 20000

Notes:
- This is for FLUX.1 because Diffusers ControlNet support is mature there.
- GGUF checkpoints are for inference; use the Diffusers BF16/FP16 model for training.
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed

# Diffusers / Transformers
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling

# LoRA (PEFT)
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

# Optional RDKit for Morgan fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_OK = True
except Exception:
    RDKIT_OK = False

# Optional safetensors for clean saving
try:
    from safetensors.torch import save_file as safetensors_save
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_OK = True
except Exception:
    SAFETENSORS_OK = False


# -------------------------
# Drug projector (1024 -> N tokens of hidden_dim)
# -------------------------
class DrugProjector(nn.Module):
    def __init__(self, fingerprint_dim: int, num_drug_tokens: int, hidden_dim: int):
        super().__init__()
        self.fingerprint_dim = fingerprint_dim
        self.num_drug_tokens = num_drug_tokens
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(fingerprint_dim, num_drug_tokens * hidden_dim),
            nn.SiLU(),
            nn.Linear(num_drug_tokens * hidden_dim, num_drug_tokens * hidden_dim),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        # fp: [B, 1024]
        x = self.net(fp)
        return x.view(fp.shape[0], self.num_drug_tokens, self.hidden_dim)


# -------------------------
# Fingerprint utils
# -------------------------
def morgan_fp_1024(smiles: str, radius: int = 2, n_bits: int = 1024) -> torch.Tensor:
    """
    Returns float32 tensor [1024] with 0/1 values.
    Requires RDKit. If RDKit missing, raises.
    """
    if not RDKIT_OK:
        raise RuntimeError("RDKit not installed. Install it or provide --fingerprint_column with precomputed 1024-bit strings.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # fall back to zeros for bad SMILES
        return torch.zeros(n_bits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = [0] * n_bits
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)


def parse_fp_string(fp_str: str, n_bits: int = 1024) -> torch.Tensor:
    """
    Parse "010101..." length 1024 into float32 tensor.
    """
    s = (fp_str or "").strip()
    if len(s) != n_bits or any(c not in "01" for c in s):
        return torch.zeros(n_bits, dtype=torch.float32)
    return torch.tensor([1.0 if c == "1" else 0.0 for c in s], dtype=torch.float32)


# -------------------------
# Dataset
# -------------------------
class PairedImageDrugCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_column: str,
        conditioning_image_column: str,
        prompt_column: Optional[str],
        smiles_column: Optional[str],
        fingerprint_column: Optional[str],
        resolution: int,
        constant_prompt: Optional[str] = None,
    ):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.image_column = image_column
        self.cond_column = conditioning_image_column
        self.prompt_column = prompt_column
        self.smiles_column = smiles_column
        self.fingerprint_column = fingerprint_column
        self.constant_prompt = constant_prompt

        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        target_path = str(row[self.image_column])
        cond_path = str(row[self.cond_column])

        target = self._load_rgb(target_path)
        cond = self._load_rgb(cond_path)

        if self.constant_prompt is not None:
            prompt = self.constant_prompt
        elif self.prompt_column and self.prompt_column in row and isinstance(row[self.prompt_column], str):
            prompt = row[self.prompt_column]
        else:
            prompt = "fluorescence microscopy image of a cell"

        # fingerprint priority: fingerprint_column > smiles_column
        if self.fingerprint_column and self.fingerprint_column in row and isinstance(row[self.fingerprint_column], str):
            fp = parse_fp_string(row[self.fingerprint_column], n_bits=1024)
        elif self.smiles_column and self.smiles_column in row and isinstance(row[self.smiles_column], str):
            fp = morgan_fp_1024(row[self.smiles_column], radius=2, n_bits=1024)
        else:
            fp = torch.zeros(1024, dtype=torch.float32)

        return {
            "pixel_values": target,
            "conditioning_pixel_values": cond,
            "prompt": prompt,
            "fingerprint": fp,
        }


# -------------------------
# Helpers (Flux flow matching noise schedule)
# -------------------------
def get_sigmas(noise_scheduler: FlowMatchEulerDiscreteScheduler, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype, device: torch.device):
    # mirrors the logic used in diffusers' FLUX ControlNet training script
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    # timesteps: [B]
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def lora_target_modules_default() -> List[str]:
    # good default set used by official FLUX LoRA scripts
    return [
        "x_embedder",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]


# -------------------------
# Saving / Loading
# -------------------------
def save_lora_weights(transformer: FluxTransformer2DModel, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state = get_peft_model_state_dict(transformer)
    if SAFETENSORS_OK:
        safetensors_save(state, os.path.join(out_dir, "transformer_lora.safetensors"))
    else:
        torch.save(state, os.path.join(out_dir, "transformer_lora.pt"))


def load_lora_weights(transformer: FluxTransformer2DModel, lora_path: str):
    if lora_path.endswith(".safetensors"):
        if not SAFETENSORS_OK:
            raise RuntimeError("safetensors not installed but .safetensors provided")
        state = safetensors_load(lora_path)
    else:
        state = torch.load(lora_path, map_location="cpu")
    set_peft_model_state_dict(transformer, state)


def save_drug_projector(drug_proj: DrugProjector, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if SAFETENSORS_OK:
        safetensors_save(drug_proj.state_dict(), os.path.join(out_dir, "drug_projector.safetensors"))
    else:
        torch.save(drug_proj.state_dict(), os.path.join(out_dir, "drug_projector.pt"))


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--image_column", type=str, default="image")
    p.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    p.add_argument("--prompt_column", type=str, default="prompt")
    p.add_argument("--smiles_column", type=str, default="smiles")
    p.add_argument("--fingerprint_column", type=str, default=None)
    p.add_argument("--constant_prompt", type=str, default=None)

    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--num_drug_tokens", type=int, default=4)
    p.add_argument("--fingerprint_dim", type=int, default=1024)

    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, default=",".join(lora_target_modules_default()))

    # ControlNet size
    p.add_argument("--num_double_layers", type=int, default=4)
    p.add_argument("--num_single_layers", type=int, default=4)

    # Logging / checkpoint
    p.add_argument("--save_every", type=int, default=1000)

    # timestep sampling scheme (same knobs as official script, kept minimal)
    p.add_argument("--weighting_scheme", type=str, default="logit_normal",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)

    # FLUX guidance embedding
    p.add_argument("--guidance_scale", type=float, default=3.5)

    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # -------------------------
    # Load tokenizers / encoders / models
    # -------------------------
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer_2", revision=args.revision)

    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = T5EncoderModel.from_pretrained(args.pretrained_model, subfolder="text_encoder_2", revision=args.revision, variant=args.variant)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", revision=args.revision, variant=args.variant)
    flux_transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model, subfolder="transformer", revision=args.revision, variant=args.variant)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler", revision=args.revision)

    # Create a trainable FLUX ControlNet (lightweight) - initialized from transformer config
    flux_controlnet = FluxControlNetModel(
        # these mirror knobs exposed in the official script
        num_double_layers=args.num_double_layers,
        num_single_layers=args.num_single_layers,
        attention_head_dim=flux_transformer.config.attention_head_dim,
        num_attention_heads=flux_transformer.config.num_attention_heads,
        joint_attention_dim=flux_transformer.config.joint_attention_dim,
        pooled_projection_dim=flux_transformer.config.pooled_projection_dim,
        guidance_embeds=flux_transformer.config.guidance_embeds,
    )

    # Freeze "brain": VAE + text encoders + base transformer
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    # -------------------------
    # Build pipeline for prompt encoding + latent packing helpers
    # -------------------------
    pipe = FluxControlNetPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=flux_transformer,
        controlnet=flux_controlnet,
    )

    # Put frozen components on device (we’ll cast later)
    # (text encoders are used by encode_prompt; keep them where pipe lives)
    pipe.to(accelerator.device)

    # Mixed precision dtype for forward
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Cast inference-only modules
    vae.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_controlnet.to(accelerator.device, dtype=weight_dtype)

    # -------------------------
    # Inject LoRA into transformer (trainable ~1%)
    # -------------------------
    target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # FluxTransformer2DModel supports adapters in diffusers, but PEFT utils still work for state dict save/load.
    flux_transformer.add_adapter(lora_cfg)

    # Ensure only LoRA params trainable
    for n, p_ in flux_transformer.named_parameters():
        p_.requires_grad = ("lora" in n.lower())

    # -------------------------
    # Create DrugProjector with correct hidden dim
    # -------------------------
    with torch.no_grad():
        dummy_prompt = ["fluorescence microscopy image of a cell"]
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(dummy_prompt, prompt_2=dummy_prompt)
        hidden_dim = prompt_embeds.shape[-1]

    drug_proj = DrugProjector(
        fingerprint_dim=args.fingerprint_dim,
        num_drug_tokens=args.num_drug_tokens,
        hidden_dim=hidden_dim,
    ).to(accelerator.device, dtype=weight_dtype)
    drug_proj.requires_grad_(True)

    # -------------------------
    # Data
    # -------------------------
    ds = PairedImageDrugCSVDataset(
        csv_path=args.train_csv,
        image_column=args.image_column,
        conditioning_image_column=args.conditioning_image_column,
        prompt_column=args.prompt_column,
        smiles_column=args.smiles_column if args.smiles_column else None,
        fingerprint_column=args.fingerprint_column,
        resolution=args.resolution,
        constant_prompt=args.constant_prompt,
    )

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        conditioning_pixel_values = torch.stack([e["conditioning_pixel_values"] for e in examples])
        fps = torch.stack([e["fingerprint"] for e in examples])
        prompts = [e["prompt"] for e in examples]
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "fingerprint": fps,
            "prompts": prompts,
        }

    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)

    # -------------------------
    # Optimizer
    # -------------------------
    params = []
    params += list(flux_controlnet.parameters())
    params += [p for p in flux_transformer.parameters() if p.requires_grad]
    params += list(drug_proj.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare with accelerate
    flux_controlnet, flux_transformer, drug_proj, optimizer, dl = accelerator.prepare(
        flux_controlnet, flux_transformer, drug_proj, optimizer, dl
    )

    # -------------------------
    # Train loop
    # -------------------------
    global_step = 0
    flux_controlnet.train()
    drug_proj.train()
    flux_transformer.train()  # only LoRA params are trainable

    # convenience: scheduler copy for sampling indices
    noise_scheduler_copy = FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config)

    while global_step < args.max_train_steps:
        for batch in dl:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(flux_controlnet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                cond_values = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                fp = batch["fingerprint"].to(accelerator.device, dtype=torch.float32)  # keep fp in fp32

                # ---- prompt embeddings (frozen text encoders) ----
                # T5 may not like autocast; do prompt encoding in no_grad then cast.
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(batch["prompts"], prompt_2=batch["prompts"])
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)          # [B, S, D]
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)  # [B, P]
                text_ids = text_ids.to(dtype=weight_dtype, device=accelerator.device)                  # [S, 3] (unbatched)

                # Expand text_ids to batch then append drug token ids
                text_ids_b = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)               # [B, S, 3]
                drug_txt_ids = torch.zeros((prompt_embeds.shape[0], args.num_drug_tokens, text_ids_b.shape[-1]),
                                           device=accelerator.device, dtype=weight_dtype)
                # ---- drug tokens ----
                drug_tokens = drug_proj(fp).to(dtype=weight_dtype)                                      # [B, N, D]
                prompt_embeds = torch.cat([prompt_embeds, drug_tokens], dim=1)                          # [B, S+N, D]
                text_ids_b = torch.cat([text_ids_b, drug_txt_ids], dim=1)                               # [B, S+N, 3]

                # ---- VAE encode target & control ----
                with torch.no_grad():
                    px_lat = vae.encode(pixel_values).latent_dist.sample()
                    px_lat = (px_lat - vae.config.shift_factor) * vae.config.scaling_factor

                    cn_lat = vae.encode(cond_values).latent_dist.sample()
                    cn_lat = (cn_lat - vae.config.shift_factor) * vae.config.scaling_factor

                # Pack latents (FLUX expects packed latents)
                pixel_latents = FluxControlNetPipeline._pack_latents(
                    px_lat, pixel_values.shape[0], px_lat.shape[1], px_lat.shape[2], px_lat.shape[3]
                )
                control_image = FluxControlNetPipeline._pack_latents(
                    cn_lat, cond_values.shape[0], cn_lat.shape[1], cn_lat.shape[2], cn_lat.shape[3]
                )

                # Latent image ids (positional ids for image tokens)
                latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                    batch_size=px_lat.shape[0],
                    height=px_lat.shape[2] // 2,
                    width=px_lat.shape[3] // 2,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)

                # sample timesteps (optionally non-uniform)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                sigmas = get_sigmas(noise_scheduler_copy, timesteps, n_dim=pixel_latents.ndim, dtype=weight_dtype, device=accelerator.device)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # guidance embedding (if enabled in config)
                if flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full((bsz,), args.guidance_scale, device=accelerator.device, dtype=weight_dtype)
                else:
                    guidance_vec = None

                # ---- ControlNet forward ----
                controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_image,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids_b[0],       # [S+N, 3] (same for whole batch)
                    img_ids=latent_image_ids,
                    return_dict=False,
                )

                # ---- Transformer forward (with ControlNet residuals) ----
                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=[s.to(dtype=weight_dtype) for s in controlnet_block_samples] if controlnet_block_samples is not None else None,
                    controlnet_single_block_samples=[s.to(dtype=weight_dtype) for s in controlnet_single_block_samples] if controlnet_single_block_samples is not None else None,
                    txt_ids=text_ids_b[0],
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # Flow-matching training target used by FLUX scripts:
                # loss against (noise - pixel_latents)
                loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"step={global_step} loss={loss.item():.6f}")

            # checkpoint
            if accelerator.is_main_process and (global_step > 0) and (global_step % args.save_every == 0):
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # Save ControlNet
                accelerator.unwrap_model(flux_controlnet).save_pretrained(os.path.join(ckpt_dir, "flux_controlnet"))

                # Save LoRA
                save_lora_weights(accelerator.unwrap_model(flux_transformer), os.path.join(ckpt_dir, "lora"))

                # Save Drug Projector
                save_drug_projector(accelerator.unwrap_model(drug_proj), os.path.join(ckpt_dir, "drug"))

                print(f"[saved] {ckpt_dir}")

            global_step += 1

    # final save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(flux_controlnet).save_pretrained(os.path.join(final_dir, "flux_controlnet"))
        save_lora_weights(accelerator.unwrap_model(flux_transformer), os.path.join(final_dir, "lora"))
        save_drug_projector(accelerator.unwrap_model(drug_proj), os.path.join(final_dir, "drug"))
        print(f"[done] saved to {final_dir}")


if __name__ == "__main__":
    main()
