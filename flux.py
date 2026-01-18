#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
BBBC021 FLUX CONTROLNET + LoRA + DRUG CONDITIONING
================================================================================
- Uses same dataset integration as your SD script:
  - metadata CSV (SPLIT, BATCH, CPD_NAME, SMILES, SAMPLE_KEY/image_path)
  - paths.csv robust lookup for .npy files
  - treated picks random DMSO control from same batch
  - Morgan 1024-bit fingerprint per CPD_NAME
- Training:
  - VAE frozen
  - Text encoders frozen
  - FLUX Transformer frozen except LoRA
  - ControlNet trainable
  - Drug projector trainable (multi-token)

NOTE (FLUX ControlNet):
  Diffusers' FLUX ControlNet expects control condition as *packed VAE latents*
  (not raw pixel images like SD ControlNet). We still load control/target images
  in pixel space using your loader, then encode both with VAE and pack.

Run:
  accelerate launch train_flux_bbbc021_controlnet_lora_drug.py \
    --pretrained_model black-forest-labs/FLUX.1-dev \
    --data_dir ./data/bbbc021_all \
    --metadata_file metadata/bbbc021_df_all.csv \
    --paths_csv ./data/bbbc021_all/paths.csv \
    --split train \
    --resolution 96 \
    --output_dir ./out_flux_bbbc021 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --max_train_steps 20000
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from accelerate import Accelerator
from accelerate.utils import set_seed

# -------------------- Diffusers / Transformers (FLUX) --------------------
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling

# -------------------- LoRA (Diffusers adapters) --------------------
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

# Optional safetensors
try:
    from safetensors.torch import save_file as safetensors_save
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_OK = True
except Exception:
    SAFETENSORS_OK = False

# Optional imageio for video generation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Optional RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_OK = True
except Exception:
    RDKIT_OK = False


# =============================================================================
# DRUG PROJECTOR
# =============================================================================
class DrugProjector(nn.Module):
    """
    Projects Morgan FP (1024) -> (num_drug_tokens * hidden_dim) then reshapes to [B, N, D]
    """
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
        x = self.net(fp)
        return x.view(fp.shape[0], self.num_drug_tokens, self.hidden_dim)


# =============================================================================
# MORGAN ENCODER (same behavior as your script; RDKit if available, else deterministic fallback)
# =============================================================================
class MorganEncoder:
    def __init__(self, n_bits=1024):
        self.n_bits = n_bits
        self.cache = {}

    def encode(self, smiles):
        if isinstance(smiles, list):
            return np.array([self.encode(s) for s in smiles])
        smiles = "" if smiles is None else str(smiles)
        if smiles in self.cache:
            return self.cache[smiles]

        if RDKIT_OK and smiles and smiles not in ["DMSO", ""]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                self.cache[smiles] = arr
                return arr
            except Exception:
                pass

        # deterministic fallback
        np.random.seed(hash(str(smiles)) % 2**32)
        arr = (np.random.rand(self.n_bits) > 0.5).astype(np.float32)
        self.cache[smiles] = arr
        return arr


# =============================================================================
# DATASET (ported from your SD code, unchanged path logic)
# =============================================================================
class PairedBBBC021Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, size=96, split="train", paths_csv=None):
        self.data_dir = Path(data_dir).resolve()
        self.size = size
        self.encoder = MorganEncoder(n_bits=1024)
        self._first_load_logged = False

        # Robust CSV loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file

        df = pd.read_csv(csv_full_path)
        if "SPLIT" in df.columns:
            df = df[df["SPLIT"].str.lower() == split.lower()]

        self.metadata = df.to_dict("records")

        # Group by batch for DMSO controls
        self.controls = {}  # batch -> [idx]
        self.treated = []   # treated idxs

        for idx, row in enumerate(self.metadata):
            batch = row.get("BATCH", "unk")
            cpd = str(row.get("CPD_NAME", "")).upper()
            if cpd == "DMSO":
                self.controls.setdefault(batch, []).append(idx)
            else:
                self.treated.append(idx)

        # Pre-encode fingerprints for each CPD_NAME using SMILES column
        self.fingerprints = {}
        if "CPD_NAME" in df.columns:
            for cpd_name in df["CPD_NAME"].unique():
                r0 = df[df["CPD_NAME"] == cpd_name].iloc[0]
                smiles = r0.get("SMILES", "")
                self.fingerprints[cpd_name] = self.encoder.encode(smiles)

        print(f"Dataset ({split}): {len(self.treated)} treated, {sum(len(v) for v in self.controls.values())} controls")

        # Load paths.csv lookup
        self.paths_lookup = {}
        self.paths_by_rel = {}
        self.paths_by_basename = {}

        if paths_csv:
            paths_csv_path = Path(paths_csv)
        else:
            paths_csv_path = Path("paths.csv")
            if not paths_csv_path.exists():
                paths_csv_path = Path(data_dir) / "paths.csv"

        if paths_csv_path.exists():
            print(f"Loading file paths from {paths_csv_path}...")
            paths_df = pd.read_csv(paths_csv_path)

            for _, row in paths_df.iterrows():
                filename = str(row["filename"])
                rel_path = str(row["relative_path"])
                basename = Path(filename).stem

                self.paths_lookup.setdefault(filename, []).append(rel_path)
                self.paths_by_rel[rel_path] = row.to_dict()
                self.paths_by_basename.setdefault(basename, []).append(rel_path)

            print(f"  Loaded {len(self.paths_lookup)} unique filenames from paths.csv")
        else:
            print("  Note: paths.csv not found; using fallback resolution")

        # Transform: -> RGB tensor normalized to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _find_file_path(self, path):
        if not path:
            return None

        path_str = str(path)
        path_obj = Path(path_str)
        filename = path_obj.name
        basename = path_obj.stem

        # Strategy 1: parse SAMPLE_KEY like WeekX_BatchY_... format
        if "_" in path_str and path_str.startswith("Week"):
            parts = path_str.replace(".0", "").split("_")
            if len(parts) >= 5:
                week_part, batch_part, table_part, image_part, object_part = parts[:5]
                expected_filename = f"{table_part}_{image_part}_{object_part}.0.npy"
                expected_dir = f"{week_part}/{batch_part}"

                if self.paths_lookup and expected_filename in self.paths_lookup:
                    for rel_path in self.paths_lookup[expected_filename]:
                        rel_path_str = str(rel_path)
                        if expected_dir in rel_path_str:
                            candidates = []
                            if self.data_dir.name in rel_path_str:
                                if rel_path_str.startswith(self.data_dir.name + "/"):
                                    rel_path_clean = rel_path_str[len(self.data_dir.name) + 1:]
                                    candidates.append(self.data_dir / rel_path_clean)
                                candidates.append(self.data_dir.parent / rel_path)
                            candidates.append(Path(rel_path).resolve())
                            candidates.append(self.data_dir / rel_path)
                            candidates.append(self.data_dir.parent / rel_path)

                            for c in dict.fromkeys(candidates):
                                if c is not None and c.exists():
                                    return c

                search_dir = self.data_dir / week_part / batch_part
                if not search_dir.exists():
                    search_dir = self.data_dir.parent / week_part / batch_part
                if search_dir.exists():
                    cand = search_dir / expected_filename
                    if cand.exists():
                        return cand

        # Strategy 2: search paths.csv by SAMPLE_KEY substring
        if self.paths_lookup:
            for rel_path_key, rel_info in self.paths_by_rel.items():
                if path_str in rel_path_key or path_str.replace(".0", "") in rel_path_key:
                    rel_path = str(rel_info["relative_path"])
                    candidates = []
                    rel_path_str = str(rel_path)
                    if self.data_dir.name in rel_path_str:
                        if rel_path_str.startswith(self.data_dir.name + "/"):
                            rel_clean = rel_path_str[len(self.data_dir.name) + 1:]
                            candidates.append(self.data_dir / rel_clean)
                        candidates.append(self.data_dir.parent / rel_path)
                    candidates.append(Path(rel_path).resolve())
                    candidates.append(self.data_dir / rel_path)
                    candidates.append(self.data_dir.parent / rel_path)

                    for c in dict.fromkeys(candidates):
                        if c is not None and c.exists():
                            return c

        # Strategy 3: exact filename match
        if self.paths_lookup and filename in self.paths_lookup:
            for rel_path in self.paths_lookup[filename]:
                candidates = []
                rel_path_str = str(rel_path)
                if self.data_dir.name in rel_path_str:
                    if rel_path_str.startswith(self.data_dir.name + "/"):
                        rel_clean = rel_path_str[len(self.data_dir.name) + 1:]
                        candidates.append(self.data_dir / rel_clean)
                    candidates.append(self.data_dir.parent / rel_path)
                candidates.append(Path(rel_path).resolve())
                candidates.append(self.data_dir / rel_path)
                candidates.append(self.data_dir.parent / rel_path)

                for c in dict.fromkeys(candidates):
                    if c is not None and c.exists():
                        return c

        # Strategy 4: basename match
        if self.paths_lookup and basename in self.paths_by_basename:
            for rel_path in self.paths_by_basename[basename]:
                candidates = []
                rel_path_str = str(rel_path)
                if self.data_dir.name in rel_path_str:
                    if rel_path_str.startswith(self.data_dir.name + "/"):
                        rel_clean = rel_path_str[len(self.data_dir.name) + 1:]
                        candidates.append(self.data_dir / rel_clean)
                    candidates.append(self.data_dir.parent / rel_path)
                candidates.append(Path(rel_path).resolve())
                candidates.append(self.data_dir / rel_path)
                candidates.append(self.data_dir.parent / rel_path)

                for c in dict.fromkeys(candidates):
                    if c is not None and c.exists():
                        return c

        # Fallback: direct
        for cand in [self.data_dir / path_str, self.data_dir / (path_str + ".npy")]:
            if cand.exists():
                return cand

        # Last resort: recursive search
        search_pattern = filename if filename.endswith(".npy") else filename + ".npy"
        matches = list(self.data_dir.rglob(search_pattern))
        if matches:
            return matches[0]

        return None

    def _load_img(self, idx):
        meta = self.metadata[idx]
        path = meta.get("image_path") or meta.get("SAMPLE_KEY")
        if not path:
            raise ValueError(f"No image path found in metadata at idx={idx}")

        full_path = self._find_file_path(path)
        if full_path is None or not full_path.exists():
            raise FileNotFoundError(
                f"Image file not found!\n"
                f"  idx={idx}\n"
                f"  meta_path={path}\n"
                f"  data_dir={self.data_dir}\n"
            )

        try:
            img = np.load(str(full_path))
            original_shape = img.shape

            # CHW -> HWC if needed
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            # normalize to uint8 0..255 for PIL
            if img.max() > 1.0:
                img = img.astype(np.uint8)
            else:
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = (img * 255).astype(np.uint8)

            # grayscale -> RGB
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            if (not self._first_load_logged) or idx < 3:
                print("\n" + "=" * 60)
                print(f"✓ Loaded image idx={idx}")
                print(f"  path: {full_path}")
                print(f"  original shape: {original_shape}")
                print(f"  processed shape: {img.shape}")
                print("=" * 60 + "\n")
                if idx >= 3:
                    self._first_load_logged = True

        except Exception as e:
            raise RuntimeError(f"Failed to load npy at {full_path}: {type(e).__name__}: {e}") from e

        pil = Image.fromarray(img)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return self.transform(pil)

    def __len__(self):
        return len(self.treated)

    def __getitem__(self, idx):
        trt_idx = self.treated[idx]
        trt_meta = self.metadata[trt_idx]
        batch = trt_meta.get("BATCH", "unk")

        # random DMSO control from same batch
        if batch in self.controls and len(self.controls[batch]) > 0:
            ctrl_idx = int(np.random.choice(self.controls[batch]))
        else:
            ctrl_idx = trt_idx

        trt_img = self._load_img(trt_idx)
        ctrl_img = self._load_img(ctrl_idx)

        cpd = trt_meta.get("CPD_NAME", "DMSO")
        fp = self.fingerprints.get(cpd, np.zeros(1024, dtype=np.float32))

        return {
            "control": ctrl_img,  # pixel space (we VAE-encode later)
            "target": trt_img,    # pixel space
            "fingerprint": torch.from_numpy(fp).float(),
            "prompt": "fluorescence microscopy image of a cell",
        }


# =============================================================================
# FLUX helpers (sigma indexing)
# =============================================================================
def get_sigmas(noise_scheduler: FlowMatchEulerDiscreteScheduler, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype, device: torch.device):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device=device)
    # timesteps: [B] values from schedule_timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def lora_target_modules_default():
    return [
        "x_embedder",
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
    ]


def save_lora(transformer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state = get_peft_model_state_dict(transformer)
    if SAFETENSORS_OK:
        safetensors_save(state, os.path.join(out_dir, "transformer_lora.safetensors"))
    else:
        torch.save(state, os.path.join(out_dir, "transformer_lora.pt"))


def save_drug_proj(drug_proj: nn.Module, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if SAFETENSORS_OK:
        safetensors_save(drug_proj.state_dict(), os.path.join(out_dir, "drug_projector.safetensors"))
    else:
        torch.save(drug_proj.state_dict(), os.path.join(out_dir, "drug_projector.pt"))


# =============================================================================
# EVALUATION FUNCTIONS (Generate samples and videos)
# =============================================================================
def generate_samples_flux(pipe, controlnet, drug_proj, control_img, fingerprint, prompt, device, weight_dtype, num_inference_steps=50):
    """Generate samples using FLUX ControlNet pipeline"""
    controlnet.eval()
    drug_proj.eval()
    
    with torch.no_grad():
        # Prepare inputs
        ctrl_px = control_img.unsqueeze(0).to(device, dtype=weight_dtype)
        fp = fingerprint.unsqueeze(0).to(device, dtype=torch.float32)
        
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt([prompt], prompt_2=[prompt])
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype, device=device)
        text_ids = text_ids.to(dtype=weight_dtype, device=device)
        
        # Add drug tokens
        drug_tokens = drug_proj(fp).to(dtype=weight_dtype)
        prompt_embeds = torch.cat([prompt_embeds, drug_tokens], dim=1)
        
        drug_txt_ids = torch.zeros((1, drug_proj.num_drug_tokens, text_ids.shape[-1]), 
                                   device=device, dtype=weight_dtype)
        text_ids_b = torch.cat([text_ids.unsqueeze(0), drug_txt_ids], dim=1)
        
        # Encode control image
        ctrl_lat = pipe.vae.encode(ctrl_px).latent_dist.sample()
        ctrl_lat = (ctrl_lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        control_latents = FluxControlNetPipeline._pack_latents(
            ctrl_lat, 1, ctrl_lat.shape[1], ctrl_lat.shape[2], ctrl_lat.shape[3]
        )
        
        # Generate using pipeline
        images = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            control_image=control_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images
        
        # Convert PIL to tensor [-1, 1]
        from torchvision.transforms import ToTensor, Normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img_tensor = transform(images[0]).unsqueeze(0)
        return img_tensor


def generate_video_flux(pipe, controlnet, drug_proj, control_img, fingerprint, prompt, save_path, device, weight_dtype, num_frames=40):
    """Generate video of FLUX generation process"""
    if not IMAGEIO_AVAILABLE:
        print("  Warning: imageio not available. Skipping video generation.")
        return
    
    controlnet.eval()
    drug_proj.eval()
    
    with torch.no_grad():
        # Prepare inputs
        ctrl_px = control_img.unsqueeze(0).to(device, dtype=weight_dtype)
        fp = fingerprint.unsqueeze(0).to(device, dtype=torch.float32)
        
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt([prompt], prompt_2=[prompt])
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype, device=device)
        text_ids = text_ids.to(dtype=weight_dtype, device=device)
        
        # Add drug tokens
        drug_tokens = drug_proj(fp).to(dtype=weight_dtype)
        prompt_embeds = torch.cat([prompt_embeds, drug_tokens], dim=1)
        
        drug_txt_ids = torch.zeros((1, drug_proj.num_drug_tokens, text_ids.shape[-1]), 
                                   device=device, dtype=weight_dtype)
        text_ids_b = torch.cat([text_ids.unsqueeze(0), drug_txt_ids], dim=1)
        
        # Encode control
        ctrl_lat = pipe.vae.encode(ctrl_px).latent_dist.sample()
        ctrl_lat = (ctrl_lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        control_latents = FluxControlNetPipeline._pack_latents(
            ctrl_lat, 1, ctrl_lat.shape[1], ctrl_lat.shape[2], ctrl_lat.shape[3]
        )
        
        # Generate with callback to save frames
        frames = []
        def callback_fn(pipe, step_index, timestep, callback_kwargs):
            if step_index % (pipe.scheduler.config.num_train_timesteps // num_frames) == 0:
                latents = callback_kwargs["latents"]
                # Decode frame
                decoded = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).float()).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(img_np)
            return callback_kwargs
        
        # Generate
        images = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            control_image=control_latents,
            num_inference_steps=50,
            guidance_scale=3.5,
            callback_on_step_end=callback_fn,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images
        
        # Add final frame
        from torchvision.transforms import ToTensor
        final_img = np.array(images[0])
        frames.append(final_img)
        
        # Create side-by-side with control
        ctrl_np = np.array(pipe.vae.decode((ctrl_lat / pipe.vae.config.scaling_factor).float()).sample[0].cpu().permute(1,2,0).numpy())
        ctrl_np = ((ctrl_np + 1) / 2 * 255).astype(np.uint8)
        separator = np.zeros((ctrl_np.shape[0], 2, 3), dtype=np.uint8)
        final_frames = [np.hstack([f, separator, ctrl_np]) for f in frames]
        
        imageio.mimsave(save_path, final_frames, fps=10)
        print(f"  ✓ Video saved to: {save_path}")


def run_evaluation(pipe, controlnet, transformer, drug_proj, eval_dataset, args, device, weight_dtype, step, output_dir, accelerator):
    """Run evaluation: generate sample grid and video"""
    if not accelerator.is_main_process:
        return
    
    # Check if eval dataset is empty
    if len(eval_dataset) == 0:
        print(f"\n{'='*60}")
        print(f"EVALUATION (Step {step}) - SKIPPED")
        print(f"{'='*60}")
        print(f"  Warning: Evaluation dataset ({args.eval_split}) is empty. Skipping evaluation.")
        print(f"{'='*60}\n")
        return
    
    print(f"\n{'='*60}")
    print(f"EVALUATION (Step {step})")
    print(f"{'='*60}")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get a sample batch
    batch_size = min(4, len(eval_dataset))
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    try:
        sample_batch = next(iter(eval_loader))
    except StopIteration:
        print(f"  Warning: Could not get sample batch from evaluation dataset. Skipping evaluation.")
        return
    
    num_samples = min(4, len(sample_batch["control"]))
    ctrl_imgs = sample_batch["control"][:num_samples].to(device, dtype=weight_dtype)
    target_imgs = sample_batch["target"][:num_samples].to(device, dtype=weight_dtype)
    fps = sample_batch["fingerprint"][:num_samples].to(device)
    prompts = sample_batch["prompt"][:num_samples] if isinstance(sample_batch["prompt"], list) else [sample_batch["prompt"]] * num_samples
    
    # Generate samples
    print("  Generating samples...")
    generated_imgs = []
    for i in range(num_samples):
        gen = generate_samples_flux(pipe, controlnet, drug_proj, ctrl_imgs[i], fps[i], prompts[i], 
                                   device, weight_dtype, num_inference_steps=50)
        generated_imgs.append(gen)
    
    generated_stack = torch.cat(generated_imgs, dim=0)
    
    # Create grid: control | generated | target
    grid = torch.cat([
        (ctrl_imgs / 2 + 0.5).clamp(0, 1),
        generated_stack,
        (target_imgs / 2 + 0.5).clamp(0, 1)
    ], dim=0)
    
    grid_path = os.path.join(plots_dir, f"step_{step}.png")
    save_image(grid, grid_path, nrow=num_samples, normalize=False)
    print(f"  ✓ Sample grid saved to: {grid_path}")
    
    # Generate video
    print("  Generating video...")
    video_path = os.path.join(plots_dir, f"video_step_{step}.mp4")
    generate_video_flux(pipe, controlnet, drug_proj, ctrl_imgs[0], fps[0], prompts[0], 
                       video_path, device, weight_dtype, num_frames=40)
    
    print(f"{'='*60}\n")


# =============================================================================
# MAIN TRAINING
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--pretrained_model", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)

    # Data (same structure as your SD script)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--metadata_file", type=str, required=True)
    p.add_argument("--paths_csv", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--resolution", type=int, default=96)

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=1000, help="Run evaluation (generate samples/video) every N steps")
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"], 
                   help="Data split to use for evaluation (default: test)")

    # Drug tokens
    p.add_argument("--fingerprint_dim", type=int, default=1024)
    p.add_argument("--num_drug_tokens", type=int, default=4)

    # Training
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Precision
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], 
                   help="Mixed precision mode. Use 'no' (fp32) for stability on GH200, 'bf16' or 'fp16' for speed if supported.")
    p.add_argument("--seed", type=int, default=42)

    # ControlNet size knobs
    p.add_argument("--num_double_layers", type=int, default=4)
    p.add_argument("--num_single_layers", type=int, default=4)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, default=",".join(lora_target_modules_default()))

    # Timestep sampling
    p.add_argument("--weighting_scheme", type=str, default="logit_normal",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)

    # Guidance embedding
    p.add_argument("--guidance_scale", type=float, default=3.5)

    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # dtype
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # ----------------- Load FLUX components -----------------
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer_2", revision=args.revision)

    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", revision=args.revision, variant=args.variant)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler", revision=args.revision)

    # ControlNet (FLUX flavor) - initialize from transformer
    # FluxControlNetModel should be initialized from the transformer config
    try:
        # Try from_transformer method (if available in newer diffusers)
        controlnet = FluxControlNetModel.from_transformer(transformer)
    except (AttributeError, TypeError):
        # Fallback: create directly with transformer config
        # The API may have changed - use the config object directly
        controlnet = FluxControlNetModel(
            transformer.config
        )

    # Freeze brain
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    transformer.requires_grad_(False)

    # Pipeline for prompt encoding + packing helpers
    pipe = FluxControlNetPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
        controlnet=controlnet,
    ).to(accelerator.device)

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    # ----------------- LoRA injection -----------------
    target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    transformer.add_adapter(lora_cfg)
    for n, p_ in transformer.named_parameters():
        p_.requires_grad = ("lora" in n.lower())

    # ----------------- Drug projector hidden dim -----------------
    with torch.no_grad():
        pe, pooled, text_ids = pipe.encode_prompt(["fluorescence microscopy image of a cell"], prompt_2=["fluorescence microscopy image of a cell"])
        hidden_dim = pe.shape[-1]

    drug_proj = DrugProjector(
        fingerprint_dim=args.fingerprint_dim,
        num_drug_tokens=args.num_drug_tokens,
        hidden_dim=hidden_dim,
    ).to(accelerator.device, dtype=weight_dtype)
    drug_proj.requires_grad_(True)

    # ----------------- Dataset / Loader -----------------
    ds = PairedBBBC021Dataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        size=args.resolution,
        split=args.split,
        paths_csv=args.paths_csv,
    )

    def collate_fn(examples):
        return {
            "control": torch.stack([e["control"] for e in examples]),
            "target": torch.stack([e["target"] for e in examples]),
            "fingerprint": torch.stack([e["fingerprint"] for e in examples]),
            "prompt": [e["prompt"] for e in examples],
        }

    dl = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ----------------- Optimizer -----------------
    params = []
    params += list(controlnet.parameters())
    params += [p for p in transformer.parameters() if p.requires_grad]
    params += list(drug_proj.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Accelerate prepare
    controlnet, transformer, drug_proj, optimizer, dl = accelerator.prepare(
        controlnet, transformer, drug_proj, optimizer, dl
    )

    # Copy scheduler for timestep sampling indices
    noise_scheduler_copy = FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config)

    controlnet.train()
    transformer.train()
    drug_proj.train()

    # Load eval dataset once (for evaluation)
    eval_dataset = None
    if accelerator.is_main_process:
        eval_dataset = PairedBBBC021Dataset(
            data_dir=args.data_dir,
            metadata_file=args.metadata_file,
            size=args.resolution,
            split=args.eval_split,
            paths_csv=args.paths_csv,
        )
        
        # If eval split is empty, try fallback splits
        if len(eval_dataset) == 0:
            print(f"  Warning: Evaluation split '{args.eval_split}' is empty.")
            # Try test if val was requested, or train as last resort
            fallback_split = "test" if args.eval_split == "val" else "train"
            print(f"  Trying fallback split '{fallback_split}' for evaluation.")
            eval_dataset = PairedBBBC021Dataset(
                data_dir=args.data_dir,
                metadata_file=args.metadata_file,
                size=args.resolution,
                split=fallback_split,
                paths_csv=args.paths_csv,
            )
        
        # Run evaluation at start of training (step 0 sanity check)
        if len(eval_dataset) > 0:
            print("\n" + "="*60)
            print("🔎 Running Step 0 Sanity Check (Untrained Model)...")
            print("="*60)
            pipe.controlnet = accelerator.unwrap_model(controlnet)
            pipe.transformer = accelerator.unwrap_model(transformer)
            
            run_evaluation(
                pipe, accelerator.unwrap_model(controlnet), accelerator.unwrap_model(transformer),
                accelerator.unwrap_model(drug_proj), eval_dataset, args, accelerator.device, 
                weight_dtype, 0, args.output_dir, accelerator
            )
            print("✅ Step 0 Check Complete. Check ./{}/plots/".format(args.output_dir))
            print("="*60 + "\n")
        else:
            print("  Warning: No evaluation data available. Skipping Step 0 check.")

    global_step = 0

    while global_step < args.max_train_steps:
        for batch in dl:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(controlnet):
                ctrl_px = batch["control"].to(accelerator.device, dtype=weight_dtype)
                tgt_px = batch["target"].to(accelerator.device, dtype=weight_dtype)
                fp = batch["fingerprint"].to(accelerator.device, dtype=torch.float32)  # keep fp stable
                prompts = batch["prompt"]

                # ---- Prompt embeddings (frozen encoders) ----
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, txt_ids = pipe.encode_prompt(prompts, prompt_2=prompts)

                prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)  # [B, S, D]
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)

                # txt_ids returned as [S, 3], not batched
                txt_ids_b = txt_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1).to(device=accelerator.device, dtype=weight_dtype)

                # ---- Drug tokens ----
                drug_tokens = drug_proj(fp).to(dtype=weight_dtype)  # [B, N, D]
                prompt_embeds = torch.cat([prompt_embeds, drug_tokens], dim=1)

                drug_txt_ids = torch.zeros((txt_ids_b.shape[0], args.num_drug_tokens, txt_ids_b.shape[-1]),
                                           device=accelerator.device, dtype=weight_dtype)
                txt_ids_b = torch.cat([txt_ids_b, drug_txt_ids], dim=1)

                # ---- VAE encode target & control (frozen) ----
                with torch.no_grad():
                    tgt_lat = vae.encode(tgt_px).latent_dist.sample()
                    tgt_lat = (tgt_lat - vae.config.shift_factor) * vae.config.scaling_factor

                    ctrl_lat = vae.encode(ctrl_px).latent_dist.sample()
                    ctrl_lat = (ctrl_lat - vae.config.shift_factor) * vae.config.scaling_factor

                # ---- Pack latents for FLUX ----
                pixel_latents = FluxControlNetPipeline._pack_latents(
                    tgt_lat, tgt_px.shape[0], tgt_lat.shape[1], tgt_lat.shape[2], tgt_lat.shape[3]
                )
                control_latents = FluxControlNetPipeline._pack_latents(
                    ctrl_lat, ctrl_px.shape[0], ctrl_lat.shape[1], ctrl_lat.shape[2], ctrl_lat.shape[3]
                )

                latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                    batch_size=tgt_lat.shape[0],
                    height=tgt_lat.shape[2] // 2,
                    width=tgt_lat.shape[3] // 2,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)

                # ---- Sample timesteps ----
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

                # ---- Guidance embedding ----
                if transformer.config.guidance_embeds:
                    guidance_vec = torch.full((bsz,), args.guidance_scale, device=accelerator.device, dtype=weight_dtype)
                else:
                    guidance_vec = None

                # ---- ControlNet forward ----
                cn_block, cn_single = controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_latents,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=txt_ids_b[0],  # same ids across batch
                    img_ids=latent_image_ids,
                    return_dict=False,
                )

                # ---- Transformer forward (with CN residuals) ----
                noise_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=cn_block,
                    controlnet_single_block_samples=cn_single,
                    txt_ids=txt_ids_b[0],
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # FLUX flow-matching target
                loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"step={global_step} loss={loss.item():.6f}")

            # evaluation (generate samples/video)
            if accelerator.is_main_process and global_step > 0 and (global_step % args.eval_every == 0) and eval_dataset is not None:
                # Update pipe with unwrapped models
                pipe.controlnet = accelerator.unwrap_model(controlnet)
                pipe.transformer = accelerator.unwrap_model(transformer)
                
                run_evaluation(
                    pipe, accelerator.unwrap_model(controlnet), accelerator.unwrap_model(transformer),
                    accelerator.unwrap_model(drug_proj), eval_dataset, args, accelerator.device, 
                    weight_dtype, global_step, args.output_dir, accelerator
                )

            # checkpoint
            if accelerator.is_main_process and global_step > 0 and (global_step % args.save_every == 0):
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(ckpt_dir, "flux_controlnet"))
                save_lora(accelerator.unwrap_model(transformer), os.path.join(ckpt_dir, "lora"))
                save_drug_proj(accelerator.unwrap_model(drug_proj), os.path.join(ckpt_dir, "drug"))

                print(f"[saved] {ckpt_dir}")

            global_step += 1

    # final save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(final_dir, "flux_controlnet"))
        save_lora(accelerator.unwrap_model(transformer), os.path.join(final_dir, "lora"))
        save_drug_proj(accelerator.unwrap_model(drug_proj), os.path.join(final_dir, "drug"))
        print(f"[done] saved to {final_dir}")


if __name__ == "__main__":
    main()
