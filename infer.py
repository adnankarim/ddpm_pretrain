"""
================================================================================
BBBC021 INFERENCE & VIDEO GENERATION
================================================================================
Generates a side-by-side video comparing:
[Control Image] -> [Generated Perturbed Cell] -> [Real Ground Truth Cell]

Usage:
    python generate_inference_video.py --checkpoint_path "ddpm_diffusers_results/checkpoints/latest.pt"
================================================================================
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import imageio
from pathlib import Path
from diffusers import UNet2DModel
import torch.nn as nn
from torch.utils.data import Dataset

# --- Dependencies Check ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not found. Chemical encoding may fail.")

# ============================================================================
# ARCHITECTURE (Must match training script exactly)
# ============================================================================

class ModifiedDiffusersUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        # Load base with default architecture (matches training script)
        # Use ignore_mismatched_sizes since we'll load our own checkpoint anyway
        self.unet = UNet2DModel.from_pretrained(
            "google/ddpm-cifar10-32",
            sample_size=image_size,
            class_embed_type="identity",
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )
        
        # Re-apply 3 -> 6 channel surgery
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0
            new_conv.bias = old_conv.bias
        self.unet.conv_in = new_conv

        # Re-apply Projection surgery - get dimension from model
        target_dim = self.unet.time_embedding.linear_1.out_features
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.SiLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, t, control, fingerprint):
        x_in = torch.cat([x, control], dim=1)
        emb = self.fingerprint_proj(fingerprint)
        output = self.unet(x_in, t, class_labels=emb).sample
        return output

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = ModifiedDiffusersUNet(
            image_size=config.image_size, 
            fingerprint_dim=config.fingerprint_dim
        ).to(config.device)
        
        self.timesteps = config.timesteps
        beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)
        alpha = 1. - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    @torch.no_grad()
    def sample_trajectory(self, control, fingerprint, num_frames=50):
        """
        Runs reverse diffusion and returns a list of frames showing the process.
        """
        self.model.eval()
        b = control.shape[0]
        # Start from random noise
        xt = torch.randn_like(control)
        frames = []
        
        # Steps to capture for video
        save_steps = np.linspace(0, self.timesteps-1, num_frames, dtype=int)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.cfg.device, dtype=torch.long)
            noise_pred = self.model(xt, t, control, fingerprint)
            
            # Coefficients
            alpha = 1 - torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.timesteps).to(self.cfg.device)[i]
            alpha_bar = self.alpha_bar[i]
            beta = 1 - alpha
            
            z = torch.randn_like(xt) if i > 0 else 0
            
            # Denoising Step: x_{t-1} = 1/sqrt(alpha) * (x_t - ...) + sigma * z
            xt = (1 / torch.sqrt(alpha)) * (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * z
            xt = torch.clamp(xt, -1, 1)
            
            if i in save_steps or i == 0:
                # Convert to numpy image [0, 255]
                img_np = ((xt[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                frames.append(img_np)
                
        return frames

# ============================================================================
# UTILITIES
# ============================================================================

class Config:
    # Minimal config for inference
    image_size = 96
    fingerprint_dim = 1024
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    device = "cuda" if torch.cuda.is_available() else "cpu"

class MorganFingerprintEncoder:
    def __init__(self, n_bits=1024):
        self.n_bits = n_bits
    def encode(self, smiles):
        if RDKIT_AVAILABLE and smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.float32)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                return arr
            except: pass
        # Fallback
        np.random.seed(hash(str(smiles)) % 2**32)
        arr = (np.random.rand(self.n_bits) > 0.5).astype(np.float32)
        return arr

def load_data_sample(data_dir, metadata_file, encoder):
    """
    Finds a random Perturbed sample and its paired Control sample from the CSV.
    """
    df = pd.read_csv(os.path.join(data_dir, metadata_file)) if os.path.exists(os.path.join(data_dir, metadata_file)) else pd.read_csv(metadata_file)
    
    # 1. Find a perturbed sample (Non-DMSO)
    perturbed_df = df[df['CPD_NAME'].str.upper() != 'DMSO']
    if len(perturbed_df) == 0:
        print("Error: No perturbed samples found in CSV.")
        return None
    
    target_row = perturbed_df.sample(1).iloc[0]
    target_batch = target_row['BATCH']
    target_compound = target_row['CPD_NAME']
    
    # 2. Find a control sample (DMSO) from SAME BATCH
    control_candidates = df[(df['BATCH'] == target_batch) & (df['CPD_NAME'].str.upper() == 'DMSO')]
    
    if len(control_candidates) == 0:
        print(f"Warning: No control found for batch {target_batch}. Using target as dummy control.")
        control_row = target_row
    else:
        control_row = control_candidates.sample(1).iloc[0]
        
    print(f"Selected Pair from Batch {target_batch}:")
    print(f"  Control: DMSO")
    print(f"  Target:  {target_compound}")

    # 3. Load Images
    def load_img(row):
        path = row.get('image_path') or row.get('SAMPLE_KEY')
        if not path:
            raise ValueError(f"No image path found in row: {row.get('CPD_NAME', 'unknown')}")
        
        print(f"  Looking for image with path: {path}")
        
        # Try multiple path variations
        data_dir_path = Path(data_dir).resolve()
        possible_paths = [
            data_dir_path / path,  # Direct path
            data_dir_path / (path + '.npy'),  # With .npy extension
            Path(path).resolve() if Path(path).is_absolute() else None,  # Absolute path
            data_dir_path / Path(path).name,  # Just filename
            data_dir_path / (Path(path).name + '.npy'),  # Just filename with .npy
        ]
        
        # Remove None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        full_path = None
        for p in possible_paths:
            if p.exists():
                full_path = p
                print(f"  Found image at: {full_path}")
                break
        
        # If still not found, try recursive search
        search_pattern = None
        if full_path is None:
            search_pattern = Path(path).name + '.npy' if not path.endswith('.npy') else Path(path).name
            matches = list(data_dir_path.rglob(search_pattern))
            if matches:
                full_path = matches[0]
                print(f"  Found image via recursive search at: {full_path}")
        
        if full_path is None:
            print(f"Error: Could not find image file. Tried paths:")
            for p in possible_paths:
                exists = "✓" if p.exists() else "✗"
                print(f"  {exists} {p}")
            if search_pattern:
                print(f"  Also searched recursively for: {search_pattern}")
            raise FileNotFoundError(f"Image file not found for: {row.get('CPD_NAME', 'unknown')} (path: {path})")
        
        img = np.load(full_path)
        if img.ndim == 3 and img.shape[-1] == 3: img = img.transpose(2, 0, 1) # HWC to CHW
        img = torch.from_numpy(img).float()
        if img.max() > 1.0: img = (img / 127.5) - 1.0 # Normalize
        return torch.clamp(img, -1, 1).unsqueeze(0) # Add batch dim

    ctrl_tensor = load_img(control_row)
    real_tensor = load_img(target_row)
    
    # 4. Encode Fingerprint
    smiles = target_row.get('SMILES', '')
    fp = encoder.encode(smiles)
    fp_tensor = torch.from_numpy(fp).float().unsqueeze(0)
    
    return ctrl_tensor, real_tensor, fp_tensor, target_compound

# ============================================================================
# MAIN INFERENCE
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="ddpm_diffusers_results/checkpoints/latest.pt", help="Path to checkpoint file (default: ddpm_diffusers_results/checkpoints/latest.pt)")
    parser.add_argument("--data_dir", type=str, default="./data/bbbc021_all")
    parser.add_argument("--metadata_file", type=str, default="metadata/bbbc021_df_all.csv")
    parser.add_argument("--output_path", type=str, default="inference_video.mp4")
    args = parser.parse_args()

    config = Config()
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    model = DiffusionModel(config)
    checkpoint = torch.load(args.checkpoint_path, map_location=config.device)
    # Use strict=False to allow partial loading if architecture differs
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
    model.eval()
    
    # 2. Load Data
    encoder = MorganFingerprintEncoder()
    data = load_data_sample(args.data_dir, args.metadata_file, encoder)
    if data is None: return
    
    ctrl, real_target, fp, compound_name = data
    ctrl = ctrl.to(config.device)
    real_target = real_target.to(config.device)
    fp = fp.to(config.device)
    
    # 3. Generate Video Frames
    print(f"Generating diffusion trajectory for {compound_name}...")
    gen_frames = model.sample_trajectory(ctrl, fp, num_frames=60)
    
    # 4. Prepare Static Images for Comparison
    # Convert Control and Real Target to [0, 255] numpy images
    ctrl_img = ((ctrl[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
    real_img = ((real_target[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
    
    # 5. Stitch Frames Together
    final_video = []
    separator = np.zeros((96, 2, 3), dtype=np.uint8) # Black line separator
    
    for frame in gen_frames:
        # Layout: [Control] | [Generated (Changing)] | [Real Ground Truth]
        combined = np.hstack([ctrl_img, separator, frame, separator, real_img])
        final_video.append(combined)
        
    # 6. Save
    imageio.mimsave(args.output_path, final_video, fps=10)
    print(f"Video saved to {args.output_path}")
    print("Left: Source (DMSO) | Middle: Generated Prediction | Right: Real Ground Truth")

if __name__ == "__main__":
    main()