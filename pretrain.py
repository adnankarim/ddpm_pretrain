"""
================================================================================
BBBC021 UNCONDITIONAL PRETRAINING (TRAIN SET ONLY)
================================================================================
Stage 1: Learning the Cell Prior
--------------------------------------------------------------------------------
This script trains a generative model on ONLY the 'train' split of the dataset.
It ignores labels and focuses solely on learning P(x) -- what a cell looks like.

Goal: Create a "Foundation Model" that understands biological textures.
Input: Random Noise (3 channels)
Output: Realistic Cell Image (3 channels)
Data: Train Split Only (No Test Leakage)
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from pathlib import Path
from PIL import Image

# --- Plotting Backend ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Dependencies ---
try:
    from diffusers import UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: 'diffusers' library not found. Install with: pip install diffusers")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    data_dir = "./data/bbbc021_all"
    metadata_file = "metadata/bbbc021_df_all.csv"
    image_size = 96
    
    # Architecture (Modern U-Net for 96x96)
    # Initialized from scratch
    channels = (128, 256, 512, 512)
    
    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training
    epochs = 300 
    batch_size = 64
    lr = 1e-4
    save_freq = 5
    eval_freq = 5
    
    output_dir = "foundation_train_only_results"
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# LOGGING UTILS
# ============================================================================

class TrainingLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = {'epoch': [], 'loss': []}
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        
    def update(self, epoch, loss):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        
        # Save CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['loss'], label='Reconstruction Loss', color='#1f77b4', linewidth=2)
        plt.title('Unconditional Training Dynamics (Train Set Only)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

# ============================================================================
# DATASET (TRAIN SPLIT ONLY)
# ============================================================================

class BBBC021TrainOnlyDataset(Dataset):
    """
    Loads only images marked as 'train' in the CSV split column.
    Ignores labels/conditions.
    """
    def __init__(self, data_dir, metadata_file, image_size=96, split='train'):
        self.data_dir = Path(data_dir).resolve()
        self.image_size = image_size
        
        # Robust CSV Loading
        csv_path = self.data_dir / metadata_file
        if not csv_path.exists(): csv_path = Path(metadata_file)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find metadata CSV at {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # --- STRICT SPLIT FILTERING ---
        if 'SPLIT' not in df.columns:
            print("Warning: 'SPLIT' column not found. Using entire dataset.")
        else:
            original_len = len(df)
            df = df[df['SPLIT'].str.lower() == split.lower()]
            print(f"Dataset Filtered: Kept {len(df)} '{split}' samples (dropped {original_len - len(df)})")
            
        self.metadata = df.to_dict('records')
        
        # Paths lookup (Robust File Finding)
        self.paths_lookup = {}
        paths_csv = self.data_dir / "paths.csv"
        if paths_csv.exists():
            print(f"Loading paths.csv optimization...")
            p_df = pd.read_csv(paths_csv)
            for _, row in p_df.iterrows():
                self.paths_lookup[row['filename']] = row['relative_path']

    def _find_file(self, path_str):
        # 1. Try direct
        p = self.data_dir / path_str
        if p.exists(): return p
        
        # 2. Try adding extension
        p_ext = self.data_dir / (path_str + ".npy")
        if p_ext.exists(): return p_ext
        
        # 3. Try lookup
        fname = Path(path_str).name
        if fname in self.paths_lookup:
            return self.data_dir / self.paths_lookup[fname]
            
        return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        path_str = meta.get('image_path') or meta.get('SAMPLE_KEY')
        
        full_path = self._find_file(path_str)
        
        if full_path is None or not full_path.exists():
            # Robust skip during training
            return self.__getitem__(np.random.randint(0, len(self)))

        try:
            img = np.load(full_path)
            if img.ndim == 3 and img.shape[-1] == 3: 
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            
            # Normalize to [-1, 1]
            if img.max() > 1.0:
                img = (img / 127.5) - 1.0
            else:
                img = (img * 2.0) - 1.0
                
            return torch.clamp(img, -1, 1)
            
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self)))

# ============================================================================
# ARCHITECTURE (MODERN U-NET)
# ============================================================================

class UnconditionalDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # Initialize Modern U-Net from Scratch
        # No conditioning inputs (in_channels=3)
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=config.channels,
            down_block_types=(
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # Attention at 24x24
                "AttnDownBlock2D",  # Attention at 12x12
            ),
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"
            ),
            resnet_time_scale_shift="scale_shift", 
            act_fn="silu", 
            norm_num_groups=32,
        ).to(config.device)
        
        # Diffusion Schedule
        self.timesteps = config.timesteps
        beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)
        alpha = 1. - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def forward(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.cfg.device)
        noise = torch.randn_like(x0)
        
        # Add Noise
        xt = self.sqrt_alpha_bar[t].view(-1,1,1,1) * x0 + \
             self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1) * noise
        
        # Predict Noise (Pure Unconditional)
        noise_pred = self.model(xt, t).sample
        
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, num_samples=16):
        self.model.eval()
        xt = torch.randn(num_samples, 3, self.cfg.image_size, self.cfg.image_size).to(self.cfg.device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=self.cfg.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(xt, t).sample
            
            # Step
            alpha = 1 - torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.timesteps).to(self.cfg.device)[i]
            alpha_bar = self.alpha_bar[i]
            beta = 1 - alpha
            z = torch.randn_like(xt) if i > 0 else 0
            
            xt = (1 / torch.sqrt(alpha)) * (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * z
            xt = torch.clamp(xt, -1, 1)
            
        return xt

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unconditional Training (Train Set Only)")
    parser.add_argument("--output_dir", type=str, default="foundation_train_only_results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    config.output_dir = args.output_dir
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: wandb.init(project="bbbc021-foundation-train-only", config=config.__dict__)

    print(f"\n{'='*60}")
    print(f"BBBC021 UNCONDITIONAL TRAINING (TRAIN SPLIT ONLY)")
    print(f"{'='*60}")
    
    # Load ONLY Train Split
    dataset = BBBC021TrainOnlyDataset(config.data_dir, config.metadata_file, split='train')
    print(f"Training Images: {len(dataset):,}")
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print("Initializing Modern U-Net (From Scratch)...")
    model = UnconditionalDiffusion(config)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.lr)
    
    start_epoch = 0
    ckpt_path = f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']

    print("Starting Training...")
    
    for epoch in range(start_epoch, config.epochs):
        model.model.train()
        losses = []
        
        for batch_idx, images in enumerate(loader):
            images = images.to(config.device)
            
            optimizer.zero_grad()
            loss = model(images)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}", end='\r')
        
        avg_loss = np.mean(losses)
        print(f"\nEpoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.5f}")
        
        logger.update(epoch+1, avg_loss)
        if WANDB_AVAILABLE: wandb.log({"loss": avg_loss, "epoch": epoch+1})
        
        # Evaluation
        if (epoch + 1) % config.eval_freq == 0:
            print("Generating Random Samples...")
            samples = model.sample(num_samples=16)
            save_image(samples, f"{config.output_dir}/plots/epoch_{epoch+1}.png", nrow=4, normalize=True, value_range=(-1, 1))
            
        # Checkpointing
        if (epoch + 1) % config.save_freq == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'config': config.__dict__ 
            }, f"{config.output_dir}/checkpoints/latest.pt")
            print("Checkpoint Saved.")

if __name__ == "__main__":
    main()