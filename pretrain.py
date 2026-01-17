"""
================================================================================
BBBC021 UNCONDITIONAL PRETRAINING (TRAIN SET ONLY)
================================================================================
Stage 1: Learning the Cell Prior
--------------------------------------------------------------------------------
This script fine-tunes a pretrained DDPM model (anton-l/ddpm-butterflies-128) 
on ONLY the 'train' split of the dataset. It ignores labels and focuses solely 
on learning P(x) -- what a cell looks like.

Goal: Create a "Foundation Model" that understands biological textures.
Input: Random Noise (3 channels)
Output: Realistic Cell Image (3 channels)
Data: Train Split Only (No Test Leakage)
Pretrained Model: anton-l/ddpm-butterflies-128
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

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    data_dir = "./data/bbbc021_all"
    metadata_file = "metadata/bbbc021_df_all.csv"
    image_size = 96
    
    # Architecture (Using pretrained DDPM model)
    pretrained_model_id = "anton-l/ddpm-butterflies-128"
    
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
        self.history = {'epoch': [], 'loss': [], 'learning_rate': []}
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        
    def update(self, epoch, loss, lr=None):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['learning_rate'].append(lr if lr is not None else 0)
        
        # Save CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Plot with dual y-axis for loss and learning rate
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss on left y-axis
        color = '#1f77b4'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss', color=color)
        line1 = ax1.plot(self.history['epoch'], self.history['loss'], 
                        label='Reconstruction Loss', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        ax1.grid(True, which="both", alpha=0.2)
        
        # Learning rate on right y-axis
        if any(lr > 0 for lr in self.history['learning_rate']):
            ax2 = ax1.twinx()
            color2 = '#ff7f0e'
            ax2.set_ylabel('Learning Rate', color=color2)
            line2 = ax2.plot(self.history['epoch'], self.history['learning_rate'], 
                            label='Learning Rate', color=color2, linewidth=2, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_yscale('log')
        
        plt.title(f'Unconditional Training Dynamics (Epoch {epoch})')
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
    Uses robust path-finding logic from train.py
    """
    def __init__(self, data_dir, metadata_file, image_size=96, split='train', paths_csv=None):
        self.data_dir = Path(data_dir).resolve()
        self.image_size = image_size
        self._first_load_logged = False  # Track if we've logged the first successful load
        
        # Robust CSV Loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file  # Try relative path
        
        if not os.path.exists(csv_full_path):
            raise FileNotFoundError(f"Cannot find metadata CSV at {csv_full_path}")
            
        df = pd.read_csv(csv_full_path)
        
        # --- STRICT SPLIT FILTERING ---
        if 'SPLIT' not in df.columns:
            print("Warning: 'SPLIT' column not found. Using entire dataset.")
        else:
            original_len = len(df)
            df = df[df['SPLIT'].str.lower() == split.lower()]
            print(f"Dataset Filtered: Kept {len(df)} '{split}' samples (dropped {original_len - len(df)})")
            
        self.metadata = df.to_dict('records')
        
        # Load paths.csv for robust file lookup (same as train.py)
        self.paths_lookup = {}  # filename -> list of relative_paths
        self.paths_by_rel = {}  # relative_path -> full info
        self.paths_by_basename = {}  # basename (without extension) -> list of paths
        
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
                filename = str(row['filename'])  # Ensure string
                rel_path = str(row['relative_path'])  # Ensure string
                basename = Path(filename).stem  # filename without extension
                
                # Lookup by exact filename
                if filename not in self.paths_lookup:
                    self.paths_lookup[filename] = []
                self.paths_lookup[filename].append(rel_path)
                
                # Lookup by relative path
                self.paths_by_rel[rel_path] = row.to_dict()
                
                # Lookup by basename (for matching without extension)
                if basename not in self.paths_by_basename:
                    self.paths_by_basename[basename] = []
                self.paths_by_basename[basename].append(rel_path)
            
            print(f"  Loaded {len(self.paths_lookup)} unique filenames from paths.csv")
        else:
            print("  Note: paths.csv not found, will use fallback path resolution")

    def _find_file_path(self, path):
        """
        Robust file path finding using paths.csv lookup (same logic as train.py).
        Returns Path object if found, None otherwise.
        """
        if not path:
            return None
        
        path_str = str(path)  # Ensure string
        path_obj = Path(path_str)
        filename = path_obj.name
        basename = path_obj.stem  # filename without extension
        
        # Strategy 1: Parse SAMPLE_KEY format (Week7_34681_7_3338_348.0 -> Week7/34681/7_3338_348.0.npy)
        if '_' in path_str and path_str.startswith('Week'):
            parts = path_str.replace('.0', '').split('_')
            if len(parts) >= 5:
                week_part = parts[0]  # Week7
                batch_part = parts[1]  # 34681
                table_part = parts[2]  # 7
                image_part = parts[3]  # 3338
                object_part = parts[4]  # 348
                
                # Construct expected filename: table_image_object.0.npy
                expected_filename = f"{table_part}_{image_part}_{object_part}.0.npy"
                expected_dir = f"{week_part}/{batch_part}"
                
                # Try to find in paths.csv
                if self.paths_lookup and expected_filename in self.paths_lookup:
                    for rel_path in self.paths_lookup[expected_filename]:
                        rel_path_str = str(rel_path)
                        # Check if this path is in the expected directory
                        if expected_dir in rel_path_str or f"{week_part}/{batch_part}" in rel_path_str:
                            # Handle path resolution
                            candidates = []
                            if self.data_dir.name in rel_path_str:
                                if rel_path_str.startswith(self.data_dir.name + '/'):
                                    rel_path_clean = rel_path_str[len(self.data_dir.name) + 1:]
                                    candidates.append(self.data_dir / rel_path_clean)
                                candidates.append(self.data_dir.parent / rel_path)
                            candidates.append(Path(rel_path).resolve())
                            candidates.append(self.data_dir / rel_path)
                            candidates.append(self.data_dir.parent / rel_path)
                            
                            candidates = list(dict.fromkeys([c for c in candidates if c is not None]))
                            for candidate in candidates:
                                if candidate.exists():
                                    return candidate
                
                # Also try direct directory search
                search_dir = self.data_dir / week_part / batch_part
                if not search_dir.exists():
                    search_dir = self.data_dir.parent / week_part / batch_part
                
                if search_dir.exists():
                    candidate = search_dir / expected_filename
                    if candidate.exists():
                        return candidate
        
        # Strategy 2: Search paths.csv by SAMPLE_KEY in relative_path
        if self.paths_lookup:
            for rel_path_key, rel_path_info in self.paths_by_rel.items():
                if path_str in rel_path_key or path_str.replace('.0', '') in rel_path_key:
                    rel_path = str(rel_path_info['relative_path'])
                    candidates = []
                    
                    rel_path_str = str(rel_path)
                    if self.data_dir.name in rel_path_str:
                        if rel_path_str.startswith(self.data_dir.name + '/'):
                            rel_path_clean = rel_path_str[len(self.data_dir.name) + 1:]
                            candidates.append(self.data_dir / rel_path_clean)
                        candidates.append(self.data_dir.parent / rel_path)
                    candidates.append(Path(rel_path).resolve())
                    candidates.append(self.data_dir / rel_path)
                    candidates.append(self.data_dir.parent / rel_path)
                    
                    candidates = list(dict.fromkeys([c for c in candidates if c is not None]))
                    for candidate in candidates:
                        if candidate.exists():
                            return candidate
        
        # Strategy 3: Exact filename match in paths.csv
        if self.paths_lookup and filename in self.paths_lookup:
            for rel_path in self.paths_lookup[filename]:
                rel_path_str = str(rel_path)
                candidates = []
                
                if self.data_dir.name in rel_path_str:
                    if rel_path_str.startswith(self.data_dir.name + '/'):
                        rel_path_clean = rel_path_str[len(self.data_dir.name) + 1:]
                        candidates.append(self.data_dir / rel_path_clean)
                    candidates.append(self.data_dir.parent / rel_path)
                
                candidates.append(Path(rel_path).resolve())
                candidates.append(self.data_dir / rel_path)
                candidates.append(self.data_dir.parent / rel_path)
                
                candidates = list(dict.fromkeys([c for c in candidates if c is not None]))
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
        
        # Strategy 4: Basename match in paths.csv
        if self.paths_lookup and basename in self.paths_by_basename:
            for rel_path in self.paths_by_basename[basename]:
                rel_path_str = str(rel_path)
                candidates = []
                
                if self.data_dir.name in rel_path_str:
                    if rel_path_str.startswith(self.data_dir.name + '/'):
                        rel_path_clean = rel_path_str[len(self.data_dir.name) + 1:]
                        candidates.append(self.data_dir / rel_path_clean)
                    candidates.append(self.data_dir.parent / rel_path)
                
                candidates.append(Path(rel_path).resolve())
                candidates.append(self.data_dir / rel_path)
                candidates.append(self.data_dir.parent / rel_path)
                
                candidates = list(dict.fromkeys([c for c in candidates if c is not None]))
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
        
        # Fallback: Direct path matching
        for candidate in [self.data_dir / path_str, self.data_dir / (path_str + '.npy')]:
            if candidate.exists():
                return candidate
        
        # Last resort: Recursive search
        search_pattern = filename if filename.endswith('.npy') else filename + '.npy'
        matches = list(self.data_dir.rglob(search_pattern))
        if matches:
            return matches[0]
        
        # Also try recursive search for SAMPLE_KEY in directory structure
        if '_' in path_str:
            parts = path_str.split('_')
            if len(parts) >= 2:
                week_part = parts[0]  # Week7
                batch_part = parts[1] if len(parts) > 1 else None  # 34681
                
                if batch_part:
                    search_dir = self.data_dir / week_part / batch_part
                    if search_dir.exists():
                        search_pattern = path_str if path_str.endswith('.npy') else path_str + '.npy'
                        matches = list(search_dir.rglob(search_pattern))
                        if matches:
                            return matches[0]
        
        return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        path = meta.get('image_path') or meta.get('SAMPLE_KEY')
        
        if not path:
            raise ValueError(
                f"CRITICAL: No image path found in metadata!\n"
                f"  Index: {idx}\n"
                f"  Metadata keys: {list(meta.keys())}"
            )
        
        # Use robust path finding (same as train.py)
        full_path = self._find_file_path(path)
        
        # CRITICAL: Check if file exists before attempting to load
        if full_path is None or not full_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Image file not found!\n"
                f"  Index: {idx}\n"
                f"  Path from metadata: {path}\n"
                f"  Data directory: {self.data_dir}\n"
                f"  Data directory exists: {self.data_dir.exists()}\n"
                f"  paths.csv loaded: {len(self.paths_lookup) > 0}"
            )
            
        try:
            # Get file size before loading
            file_size_bytes = full_path.stat().st_size if full_path.exists() else 0
            
            img = np.load(str(full_path))  # Convert Path to string for np.load
            original_shape = img.shape
            original_dtype = img.dtype
            original_min = float(img.min())
            original_max = float(img.max())
            
            # Handle [H, W, C] -> [C, H, W]
            if img.ndim == 3 and img.shape[-1] == 3: 
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            
            # Normalize [0, 255] or [0, 1] -> [-1, 1]
            if img.max() > 1.0: 
                img = (img / 127.5) - 1.0
            else:
                img = (img * 2.0) - 1.0
                
            img = torch.clamp(img, -1, 1)
            
            # Log details for first successful load (or first few)
            if not self._first_load_logged or idx < 3:
                print(f"\n{'='*60}", flush=True)
                print(f"✓ Successfully loaded image #{idx}", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"  File path: {full_path}", flush=True)
                print(f"  File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)", flush=True)
                print(f"  Original shape: {original_shape} (dtype: {original_dtype})", flush=True)
                print(f"  Original range: [{original_min:.2f}, {original_max:.2f}]", flush=True)
                print(f"  Processed shape: {img.shape} (dtype: {img.dtype})", flush=True)
                print(f"  Processed range: [{img.min():.2f}, {img.max():.2f}]", flush=True)
                print(f"{'='*60}\n", flush=True)
                if idx >= 3:
                    self._first_load_logged = True
                    
        except Exception as e:
            # Show the actual error instead of silently failing
            raise RuntimeError(
                f"CRITICAL: Failed to load image file!\n"
                f"  Index: {idx}\n"
                f"  File path: {full_path}\n"
                f"  Original error: {type(e).__name__}: {str(e)}"
            ) from e
        
        return img

# ============================================================================
# ARCHITECTURE (MODERN U-NET)
# ============================================================================

class UnconditionalDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # Load pretrained U-Net from anton-l/ddpm-butterflies-128
        # The pretrained model is for 128x128, but we'll use it for 96x96 (should work with resizing)
        print(f"Loading pretrained model from {config.pretrained_model_id}...")
        self.model = UNet2DModel.from_pretrained(
            config.pretrained_model_id,
            sample_size=config.image_size,  # Use our image size (96x96)
            ignore_mismatched_sizes=True  # Allow size mismatch
        ).to(config.device)
        print("  ✓ Pretrained model loaded successfully")
        
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
        """Generate samples using reverse diffusion"""
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
    
    @torch.no_grad()
    def sample_trajectory(self, num_samples=1, num_frames=60):
        """
        Generate samples and return frames showing the reverse diffusion process.
        Useful for creating videos.
        """
        self.model.eval()
        xt = torch.randn(num_samples, 3, self.cfg.image_size, self.cfg.image_size).to(self.cfg.device)
        frames = []
        
        # Capture frames at equidistant timesteps
        save_steps = np.linspace(0, self.timesteps-1, num_frames, dtype=int)
        
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
            
            # Save frame if at a capture point
            if i in save_steps or i == 0:
                img_np = ((xt[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                frames.append(img_np)
        
        return frames

# ============================================================================
# UTILITIES
# ============================================================================

def generate_video(model, dataset, save_path, num_frames=60):
    """
    Generate a video showing the reverse diffusion process.
    Shows: [Generated (changing)] | [Real Ground Truth from dataset]
    """
    if not IMAGEIO_AVAILABLE:
        print("Warning: imageio not available. Skipping video generation.")
        return
    
    try:
        print(f"  Generating diffusion video...", flush=True)
        
        # Get a random real image from dataset for comparison
        import random
        random_idx = random.randint(0, len(dataset) - 1)
        real_img_tensor = dataset[random_idx]  # This is already normalized to [-1, 1]
        
        # Convert real image to numpy [0, 255]
        if isinstance(real_img_tensor, torch.Tensor):
            real_img_np = ((real_img_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        else:
            real_img_np = ((real_img_tensor['image'].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        real_img_np = np.clip(real_img_np, 0, 255)
        
        # Generate diffusion trajectory frames
        frames = model.sample_trajectory(num_samples=1, num_frames=num_frames)
        
        if frames:
            # Create separator line
            separator = np.zeros((model.cfg.image_size, 2, 3), dtype=np.uint8)
            
            # Stitch frames together: [Generated] | [Real]
            final_frames = []
            for frame in frames:
                combined = np.hstack([frame, separator, real_img_np])
                final_frames.append(combined)
            
            imageio.mimsave(save_path, final_frames, fps=10)
            print(f"  ✓ Video saved to: {save_path}", flush=True)
            print(f"  Layout: [Generated (changing)] | [Real Ground Truth]", flush=True)
        else:
            print(f"  ✗ Failed to generate video frames", flush=True)
    except Exception as e:
        print(f"  ✗ Error generating video: {e}", flush=True)
        import traceback
        traceback.print_exc()

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
    dataset = BBBC021TrainOnlyDataset(config.data_dir, config.metadata_file, split='train', paths_csv=None)
    print(f"Training Images: {len(dataset):,}")
    
    # Log paths.csv status
    print(f"\n{'='*60}", flush=True)
    print(f"File Path Resolution Status:", flush=True)
    print(f"{'='*60}", flush=True)
    if len(dataset.paths_lookup) > 0:
        print(f"  ✓ paths.csv loaded successfully", flush=True)
        print(f"  - Unique filenames in lookup: {len(dataset.paths_lookup):,}", flush=True)
        print(f"  - Total paths indexed: {len(dataset.paths_by_rel):,}", flush=True)
        print(f"  - Basename lookups: {len(dataset.paths_by_basename):,}", flush=True)
    else:
        print(f"  ⚠ paths.csv not found - using fallback path resolution", flush=True)
    print(f"  - Data directory: {dataset.data_dir}", flush=True)
    print(f"  - Data directory exists: {dataset.data_dir.exists()}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Save a random dataset sample image
    print("\nSaving random dataset sample image...", flush=True)
    try:
        import random
        random_idx = random.randint(0, len(dataset) - 1)
        sample_img = dataset[random_idx]
        
        # Convert tensor to numpy image
        if isinstance(sample_img, torch.Tensor):
            img_tensor = sample_img
        else:
            img_tensor = sample_img['image'] if isinstance(sample_img, dict) else sample_img
        
        img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        img_np = np.clip(img_np, 0, 255)
        
        img_pil = Image.fromarray(img_np)
        sample_filename = f"pretrain_dataset_sample.jpg"
        absolute_path = os.path.abspath(sample_filename)
        img_pil.save(absolute_path, "JPEG", quality=95)
        print(f"  ✓ Saved random sample to: {absolute_path}", flush=True)
        print(f"  (Current working directory: {os.getcwd()})", flush=True)
        print(f"  Image shape: {img_tensor.shape}", flush=True)
    except Exception as e:
        print(f"  ⚠ Warning: Could not save sample image: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print("Initializing Modern U-Net (From Scratch)...")
    model = UnconditionalDiffusion(config)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # Learning Rate Scheduler - Cosine Annealing (recommended for diffusion models)
    # Gradually reduces LR from initial to near-zero over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,  # Full training cycle
        eta_min=1e-6  # Minimum learning rate
    )
    
    start_epoch = 0
    ckpt_path = f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f"  Resumed from epoch {start_epoch}, current LR: {scheduler.get_last_lr()[0]:.2e}")

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
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.5f} | LR: {current_lr:.2e}")
        
        # Step scheduler
        scheduler.step()
        
        logger.update(epoch+1, avg_loss, current_lr)
        if WANDB_AVAILABLE: 
            wandb.log({
                "loss": avg_loss, 
                "epoch": epoch+1,
                "learning_rate": current_lr
            })
        
        # Evaluation
        if (epoch + 1) % config.eval_freq == 0:
            print("\n" + "="*60, flush=True)
            print(f"EVALUATION (Epoch {epoch+1})", flush=True)
            print("="*60, flush=True)
            
            print("  Generating sample grid...", flush=True)
            samples = model.sample(num_samples=16)
            grid_path = f"{config.output_dir}/plots/epoch_{epoch+1}.png"
            save_image(samples, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"  ✓ Sample grid saved to: {grid_path}", flush=True)
            
            # Generate video (with ground truth comparison)
            video_path = f"{config.output_dir}/plots/video_{epoch+1}.mp4"
            generate_video(model, dataset, video_path, num_frames=60)
            
            print("="*60 + "\n", flush=True)
            
        # Checkpointing
        if (epoch + 1) % config.save_freq == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1,
                'config': config.__dict__ 
            }, f"{config.output_dir}/checkpoints/latest.pt")
            print(f"Checkpoint Saved. (LR: {scheduler.get_last_lr()[0]:.2e})")

if __name__ == "__main__":
    main()