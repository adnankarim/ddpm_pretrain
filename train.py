"""
================================================================================
BBBC021 PRETRAINING WITH MODIFIED PRETRAINED U-NET (Diffusers)
================================================================================
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from pathlib import Path

# --- Plotting Backend (Prevents crashes on headless servers) ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt

# --- NEW IMPORTS ---
try:
    from diffusers import UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: 'diffusers' library not found. Install with: pip install diffusers")
    sys.exit(1)

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    data_dir = "./data/bbbc021_all"
    metadata_file = "metadata/bbbc021_df_all.csv"
    image_size = 96
    
    # Architecture (Modified Diffusers)
    base_model_id = "google/ddpm-cifar10-32" 
    
    # Embeddings
    perturbation_emb_dim = 128 
    fingerprint_dim = 1024
    
    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training
    epochs = 500
    batch_size = 64
    lr = 1e-4
    save_freq = 1
    eval_freq = 5
    
    output_dir = "ddpm_diffusers_results"
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# LOGGING UTILS (NEW)
# ============================================================================

class TrainingLogger:
    """
    Logs training metrics to CSV and generates plots every epoch.
    Uses log scale for better visualization of diffusion loss dynamics.
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = {
            'epoch': [],
            'train_loss': []
        }
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        
    def update(self, epoch, loss):
        # Update internal history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(loss)
        
        # Save to CSV immediately
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Generate Plot
        self._plot()
        
    def _plot(self):
        plt.figure(figsize=(10, 6))
        # Plot Loss with log scale (better for diffusion training)
        plt.plot(self.history['epoch'], self.history['train_loss'], 
                 label='MSE Loss (Proxy for KL)', color='#1f77b4', linewidth=2)
        
        plt.title(f'DDPM Training Dynamics (Epoch {self.history["epoch"][-1]})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale is often better for diffusion loss
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        # Save
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

# ============================================================================
# DATASET & ENCODER
# ============================================================================

class MorganFingerprintEncoder:
    def __init__(self, n_bits=1024):
        self.n_bits = n_bits
        self.cache = {}

    def encode(self, smiles):
        if isinstance(smiles, list): return np.array([self.encode(s) for s in smiles])
        if smiles in self.cache: return self.cache[smiles]

        if RDKIT_AVAILABLE and smiles and smiles not in ['DMSO', '']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.float32)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                self.cache[smiles] = arr
                return arr
            except: pass
        
        np.random.seed(hash(str(smiles)) % 2**32)
        arr = (np.random.rand(self.n_bits) > 0.5).astype(np.float32)
        self.cache[smiles] = arr
        return arr

class BBBC021Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, image_size=96, split='train', encoder=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.encoder = encoder
        
        # Robust CSV loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file  # Try relative path
            
        df = pd.read_csv(csv_full_path)
        if 'SPLIT' in df.columns: 
            df = df[df['SPLIT'].str.lower() == split.lower()]
        
        self.metadata = df.to_dict('records')
        self.batch_map = self._group_by_batch()
        
        # Pre-encode chemicals
        self.fingerprints = {}
        if 'CPD_NAME' in df.columns:
            for cpd in df['CPD_NAME'].unique():
                row = df[df['CPD_NAME'] == cpd].iloc[0]
                smiles = row.get('SMILES', '')
                self.fingerprints[cpd] = self.encoder.encode(smiles)

    def _group_by_batch(self):
        groups = {}
        for idx, row in enumerate(self.metadata):
            b = row.get('BATCH', 'unknown')
            if b not in groups: groups[b] = {'ctrl': [], 'trt': []}
            
            cpd = str(row.get('CPD_NAME', '')).upper()
            if cpd == 'DMSO': 
                groups[b]['ctrl'].append(idx)
            else: 
                groups[b]['trt'].append(idx)
        return groups

    def get_perturbed_indices(self):
        return [i for i, m in enumerate(self.metadata) if str(m.get('CPD_NAME', '')).upper() != 'DMSO']

    def get_paired_sample(self, trt_idx):
        batch = self.metadata[trt_idx].get('BATCH', 'unknown')
        if batch in self.batch_map and self.batch_map[batch]['ctrl']:
            ctrls = self.batch_map[batch]['ctrl']
            return (np.random.choice(ctrls), trt_idx)
        return (trt_idx, trt_idx)  # Fallback: use self as control if none found

    def __len__(self): return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        path = meta.get('image_path') or meta.get('SAMPLE_KEY')
        
        # Try finding the file
        full_path = self.data_dir / path
        if not full_path.exists():
            full_path = self.data_dir / (str(path) + '.npy')
            
        try:
            img = np.load(full_path)
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
        except Exception:
            # Silent fallback for robust training
            img = torch.zeros((3, self.image_size, self.image_size))

        cpd = meta.get('CPD_NAME', 'DMSO')
        fp = self.fingerprints.get(cpd, np.zeros(1024))
        
        return {
            'image': img, 
            'fingerprint': torch.from_numpy(fp).float(), 
            'compound': cpd
        }

class PairedDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.ds = dataset
        self.bs = batch_size
        self.indices = self.ds.get_perturbed_indices()
        self.shuffle = shuffle
        if len(self.indices) == 0:
            print("Warning: No perturbed samples found. DataLoader will be empty.")
    
    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.bs):
            batch_idx = self.indices[i:i+self.bs]
            ctrls, trts, fps, names = [], [], [], []
            for tidx in batch_idx:
                cidx, tidx = self.ds.get_paired_sample(tidx)
                ctrls.append(self.ds[cidx]['image'])
                trts.append(self.ds[tidx]['image'])
                fps.append(self.ds[tidx]['fingerprint'])
                names.append(self.ds[tidx]['compound'])
            
            if not ctrls: continue
            
            yield {
                'control': torch.stack(ctrls), 
                'perturbed': torch.stack(trts), 
                'fingerprint': torch.stack(fps), 
                'compound': names
            }
            
    def __len__(self): 
        if self.bs == 0: return 0
        return (len(self.indices) + self.bs - 1) // self.bs

# ============================================================================
# ARCHITECTURE
# ============================================================================

class ModifiedDiffusersUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        print(f"Loading base architecture from {Config.base_model_id}...")
        # Load pretrained model with its default architecture, then modify
        self.unet = UNet2DModel.from_pretrained(
            Config.base_model_id,
            sample_size=image_size,
            class_embed_type="identity"
        )
        
        # 
        # Surgery: 3 -> 6 channels
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
        print("  ✓ Network Surgery: Input expanded 3 -> 6 channels")

        # Surgery: Projection - get actual class embedding dimension from model
        # The model uses class_labels, so we need to match its embedding dimension
        # For identity class_embed_type, it uses time_embedding_dim
        target_dim = self.unet.time_embedding.linear_1.out_features
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.SiLU(),
            nn.Linear(512, target_dim)
        )
        print(f"  ✓ Projection Added: {fingerprint_dim} -> {target_dim}")

    def forward(self, x, t, control, fingerprint):
        # Concatenate noisy image with control
        x_in = torch.cat([x, control], dim=1)
        # Project fingerprint to time embedding dimension
        emb = self.fingerprint_proj(fingerprint)
        # Pass fingerprint embedding as 'class_labels'
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
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def forward(self, x0, control, fingerprint):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.cfg.device)
        noise = torch.randn_like(x0)
        
        # Forward Diffusion: q(x_t | x_0)
        xt = self.sqrt_alpha_bar[t].view(-1,1,1,1) * x0 + \
             self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1) * noise
        
        # Prediction
        noise_pred = self.model(xt, t, control, fingerprint)
        
        # Simple MSE Loss (Proxy for KL Divergence)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, control, fingerprint):
        """Generate a sample using reverse diffusion"""
        self.model.eval()
        b = control.shape[0]
        xt = torch.randn_like(control)
        
        # Reverse Diffusion: p(x_{t-1} | x_t)
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.cfg.device, dtype=torch.long)
            noise_pred = self.model(xt, t, control, fingerprint)
            
            alpha = 1 - torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.timesteps).to(self.cfg.device)[i]
            alpha_bar = self.alpha_bar[i]
            beta = 1 - alpha
            z = torch.randn_like(xt) if i > 0 else 0
            
            # Update step
            xt = (1 / torch.sqrt(alpha)) * (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * z
            xt = torch.clamp(xt, -1, 1)
        return xt

# ============================================================================
# UTILITIES
# ============================================================================

def generate_video(model, control, fingerprint, save_path):
    """Generate a video showing the reverse diffusion process"""
    if not IMAGEIO_AVAILABLE: return
    model.model.eval()
    xt = torch.randn_like(control)
    frames = []
    
    # Capture 40 equidistant frames
    save_steps = np.linspace(0, model.timesteps-1, 40, dtype=int)
    
    with torch.no_grad():
        for i in reversed(range(model.timesteps)):
            t = torch.full((1,), i, device=model.cfg.device, dtype=torch.long)
            noise_pred = model.model(xt, t, control, fingerprint)
            
            alpha = 1 - torch.linspace(model.cfg.beta_start, model.cfg.beta_end, model.timesteps).to(model.cfg.device)[i]
            alpha_bar = model.alpha_bar[i]
            beta = 1 - alpha
            z = torch.randn_like(xt) if i > 0 else 0
            xt = (1 / torch.sqrt(alpha)) * (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * z
            xt = torch.clamp(xt, -1, 1)
            
            if i in save_steps or i==0:
                img_np = ((xt[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                frames.append(img_np)
    
    ctrl_np = ((control[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
    final_frames = [np.concatenate([f, ctrl_np], axis=1) for f in frames]
    imageio.mimsave(save_path, final_frames, fps=10)

def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path): return 0
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=model.cfg.device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train DDPM model on BBBC021 dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume from (default: auto-loads latest.pt)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: ddpm_diffusers_results)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()
    
    config = Config()
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
        print(f"Using output directory: {config.output_dir}")
    
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    
    # Initialize Logger
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: wandb.init(project="bbbc021-diffusers-pretrain", config=config.__dict__)

    print("Loading Dataset...")
    import sys
    sys.stdout.flush()  # Ensure output is flushed
    
    encoder = MorganFingerprintEncoder()
    train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder)
    if len(train_ds) == 0: train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='', encoder=encoder)
    val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='val', encoder=encoder)
    if len(val_ds) == 0: val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='test', encoder=encoder)

    # Print dataset details
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset Details:", flush=True)
    print(f"{'='*60}", flush=True)
    
    try:
        train_count = len(train_ds)
        val_count = len(val_ds)
        print(f"Train split: {train_count} samples", flush=True)
        print(f"Val/Test split: {val_count} samples", flush=True)
        print(f"Total samples: {train_count + val_count}", flush=True)
        
        # Count compounds
        if hasattr(train_ds, 'metadata') and train_ds.metadata:
            train_compounds = len(set([m.get('CPD_NAME', '') for m in train_ds.metadata]))
            val_compounds = len(set([m.get('CPD_NAME', '') for m in val_ds.metadata]))
            print(f"Train compounds: {train_compounds}", flush=True)
            print(f"Val/Test compounds: {val_compounds}", flush=True)
            
            # Count batches
            train_batches = len(set([m.get('BATCH', '') for m in train_ds.metadata]))
            val_batches = len(set([m.get('BATCH', '') for m in val_ds.metadata]))
            print(f"Train batches: {train_batches}", flush=True)
            print(f"Val/Test batches: {val_batches}", flush=True)
            
            # Count DMSO vs perturbed
            train_dmso = sum([1 for m in train_ds.metadata if str(m.get('CPD_NAME', '')).upper() == 'DMSO'])
            train_perturbed = len(train_ds.metadata) - train_dmso
            val_dmso = sum([1 for m in val_ds.metadata if str(m.get('CPD_NAME', '')).upper() == 'DMSO'])
            val_perturbed = len(val_ds.metadata) - val_dmso
            print(f"Train - DMSO: {train_dmso}, Perturbed: {train_perturbed}", flush=True)
            print(f"Val/Test - DMSO: {val_dmso}, Perturbed: {val_perturbed}", flush=True)
        else:
            print("Warning: Could not access dataset metadata for detailed statistics", flush=True)
        
        print(f"{'='*60}\n", flush=True)
    except Exception as e:
        print(f"Error printing dataset details: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n", flush=True)

    # Save a random dataset image as JPG
    print("Saving random dataset sample image...")
    try:
        import random
        from PIL import Image
        
        # Get a random sample from train dataset
        random_idx = random.randint(0, len(train_ds) - 1)
        sample = train_ds[random_idx]
        
        # Convert tensor to numpy image
        img_tensor = sample['image']  # Shape: [3, H, W], range [-1, 1]
        img_np = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        img_np = np.clip(img_np, 0, 255)
        
        # Save as JPG in current working directory (make path absolute)
        img_pil = Image.fromarray(img_np)
        sample_filename = f"dataset_sample_{sample['compound'].replace('/', '_').replace(' ', '_')}.jpg"
        sample_path = os.path.abspath(sample_filename)  # Get absolute path
        img_pil.save(sample_path, "JPEG", quality=95)
        print(f"  Saved random sample to: {sample_path}")
        print(f"  (Current working directory: {os.getcwd()})")
        print(f"  Compound: {sample['compound']}")
        print(f"  Image shape: {img_tensor.shape}")
    except Exception as e:
        print(f"  Warning: Could not save sample image: {e}")
        import traceback
        traceback.print_exc()

    train_loader = PairedDataLoader(train_ds, config.batch_size, shuffle=True)
    val_loader = PairedDataLoader(val_ds, batch_size=8, shuffle=True)

    print(f"Initializing Modified Diffusers U-Net...")
    model = DiffusionModel(config)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.lr)

    # Load checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint else f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume or args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch+1}...")
        else:
            print(f"Starting training from epoch 1 (checkpoint not found or empty)...")
    else:
        # Only auto-load if checkpoint exists
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        if start_epoch > 0:
            print(f"Found checkpoint, resuming from epoch {start_epoch+1}...")
        else:
            print(f"Starting training from epoch 1...")

    for epoch in range(start_epoch, config.epochs):
        model.model.train()
        losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            ctrl = batch['control'].to(config.device)
            target = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            loss = model(target, ctrl, fp)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.5f}")
        
        # LOGGING & PLOTTING
        logger.update(epoch+1, avg_loss)
        if WANDB_AVAILABLE: wandb.log({"loss": avg_loss, "epoch": epoch+1})

        if (epoch + 1) % config.eval_freq == 0:
            print("Evaluation & Visualization...")
            val_iter = iter(val_loader)
            batch = next(val_iter)
            ctrl = batch['control'].to(config.device)
            real_t = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            fakes = model.sample(ctrl, fp)
            
            grid = torch.cat([ctrl[:8], fakes[:8], real_t[:8]], dim=0)
            save_image(grid, f"{config.output_dir}/plots/epoch_{epoch+1}.png", nrow=8, normalize=True, value_range=(-1,1))
            generate_video(model, ctrl[0:1], fp[0:1], f"{config.output_dir}/plots/video_{epoch+1}.mp4")

        # CHECKPOINTING
        if (epoch + 1) % config.save_freq == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1}, 
                       f"{config.output_dir}/checkpoints/latest.pt")

if __name__ == "__main__":
    main()