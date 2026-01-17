"""
================================================================================
BBBC021 PARTIAL FINE-TUNING (CONDITIONAL U-NET)
================================================================================
Strategy: Partial Fine-Tuning (Sparse Tuning)
--------------------------------------------------------------------------------
Instead of training the entire U-Net (slow, overfitting risk) or just LoRA (low capacity),
we fine-tune SPECIFIC layers crucial for the task:
1. Input Layer (Adapter for 6-channel input)
2. Cross-Attention Layers (The "Drug Listeners")
3. Output Layer (Pixel refinement)
Everything else (ResNet blocks, Self-Attention) is FROZEN.

Input: Control Image + Drug Fingerprint + Noise
Output: Perturbed Image
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
    
    # Architecture
    base_model_id = "google/ddpm-cifar10-32" 
    
    # Embeddings
    fingerprint_dim = 1024
    
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
    
    output_dir = "partial_finetune_results"
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
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Plot with dual y-axis for loss and learning rate
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss on left y-axis
        color = '#1f77b4'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss', color=color)
        line1 = ax1.plot(self.history['epoch'], self.history['loss'], 
                        label='MSE Loss', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        # Learning rate on right y-axis
        if any(lr > 0 for lr in self.history['learning_rate']):
            ax2 = ax1.twinx()
            color2 = '#ff7f0e'
            ax2.set_ylabel('Learning Rate', color=color2)
            line2 = ax2.plot(self.history['epoch'], self.history['learning_rate'], 
                            label='Learning Rate', color=color2, linewidth=2, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_yscale('log')
        
        plt.title(f'Partial Fine-Tuning Dynamics (Epoch {epoch})')
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
    def __init__(self, data_dir, metadata_file, image_size=96, split='train', encoder=None, paths_csv=None):
        self.data_dir = Path(data_dir).resolve()
        self.image_size = image_size
        self.encoder = encoder
        self._first_load_logged = False
        
        # Robust CSV Loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file
            
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
        
        # Load paths.csv for robust file lookup (same as train.py)
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
                filename = str(row['filename'])
                rel_path = str(row['relative_path'])
                basename = Path(filename).stem
                
                if filename not in self.paths_lookup:
                    self.paths_lookup[filename] = []
                self.paths_lookup[filename].append(rel_path)
                
                self.paths_by_rel[rel_path] = row.to_dict()
                
                if basename not in self.paths_by_basename:
                    self.paths_by_basename[basename] = []
                self.paths_by_basename[basename].append(rel_path)
            
            print(f"  Loaded {len(self.paths_lookup)} unique filenames from paths.csv")
        else:
            print("  Note: paths.csv not found, will use fallback path resolution")

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
        return (trt_idx, trt_idx)

    def __len__(self): return len(self.metadata)

    def _find_file_path(self, path):
        """Robust file path finding using paths.csv lookup (same logic as train.py)."""
        if not path:
            return None
        
        path_str = str(path)
        path_obj = Path(path_str)
        filename = path_obj.name
        basename = path_obj.stem
        
        # Strategy 1: Parse SAMPLE_KEY format
        if '_' in path_str and path_str.startswith('Week'):
            parts = path_str.replace('.0', '').split('_')
            if len(parts) >= 5:
                week_part = parts[0]
                batch_part = parts[1]
                table_part = parts[2]
                image_part = parts[3]
                object_part = parts[4]
                
                expected_filename = f"{table_part}_{image_part}_{object_part}.0.npy"
                expected_dir = f"{week_part}/{batch_part}"
                
                if self.paths_lookup and expected_filename in self.paths_lookup:
                    for rel_path in self.paths_lookup[expected_filename]:
                        rel_path_str = str(rel_path)
                        if expected_dir in rel_path_str:
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
        
        # Strategy 3: Exact filename match
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
        
        # Strategy 4: Basename match
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
        
        return None

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        path = meta.get('image_path') or meta.get('SAMPLE_KEY')
        
        if not path:
            raise ValueError(
                f"CRITICAL: No image path found in metadata!\n"
                f"  Index: {idx}\n"
                f"  Compound: {meta.get('CPD_NAME', 'unknown')}\n"
                f"  Metadata keys: {list(meta.keys())}"
            )
        
        full_path = self._find_file_path(path)
        
        if full_path is None or not full_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Image file not found!\n"
                f"  Index: {idx}\n"
                f"  Compound: {meta.get('CPD_NAME', 'unknown')}\n"
                f"  Path from metadata: {path}\n"
                f"  Data directory: {self.data_dir}\n"
                f"  paths.csv loaded: {len(self.paths_lookup) > 0}"
            )
        
        try:
            file_size_bytes = full_path.stat().st_size if full_path.exists() else 0
            img = np.load(str(full_path))
            original_shape = img.shape
            original_dtype = img.dtype
            
            if img.ndim == 3 and img.shape[-1] == 3: 
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            
            if img.max() > 1.0: 
                img = (img / 127.5) - 1.0
            else:
                img = (img * 2.0) - 1.0
            img = torch.clamp(img, -1, 1)
            
            if not self._first_load_logged or idx < 3:
                print(f"\n{'='*60}", flush=True)
                print(f"✓ Successfully loaded image #{idx}", flush=True)
                print(f"  Compound: {meta.get('CPD_NAME', 'unknown')}", flush=True)
                print(f"  File path: {full_path}", flush=True)
                print(f"  File size: {file_size_bytes:,} bytes", flush=True)
                print(f"  Original shape: {original_shape} (dtype: {original_dtype})", flush=True)
                print(f"  Processed shape: {img.shape}", flush=True)
                print(f"{'='*60}\n", flush=True)
                if idx >= 3:
                    self._first_load_logged = True
                    
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load image file!\n"
                f"  Index: {idx}\n"
                f"  File path: {full_path}\n"
                f"  Original error: {type(e).__name__}: {str(e)}"
            ) from e

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
# ARCHITECTURE (PARTIAL FINE-TUNING)
# ============================================================================

class ConditionalUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        print(f"Loading base architecture from {Config.base_model_id}...")
        
        # 1. Load Pre-trained U-Net
        self.unet = UNet2DModel.from_pretrained(
            Config.base_model_id,
            sample_size=image_size,
            class_embed_type="identity"
        )
        
        # 2. Input Layer Surgery (3 -> 6 channels)
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            in_channels=6, # 3 Noise + 3 Control
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        # Initialize: Copy old weights to first 3 channels, zero others
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0 
            new_conv.bias = old_conv.bias
        self.unet.conv_in = new_conv
        
        # 3. Drug Projection (Fingerprint -> Time Embed Dim)
        target_dim = self.unet.time_embedding.linear_1.out_features
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.SiLU(),
            nn.Linear(512, target_dim)
        )

        # 4. PARTIAL FREEZING (The Magic Step)
        print("\nConfiguring Partial Fine-Tuning...")
        
        # A. Freeze EVERYTHING first
        for param in self.unet.parameters():
            param.requires_grad = False
            
        # B. Unfreeze Input Layer (Must train to adapt to 6 channels)
        for param in self.unet.conv_in.parameters():
            param.requires_grad = True
            
        # C. Unfreeze Output Layer (Refine pixel texture)
        for param in self.unet.conv_out.parameters():
            param.requires_grad = True
            
        # D. Unfreeze Attention Layers (The "Drug Listeners")
        unfrozen_attn_count = 0
        for name, module in self.unet.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
                unfrozen_attn_count += 1
        
        # E. Drug Projection (Must be trainable)
        for param in self.fingerprint_proj.parameters():
            param.requires_grad = True

        # Stats
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total Params: {total_params:,}")
        print(f"  Trainable:    {trainable_params:,} ({(trainable_params/total_params)*100:.1f}%)")
        print(f"  Unfrozen Attention Blocks: {unfrozen_attn_count}")

    def forward(self, x, t, control, fingerprint):
        x_in = torch.cat([x, control], dim=1)
        emb = self.fingerprint_proj(fingerprint)
        return self.unet(x_in, t, class_labels=emb).sample

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = ConditionalUNet(
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
        xt = self.sqrt_alpha_bar[t].view(-1,1,1,1) * x0 + \
             self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1) * noise
        noise_pred = self.model(xt, t, control, fingerprint)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, control, fingerprint):
        self.model.eval()
        b = control.shape[0]
        xt = torch.randn_like(control)
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.cfg.device, dtype=torch.long)
            noise_pred = self.model(xt, t, control, fingerprint)
            
            alpha = 1 - torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.timesteps).to(self.cfg.device)[i]
            alpha_bar = self.alpha_bar[i]
            beta = 1 - alpha
            z = torch.randn_like(xt) if i > 0 else 0
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

def load_checkpoint(model, optimizer, path, scheduler=None):
    if not os.path.exists(path): return 0
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=model.cfg.device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch']

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Partial Fine-Tuning of Conditional U-Net")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file")
    args = parser.parse_args()
    
    config = Config()
    if args.output_dir:
        config.output_dir = args.output_dir
    
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: wandb.init(project="bbbc021-partial-finetune", config=config.__dict__)

    print("Loading Dataset...")
    encoder = MorganFingerprintEncoder()
    train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder, paths_csv=args.paths_csv)
    val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='val', encoder=encoder, paths_csv=args.paths_csv)
    if len(val_ds) == 0: val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='test', encoder=encoder, paths_csv=args.paths_csv)
    
    train_loader = PairedDataLoader(train_ds, config.batch_size, shuffle=True)
    val_loader = PairedDataLoader(val_ds, batch_size=8, shuffle=True)

    print("Initializing Model...")
    model = DiffusionModel(config)
    # Important: Optimize ONLY trainable parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=config.lr, weight_decay=0.01)
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,
        eta_min=1e-6
    )

    # Load checkpoint
    checkpoint_path = f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch+1}...")
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Starting training from epoch 1...")
    else:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"Found checkpoint, resuming from epoch {start_epoch+1}...")
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Starting training from epoch 1...")

    print("Starting Partial Fine-Tuning...")
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
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.5f} | LR: {current_lr:.2e}", flush=True)
        logger.update(epoch+1, avg_loss, current_lr)
        if WANDB_AVAILABLE: 
            wandb.log({
                "loss": avg_loss,
                "epoch": epoch+1,
                "learning_rate": current_lr
            })

        if (epoch + 1) % config.eval_freq == 0:
            # Visualization
            model.model.eval()
            val_iter = iter(val_loader)
            batch = next(val_iter)
            ctrl = batch['control'].to(config.device)
            real_t = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            fakes = model.sample(ctrl[:8], fp[:8])
            grid = torch.cat([ctrl[:8], fakes[:8], real_t[:8]], dim=0)
            save_image(grid, f"{config.output_dir}/plots/epoch_{epoch+1}.png", nrow=8, normalize=True, value_range=(-1,1))
            generate_video(model, ctrl[0:1], fp[0:1], f"{config.output_dir}/plots/video_{epoch+1}.mp4")

        if (epoch + 1) % config.save_freq == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1
            }, f"{config.output_dir}/checkpoints/latest.pt")
            print(f"Checkpoint Saved. (LR: {current_lr:.2e})", flush=True)

if __name__ == "__main__":
    main()
