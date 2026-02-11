"""
================================================================================
BBBC021 PRETRAINING WITH MODIFIED PRETRAINED U-NET (Diffusers)
================================================================================
"""

import os
import sys
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

# --- Plotting Backend (Prevents crashes on headless servers) ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt

# --- NEW IMPORTS ---
import inspect
try:
    from diffusers import UNet2DConditionModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: 'diffusers' library not found. Install with: pip install diffusers")
    sys.exit(1)


from diffusers.utils import is_peft_available

try:
    from peft import LoraConfig
    DIFFUSERS_LORA_AVAILABLE = True
except ImportError:
    try:
        from diffusers.models.lora import LoraConfig
        DIFFUSERS_LORA_AVAILABLE = True
    except ImportError:
        try:
            from diffusers import LoraConfig
            DIFFUSERS_LORA_AVAILABLE = True
        except ImportError:
            DIFFUSERS_LORA_AVAILABLE = False



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
    from rdkit import Chem, DataStructs
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

    # Architecture / Attention
    cross_attention_dim = 128
    attention_head_dim = 64
    unet_block_out_channels = (128, 256, 512, 512)

    # Embeddings
    fingerprint_dim = 1024
    num_fp_tokens = 4
    num_ctrl_tokens = 4
    
    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training
    epochs = 500
    epochs_each = 50  # per-phase (theta / phi)
    batch_size = 64
    lr = 3e-5  # Lower LR when using pretrained weights to prevent drift
    save_freq = 1
    eval_freq = 5
    calculate_fid = True  # Set to True to enable FID calculation (slower evaluation)
    skip_metrics_during_training = False  # If True, skip metric calculations during training (only generate samples/video)

    # LoRA / trainable-selection
    lora_rank = 16
    train_conv_in = False  # optional fine-tuning of conv_in
    
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
            'train_loss': [],
            'kl_divergence': [],
            'mse_loss': [],
            'psnr': [],
            'ssim': [],
            'fid': [],
            'cfid': [],
            'kid_mean': [],
            'kid_std': [],
            'avg_fid': [],
            'avg_kid_mean': [],
            'avg_kid_std': [],
            'learning_rate': []
        }
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        self.metrics_plot_path = os.path.join(save_dir, "training_metrics.png")
        self.metrics_csv_path = os.path.join(save_dir, "evaluation_metrics.csv")
        
    def update(self, epoch, loss, metrics=None, lr=None):
        """
        Update logger with training loss and optional metrics.
        
        Args:
            epoch: Current epoch number
            loss: Training loss (MSE)
            metrics: Optional dict with keys like 'kl_divergence', 'psnr', 'ssim'
            lr: Current learning rate
        """
        # Update internal history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(loss)
        self.history['mse_loss'].append(loss)  # MSE is the training loss
        self.history['learning_rate'].append(lr if lr is not None else 0)
        
        # Add metrics if provided
        if metrics:
            self.history['kl_divergence'].append(metrics.get('kl_divergence', None))
            self.history['psnr'].append(metrics.get('psnr', None))
            self.history['ssim'].append(metrics.get('ssim', None))
            self.history['fid'].append(metrics.get('fid', None))
            self.history['cfid'].append(metrics.get('cfid', None))
            self.history['kid_mean'].append(metrics.get('kid_mean', None))
            self.history['kid_std'].append(metrics.get('kid_std', None))
            self.history['avg_fid'].append(metrics.get('avg_fid', None))
            self.history['avg_kid_mean'].append(metrics.get('avg_kid_mean', None))
            self.history['avg_kid_std'].append(metrics.get('avg_kid_std', None))
        else:
            self.history['kl_divergence'].append(None)
            self.history['psnr'].append(None)
            self.history['ssim'].append(None)
            self.history['fid'].append(None)
            self.history['cfid'].append(None)
            self.history['kid_mean'].append(None)
            self.history['kid_std'].append(None)
            self.history['avg_fid'].append(None)
            self.history['avg_kid_mean'].append(None)
            self.history['avg_kid_std'].append(None)
        
        # Save to CSV immediately
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Also save metrics to a separate file for easy tracking (only when metrics exist)
        if metrics and any(v is not None for v in metrics.values()):
            metrics_df = pd.DataFrame([{
                'epoch': epoch,
                'kl_divergence': metrics.get('kl_divergence'),
                'mse_gen_real': metrics.get('mse'),
                'psnr': metrics.get('psnr'),
                'ssim': metrics.get('ssim'),
                'fid': metrics.get('fid'),
                'kid_mean': metrics.get('kid_mean'),
                'kid_std': metrics.get('kid_std'),
                'cfid': metrics.get('cfid'), # This will now be average per-class FID
                'avg_fid': metrics.get('avg_fid'), # Same as cfid basically, but explicit
                'avg_kid_mean': metrics.get('avg_kid_mean'),
                'avg_kid_std': metrics.get('avg_kid_std')
            }])
            metrics_csv = os.path.join(self.save_dir, "evaluation_metrics.csv")
            # Append to file if it exists, otherwise create new
            if os.path.exists(metrics_csv):
                metrics_df.to_csv(metrics_csv, mode='a', header=False, index=False)
            else:
                metrics_df.to_csv(metrics_csv, index=False)
        
        # Generate Plots
        self._plot_loss()
        if metrics and any(v is not None for v in metrics.values()):
            self._plot_metrics()
        
    def _plot_loss(self):
        """Plot training loss curve with learning rate"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss on left y-axis
        color = '#1f77b4'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss (Proxy for KL)', color=color)
        line1 = ax1.plot(self.history['epoch'], self.history['train_loss'], 
                        label='MSE Loss', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')  # Log scale is often better for diffusion loss
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
        
        plt.title(f'DDPM Training Loss & Learning Rate (Epoch {self.history["epoch"][-1]})')
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()
        
    def _plot_metrics(self):
        """Plot additional metrics (KL, PSNR, SSIM)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Filter out None values for plotting
        epochs = self.history['epoch']
        
        # KL Divergence
        kl_values = [v for v in self.history['kl_divergence'] if v is not None]
        kl_epochs = [epochs[i] for i, v in enumerate(self.history['kl_divergence']) if v is not None]
        if kl_values:
            axes[0, 0].plot(kl_epochs, kl_values, label='KL Divergence', color='#ff7f0e', linewidth=2)
            axes[0, 0].set_title('KL Divergence')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('KL Divergence')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.2)
            axes[0, 0].legend()
        
        # PSNR
        psnr_values = [v for v in self.history['psnr'] if v is not None]
        psnr_epochs = [epochs[i] for i, v in enumerate(self.history['psnr']) if v is not None]
        if psnr_values:
            axes[0, 1].plot(psnr_epochs, psnr_values, label='PSNR', color='#2ca02c', linewidth=2)
            axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PSNR (dB)')
            axes[0, 1].grid(True, alpha=0.2)
            axes[0, 1].legend()
        
        # SSIM
        ssim_values = [v for v in self.history['ssim'] if v is not None]
        ssim_epochs = [epochs[i] for i, v in enumerate(self.history['ssim']) if v is not None]
        if ssim_values:
            axes[1, 0].plot(ssim_epochs, ssim_values, label='SSIM', color='#d62728', linewidth=2)
            axes[1, 0].set_title('Structural Similarity Index')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('SSIM')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.2)
            axes[1, 0].legend()
        
        # Combined metrics view
        axes[1, 1].plot(self.history['epoch'], self.history['train_loss'], 
                       label='MSE Loss', color='#1f77b4', linewidth=2)
        if kl_values:
            axes[1, 1].plot(kl_epochs, kl_values, label='KL Divergence', color='#ff7f0e', linewidth=2)
        axes[1, 1].set_title('Loss Metrics Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.2)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.metrics_plot_path, dpi=150)
        plt.close()

# ============================================================================
# DATASET & ENCODER
# ============================================================================

class MorganFingerprintEncoder:
    def __init__(self, n_bits=1024):
        if not RDKIT_AVAILABLE:
            raise ImportError("CRITICAL: RDKit is required for Morgan fingerprints. Install RDKit.")
        self.n_bits = n_bits
        self.cache = {}

    def encode(self, smiles: str):
        if smiles is None or str(smiles).strip() == "":
            raise ValueError("CRITICAL: Empty SMILES encountered (cannot compute fingerprint).")

        smiles = str(smiles).strip()
        if smiles in self.cache:
            return self.cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"CRITICAL: RDKit failed to parse SMILES: '{smiles}'")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        self.cache[smiles] = arr
        return arr

class BBBC021Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, image_size=96, split='train', encoder=None, paths_csv=None):
        self.data_dir = Path(data_dir).resolve()
        self.image_size = image_size
        if encoder is None:
            raise ValueError("CRITICAL: encoder must be provided (MorganFingerprintEncoder).")
        self.encoder = encoder
        self._first_load_logged = False  # Track if we've logged the first successful load
        
        # Robust CSV loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file  # Try relative path

        df = pd.read_csv(csv_full_path)
        # Optional split filtering: if split is empty/None, use all rows
        if 'SPLIT' in df.columns and split is not None and str(split).strip() != "":
            split_l = str(split).strip().lower()
            df_split = df['SPLIT'].astype(str).str.lower()
            df = df[df_split == split_l]
        
        self.metadata = df.to_dict('records')
        self.batch_map = self._group_by_batch()
        
        # Pre-encode chemicals
        self.fingerprints = {}
        if 'CPD_NAME' in df.columns:
            for cpd in df['CPD_NAME'].unique():
                row = df[df['CPD_NAME'] == cpd].iloc[0]
                smiles = row.get('SMILES', None)
                if smiles is None or str(smiles).strip() == "":
                    raise ValueError(f"CRITICAL: Missing SMILES for compound '{cpd}' (cannot compute fingerprint).")
                try:
                    self.fingerprints[cpd] = self.encoder.encode(smiles)
                except Exception as e:
                    raise RuntimeError(f"CRITICAL: Failed to encode fingerprint for compound '{cpd}': {e}") from e
        
        # Load paths.csv for robust file lookup (same as infer.py)
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
                filename = row['filename']
                rel_path = row['relative_path']
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

    def get_paired_sample(self, trt_idx, deterministic: bool = False):
        batch = self.metadata[trt_idx].get('BATCH', 'unknown')
        if batch in self.batch_map and self.batch_map[batch]['ctrl']:
            ctrls = self.batch_map[batch]['ctrl']
            if deterministic:
                # stable pairing: same treated idx -> same control idx (given fixed metadata)
                cidx = ctrls[trt_idx % len(ctrls)]
            else:
                cidx = np.random.choice(ctrls)
            return (cidx, trt_idx)
        raise RuntimeError(
            "CRITICAL: No control (DMSO) found for this batch; cannot form paired sample.\n"
            f"  batch: {batch}\n"
            f"  trt_idx: {trt_idx}\n"
        )

    def __len__(self): return len(self.metadata)

    def _find_file_path(self, path):
        """
        Robust file path finding using paths.csv lookup (same logic as infer.py).
        Returns Path object if found, None otherwise.
        """
        if not path:
            return None
        
        path_obj = Path(path)
        filename = path_obj.name
        basename = path_obj.stem  # filename without extension
        
        # Strategy 1: Parse SAMPLE_KEY format (Week7_34681_7_3338_348.0 -> Week7/34681/7_3338_348.0.npy)
        if '_' in path and path.startswith('Week'):
            parts = path.replace('.0', '').split('_')
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
                            # Handle path resolution (same as infer.py)
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
                if path in rel_path_key or path.replace('.0', '') in rel_path_key:
                    rel_path = rel_path_info['relative_path']
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
        for candidate in [self.data_dir / path, self.data_dir / (path + '.npy')]:
            if candidate.exists():
                return candidate
        
        # Last resort: Recursive search
        search_pattern = filename if filename.endswith('.npy') else filename + '.npy'
        matches = list(self.data_dir.rglob(search_pattern))
        if matches:
            return matches[0]
        
        # Also try recursive search for SAMPLE_KEY in directory structure
        if '_' in path:
            parts = path.split('_')
            if len(parts) >= 2:
                week_part = parts[0]  # Week7
                batch_part = parts[1] if len(parts) > 1 else None  # 34681
                
                if batch_part:
                    search_dir = self.data_dir / week_part / batch_part
                    if search_dir.exists():
                        search_pattern = path if path.endswith('.npy') else path + '.npy'
                        matches = list(search_dir.rglob(search_pattern))
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
        
        # Use robust path finding (same as infer.py)
        full_path = self._find_file_path(path)
        
        # CRITICAL: Check if file exists before attempting to load
        if full_path is None or not full_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Image file not found!\n"
                f"  Index: {idx}\n"
                f"  Compound: {meta.get('CPD_NAME', 'unknown')}\n"
                f"  Path from metadata: {path}\n"
                f"  Data directory: {self.data_dir}\n"
                f"  Data directory exists: {self.data_dir.exists()}\n"
                f"  paths.csv loaded: {len(self.paths_lookup) > 0}"
            )
            
        try:
            # Get file size before loading
            file_size_bytes = full_path.stat().st_size if full_path.exists() else 0
            
            img = np.load(full_path)
            original_shape = img.shape
            original_dtype = img.dtype
            original_min = float(img.min())
            original_max = float(img.max())
            
            # Handle [H, W, C] -> [C, H, W]
            if img.ndim == 3 and img.shape[-1] == 3: 
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            
            # Normalize [0, 255] or [0, 1] -> [-1, 1], or leave [-1,1] as-is
            mx = float(img.max())
            mn = float(img.min())
            if mx > 1.0:
                # assume [0,255]
                img = (img / 127.5) - 1.0
            elif mn >= 0.0 and mx <= 1.0:
                # assume [0,1]
                img = (img * 2.0) - 1.0
            else:
                # assume already roughly in [-1,1]
                img = img
                
            img = torch.clamp(img, -1, 1)
            
            # Log details for first successful load (or first few)
            if not self._first_load_logged or idx < 3:
                print(f"\n{'='*60}", flush=True)
                print(f"✓ Successfully loaded image #{idx}", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"  Compound: {meta.get('CPD_NAME', 'unknown')}", flush=True)
                print(f"  File path: {full_path}", flush=True)
                print(f"  File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)", flush=True)
                print(f"  Original shape: {original_shape} (dtype: {original_dtype})", flush=True)
                print(f"  Original range: [{original_min:.2f}, {original_max:.2f}]", flush=True)
                print(f"  Processed shape: {img.shape} (dtype: {img.dtype})", flush=True)
                print(f"  Processed range: [{img.min():.2f}, {img.max():.2f}]", flush=True)
                cpd_name = meta.get('CPD_NAME', 'DMSO')
                if cpd_name not in self.fingerprints:
                    raise KeyError(f"CRITICAL: Fingerprint for compound '{cpd_name}' not precomputed in dataset.")
                print(f"  Fingerprint shape: {self.fingerprints[cpd_name].shape}", flush=True)
                print(f"{'='*60}\n", flush=True)
                if idx >= 3:
                    self._first_load_logged = True
                    
        except Exception as e:
            # Show the actual error instead of silently failing
            raise RuntimeError(
                f"CRITICAL: Failed to load image file!\n"
                f"  Index: {idx}\n"
                f"  Compound: {meta.get('CPD_NAME', 'unknown')}\n"
                f"  File path: {full_path}\n"
                f"  Original error: {type(e).__name__}: {str(e)}"
            ) from e

        cpd = meta.get('CPD_NAME', 'DMSO')
        if cpd not in self.fingerprints:
            raise KeyError(f"CRITICAL: Fingerprint for compound '{cpd}' not found in precomputed fingerprints.")
        fp = self.fingerprints[cpd]
        
        return {
            'image': img, 
            'fingerprint': torch.from_numpy(fp).float(), 
            'compound': cpd
        }

class PairedDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, deterministic_ctrl: bool = False):
        self.ds = dataset
        self.bs = batch_size
        self.indices = self.ds.get_perturbed_indices()
        self.shuffle = shuffle
        self.deterministic_ctrl = deterministic_ctrl
        if len(self.indices) == 0:
            print("Warning: No perturbed samples found. DataLoader will be empty.")
    
    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.bs):
            batch_idx = self.indices[i:i+self.bs]
            ctrls, trts, fps, names = [], [], [], []
            for tidx in batch_idx:
                cidx, tidx = self.ds.get_paired_sample(tidx, deterministic=self.deterministic_ctrl)
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

class FingerprintToTokens(nn.Module):
    def __init__(self, fp_dim=1024, token_dim=128, num_tokens=4, hidden=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_tokens * token_dim),
        )

    def forward(self, fp):
        x = self.net(fp)
        return x.view(fp.size(0), self.num_tokens, self.token_dim)


class CtrlImageToTokens(nn.Module):
    """Small encoder -> M tokens for cross-attn context."""
    def __init__(self, in_ch=3, token_dim=128, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((num_tokens, 1)),  # [B,128,M,1]
        )
        self.proj = nn.Linear(128, token_dim)

    def forward(self, x):
        h = self.conv(x).squeeze(-1).transpose(1, 2)  # [B,M,128]
        return self.proj(h)  # [B,M,D]



def enable_unet_lora(unet: UNet2DConditionModel, rank: int = 8):
    """
    Diffusers-native LoRA for attention layers.
    Works with diffusers>=0.20 and especially 0.36.0.
    """
    if not is_peft_available():
        raise RuntimeError(
            "PEFT not available but required for diffusers LoRA adapters.\n"
            "Install: pip install peft"
        )
    
    # Check if LoraConfig is available
    if not DIFFUSERS_LORA_AVAILABLE:
         raise RuntimeError(
            "LoraConfig not available. Please upgrade diffusers."
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    # unet.enable_adapters() # Some versions might not have this or it might be default

def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]



def load_uncond_init_into_cond_unet(unet: UNet2DConditionModel, ckpt_path: str, skip_conv_in: bool = True):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"CRITICAL: init checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")

    # unwrap common checkpoint formats
    if isinstance(sd, dict) and ("ema_model" in sd or "model" in sd):
        sd = sd["ema_model"] if sd.get("ema_model") is not None else sd["model"]

    if not isinstance(sd, dict):
        raise RuntimeError(f"CRITICAL: init checkpoint is not a state_dict: {type(sd)}")

    # Accept these common prefixes, in priority order:
    prefixes = [
        "unet.",
        "model.unet.",
        "model.model.unet.",
        "module.unet.",
        "module.model.unet.",
        "module.model.model.unet.",
    ]

    found_prefix = None
    for p in prefixes:
        if any(isinstance(k, str) and k.startswith(p) for k in sd.keys()):
            found_prefix = p
            break

    if found_prefix is None:
        # show a few keys to debug
        some = [k for k in list(sd.keys())[:20]]
        raise RuntimeError(
            "CRITICAL: Could not find UNet weights in init checkpoint.\n"
            f"  Expected a prefix in {prefixes}\n"
            f"  Example keys: {some}"
        )

    unet_sd = {k[len(found_prefix):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith(found_prefix)}

    # Optionally skip conv_in (used for 3->6 surgery)
    if skip_conv_in:
        unet_sd.pop("conv_in.weight", None)
        unet_sd.pop("conv_in.bias", None)

    missing, unexpected = unet.load_state_dict(unet_sd, strict=False)

    # Hard fail if we basically loaded nothing
    loaded = len(unet_sd)
    if loaded < 50:
        raise RuntimeError(
            "CRITICAL: init checkpoint load produced too few keys.\n"
            f"  prefix used: {found_prefix}\n"
            f"  keys loaded: {loaded}\n"
            f"  missing: {missing[:20]}"
        )

    # Unexpected should be empty if prefix was correct
    if len(unexpected) != 0:
        raise RuntimeError(f"CRITICAL: unexpected keys when loading init: {unexpected[:10]}")


class ConditionalWarmupUNet(nn.Module):
    """
    Input: concat(noisy_target, cond_image) => 6ch
    Cross-attn tokens: [ctrl_tokens, fp_tokens]
    """
    def __init__(self, config: Config, init_ckpt_path: str):
        super().__init__()

        down_block_types = ("DownBlock2D","DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D")
        up_block_types   = ("CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D","UpBlock2D")

        self.unet = UNet2DConditionModel(
            sample_size=config.image_size,
            in_channels=6,             # noisy(3)+cond(3)
            out_channels=3,
            layers_per_block=2,
            block_out_channels=config.unet_block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
        )

        # conv_in surgery: init from 3ch weights
        # Load init weights FIRST into a temporary 3ch model weights, then copy conv_in
        tmp = UNet2DConditionModel(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=config.unet_block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
        )
        load_uncond_init_into_cond_unet(tmp, init_ckpt_path, skip_conv_in=False)

        old_conv = tmp.conv_in
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)  # noisy channels
            new_conv.weight[:, 3:].zero_()                 # cond channels start at 0
            new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv

        # load rest weights (excluding conv_in which we already handled)
        load_uncond_init_into_cond_unet(self.unet, init_ckpt_path, skip_conv_in=True)

        # token modules
        self.fp_tokens = FingerprintToTokens(
            fp_dim=config.fingerprint_dim,
            token_dim=config.cross_attention_dim,
            num_tokens=config.num_fp_tokens,
        )
        self.ctrl_tokens = CtrlImageToTokens(
            in_ch=3,
            token_dim=config.cross_attention_dim,
            num_tokens=config.num_ctrl_tokens,
        )

        # freeze everything (we will unfreeze LoRA + token modules + optional conv_in)

        # freeze everything (we will unfreeze LoRA + token modules + optional conv_in)
        self.unet.requires_grad_(False)

        # Enable LoRA adapters (this sets LoRA params to trainable)
        enable_unet_lora(self.unet, rank=config.lora_rank)

        # allow training selected params
        for p in self.fp_tokens.parameters():
            p.requires_grad_(True)
        for p in self.ctrl_tokens.parameters():
            p.requires_grad_(True)
        
        # Optional conv_in
        if config.train_conv_in:
            for p in self.unet.conv_in.parameters():
                p.requires_grad_(True)


    def forward(self, x_noisy, t, cond_img, fingerprint):
        x_in = torch.cat([x_noisy, cond_img], dim=1)  # [B,6,H,W]
        tok_ctrl = self.ctrl_tokens(cond_img)         # [B,M,D]
        tok_fp   = self.fp_tokens(fingerprint)        # [B,K,D]
        tokens = torch.cat([tok_ctrl, tok_fp], dim=1) # [B,M+K,D]
        return self.unet(x_in, t, encoder_hidden_states=tokens).sample

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = ConditionalWarmupUNet(config, init_ckpt_path=config.init_ckpt).to(config.device)
        
        # Use Diffusers' DDPMScheduler for consistent training/sampling
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        self.timesteps = config.timesteps

    def forward(self, x0, control, fingerprint):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.cfg.device).long()
        noise = torch.randn_like(x0)
        
        # Forward Diffusion: Use scheduler's add_noise for consistency
        xt = self.noise_scheduler.add_noise(x0, noise, t)
        
        # Prediction
        noise_pred = self.model(xt, t, control, fingerprint)
        
        # Simple MSE Loss (Proxy for KL Divergence)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, control, fingerprint, num_inference_steps=None, generator: torch.Generator = None):
        """Generate a sample using reverse diffusion with DDPMScheduler"""
        self.model.eval()
        b, c, h, w = control.shape
        xt = torch.randn((b, 3, h, w), device=self.cfg.device, generator=generator)
        
        # Use scheduler for consistent sampling
        steps = num_inference_steps or self.timesteps
        self.noise_scheduler.set_timesteps(steps, device=self.cfg.device)
        
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((b,), t, device=self.cfg.device, dtype=torch.long)
            noise_pred = self.model(xt, t_batch, control, fingerprint)
            xt = self.noise_scheduler.step(noise_pred, t, xt).prev_sample
        
        # Clamp only once at the end
        return xt.clamp(-1, 1)

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_kl_divergence(noise_pred, noise_true):
    """
    Calculate KL divergence between predicted and true noise distributions.
    For diffusion models, this is approximated as the MSE loss scaled appropriately.
    """
    # KL divergence approximation: 0.5 * MSE (assuming Gaussian distributions)
    mse = F.mse_loss(noise_pred, noise_true)
    # More accurate KL: 0.5 * (MSE / variance) where variance is typically 1.0
    kl = 0.5 * mse
    return kl.item()


def seed_all(seed: int = 42):
    """Seed python, numpy, and torch (CPU/GPU) for deterministic eval."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Stronger determinism for eval (may affect performance but improves repeatability)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_metrics(model, val_loader, device, num_samples=1000, calculate_fid_flag=False, num_inference_steps=200, skip_other_metrics=False, direction="theta"):
    """
    Calculate evaluation metrics on validation set.
    
    Args:
        model: The diffusion model
        val_loader: Validation data loader
        device: torch device
        num_samples: Number of samples to use for evaluation
        calculate_fid_flag: If True, calculate FID and KID (slower).
        num_inference_steps: Number of inference steps for generation
        skip_other_metrics: If True, skip KL, MSE, PSNR, SSIM and only calculate FID/KID
    
    Returns:
        dict with keys: 'kl_divergence', 'mse', 'psnr', 'ssim', 'fid', 'kid', 'cfid', 'avg_fid', ...
    """
    # Make evaluation as deterministic as reasonably possible
    seed_all(42)
    model.model.eval()
    metrics = {
        'kl_divergence': [],
        'mse': [],
        'psnr': [],
        'ssim': [],
        'fid': None,
        'kid_mean': None,
        'kid_std': None,
        'cfid': None,
        'avg_fid': None,
        'avg_kid_mean': None,
        'avg_kid_std': None
    }
    
    # Try to import scikit-image for SSIM/PSNR
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
    
    # FID/KID metrics (initialized lazily after we know how many samples we'll use)
    fid_metric = None
    kid_metric = None

    # Dedicated generators so sampling, KL noise, and timestep draws don't interfere
    gen_sample = torch.Generator(device=device).manual_seed(1234)
    gen_kl = torch.Generator(device=device).manual_seed(5678)
    gen_t = torch.Generator(device=device).manual_seed(9012)
    
    # Stream over val_loader with reservoir sampling to bound memory and keep uniform randomness
    all_samples = []
    seen = 0
    with torch.no_grad():
        for batch in val_loader:
            b_size = batch['control'].shape[0]
            for i in range(b_size):
                sample = {
                    'control': batch['control'][i:i+1],
                    'perturbed': batch['perturbed'][i:i+1],
                    'fingerprint': batch['fingerprint'][i:i+1],
                    'compound': batch['compound'][i],
                }
                seen += 1

                if len(all_samples) < num_samples:
                    all_samples.append(sample)
                else:
                    j = random.randint(1, seen)
                    if j <= num_samples:
                        all_samples[j - 1] = sample

    print(f"  Using {len(all_samples)} samples for evaluation (requested: {num_samples})", flush=True)

    # Initialize FID and KID metrics now that we know how many samples we have
    if calculate_fid_flag and len(all_samples) >= 2:
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        kid_subset = min(100, len(all_samples))
        kid_metric = KernelInceptionDistance(subset_size=kid_subset, normalize=True).to(device)
    elif calculate_fid_flag:
        print("  Warning: Not enough samples for FID/KID (need >= 2). Skipping FID/KID.", flush=True)
    
    # Data structures for per-class FID/KID
    # compound -> {'gen': [], 'real': []}
    class_samples = {}
    
    sample_count = 0
    with torch.no_grad():
        for sample in tqdm(all_samples, desc="  Evaluating", leave=False):
            if sample_count >= num_samples:
                break
            
            ctrl = sample['control'].to(device)
            real_t = sample['perturbed'].to(device)
            fp = sample['fingerprint'].to(device)
            compound = sample['compound']  # compound name

            # Choose conditioning image and target image based on direction
            if direction == "theta":
                cond_img = ctrl
                target_img = real_t
            else:
                cond_img = real_t
                target_img = ctrl
            
            # Generate samples
            generated = model.sample(cond_img, fp, num_inference_steps=num_inference_steps, generator=gen_sample)
            
            # ----------------------------------------------------------------
            # 1. Standard Metrics (KL, MSE, PSNR, SSIM)
            # ----------------------------------------------------------------
            if not skip_other_metrics:
                # KL Divergence (approx)
                b = ctrl.shape[0]
                t = torch.randint(0, model.timesteps, (b,), device=device, generator=gen_t).long()
                noise = torch.randn_like(target_img, generator=gen_kl)
                xt = model.noise_scheduler.add_noise(target_img, noise, t)
                noise_pred = model.model(xt, t, cond_img, fp)
                kl = calculate_kl_divergence(noise_pred, noise)
                metrics['kl_divergence'].append(kl)
                
                # MSE
                mse = F.mse_loss(generated, target_img).item()
                metrics['mse'].append(mse)

                # PSNR / SSIM
                if SKIMAGE_AVAILABLE:
                    for i in range(generated.shape[0]):
                        gen_np = ((generated[i].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                        real_np = ((target_img[i].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                        gen_gray = np.mean(gen_np, axis=2)
                        real_gray = np.mean(real_np, axis=2)
                        
                        metrics['ssim'].append(ssim(real_gray, gen_gray, data_range=255))
                        metrics['psnr'].append(psnr(real_np, gen_np, data_range=255))

            # ----------------------------------------------------------------
            # 2. FID / KID Preparation
            # ----------------------------------------------------------------
            if calculate_fid_flag and fid_metric is not None and kid_metric is not None:
                # Preprocess as per paper implementation: [-1, 1] -> [0, 1] quantized
                # Real
                real_norm = torch.clamp(target_img * 0.5 + 0.5, min=0.0, max=1.0)
                real_norm = torch.floor(real_norm * 255).to(torch.float32) / 255.0
                
                # Generated
                gen_norm = torch.clamp(generated * 0.5 + 0.5, min=0.0, max=1.0)
                gen_norm = torch.floor(gen_norm * 255).to(torch.float32) / 255.0
                
                # Update Overall Metrics
                fid_metric.update(real_norm, real=True)
                fid_metric.update(gen_norm, real=False)
                kid_metric.update(real_norm, real=True)
                kid_metric.update(gen_norm, real=False)
                
                # Store for Per-Class
                if compound not in class_samples:
                    class_samples[compound] = {'gen': [], 'real': []}
                
                # Move to CPU to save memory during loop
                class_samples[compound]['real'].append(real_norm.cpu())
                class_samples[compound]['gen'].append(gen_norm.cpu())
            
            sample_count += generated.shape[0]

    # ----------------------------------------------------------------
    # 3. Compute Final Metrics
    # ----------------------------------------------------------------
    
    # Average standard metrics
    metrics['kl_divergence'] = np.mean(metrics['kl_divergence']) if metrics['kl_divergence'] else None
    metrics['mse'] = np.mean(metrics['mse']) if metrics['mse'] else None
    metrics['psnr'] = np.mean(metrics['psnr']) if metrics['psnr'] else None
    metrics['ssim'] = np.mean(metrics['ssim']) if metrics['ssim'] else None
    
    if calculate_fid_flag and fid_metric is not None and kid_metric is not None:
        print("  Calculating Overall FID/KID...", flush=True)
        # Compute Overall
        try:
            metrics['fid'] = fid_metric.compute().item()
            kid_mean, kid_std = kid_metric.compute()
            metrics['kid_mean'] = kid_mean.item()
            metrics['kid_std'] = kid_std.item()
            print(f"    Overall FID: {metrics['fid']:.4f}", flush=True)
            print(f"    Overall KID: {metrics['kid_mean']:.5f} (±{metrics['kid_std']:.5f})", flush=True)
        except Exception as e:
            print(f"    Warning: Overall FID/KID calculation failed: {e}", flush=True)

        # Compute Per-Class (FID c)
        print("  Calculating Per-Class FID/KID...", flush=True)
        fid_per_class = {}
        kid_per_class = {}
        
        # We need a new instance or reset? 
        # torchmetrics reset() clears state.
        
        for cls, samples in tqdm(class_samples.items(), desc="    Classes", leave=False):
            # Skip classes with too few samples
            if len(samples['real']) < 2 or len(samples['gen']) < 2:
                continue
                
            # Stack and move to device
            real_stack = torch.cat(samples['real'], dim=0).to(device)
            gen_stack = torch.cat(samples['gen'], dim=0).to(device)
            
            # Reset metrics
            fid_metric.reset()
            
            # Update with class samples
            fid_metric.update(real_stack, real=True)
            fid_metric.update(gen_stack, real=False)
            
            # KID subset size adjustment (respect both real and generated counts)
            current_subset_size = min(len(samples['real']), len(samples['gen']), 100)
            if current_subset_size < 2:
                continue
            # Creating new instance per class is safer for KID subset_size
            kid_metric_class = KernelInceptionDistance(subset_size=current_subset_size, normalize=True).to(device)
            kid_metric_class.update(real_stack, real=True)
            kid_metric_class.update(gen_stack, real=False)
            
            try:
                # FID
                val_fid = fid_metric.compute().item()
                fid_per_class[cls] = val_fid
                
                # KID
                val_kid_mu, val_kid_sigma = kid_metric_class.compute()
                kid_per_class[cls] = {'mean': val_kid_mu.item(), 'std': val_kid_sigma.item()}
            except Exception as e:
                # print(f"Warning: Failed for class {cls}: {e}")
                continue
        
        # Calculate Averages (FID c / Average KID)
        if fid_per_class:
            avg_fid = np.mean(list(fid_per_class.values()))
            metrics['cfid'] = avg_fid # Storing as cfid to match existing logging key expectation if any
            metrics['avg_fid'] = avg_fid
            print(f"    Average FID (FID c): {avg_fid:.4f}", flush=True)
        
        if kid_per_class:
            avg_kid_mean = np.mean([v['mean'] for v in kid_per_class.values()])
            avg_kid_std = np.mean([v['std'] for v in kid_per_class.values()])
            metrics['avg_kid_mean'] = avg_kid_mean
            metrics['avg_kid_std'] = avg_kid_std
            print(f"    Average KID: {avg_kid_mean:.5f} (±{avg_kid_std:.5f})", flush=True)
            
        # Log to file detailed breakdown if needed? 
        # (Maybe to a separate JSON like snippet? For now staying compatible with current return)
    
    return metrics

# ============================================================================
# UTILITIES
# ============================================================================

def generate_video(model, control, fingerprint, save_path, generator: torch.Generator = None):
    """Generate a video showing the reverse diffusion process using DDPMScheduler"""
    if not IMAGEIO_AVAILABLE: return
    model.model.eval()
    b, c, h, w = control.shape
    xt = torch.randn((b, 3, h, w), device=model.cfg.device, generator=generator)
    frames = []
    
    # Use scheduler for consistent sampling
    model.noise_scheduler.set_timesteps(model.timesteps, device=model.cfg.device)
    save_steps = np.linspace(0, len(model.noise_scheduler.timesteps) - 1, 40, dtype=int)
    
    with torch.no_grad():
        for i, t in enumerate(model.noise_scheduler.timesteps):
            t_batch = torch.full((b,), t, device=model.cfg.device, dtype=torch.long)
            noise_pred = model.model(xt, t_batch, control, fingerprint)
            xt = model.noise_scheduler.step(noise_pred, t, xt).prev_sample
            
            if i in save_steps or i == len(model.noise_scheduler.timesteps) - 1:
                img_np = ((xt[0].cpu().permute(1,2,0).clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
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
    parser = argparse.ArgumentParser(
        description="Train DDPM model on BBBC021 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python train.py --output_dir ./results

  # Resume training (runs evaluation first, then continues)
  python train.py --resume --output_dir ./results

  # Resume from specific checkpoint
  python train.py --checkpoint ./results/checkpoints/checkpoint_epoch_10.pt

  # Evaluate only FID on latest checkpoint with 1000 timesteps
  python train.py --eval_only --output_dir ./results --calculate_fid --fid_only --inference_steps 1000
        """
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume from (default: auto-loads latest.pt)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: ddpm_diffusers_results)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint (runs evaluation first before continuing)")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file (auto-detected if not specified)")
    parser.add_argument("--calculate_fid", action="store_true", help="Enable FID and cFID calculation during evaluation (slower but more comprehensive metrics)")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only (no training). Loads latest checkpoint from output_dir.")
    parser.add_argument("--inference_steps", type=int, default=200, help="Number of inference steps for generation (default: 200, use 1000 for full quality)")
    parser.add_argument("--fid_only", action="store_true", help="Only calculate FID metrics (skip KL, MSE, PSNR, SSIM for faster evaluation)")
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"], help="Data split to use for evaluation (default: val, falls back to test if val is empty)")
    parser.add_argument("--num_eval_samples", type=int, default=1000, help="Max number of samples to use for evaluation metrics (default: 1000)")
    parser.add_argument(
        "--direction",
        type=str,
        default="theta",
        choices=["theta", "phi"],
        help="theta: train p(X|Y,c) (treated|control). phi: train p(Y|X,c) (control|treated, specular)."
    )
    parser.add_argument("--overall_init", type=str, default="ddpm_uncond_all/theta_ref_ema_best.pt")
    parser.add_argument("--ctrl_init", type=str, default="ddpm_uncond_ctrl/theta_ctrl_ema_best.pt")
    parser.add_argument("--epochs_each", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=5, help="Frequency of evaluation (epochs)")
    parser.add_argument("--train_conv_in", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    config.epochs_each = args.epochs_each
    config.eval_freq = args.eval_freq
    config.train_conv_in = args.train_conv_in
    
    # Override calculate_fid from command line
    if args.calculate_fid:
        config.calculate_fid = True
    
    # Check if eval_only mode
    if args.eval_only:
        if not args.calculate_fid:
            print("WARNING: --eval_only specified but --calculate_fid not enabled. FID will not be calculated.")
            print("  Use --calculate_fid to enable FID calculation.")
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
        print(f"Using output directory: {config.output_dir}")

    if WANDB_AVAILABLE: wandb.init(project="bbbc021-diffusers-pretrain", config=config.__dict__)

    print("Loading Dataset...")
    import sys
    sys.stdout.flush()  # Ensure output is flushed
    
    encoder = MorganFingerprintEncoder()
    train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder, paths_csv=args.paths_csv)
    if len(train_ds) == 0: train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='', encoder=encoder, paths_csv=args.paths_csv)
    val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split=args.eval_split, encoder=encoder, paths_csv=args.paths_csv)
    if len(val_ds) == 0 and args.eval_split == "val": val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='test', encoder=encoder, paths_csv=args.paths_csv)
    
    # Log paths.csv status
    print(f"\n{'='*60}", flush=True)
    print(f"File Path Resolution Status:", flush=True)
    print(f"{'='*60}", flush=True)
    if len(train_ds.paths_lookup) > 0:
        print(f"  ✓ paths.csv loaded successfully", flush=True)
        print(f"  - Unique filenames in lookup: {len(train_ds.paths_lookup):,}", flush=True)
        print(f"  - Total paths indexed: {len(train_ds.paths_by_rel):,}", flush=True)
        print(f"  - Basename lookups: {len(train_ds.paths_by_basename):,}", flush=True)
    else:
        print(f"  ⚠ paths.csv not found - using fallback path resolution", flush=True)
    print(f"  - Data directory: {train_ds.data_dir}", flush=True)
    print(f"  - Data directory exists: {train_ds.data_dir.exists()}", flush=True)
    print(f"{'='*60}\n", flush=True)

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

    train_loader = PairedDataLoader(train_ds, config.batch_size, shuffle=True, deterministic_ctrl=False)
    val_loader = PairedDataLoader(val_ds, batch_size=8, shuffle=False, deterministic_ctrl=True)

    # Handle eval_only mode (single-direction evaluation of a trained conditional model)
    if args.eval_only:
        # For evaluation, choose init according to direction (theta/phi)
        if args.direction == "theta":
            config.init_ckpt = args.overall_init
        else:
            config.init_ckpt = args.ctrl_init

        os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        logger = TrainingLogger(config.output_dir)

        print(f"Initializing ConditionalWarmupUNet for evaluation ({args.direction})...")
        model = DiffusionModel(config)

        # Optimizer is only needed to load, not to step
        lora_params = get_lora_params(model.model.unet)
        param_groups = [
            {"params": model.model.fp_tokens.parameters(), "lr": config.lr},
            {"params": model.model.ctrl_tokens.parameters(), "lr": config.lr},
            {"params": lora_params, "lr": config.lr},
        ]
        if config.train_conv_in:
            param_groups.append({"params": model.model.unet.conv_in.parameters(), "lr": config.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

        checkpoint_path = args.checkpoint if args.checkpoint else f"{config.output_dir}/checkpoints/latest.pt"
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print(f"  Please specify --checkpoint or ensure latest.pt exists in {config.output_dir}/checkpoints/")
            return

        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATION ONLY MODE", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Loading checkpoint: {checkpoint_path}", flush=True)
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler=None)

        if start_epoch == 0:
            print("ERROR: Failed to load checkpoint or checkpoint is empty")
            return

        print(f"  Loaded checkpoint from epoch {start_epoch}", flush=True)
        print(f"  Evaluation split: {args.eval_split}", flush=True)
        print(f"  Inference steps: {args.inference_steps}", flush=True)
        print(f"  FID calculation: {'Enabled' if args.calculate_fid else 'Disabled'}", flush=True)
        print(f"  FID only mode: {'Yes' if args.fid_only else 'No'}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Run evaluation
        print("Running evaluation...", flush=True)
        metrics = calculate_metrics(
            model,
            val_loader,
            config.device,
            num_samples=args.num_eval_samples,
            calculate_fid_flag=config.calculate_fid,
            num_inference_steps=args.inference_steps,
            skip_other_metrics=args.fid_only,
            direction=args.direction,
        )

        # Print metrics
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATION RESULTS", flush=True)
        print(f"{'='*60}", flush=True)
        if not args.fid_only:
            if metrics['kl_divergence'] is not None:
                print(f"  KL Divergence:     {metrics['kl_divergence']:.6f}", flush=True)
            if metrics['mse'] is not None:
                print(f"  MSE (gen vs real): {metrics['mse']:.6f}", flush=True)
            if metrics['psnr'] is not None:
                print(f"  PSNR:              {metrics['psnr']:.2f} dB", flush=True)
            if metrics['ssim'] is not None:
                print(f"  SSIM:              {metrics['ssim']:.4f}", flush=True)
        if metrics['fid'] is not None:
            print(f"  FID (Overall):     {metrics['fid']:.2f}", flush=True)
        if metrics['kid_mean'] is not None:
            print(f"  KID (Overall):     {metrics['kid_mean']:.5f} (±{metrics['kid_std']:.5f})", flush=True)
        if metrics['cfid'] is not None:
            print(f"  cFID (Conditional): {metrics['cfid']:.2f}", flush=True)
        if metrics['avg_kid_mean'] is not None:
            print(f"  Avg KID (Per-Class): {metrics['avg_kid_mean']:.5f} (±{metrics['avg_kid_std']:.5f})", flush=True)
        print(f"{'='*60}", flush=True)

        # Log metrics (use optimizer LR for logging)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.update(start_epoch, 0.0, metrics, current_lr)
        print(f"\n✅ Evaluation complete! Results saved to {logger.metrics_csv_path}", flush=True)
        return

    # ---------------------------
    # Two-phase warm-start training
    # ---------------------------

    def run_phase(direction, init_ckpt, out_dir):
        phase_config = Config()
        # Explicitly copy over dynamic fields from base config
        phase_config.calculate_fid = config.calculate_fid
        phase_config.skip_metrics_during_training = config.skip_metrics_during_training
        phase_config.batch_size = config.batch_size
        phase_config.lr = config.lr
        phase_config.eval_freq = config.eval_freq
        phase_config.timesteps = config.timesteps
        phase_config.beta_start = config.beta_start
        phase_config.beta_end = config.beta_end
        phase_config.data_dir = config.data_dir
        phase_config.metadata_file = config.metadata_file
        phase_config.image_size = config.image_size
        phase_config.cross_attention_dim = config.cross_attention_dim
        phase_config.attention_head_dim = config.attention_head_dim
        phase_config.unet_block_out_channels = config.unet_block_out_channels
        phase_config.fingerprint_dim = config.fingerprint_dim
        phase_config.num_fp_tokens = config.num_fp_tokens
        phase_config.num_ctrl_tokens = config.num_ctrl_tokens
        phase_config.lora_rank = config.lora_rank
        phase_config.output_dir = out_dir
        phase_config.epochs = phase_config.epochs_each = args.epochs_each
        phase_config.init_ckpt = init_ckpt
        phase_config.train_conv_in = args.train_conv_in

        os.makedirs(f"{phase_config.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{phase_config.output_dir}/checkpoints", exist_ok=True)
        logger = TrainingLogger(phase_config.output_dir)

        print(f"Initializing ConditionalWarmupUNet for phase '{direction}' with init '{init_ckpt}'...")
        model = DiffusionModel(phase_config)


        unet_trainable = [p for p in model.model.unet.parameters() if p.requires_grad]

        param_groups = [
            {"params": model.model.fp_tokens.parameters(), "lr": phase_config.lr},
            {"params": model.model.ctrl_tokens.parameters(), "lr": phase_config.lr},
            {"params": unet_trainable, "lr": phase_config.lr},
        ]
        
        if phase_config.train_conv_in:
            # conv_in already included above if requires_grad=True, so no extra group needed unless you want specific LR
            pass

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=phase_config.epochs,
            eta_min=1e-6,
        )

        # Sanity check: ensure we actually have trainable parameters
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if len(trainable) == 0:
            raise RuntimeError("CRITICAL: No trainable parameters. LoRA/token modules not set correctly.")
        print(f"Trainable params ({len(trainable)} tensors):", flush=True)
        print("\n".join(trainable[:30]), flush=True)

        # Optional resume: load latest phase checkpoint if requested
        start_epoch = 0
        if args.resume or args.checkpoint:
            ckpt_path = args.checkpoint if args.checkpoint else f"{phase_config.output_dir}/checkpoints/latest.pt"
            if os.path.exists(ckpt_path):
                print(f"[{direction}] Resuming from: {ckpt_path}", flush=True)
                start_epoch = load_checkpoint(model, optimizer, ckpt_path, scheduler)
                print(f"[{direction}] Resumed at epoch {start_epoch}", flush=True)
            else:
                print(f"[{direction}] Resume requested but checkpoint not found: {ckpt_path}", flush=True)

        for epoch in range(start_epoch, phase_config.epochs):
            model.model.train()
            losses = []

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                ctrl = batch['control'].to(phase_config.device)
                trt = batch['perturbed'].to(phase_config.device)
                fp = batch['fingerprint'].to(phase_config.device)

                if direction == "theta":
                    # p_theta(X | Y, c): target = treated, condition = control
                    loss = model(trt, ctrl, fp)
                else:
                    # p_phi(Y | X, c): target = control, condition = treated
                    loss = model(ctrl, trt, fp)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            avg_loss = float(np.mean(losses)) if losses else 0.0

            metrics = None
            if (epoch + 1) % phase_config.eval_freq == 0:
                print("\n" + "="*60, flush=True)
                print(f"[{direction}] EVALUATION (Epoch {epoch+1})", flush=True)
                print("="*60, flush=True)

                if not phase_config.skip_metrics_during_training:
                    print("  Calculating metrics on validation set...", flush=True)
                    metrics = calculate_metrics(
                        model,
                        val_loader,
                        phase_config.device,
                        num_samples=args.num_eval_samples,
                        calculate_fid_flag=phase_config.calculate_fid,
                        num_inference_steps=args.inference_steps,
                        skip_other_metrics=args.fid_only,
                        direction=direction,
                    )

                    print(f"\n  📊 EVALUATION METRICS:", flush=True)
                    print(f"  {'-'*58}", flush=True)
                    if metrics['kl_divergence'] is not None:
                        print(f"    KL Divergence:     {metrics['kl_divergence']:.6f}", flush=True)
                    if metrics['mse'] is not None:
                        print(f"    MSE (gen vs real): {metrics['mse']:.6f}", flush=True)
                    if metrics['psnr'] is not None:
                        print(f"    PSNR:              {metrics['psnr']:.2f} dB", flush=True)
                    if metrics['ssim'] is not None:
                        print(f"    SSIM:              {metrics['ssim']:.4f}", flush=True)
                    if metrics['fid'] is not None:
                        print(f"    FID (Overall):     {metrics['fid']:.2f}", flush=True)
                    if metrics['kid_mean'] is not None:
                        print(f"    KID (Overall):     {metrics['kid_mean']:.5f} (±{metrics['kid_std']:.5f})", flush=True)
                    if metrics['cfid'] is not None:
                        print(f"    cFID (Conditional): {metrics['cfid']:.2f}", flush=True)
                    if metrics['avg_kid_mean'] is not None:
                        print(f"    Avg KID (Per-Class): {metrics['avg_kid_mean']:.5f} (±{metrics['avg_kid_std']:.5f})", flush=True)
                    print(f"  {'-'*58}", flush=True)
                else:
                    print("  Skipping metric calculations (only generating samples/video)...", flush=True)

                print("="*60 + "\n", flush=True)

                # Visualization
                print("  Generating sample grid and video...", flush=True)
                val_iter = iter(val_loader)
                batch = next(val_iter)
                ctrl = batch['control'].to(phase_config.device)
                real_t = batch['perturbed'].to(phase_config.device)
                fp = batch['fingerprint'].to(phase_config.device)

                if direction == "theta":
                    fakes = model.sample(ctrl, fp, num_inference_steps=200)
                    grid = torch.cat([ctrl[:8], fakes[:8], real_t[:8]], dim=0)
                    video_cond = ctrl[0:1]
                else:
                    fakes = model.sample(real_t, fp, num_inference_steps=200)
                    grid = torch.cat([real_t[:8], fakes[:8], ctrl[:8]], dim=0)
                    video_cond = real_t[0:1]
                tag = direction
                grid_path = f"{phase_config.output_dir}/plots/{tag}_epoch_{epoch+1}.png"
                save_image(grid, grid_path, nrow=8, normalize=True, value_range=(-1,1))
                print(f"  ✓ Sample grid saved to: {grid_path}", flush=True)
                video_path = f"{phase_config.output_dir}/plots/{tag}_video_{epoch+1}.mp4"
                # Use a dedicated generator for video noise so it is reproducible
                gen_video = torch.Generator(device=phase_config.device).manual_seed(1357 + epoch)
                generate_video(model, video_cond, fp[0:1], video_path, generator=gen_video)
                print(f"  ✓ Video saved to: {video_path}", flush=True)

            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(f"[{direction}] Epoch {epoch+1}/{phase_config.epochs} | Loss: {avg_loss:.5f} | LR: {current_lr:.2e}", flush=True)
            logger.update(epoch+1, avg_loss, metrics, current_lr)

            if WANDB_AVAILABLE:
                log_dict = {
                    "loss": avg_loss,
                    "epoch": epoch+1,
                    "mse_loss": avg_loss,
                    "learning_rate": current_lr,
                    "phase": direction,
                }
                if metrics:
                    if metrics['kl_divergence'] is not None:
                        log_dict['kl_divergence'] = metrics['kl_divergence']
                    if metrics['mse'] is not None:
                        log_dict['mse_gen_real'] = metrics['mse']
                    if metrics['psnr'] is not None:
                        log_dict['psnr'] = metrics['psnr']
                    if metrics['ssim'] is not None:
                        log_dict['ssim'] = metrics['ssim']
                    if metrics['fid'] is not None:
                        log_dict['fid'] = metrics['fid']
                    if metrics['kid_mean'] is not None:
                        log_dict['kid_mean'] = metrics['kid_mean']
                    if metrics['cfid'] is not None:
                        log_dict['cfid'] = metrics['cfid']
                wandb.log(log_dict)

            # CHECKPOINTING (Save every epoch)
            epoch_checkpoint_path = f"{phase_config.output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1
            }, epoch_checkpoint_path)

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1
            }, f"{phase_config.output_dir}/checkpoints/latest.pt")

            print(f"  ✓ Checkpoint saved: {epoch_checkpoint_path} (LR: {current_lr:.2e})", flush=True)

    # Run requested phase based on direction
    if args.direction == "theta":
        run_phase("theta", args.overall_init, config.output_dir + "_theta")
    else:
        run_phase("phi",   args.ctrl_init,    config.output_dir + "_phi")

if __name__ == "__main__":
    main()