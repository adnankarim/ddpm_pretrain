"""
================================================================================
BBBC021 UNCONDITIONAL PRETRAINING - TWO SEPARATE DDPM MODELS
================================================================================
Trains two unconditional DDPM models:
1. Control model: trained on DMSO (control) images only
2. Perturbed model: trained on treated (perturbed) images only
Each model is trained for 100 epochs unconditionally (no conditioning).
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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

# --- Plotting Backend (Prevents crashes on headless servers) ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt

# --- NEW IMPORTS ---
try:
    from diffusers import UNet2DModel, DDPMScheduler
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
    # Note: anton-l/ddpm-butterflies-128 doesn't exist, will fallback to google/ddpm-cifar10-32
    base_model_id = "google/ddpm-cifar10-32"  # Pretrained DDPM model (will try anton-l/ddpm-butterflies-128 first if available) 
    
    # Embeddings
    perturbation_emb_dim = 128 
    fingerprint_dim = 1024
    
    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training
    epochs = 100  # Train each model for 100 epochs
    batch_size = 64
    lr = 3e-5  # Lower LR when using pretrained weights to prevent drift
    save_freq = 1
    eval_freq = 10  # Evaluate every 10 epochs
    calculate_fid = False  # Set to True to enable FID calculation (slower evaluation)
    skip_metrics_during_training = True  # If True, skip metric calculations during training (only generate samples/video)
    
    # Output directories for two models
    output_dir_control = "ddpm_uncond_control"
    output_dir_perturbed = "ddpm_uncond_perturbed"
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
        self._first_load_logged = False  # Track if we've logged the first successful load
        
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

    def get_paired_sample(self, trt_idx):
        batch = self.metadata[trt_idx].get('BATCH', 'unknown')
        if batch in self.batch_map and self.batch_map[batch]['ctrl']:
            ctrls = self.batch_map[batch]['ctrl']
            return (np.random.choice(ctrls), trt_idx)
        return (trt_idx, trt_idx)  # Fallback: use self as control if none found

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
                print(f"  Compound: {meta.get('CPD_NAME', 'unknown')}", flush=True)
                print(f"  File path: {full_path}", flush=True)
                print(f"  File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)", flush=True)
                print(f"  Original shape: {original_shape} (dtype: {original_dtype})", flush=True)
                print(f"  Original range: [{original_min:.2f}, {original_max:.2f}]", flush=True)
                print(f"  Processed shape: {img.shape} (dtype: {img.dtype})", flush=True)
                print(f"  Processed range: [{img.min():.2f}, {img.max():.2f}]", flush=True)
                print(f"  Fingerprint shape: {self.fingerprints.get(meta.get('CPD_NAME', 'DMSO'), np.zeros(1024)).shape}", flush=True)
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
# UNCONDITIONAL DATASET
# ============================================================================

class UnconditionalDataset(Dataset):
    """Simple dataset that returns only images (no pairs, no fingerprints)"""
    def __init__(self, base_dataset, filter_type='control'):
        """
        Args:
            base_dataset: BBBC021Dataset instance
            filter_type: 'control' for DMSO images, 'perturbed' for treated images
        """
        self.base_dataset = base_dataset
        self.filter_type = filter_type
        
        # Filter indices based on compound type
        if filter_type == 'control':
            # DMSO (control) images
            self.indices = [i for i, m in enumerate(base_dataset.metadata) 
                          if str(m.get('CPD_NAME', '')).upper() == 'DMSO']
        else:  # 'perturbed'
            # Treated (perturbed) images
            self.indices = [i for i, m in enumerate(base_dataset.metadata) 
                          if str(m.get('CPD_NAME', '')).upper() != 'DMSO']
        
        print(f"  UnconditionalDataset ({filter_type}): {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.base_dataset[actual_idx]
        return sample['image']  # Return only the image tensor

# ============================================================================
# ARCHITECTURE
# ============================================================================

class UnconditionalUNet(nn.Module):
    """Unconditional UNet - no fingerprint, no control image conditioning"""
    def __init__(self, image_size=96):
        super().__init__()
        base_model_id = Config.base_model_id
        print(f"Attempting to load base architecture from {base_model_id}...")
        
        # Try to load pretrained UNet weights, with fallback
        try:
            unet_pre = UNet2DModel.from_pretrained(base_model_id)
            print(f"  ✓ Successfully loaded pretrained model from {base_model_id}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load {base_model_id}: {e}")
            print(f"  Falling back to google/ddpm-cifar10-32...")
            base_model_id = "google/ddpm-cifar10-32"
            unet_pre = UNet2DModel.from_pretrained(base_model_id)
            print(f"  ✓ Loaded fallback model from {base_model_id}")
        
        # Standard UNet: 3 channels in, 3 channels out (no conditioning)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=3,  # Standard RGB
            out_channels=unet_pre.config.out_channels,  # usually 3
            layers_per_block=unet_pre.config.layers_per_block,
            block_out_channels=unet_pre.config.block_out_channels,
            down_block_types=unet_pre.config.down_block_types,
            up_block_types=unet_pre.config.up_block_types,
            dropout=unet_pre.config.dropout,
            attention_head_dim=getattr(unet_pre.config, "attention_head_dim", None),
            norm_num_groups=unet_pre.config.norm_num_groups,
        )
        
        # Load pretrained weights (excluding conv_in if channel mismatch)
        pretrained_state = unet_pre.state_dict()
        # Only load if channels match
        if unet_pre.config.in_channels == 3:
            missing, unexpected = self.unet.load_state_dict(pretrained_state, strict=False)
            print(f"  ✓ Loaded pretrained weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
        else:
            # Filter out conv_in if channels don't match
            filtered_state = {k: v for k, v in pretrained_state.items() 
                             if not k.startswith('conv_in.')}
            missing, unexpected = self.unet.load_state_dict(filtered_state, strict=False)
            print(f"  ✓ Loaded pretrained weights (conv_in excluded, missing: {len(missing)}, unexpected: {len(unexpected)})")
    
    def forward(self, x, t):
        """Forward pass: x is noisy image, t is timestep"""
        return self.unet(x, t).sample

class ModifiedDiffusersUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        base_model_id = Config.base_model_id
        print(f"Attempting to load base architecture from {base_model_id}...")
        
        # Try to load pretrained UNet weights, with fallback
        try:
            unet_pre = UNet2DModel.from_pretrained(base_model_id)
            print(f"  ✓ Successfully loaded pretrained model from {base_model_id}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load {base_model_id}: {e}")
            print(f"  Falling back to google/ddpm-cifar10-32...")
            base_model_id = "google/ddpm-cifar10-32"
            unet_pre = UNet2DModel.from_pretrained(base_model_id)
            print(f"  ✓ Loaded fallback model from {base_model_id}")
        
        # Rebuild a UNet with YOUR desired sample_size and channels
        # This ensures config matches your training setup (96x96, 6 channels)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=6,  # Your modified input (3 noise + 3 control)
            out_channels=unet_pre.config.out_channels,  # usually 3
            layers_per_block=unet_pre.config.layers_per_block,
            block_out_channels=unet_pre.config.block_out_channels,
            down_block_types=unet_pre.config.down_block_types,
            up_block_types=unet_pre.config.up_block_types,
            dropout=unet_pre.config.dropout,
            attention_head_dim=getattr(unet_pre.config, "attention_head_dim", None),
            norm_num_groups=unet_pre.config.norm_num_groups,
            class_embed_type="identity"  # For fingerprint conditioning
        )
        
        # --- conv_in surgery: 3 -> 6 (do this BEFORE loading state dict) ---
        old_conv = unet_pre.conv_in
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        with torch.no_grad():
            # Copy first 3 channels from pretrained, zero-init last 3 (control)
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0 
            new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        print("  ✓ Network Surgery: Input expanded 3 -> 6 channels (pretrained weights preserved)")
        
        # Copy weights where shapes match (excluding conv_in which we already handled)
        # Filter out conv_in from the state dict to avoid size mismatch
        pretrained_state = unet_pre.state_dict()
        # Create a new dict without conv_in keys to avoid size mismatch
        filtered_state = {k: v for k, v in pretrained_state.items() 
                         if not k.startswith('conv_in.')}
        
        missing, unexpected = self.unet.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded weights with strict=False (conv_in excluded)")
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # Fingerprint projection - get actual class embedding dimension from model
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

class UnconditionalDiffusionModel(nn.Module):
    """Unconditional DDPM model - no conditioning"""
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = UnconditionalUNet(image_size=config.image_size).to(config.device)
        
        # Use Diffusers' DDPMScheduler for consistent training/sampling
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        self.timesteps = config.timesteps

    def forward(self, x0):
        """Training forward pass: x0 is clean image"""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.cfg.device).long()
        noise = torch.randn_like(x0)
        
        # Forward Diffusion: Use scheduler's add_noise for consistency
        xt = self.noise_scheduler.add_noise(x0, noise, t)
        
        # Prediction (unconditional)
        noise_pred = self.model(xt, t)
        
        # Simple MSE Loss (Proxy for KL Divergence)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, batch_size=1, num_inference_steps=None, image_size=None):
        """Generate unconditional samples"""
        self.model.eval()
        if image_size is None:
            image_size = self.cfg.image_size
        h = w = image_size
        xt = torch.randn((batch_size, 3, h, w), device=self.cfg.device)
        
        # Use scheduler for consistent sampling
        steps = num_inference_steps or self.timesteps
        self.noise_scheduler.set_timesteps(steps, device=self.cfg.device)
        
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=self.cfg.device, dtype=torch.long)
            noise_pred = self.model(xt, t_batch)
            xt = self.noise_scheduler.step(noise_pred, t, xt).prev_sample
        
        # Clamp only once at the end
        return xt.clamp(-1, 1)

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = ModifiedDiffusersUNet(
            image_size=config.image_size, 
            fingerprint_dim=config.fingerprint_dim
        ).to(config.device)
        
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
    def sample(self, control, fingerprint, num_inference_steps=None):
        """Generate a sample using reverse diffusion with DDPMScheduler"""
        self.model.eval()
        b, c, h, w = control.shape
        xt = torch.randn((b, 3, h, w), device=self.cfg.device)
        
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
    
    # Initialize FID and KID metrics if enabled
    if calculate_fid_flag:
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        kid_metric = KernelInceptionDistance(subset_size=100, normalize=True).to(device)
    
    # Collect all batches first, then randomly sample to respect num_samples
    all_batches = []
    with torch.no_grad():
        for batch in val_loader:
            all_batches.append(batch)
    
    # Flatten all samples
    all_samples = []
    for batch in all_batches:
        b_size = batch['control'].shape[0]
        for i in range(b_size):
            all_samples.append({
                'control': batch['control'][i:i+1],
                'perturbed': batch['perturbed'][i:i+1],
                'fingerprint': batch['fingerprint'][i:i+1],
                'compound': batch['compound'][i]
            })
    
    # Randomly sample up to num_samples
    if len(all_samples) > num_samples:
        import random
        random.seed(42)
        sampled_indices = random.sample(range(len(all_samples)), num_samples)
        all_samples = [all_samples[i] for i in sampled_indices]
    
    print(f"  Using {len(all_samples)} samples for evaluation (requested: {num_samples})", flush=True)
    
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
            化合物 = sample['compound'] # compound name

            # Choose conditioning image and target image based on direction
            if direction == "theta":
                cond_img = ctrl
                target_img = real_t
            else:
                cond_img = real_t
                target_img = ctrl
            
            # Generate samples
            generated = model.sample(cond_img, fp, num_inference_steps=num_inference_steps)
            
            # ----------------------------------------------------------------
            # 1. Standard Metrics (KL, MSE, PSNR, SSIM)
            # ----------------------------------------------------------------
            if not skip_other_metrics:
                # KL Divergence (approx)
                b = ctrl.shape[0]
                t = torch.randint(0, model.timesteps, (b,), device=device).long()
                noise = torch.randn_like(target_img)
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
            if calculate_fid_flag:
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
                if 化合物 not in class_samples:
                    class_samples[化合物] = {'gen': [], 'real': []}
                
                # Move to CPU to save memory during loop
                class_samples[化合物]['real'].append(real_norm.cpu())
                class_samples[化合物]['gen'].append(gen_norm.cpu())
            
            sample_count += generated.shape[0]

    # ----------------------------------------------------------------
    # 3. Compute Final Metrics
    # ----------------------------------------------------------------
    
    # Average standard metrics
    metrics['kl_divergence'] = np.mean(metrics['kl_divergence']) if metrics['kl_divergence'] else None
    metrics['mse'] = np.mean(metrics['mse']) if metrics['mse'] else None
    metrics['psnr'] = np.mean(metrics['psnr']) if metrics['psnr'] else None
    metrics['ssim'] = np.mean(metrics['ssim']) if metrics['ssim'] else None
    
    if calculate_fid_flag:
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
            kid_metric.reset() # KID might need reset too
            
            # Update with class samples
            fid_metric.update(real_stack, real=True)
            fid_metric.update(gen_stack, real=False)
            
            # KID subset size adjustment
            current_subset_size = min(len(samples['gen']), 100)
            # We can't easily change subset_size of existing metric without re-init?
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

def generate_video(model, control, fingerprint, save_path):
    """Generate a video showing the reverse diffusion process using DDPMScheduler"""
    if not IMAGEIO_AVAILABLE: return
    model.model.eval()
    b, c, h, w = control.shape
    xt = torch.randn((b, 3, h, w), device=model.cfg.device)
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

def train_unconditional_model(model, train_loader, config, output_dir, model_name):
    """Train a single unconditional DDPM model"""
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    logger = TrainingLogger(output_dir)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} Model")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"{'='*60}\n")
    
    for epoch in range(config.epochs):
        model.model.train()
        losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            images = batch.to(config.device)  # [B, 3, H, W]
            loss = model(images)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.5f} | LR: {current_lr:.2e}")
        logger.update(epoch+1, avg_loss, None, current_lr)
        
        # Generate samples every eval_freq epochs
        if (epoch + 1) % config.eval_freq == 0:
            print(f"  Generating samples...")
            with torch.no_grad():
                samples = model.sample(batch_size=16, num_inference_steps=200)
                save_image(samples, f"{output_dir}/plots/samples_epoch_{epoch+1}.png", 
                          nrow=4, normalize=True, value_range=(-1, 1))
                print(f"  ✓ Samples saved to {output_dir}/plots/samples_epoch_{epoch+1}.png")
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1
        }, checkpoint_path)
        
        # Also save latest
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1
        }, f"{output_dir}/checkpoints/latest.pt")
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path}\n")
    
    print(f"\n✅ {model_name} model training complete!")
    print(f"   Final checkpoint: {output_dir}/checkpoints/checkpoint_epoch_{config.epochs}.pt\n")

def main():
    parser = argparse.ArgumentParser(
        description="Train two unconditional DDPM models on BBBC021 dataset (control and perturbed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both models (default)
  python uncond.py

  # Specify custom output directories
  python uncond.py --output_dir_control ./control_model --output_dir_perturbed ./perturbed_model
        """
    )
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file (auto-detected if not specified)")
    parser.add_argument("--output_dir_control", type=str, default=None, help="Output directory for control model (default: ddpm_uncond_control)")
    parser.add_argument("--output_dir_perturbed", type=str, default=None, help="Output directory for perturbed model (default: ddpm_uncond_perturbed)")
    parser.add_argument("--train_control_only", action="store_true", help="Train only the control model")
    parser.add_argument("--train_perturbed_only", action="store_true", help="Train only the perturbed model")
    args = parser.parse_args()
    
    config = Config()
    
    # Override output directories if specified
    if args.output_dir_control:
        config.output_dir_control = args.output_dir_control
    if args.output_dir_perturbed:
        config.output_dir_perturbed = args.output_dir_perturbed
    
    print("\n" + "="*60)
    print("UNCONDITIONAL DDPM PRETRAINING")
    print("="*60)
    print("Training two separate unconditional models:")
    print("  1. Control model (DMSO images)")
    print("  2. Perturbed model (treated images)")
    print("="*60 + "\n")
    
    # Load base dataset
    print("Loading Dataset...")
    encoder = MorganFingerprintEncoder()  # Still needed for dataset initialization
    base_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder, paths_csv=args.paths_csv)
    if len(base_ds) == 0:
        base_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='', encoder=encoder, paths_csv=args.paths_csv)
    
    # Create unconditional datasets
    print("\nCreating unconditional datasets...")
    control_ds = UnconditionalDataset(base_ds, filter_type='control')
    perturbed_ds = UnconditionalDataset(base_ds, filter_type='perturbed')
    
    # Create data loaders
    control_loader = DataLoader(control_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    perturbed_loader = DataLoader(perturbed_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    print(f"\nDataset Summary:")
    print(f"  Control (DMSO) samples: {len(control_ds)}")
    print(f"  Perturbed (treated) samples: {len(perturbed_ds)}")
    print()
    
    # Train Control Model
    if not args.train_perturbed_only:
        print("\n" + "="*60)
        print("TRAINING CONTROL MODEL (DMSO)")
        print("="*60)
        control_model = UnconditionalDiffusionModel(config)
        train_unconditional_model(
            control_model, 
            control_loader, 
            config, 
            config.output_dir_control,
            "Control (DMSO)"
        )
    
    # Train Perturbed Model
    if not args.train_control_only:
        print("\n" + "="*60)
        print("TRAINING PERTURBED MODEL (TREATED)")
        print("="*60)
        perturbed_model = UnconditionalDiffusionModel(config)
        train_unconditional_model(
            perturbed_model, 
            perturbed_loader, 
            config, 
            config.output_dir_perturbed,
            "Perturbed (Treated)"
        )
    
    print("\n" + "="*60)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*60)
    if not args.train_perturbed_only:
        print(f"  Control model: {config.output_dir_control}/checkpoints/checkpoint_epoch_{config.epochs}.pt")
    if not args.train_control_only:
        print(f"  Perturbed model: {config.output_dir_perturbed}/checkpoints/checkpoint_epoch_{config.epochs}.pt")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
