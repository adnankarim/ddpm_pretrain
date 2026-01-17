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
    epochs = 500
    batch_size = 64
    lr = 3e-5  # Lower LR when using pretrained weights to prevent drift
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
            'train_loss': [],
            'kl_divergence': [],
            'mse_loss': [],
            'psnr': [],
            'ssim': [],
            'fid': [],
            'cfid': [],
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
        else:
            self.history['kl_divergence'].append(None)
            self.history['psnr'].append(None)
            self.history['ssim'].append(None)
            self.history['fid'].append(None)
            self.history['cfid'].append(None)
        
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
                'cfid': metrics.get('cfid')
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
# ARCHITECTURE
# ============================================================================

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

def calculate_fid(real_images, fake_images, device):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images: Tensor of shape [N, 3, H, W] in range [-1, 1]
        fake_images: Tensor of shape [N, 3, H, W] in range [-1, 1]
        device: torch device
    
    Returns:
        FID score (float) or None if calculation fails
    """
    try:
        from torchvision.models import inception_v3
        from scipy import linalg
        
        # Load Inception v3 model
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        inception.fc = torch.nn.Identity()  # Remove final classification layer
        
        def get_features(images):
            """Extract features from Inception v3"""
            # Normalize to [0, 1] for Inception
            img_norm = (images + 1.0) / 2.0
            # Resize to 299x299 (Inception input size)
            img_resized = torch.nn.functional.interpolate(
                img_norm, size=(299, 299), mode='bilinear', align_corners=False
            )
            with torch.no_grad():
                features = inception(img_resized)
            return features.cpu().numpy()
        
        # Extract features
        real_features = get_features(real_images)
        fake_features = get_features(fake_images)
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            # Handle numerical issues
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(np.real(fid))
        
    except ImportError:
        return None
    except Exception as e:
        print(f"  Warning: FID calculation failed: {e}", flush=True)
        return None

def calculate_metrics(model, val_loader, device, num_samples=1000):
    """
    Calculate evaluation metrics on validation set.
    
    Returns:
        dict with keys: 'kl_divergence', 'mse', 'psnr', 'ssim', 'fid', 'cfid'
    """
    model.model.eval()
    metrics = {
        'kl_divergence': [],
        'mse': [],
        'psnr': [],
        'ssim': [],
        'fid': [],
        'cfid': []
    }
    
    # Try to import scikit-image for SSIM/PSNR
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
    
    # Collect all real and generated images for overall FID
    all_real_images = []
    all_generated_images = []
    
    # Collect per-condition images for conditional FID
    # Group by condition (control + fingerprint combination)
    condition_groups = {}  # (ctrl_hash, fp_hash) -> {'real': [...], 'gen': [...]}
    
    # Collect all batches first, then randomly sample
    all_batches = []
    with torch.no_grad():
        for batch in val_loader:
            all_batches.append(batch)
    
    # Flatten all samples and randomly sample
    all_samples = []
    for batch in all_batches:
        b_size = batch['control'].shape[0]
        for i in range(b_size):
            all_samples.append({
                'control': batch['control'][i:i+1],
                'perturbed': batch['perturbed'][i:i+1],
                'fingerprint': batch['fingerprint'][i:i+1]
            })
    
    # Randomly sample up to num_samples
    if len(all_samples) > num_samples:
        import random
        random.seed(42)  # For reproducibility
        sampled_indices = random.sample(range(len(all_samples)), num_samples)
        all_samples = [all_samples[i] for i in sampled_indices]
    
    print(f"  Using {len(all_samples)} samples for evaluation (requested: {num_samples})", flush=True)
    
    sample_count = 0
    with torch.no_grad():
        for sample in all_samples:
            if sample_count >= num_samples:
                break
            
            ctrl = sample['control'].to(device)
            real_t = sample['perturbed'].to(device)
            fp = sample['fingerprint'].to(device)
            
            # Generate samples (use fewer steps for faster evaluation)
            generated = model.sample(ctrl, fp, num_inference_steps=200)
            
            # Calculate KL divergence (using forward pass on a sample)
            # Sample a random timestep and compute KL
            b = ctrl.shape[0]
            t = torch.randint(0, model.timesteps, (b,), device=device).long()
            noise = torch.randn_like(real_t)
            # Use scheduler's add_noise for consistency
            xt = model.noise_scheduler.add_noise(real_t, noise, t)
            noise_pred = model.model(xt, t, ctrl, fp)
            kl = calculate_kl_divergence(noise_pred, noise)
            metrics['kl_divergence'].append(kl)
            
            # Calculate MSE between generated and real
            mse = F.mse_loss(generated, real_t).item()
            metrics['mse'].append(mse)
            
            # Collect images for overall FID (keep on device for efficiency)
            all_real_images.append(real_t)
            all_generated_images.append(generated)
            
            # Group by condition for conditional FID
            # Create a hash of control and fingerprint to group similar conditions
            for i in range(b):
                # Use a simple hash: mean of control and fingerprint
                ctrl_hash = tuple(ctrl[i].mean(dim=(1, 2)).cpu().numpy().round(decimals=2))
                fp_hash = tuple(fp[i].cpu().numpy()[:10].round(decimals=2))  # Use first 10 dims for hash
                cond_key = (ctrl_hash, fp_hash)
                
                if cond_key not in condition_groups:
                    condition_groups[cond_key] = {'real': [], 'gen': []}
                
                condition_groups[cond_key]['real'].append(real_t[i:i+1])
                condition_groups[cond_key]['gen'].append(generated[i:i+1])
            
            # Calculate PSNR and SSIM if available
            if SKIMAGE_AVAILABLE:
                for i in range(generated.shape[0]):
                    # Convert to numpy [0, 255] range
                    gen_np = ((generated[i].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                    real_np = ((real_t[i].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                    
                    # Convert to grayscale for SSIM
                    gen_gray = np.mean(gen_np, axis=2)
                    real_gray = np.mean(real_np, axis=2)
                    
                    # Calculate SSIM
                    ssim_val = ssim(real_gray, gen_gray, data_range=255)
                    metrics['ssim'].append(ssim_val)
                    
                    # Calculate PSNR
                    psnr_val = psnr(real_np, gen_np, data_range=255)
                    metrics['psnr'].append(psnr_val)
            
            sample_count += generated.shape[0]
    
    # Calculate Overall FID (all images regardless of condition)
    fid_score = None
    if len(all_real_images) > 0 and len(all_generated_images) > 0:
        real_stack = torch.cat(all_real_images, dim=0).to(device)
        fake_stack = torch.cat(all_generated_images, dim=0).to(device)
        # Need at least 2 samples for FID
        if real_stack.shape[0] >= 2 and fake_stack.shape[0] >= 2:
            print("  Calculating Overall FID...", flush=True)
            fid_score = calculate_fid(real_stack, fake_stack, device)
            if fid_score is not None:
                metrics['fid'].append(fid_score)
    
    # Calculate Conditional FID (per-condition comparison)
    cfid_scores = []
    if len(condition_groups) > 0:
        print(f"  Calculating Conditional FID across {len(condition_groups)} conditions...", flush=True)
        for cond_key, group in condition_groups.items():
            if len(group['real']) >= 2 and len(group['gen']) >= 2:
                real_cond = torch.cat(group['real'], dim=0).to(device)
                gen_cond = torch.cat(group['gen'], dim=0).to(device)
                cfid = calculate_fid(real_cond, gen_cond, device)
                if cfid is not None:
                    cfid_scores.append(cfid)
    
    # Average conditional FID across all conditions
    cfid_score = np.mean(cfid_scores) if len(cfid_scores) > 0 else None
    
    # Average metrics
    result = {
        'kl_divergence': np.mean(metrics['kl_divergence']) if metrics['kl_divergence'] else None,
        'mse': np.mean(metrics['mse']) if metrics['mse'] else None,
        'psnr': np.mean(metrics['psnr']) if metrics['psnr'] else None,
        'ssim': np.mean(metrics['ssim']) if metrics['ssim'] else None,
        'fid': fid_score if fid_score is not None else None,
        'cfid': cfid_score if cfid_score is not None else None
    }
    
    return result

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

def main():
    parser = argparse.ArgumentParser(description="Train DDPM model on BBBC021 dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume from (default: auto-loads latest.pt)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: ddpm_diffusers_results)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file (auto-detected if not specified)")
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
    train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder, paths_csv=args.paths_csv)
    if len(train_ds) == 0: train_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='', encoder=encoder, paths_csv=args.paths_csv)
    val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='val', encoder=encoder, paths_csv=args.paths_csv)
    if len(val_ds) == 0: val_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='test', encoder=encoder, paths_csv=args.paths_csv)
    
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

    train_loader = PairedDataLoader(train_ds, config.batch_size, shuffle=True)
    val_loader = PairedDataLoader(val_ds, batch_size=8, shuffle=True)

    print(f"Initializing Modified Diffusers U-Net...")
    model = DiffusionModel(config)
    
    # Use parameter groups: higher LR for new layers (fingerprint_proj, conv_in)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.model.unet.named_parameters() if 'conv_in' not in n], "lr": config.lr},
        {"params": model.model.fingerprint_proj.parameters(), "lr": config.lr * 10},  # 10x LR for new projection
        {"params": model.model.unet.conv_in.parameters(), "lr": config.lr * 10},  # 10x LR for modified input layer
    ], weight_decay=0.01)
    
    # Learning Rate Scheduler - Cosine Annealing (recommended for diffusion models)
    # Gradually reduces LR from initial to near-zero over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,  # Full training cycle
        eta_min=1e-6  # Minimum learning rate
    )

    # Load checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint else f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume or args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch+1}...")
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Starting training from epoch 1 (checkpoint not found or empty)...")
    else:
        # Only auto-load if checkpoint exists
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"Found checkpoint, resuming from epoch {start_epoch+1}...")
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
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
        
        # Calculate metrics during evaluation
        metrics = None
        if (epoch + 1) % config.eval_freq == 0:
            print("\n" + "="*60, flush=True)
            print(f"EVALUATION (Epoch {epoch+1})", flush=True)
            print("="*60, flush=True)
            print("  Calculating metrics on validation set...", flush=True)
            metrics = calculate_metrics(model, val_loader, config.device, num_samples=1000)
            
            # Print metrics prominently
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
            if metrics['cfid'] is not None:
                print(f"    cFID (Conditional): {metrics['cfid']:.2f}", flush=True)
            print(f"  {'-'*58}", flush=True)
            print(f"  ✓ Metrics saved to: {logger.csv_path}", flush=True)
            print(f"  ✓ Metrics also saved to: {logger.metrics_csv_path}", flush=True)
            print(f"  ✓ Metrics plot: {logger.metrics_plot_path}", flush=True)
            print("="*60 + "\n", flush=True)
            
            # Visualization
            val_iter = iter(val_loader)
            batch = next(val_iter)
            ctrl = batch['control'].to(config.device)
            real_t = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            # Use fewer inference steps for faster evaluation (can increase later)
            fakes = model.sample(ctrl, fp, num_inference_steps=200)
            
            grid = torch.cat([ctrl[:8], fakes[:8], real_t[:8]], dim=0)
            save_image(grid, f"{config.output_dir}/plots/epoch_{epoch+1}.png", nrow=8, normalize=True, value_range=(-1,1))
            generate_video(model, ctrl[0:1], fp[0:1], f"{config.output_dir}/plots/video_{epoch+1}.mp4")

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # LOGGING & PLOTTING
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.5f} | LR: {current_lr:.2e}", flush=True)
        logger.update(epoch+1, avg_loss, metrics, current_lr)
        
        # Log to wandb
        if WANDB_AVAILABLE:
            log_dict = {
                "loss": avg_loss, 
                "epoch": epoch+1, 
                "mse_loss": avg_loss,
                "learning_rate": current_lr
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
                if metrics['cfid'] is not None:
                    log_dict['cfid'] = metrics['cfid']
            wandb.log(log_dict)

        # CHECKPOINTING (Save every epoch)
        # Save epoch-specific checkpoint
        epoch_checkpoint_path = f"{config.output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1
        }, epoch_checkpoint_path)
        
        # Also update latest.pt for easy resuming
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1
        }, f"{config.output_dir}/checkpoints/latest.pt")
        
        print(f"  ✓ Checkpoint saved: {epoch_checkpoint_path} (LR: {current_lr:.2e})", flush=True)

if __name__ == "__main__":
    main()