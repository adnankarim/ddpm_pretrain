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

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: 'transformers' library not found. Install with: pip install transformers")
    sys.exit(1)

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
    fingerprint_dim = 768  # MolFormer native dimension (no padding needed)
    
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
    calculate_fid = False  # Set to True to enable FID calculation (slower evaluation)
    skip_metrics_during_training = True  # If True, skip metric calculations during training (only generate samples/video)
    
    output_dir = "ddpm_diffusers_results_mol"
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

class MolFormerEncoder:
    def __init__(self, target_dim=768, model_name="ibm/MoLFormer-XL-both-10pct"):
        self.target_dim = target_dim
        self.cache = {}
        
        print(f"Loading MolFormer ({model_name})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.eval()
            
            # Move to GPU if available for faster preprocessing
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"  ✓ MolFormer loaded on {self.device}")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading MolFormer: {e}")
            sys.exit(1)

    def encode(self, smiles):
        if isinstance(smiles, list): 
            # Handle list processing
            return np.array([self.encode(s) for s in smiles])
            
        if smiles in self.cache: 
            return self.cache[smiles]

        if not smiles or smiles == 'DMSO':
            # Return zero vector for empty/DMSO
            return np.zeros((self.target_dim,), dtype=np.float32)

        try:
            # Tokenize and run inference
            inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the pooler_output (CLS token equivalent) - shape [1, 768]
            # Some versions of MolFormer use 'pooler_output', others 'last_hidden_state'
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                # Mean pooling of last hidden state as fallback
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            embedding = embedding.cpu().numpy().flatten() # Shape (768,)

            # Use MolFormer's native 768 dimensions (no padding)
            # Standard MolFormer outputs 768 dimensions - use as-is
            if embedding.shape[0] != self.target_dim:
                # Only handle mismatch if model outputs unexpected size
                if embedding.shape[0] < self.target_dim:
                    # Pad with zeros if smaller (shouldn't happen for standard MolFormer)
                    padded_emb = np.zeros(self.target_dim, dtype=np.float32)
                    padded_emb[:embedding.shape[0]] = embedding
                    embedding = padded_emb
                else:
                    # Truncate if larger (shouldn't happen for standard MolFormer)
                    embedding = embedding[:self.target_dim]

            self.cache[smiles] = embedding.astype(np.float32)
            return self.cache[smiles]

        except Exception as e:
            print(f"Warning: Failed to encode SMILES '{smiles}': {e}")
            # Return random fingerprint on failure to prevent crash
            np.random.seed(hash(str(smiles)) % 2**32)
            return (np.random.rand(self.target_dim) > 0.5).astype(np.float32)

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
        
        # Pre-encode SMILES directly from each row in metadata
        # Check if SMILES column exists (case-insensitive)
        smiles_col = None
        for col in df.columns:
            if col.upper() == 'SMILES':
                smiles_col = col
                break
        
        self.fingerprints = {}  # Will store fingerprints by index or by SMILES string
        if smiles_col:
            print(f"  Using '{smiles_col}' column for SMILES encoding...", flush=True)
            # Encode SMILES for each row directly (cache by SMILES string for efficiency)
            unique_smiles_encoded = 0
            for idx, row in df.iterrows():
                smiles = str(row.get(smiles_col, '')).strip()
                if not smiles or smiles == 'nan' or smiles.lower() == 'dmso':
                    smiles = 'DMSO'
                
                # Cache by SMILES string to avoid re-encoding identical SMILES
                if smiles not in self.fingerprints:
                    self.fingerprints[smiles] = self.encoder.encode(smiles)
                    unique_smiles_encoded += 1
                    if unique_smiles_encoded <= 5:  # Log first few
                        cpd = row.get('CPD_NAME', 'unknown')
                        print(f"    Encoded SMILES for {cpd}: length={len(smiles)}", flush=True)
            
            print(f"  Encoded {unique_smiles_encoded} unique SMILES strings (from {len(df)} total rows)", flush=True)
        else:
            print(f"  WARNING: SMILES column not found in metadata. Available columns: {list(df.columns)}", flush=True)
            # Fallback: encode compound names as strings (not ideal, but prevents crash)
            for idx, row in df.iterrows():
                cpd = str(row.get('CPD_NAME', 'DMSO')).strip()
                if not cpd or cpd.upper() == 'DMSO':
                    smiles = 'DMSO'
                else:
                    smiles = cpd  # Use compound name as fallback
                if smiles not in self.fingerprints:
                    self.fingerprints[smiles] = self.encoder.encode(smiles)
        
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
                # Get fingerprint for logging
                smiles_col = None
                for col in meta.keys():
                    if col.upper() == 'SMILES':
                        smiles_col = col
                        break
                if smiles_col:
                    smiles = str(meta.get(smiles_col, '')).strip()
                    if not smiles or smiles == 'nan' or smiles.lower() == 'dmso':
                        smiles = 'DMSO'
                    fp_shape = self.fingerprints.get(smiles, np.zeros(768)).shape
                else:
                    fp_shape = np.zeros(768).shape
                print(f"  Fingerprint shape: {fp_shape}", flush=True)
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

        # Get fingerprint directly from SMILES in this row (not from CPD_NAME)
        smiles_col = None
        for col in meta.keys():
            if col.upper() == 'SMILES':
                smiles_col = col
                break
        
        if smiles_col:
            smiles = str(meta.get(smiles_col, '')).strip()
            if not smiles or smiles == 'nan' or smiles.lower() == 'dmso':
                smiles = 'DMSO'
            # Get from cache (already encoded during __init__)
            fp = self.fingerprints.get(smiles, np.zeros(768))
        else:
            # Fallback: use DMSO
            fp = self.fingerprints.get('DMSO', np.zeros(768))
            smiles = 'DMSO'
        
        cpd = meta.get('CPD_NAME', 'DMSO')
        
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

def calculate_metrics_torchmetrics(model, val_loader, device, num_samples=20480, num_inference_steps=200):
    """
    Calculate evaluation metrics using torchmetrics (FID/KID) following reference pattern.
    
    Args:
        model: The diffusion model
        val_loader: Validation data loader
        device: torch device
        num_samples: Number of samples to use for evaluation
        num_inference_steps: Number of inference steps for generation
    
    Returns:
        dict with keys: 'overall_fid', 'overall_kid', 'fid_per_class', 'kid_per_class', 'average_fid', 'average_kid'
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        TORCHMETRICS_AVAILABLE = True
    except ImportError:
        print("  Warning: torchmetrics not available. Install with: pip install torchmetrics")
        return None
    
    model.model.eval()
    
    # Initialize metrics
    fid_metric = FrechetInceptionDistance(normalize=True).to(device, non_blocking=True)
    kid_metric = KernelInceptionDistance(subset_size=100, normalize=True).to(device, non_blocking=True)
    
    # Group samples by compound for per-class metrics
    generated_samples = {}  # compound_name -> list of tensors
    target_samples = {}     # compound_name -> list of tensors
    
    sample_count = 0
    with torch.no_grad():
        from tqdm.auto import tqdm
        for batch in tqdm(val_loader, desc="  Evaluating", leave=False):
            if sample_count >= num_samples:
                break
            
            ctrl = batch['control'].to(device)
            real_t = batch['perturbed'].to(device)
            fp = batch['fingerprint'].to(device)
            compounds = batch['compound']  # List of compound names
            
            # Generate samples
            generated = model.sample(ctrl, fp, num_inference_steps=num_inference_steps)
            
            # Normalize to [0, 1] range for torchmetrics (expects [0, 255] or [0, 1])
            real_norm = torch.clamp(real_t * 0.5 + 0.5, min=0.0, max=1.0)
            gen_norm = torch.clamp(generated * 0.5 + 0.5, min=0.0, max=1.0)
            
            # Convert to [0, 255] range for torchmetrics
            real_uint8 = torch.floor(real_norm * 255.0).to(torch.uint8)
            gen_uint8 = torch.floor(gen_norm * 255.0).to(torch.uint8)
            
            # Update overall metrics
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(gen_uint8, real=False)
            kid_metric.update(real_uint8, real=True)
            kid_metric.update(gen_uint8, real=False)
            
            # Group by compound for per-class metrics
            for i in range(real_t.shape[0]):
                compound = compounds[i] if isinstance(compounds, list) else str(compounds[i])
                if compound not in generated_samples:
                    generated_samples[compound] = []
                    target_samples[compound] = []
                generated_samples[compound].append(gen_uint8[i])
                target_samples[compound].append(real_uint8[i])
            
            sample_count += real_t.shape[0]
    
    # Compute overall FID and KID
    fid = fid_metric.compute()
    kid_mean, kid_std = kid_metric.compute()
    
    # Compute per-class FID and KID
    fid_per_class = {}
    kid_per_class = {}
    
    for compound in tqdm(generated_samples.keys(), desc="  Computing per-class metrics", leave=False):
        if len(generated_samples[compound]) == 0:
            continue
        
        try:
            # Stack samples for this compound
            gen_stack = torch.stack(generated_samples[compound]).to(device)
            target_stack = torch.stack(target_samples[compound]).to(device)
            
            # Compute FID for this class
            fid_metric_class = FrechetInceptionDistance(normalize=True).to(device, non_blocking=True)
            fid_metric_class.update(target_stack, real=True)
            fid_metric_class.update(gen_stack, real=False)
            fid_per_class[compound] = float(fid_metric_class.compute().cpu().item())
            
            # Compute KID for this class
            dynamic_subset_size = min(len(generated_samples[compound]), 100)
            kid_metric_class = KernelInceptionDistance(subset_size=dynamic_subset_size, normalize=True).to(device, non_blocking=True)
            kid_metric_class.update(target_stack, real=True)
            kid_metric_class.update(gen_stack, real=False)
            kid_mean_class, kid_std_class = kid_metric_class.compute()
            kid_per_class[compound] = {
                "mean": float(kid_mean_class.cpu().item()),
                "std": float(kid_std_class.cpu().item())
            }
            
            print(f"  {compound} ({len(generated_samples[compound])}): FID={fid_per_class[compound]:.2f}, KID={kid_per_class[compound]['mean']:.4f}±{kid_per_class[compound]['std']:.4f}", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to compute metrics for {compound}: {e}", flush=True)
            continue
    
    # Calculate averages
    avg_fid = np.mean(list(fid_per_class.values())) if fid_per_class else None
    avg_kid_mean = np.mean([v["mean"] for v in kid_per_class.values()]) if kid_per_class else None
    avg_kid_std = np.mean([v["std"] for v in kid_per_class.values()]) if kid_per_class else None
    
    result = {
        "overall_fid": float(fid.item()),
        "overall_kid": {"mean": float(kid_mean.item()), "std": float(kid_std.item())},
        "fid_per_class": fid_per_class,
        "kid_per_class": kid_per_class,
        "average_fid": float(avg_fid) if avg_fid is not None else None,
        "average_kid": {"mean": avg_kid_mean, "std": avg_kid_std} if avg_kid_mean is not None else None,
    }
    
    return result

def calculate_metrics(model, val_loader, device, num_samples=1000, calculate_fid=False, num_inference_steps=200, skip_other_metrics=False):
    """
    Calculate evaluation metrics on validation set.
    
    Args:
        model: The diffusion model
        val_loader: Validation data loader
        device: torch device
        num_samples: Number of samples to use for evaluation
        calculate_fid: If True, calculate FID and cFID (slower). If False, skip FID calculation.
        num_inference_steps: Number of inference steps for generation (default: 200)
        skip_other_metrics: If True, skip KL, MSE, PSNR, SSIM and only calculate FID (faster)
    
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
        # Add progress bar for evaluation
        from tqdm.auto import tqdm
        for sample in tqdm(all_samples, desc="  Evaluating", leave=False):
            if sample_count >= num_samples:
                break
            
            ctrl = sample['control'].to(device)
            real_t = sample['perturbed'].to(device)
            fp = sample['fingerprint'].to(device)
            
            # Generate samples with specified inference steps
            generated = model.sample(ctrl, fp, num_inference_steps=num_inference_steps)
            
            # Calculate other metrics only if not skipping
            if not skip_other_metrics:
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
            
            # Collect images for FID only if enabled (saves memory and time)
            if calculate_fid:
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
            
            # Calculate PSNR and SSIM if available (only if not skipping)
            if not skip_other_metrics and SKIMAGE_AVAILABLE:
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
    
    # Calculate FID only if enabled (can be slow)
    fid_score = None
    cfid_score = None
    if calculate_fid:
        # Calculate Overall FID (all images regardless of condition)
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
    else:
        print("  Skipping FID calculation (use --calculate_fid to enable)", flush=True)
    
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
    parser.add_argument("--num_samples", type=int, default=20480, help="Number of samples to calculate FID and KID (default: 20480)")
    parser.add_argument("--inference_steps", type=int, default=200, help="Number of inference steps for generation (default: 200, use 1000 for full quality)")
    parser.add_argument("--fid_only", action="store_true", help="Only calculate FID metrics (skip KL, MSE, PSNR, SSIM for faster evaluation)")
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"], help="Data split to use for evaluation (default: val, falls back to test if val is empty)")
    args = parser.parse_args()
    
    config = Config()
    
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
    
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    
    # Initialize Logger
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: wandb.init(project="bbbc021-diffusers-pretrain", config=config.__dict__)

    print("Loading Dataset...")
    import sys
    sys.stdout.flush()  # Ensure output is flushed
    
    # --- CHANGED: Use MolFormer instead of Morgan ---
    # target_dim=1024 ensures compatible output for your U-Net
    encoder = MolFormerEncoder(target_dim=config.fingerprint_dim)
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

    # Handle eval_only mode
    if args.eval_only:
        checkpoint_path = args.checkpoint if args.checkpoint else f"{config.output_dir}/checkpoints/latest.pt"
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print(f"  Please specify --checkpoint or ensure latest.pt exists in {config.output_dir}/checkpoints/")
            return
        
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATION ONLY MODE", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Loading checkpoint: {checkpoint_path}", flush=True)
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        
        if start_epoch == 0:
            print("ERROR: Failed to load checkpoint or checkpoint is empty")
            return
        
        print(f"  Loaded checkpoint from epoch {start_epoch}", flush=True)
        print(f"  Evaluation split: {args.eval_split}", flush=True)
        print(f"  Inference steps: {args.inference_steps}", flush=True)
        print(f"  FID calculation: {'Enabled' if args.calculate_fid else 'Disabled'}", flush=True)
        print(f"  FID only mode: {'Yes' if args.fid_only else 'No'}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Run evaluation using torchmetrics
        print("Running evaluation...", flush=True)
        import json
        metrics = calculate_metrics_torchmetrics(model, val_loader, config.device, 
                                               num_samples=args.num_samples,
                                               num_inference_steps=args.inference_steps)
        
        if metrics is None:
            print("  Error: Failed to compute metrics. Make sure torchmetrics is installed.")
            return
        
        # Print metrics
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATION RESULTS", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Overall FID:        {metrics['overall_fid']:.2f}", flush=True)
        print(f"  Overall KID:        mean={metrics['overall_kid']['mean']:.4f}, std={metrics['overall_kid']['std']:.4f}", flush=True)
        if metrics['average_fid'] is not None:
            print(f"  Average FID:        {metrics['average_fid']:.2f}", flush=True)
        if metrics['average_kid'] is not None:
            print(f"  Average KID:        mean={metrics['average_kid']['mean']:.4f}, std={metrics['average_kid']['std']:.4f}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Save results to JSON
        output_name = f"mol_eval_{args.eval_split}_{args.num_samples}_{args.inference_steps}"
        os.makedirs("outputs/evaluation", exist_ok=True)
        json_path = f"outputs/evaluation/{output_name}.json"
        
        results = {
            "model": "mol.py",
            "checkpoint": checkpoint_path,
            "eval_split": args.eval_split,
            "num_samples": args.num_samples,
            "inference_steps": args.inference_steps,
            **metrics
        }
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Evaluation complete! Results saved to {json_path}", flush=True)
        return
    
    # Load checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint else f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume or args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"\n{'='*60}", flush=True)
            print(f"RESUMING TRAINING FROM EPOCH {start_epoch}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
            print(f"  Checkpoint: {checkpoint_path}", flush=True)
            
            # Run evaluation first when resuming (only generate samples/video, skip metrics by default)
            print(f"\n🔍 Running evaluation on loaded checkpoint before continuing training...", flush=True)
            print(f"{'='*60}", flush=True)
            
            metrics = None
            if not config.skip_metrics_during_training:
                metrics = calculate_metrics(model, val_loader, config.device, num_samples=1000, 
                                          calculate_fid=config.calculate_fid, 
                                          num_inference_steps=args.inference_steps,
                                          skip_other_metrics=args.fid_only)
                
                # Print metrics prominently
                print(f"\n  📊 EVALUATION METRICS (Resume Checkpoint):", flush=True)
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
            else:
                print("  Skipping metric calculations (only generating samples/video)...", flush=True)
            
            # Generate visualization for resume checkpoint
            val_iter = iter(val_loader)
            batch = next(val_iter)
            ctrl = batch['control'].to(config.device)
            real_t = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            fakes = model.sample(ctrl, fp, num_inference_steps=200)
            grid = torch.cat([ctrl[:8], fakes[:8], real_t[:8]], dim=0)
            resume_grid_path = f"{config.output_dir}/plots/resume_epoch_{start_epoch}.png"
            save_image(grid, resume_grid_path, nrow=8, normalize=True, value_range=(-1,1))
            print(f"  ✓ Resume checkpoint grid saved to: {resume_grid_path}", flush=True)
            
            generate_video(model, ctrl[0:1], fp[0:1], f"{config.output_dir}/plots/video_resume_epoch_{start_epoch}.mp4")
            print(f"  ✓ Resume checkpoint video saved", flush=True)
            
            # Log metrics for resume checkpoint
            logger.update(start_epoch, 0.0, metrics, scheduler.get_last_lr()[0])
            print(f"{'='*60}\n", flush=True)
            print(f"✅ Evaluation complete. Continuing training from epoch {start_epoch+1}...\n", flush=True)
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
        
        # Generate samples and video during evaluation (metrics optional)
        metrics = None
        if (epoch + 1) % config.eval_freq == 0:
            print("\n" + "="*60, flush=True)
            print(f"EVALUATION (Epoch {epoch+1})", flush=True)
            print("="*60, flush=True)
            
            # Calculate metrics only if not skipped
            if not config.skip_metrics_during_training:
                print("  Calculating metrics on validation set...", flush=True)
                metrics = calculate_metrics(model, val_loader, config.device, num_samples=1000, 
                                          calculate_fid=config.calculate_fid,
                                          num_inference_steps=args.inference_steps,
                                          skip_other_metrics=args.fid_only)
                
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
            else:
                print("  Skipping metric calculations (only generating samples/video)...", flush=True)
            
            print("="*60 + "\n", flush=True)
            
            # Visualization (always generate samples and video)
            print("  Generating sample grid and video...", flush=True)
            val_iter = iter(val_loader)
            batch = next(val_iter)
            ctrl = batch['control'].to(config.device)
            real_t = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            # Use fewer inference steps for faster evaluation (can increase later)
            fakes = model.sample(ctrl, fp, num_inference_steps=200)
            
            grid = torch.cat([ctrl[:8], fakes[:8], real_t[:8]], dim=0)
            save_image(grid, f"{config.output_dir}/plots/epoch_{epoch+1}.png", nrow=8, normalize=True, value_range=(-1,1))
            print(f"  ✓ Sample grid saved to: {config.output_dir}/plots/epoch_{epoch+1}.png", flush=True)
            generate_video(model, ctrl[0:1], fp[0:1], f"{config.output_dir}/plots/video_{epoch+1}.mp4")
            print(f"  ✓ Video saved to: {config.output_dir}/plots/video_{epoch+1}.mp4", flush=True)

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