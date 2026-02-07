"""
================================================================================
BBBC021 UNCONDITIONAL PRETRAINING - SINGLE MODEL
================================================================================
Trains a single unconditional DDPM model on ALL available images (control + perturbed).
Uses a null-token conditioning strategy for future compatibility with cross-attention control.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance

try:
    from diffusers import UNet2DConditionModel, DDPMScheduler
except ImportError:
    print("CRITICAL: 'diffusers' library not found. Install with: pip install diffusers")
    sys.exit(1)

# --- Plotting Backend (Prevents crashes on headless servers) ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt





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

    # Diffusion
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 2e-2

    # Training
    epochs = 200
    batch_size = 128
    lr = 3e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    output_dir = "ddpm_uncond_all"

    # Sampling / eval
    sample_every_epochs = 1           # save gen samples
    preview_real_every_epochs = 1     # save real batches
    fid_every_epochs = 10             # <-- N epochs
    fid_num_real = 1000               # train only
    fid_num_gen  = 1000               # generated
    fid_inference_steps = 50          # <-- Reduced for faster monitoring

    # Future-proof attention
    use_cross_attn = True
    cross_attention_dim = 128
    attention_head_dim = 64           # Explicitly defined
    unet_block_out_channels = (128, 256, 512, 512)

    # EMA (recommended; use EMA for sampling + FID + theta_ref)
    use_ema = True
    ema_decay = 0.9999

    # EMA (recommended; use EMA for sampling + FID + theta_ref)
    # use_ema = True  # Already defined above
    # ema_decay = 0.9999

    # --- Stage 1B control-anchor ---
    stage = "global"  # "global" or "ctrl"
    global_ckpt = "ddpm_uncond_all/checkpoints/best.pt"  # or latest.pt
    global_ema_path = "ddpm_uncond_all/theta_ref_ema_best.pt"  # preferred if exists

    ctrl_epochs = 50            # short fine-tune (10–50 typical)
    ctrl_lr = 5e-6              # tiny LR (1e-6 to 1e-5)
    ctrl_patience = 5           # early stop patience (FID or loss)
    ctrl_min_delta = 0.2        # FID improvement threshold
    ctrl_output_dir = "ddpm_uncond_ctrl"
# ============================================================================

class TrainingLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = {
            "epoch": [],
            "train_loss": [],
            "learning_rate": [],
            "fid": [],
        }
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        
    def update(self, epoch, loss, metrics=None, lr=None):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(loss)
        self.history['learning_rate'].append(lr if lr is not None else 0)
        
        fid_val = metrics.get('fid') if metrics else None
        self.history['fid'].append(fid_val)
        
        # Save to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Plot
        self._plot_loss()
        
    def _plot_loss(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Loss
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(self.history['epoch'], self.history['train_loss'], color='tab:blue', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # FID
        if any(v is not None for v in self.history['fid']):
            ax2 = ax1.twinx()
            ax2.set_ylabel('FID', color='tab:red')
            # Filter None values for plotting
            valid_epochs = [e for e, f in zip(self.history['epoch'], self.history['fid']) if f is not None]
            valid_fids = [f for f in self.history['fid'] if f is not None]
            ax2.plot(valid_epochs, valid_fids, color='tab:red', marker='o', label='FID')
            ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Training Progress')
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

# ============================================================================
# HELPERS
# ============================================================================

def make_ema_copy(model: nn.Module):
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.inference_mode()
def ema_update(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p, alpha=1 - decay)
    # Copy buffers (running stats) occasionally or always (cheap)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)

def load_global_ema_into_model(model, config):
    # Prefer loading EMA weights directly
    if config.global_ema_path and os.path.exists(config.global_ema_path):
        sd = torch.load(config.global_ema_path, map_location="cpu")
        model.model.load_state_dict(sd, strict=True)
        if config.use_ema and model.ema_model is not None:
            model.ema_model.load_state_dict(sd, strict=True)
        print(f"✓ Loaded global EMA weights from {config.global_ema_path}")
        return

    # Otherwise try checkpoint dict
    if not config.global_ckpt or not os.path.exists(config.global_ckpt):
        print(f"WARNING: Global checkpoint not found ({config.global_ckpt}). Starting from scratch.")
        return

    ckpt = torch.load(config.global_ckpt, map_location="cpu")
    if ckpt.get("ema_model") is not None:
        model.model.load_state_dict(ckpt["ema_model"], strict=True)
        if config.use_ema and model.ema_model is not None:
            model.ema_model.load_state_dict(ckpt["ema_model"], strict=True)
        print(f"✓ Loaded EMA from checkpoint {config.global_ckpt}")
    else:
        model.model.load_state_dict(ckpt["model"], strict=True)
        if config.use_ema and model.ema_model is not None:
            model.ema_model.load_state_dict(ckpt["model"], strict=True)
        print(f"✓ Loaded model weights (no EMA) from {config.global_ckpt}")

import json
def dump_config(config, output_dir):
    cfg = {}
    for k in dir(config):
        if k.startswith("_"): continue
        v = getattr(config, k)
        if callable(v): continue
        cfg[k] = v
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


@torch.inference_mode()
def save_real_batch_preview(loader, save_path, max_images=16):
    batch = next(iter(loader))
    imgs = batch["image"] if isinstance(batch, dict) else batch
    imgs = imgs[:max_images]
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1) # [-1,1] -> [0,1]
    save_image(imgs, save_path, nrow=int(math.sqrt(max_images)), normalize=False)

def make_fixed_subset(dataset, n=1000, seed=42, save_path=None):
    g = torch.Generator().manual_seed(seed)
    n = min(n, len(dataset))
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, np.array(idx, dtype=np.int64))
    return Subset(dataset, idx)

@torch.inference_mode()
def compute_fid_uncond(model, real_loader, device,
                       num_real=1000, num_gen=1000,
                       gen_bs=64, steps=200):
    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Real (train subset)
    seen = 0
    for imgs in real_loader:
        if isinstance(imgs, dict):
            imgs = imgs["image"]
        imgs = imgs.to(device)
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        # imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False) # Handled by TorchMetrics
        
        # Exact counting: slice if last batch exceeds num_real
        remaining = num_real - seen
        if imgs.size(0) > remaining:
            imgs = imgs[:remaining]
            
        fid.update(imgs, real=True)
        seen += imgs.size(0)
        if seen >= num_real:
            break

    # Generated
    seen = 0
    while seen < num_gen:
        bs = min(gen_bs, num_gen - seen)
        gen = model.sample(batch_size=bs, num_inference_steps=steps)
        gen = (gen * 0.5 + 0.5).clamp(0, 1)
        # gen = F.interpolate(gen, size=(299, 299), mode="bilinear", align_corners=False) # Handled by TorchMetrics
        fid.update(gen, real=False)
        seen += bs

    return fid.compute().item()

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
        if split and 'SPLIT' in df.columns: 
            df = df[df['SPLIT'].astype(str).str.lower() == split.lower()]
        
        self.metadata = df.to_dict('records')
        self.batch_map = self._group_by_batch()
        
        # Pre-encode chemicals
        self.fingerprints = {}
        if self.encoder is not None and 'CPD_NAME' in df.columns:
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
            
            # Build strict sample_key index for Strategy 2 (O(1), low collision)
            self.sample_key_to_rel = {}
            for rel_path in paths_df['relative_path']:
                # Expected format: .../WeekX/BATCH/TABLE_IMAGE_OBJECT.0.npy
                try:
                    p = Path(rel_path)
                    parts = p.parts
                    if len(parts) >= 3 and parts[-3].startswith("Week"):
                        week = parts[-3]
                        batch = parts[-2]
                        stem = p.stem # "7_3338_348.0" (or similar)
                        
                        # Store "Week7_34681_7_3338_348.0"
                        key_full = f"{week}_{batch}_{stem}"
                        if key_full not in self.sample_key_to_rel:
                            self.sample_key_to_rel[key_full] = []
                        self.sample_key_to_rel[key_full].append(rel_path)
                        
                        # Store "Week7_34681_7_3338_348" (no .0)
                        key_clean = key_full[:-2] if key_full.endswith(".0") else key_full
                        if key_clean != key_full:
                            if key_clean not in self.sample_key_to_rel:
                                self.sample_key_to_rel[key_clean] = []
                            self.sample_key_to_rel[key_clean].append(rel_path)
                except Exception:
                    continue
            
            # Dedupe once after building
            for k, v in self.sample_key_to_rel.items():
                self.sample_key_to_rel[k] = list(dict.fromkeys(v))
            
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
            key = path[:-2] if path.endswith(".0") else path
            parts = key.split('_')
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
        
        # Strategy 2: Fast Lookup via sample_key (O(1))
        # path is like "Week7_34681_7_3338_348.0"
        if getattr(self, 'sample_key_to_rel', None):
            # We try the path as-is, and maybe without .0 if user query didn't have it
            # But usually path HAS .0 if it came from metadata
            key2 = path[:-2] if path.endswith(".0") else path
            candidates_keys = [path, key2]
            
            for key in candidates_keys:
                if key in self.sample_key_to_rel:
                    for rel_path in self.sample_key_to_rel[key]:
                        # Resolve full path candidates
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
            # Optimized loading: skip stats unless logging
            img = np.load(full_path)
            
            do_log = (not self._first_load_logged) or (idx < 3)
            if do_log:
                original_shape = img.shape
                original_dtype = img.dtype
                file_size_bytes = full_path.stat().st_size if full_path.exists() else 0
                original_min = float(img.min())
                original_max = float(img.max())
            
            # Handle [H, W, C] -> [C, H, W]
            if img.ndim == 3 and img.shape[-1] == 3: 
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            
            # Normalize [0, 255] or [0, 1] -> [-1, 1]
            if float(img.max()) > 1.0: 
                img = (img / 127.5) - 1.0
            else:
                img = (img * 2.0) - 1.0
                
            img = torch.clamp(img, -1, 1)

            # Ensure CHW
            if img.ndim == 2:
                img = img.unsqueeze(0)  # [1,H,W]
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)  # [3,H,W]
            if img.ndim != 3 or img.shape[0] != 3:
                raise RuntimeError(f"Unexpected image shape after processing: {img.shape} for {full_path}")
            
            # Log details for first successful load (or first few)
            if do_log:
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
                if self.encoder:
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
        fp = self.fingerprints.get(cpd, np.zeros(1024)) if self.encoder else np.zeros(1024) # Return dummy if no encoder
        
        return {
            'image': img, 
            'fingerprint': torch.from_numpy(fp).float(), 
            'compound': cpd
        }

class AllImagesDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx]["image"]  # tensor [-1,1], shape [3,H,W]

class ControlOnlyImagesDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        # indices where CPD_NAME == DMSO
        self.ctrl_indices = [
            i for i, m in enumerate(self.base.metadata)
            if str(m.get("CPD_NAME", "")).upper() == "DMSO"
        ]

    def __len__(self):
        return len(self.ctrl_indices)

    def __getitem__(self, idx):
        real_idx = self.ctrl_indices[idx]
        return self.base[real_idx]["image"]

# ============================================================================
# ARCHITECTURE
# ============================================================================

class UnconditionalUNet(nn.Module):
    def __init__(self, image_size=96, cross_attention_dim=128, attention_head_dim=64,
                 block_out_channels=(128, 256, 512, 512)):
        super().__init__()

        # Cross-attn only at low-res blocks (future-proof)
        down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        )
        up_block_types = (
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )

        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim, 
        )

        # Learnable null token: [1, 1, cross_attention_dim]
        # Initialize to small noise to break symmetry
        self.null_token = nn.Parameter(0.01 * torch.randn(1, 1, cross_attention_dim))

    def forward(self, x, t):
        b = x.shape[0]
        null_ctx = self.null_token.expand(b, 1, -1)  # [B,1,D]
        return self.unet(x, t, encoder_hidden_states=null_ctx).sample

class UnconditionalDiffusionModel(nn.Module):
    """Unconditional DDPM model - no conditioning"""
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = UnconditionalUNet(
            image_size=config.image_size, 
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
            block_out_channels=config.unet_block_out_channels,
        ).to(config.device)
        
        if config.use_ema:
            self.ema_model = make_ema_copy(self.model)
        else:
            self.ema_model = None
        
        # Use Diffusers' DDPMScheduler for consistent training/sampling
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        self.timesteps = config.timesteps
        self._cached_steps = None

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

    @torch.inference_mode()
    def sample(self, batch_size=1, num_inference_steps=None, image_size=None):
        """Generate unconditional samples"""
        # Select model (EMA or regular)
        net = self.ema_model if (self.cfg.use_ema and self.ema_model is not None) else self.model
        net.eval()
        
        if image_size is None:
            image_size = self.cfg.image_size
        h = w = image_size
        xt = torch.randn((batch_size, 3, h, w), device=self.cfg.device)
        
        # Use scheduler for consistent sampling (cached)
        steps = num_inference_steps or self.timesteps
        if steps != self._cached_steps:
            self.noise_scheduler.set_timesteps(steps, device=self.cfg.device)
            self._cached_steps = steps
        
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=self.cfg.device, dtype=torch.long)
            noise_pred = net(xt, t_batch)
            xt = self.noise_scheduler.step(noise_pred, t, xt).prev_sample
        
        # Clamp only once at the end
        return xt.clamp(-1, 1)

# ============================================================================
# MAIN
# ============================================================================

def train_unconditional_model(model, train_loader, fid_loader, config, output_dir, resume_ckpt=None):
    """Train a single unconditional DDPM model"""
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    logger = TrainingLogger(output_dir)
    use_amp = config.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Create a consistent preview loader (non-shuffled)
    preview_loader = DataLoader(train_loader.dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Dump effective config (includes overrides)
    dump_config(config, output_dir)

    # Optimizer only for model parameters (exclude EMA)
    wd = 0.0 if config.stage == "ctrl" else 0.01
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.lr, weight_decay=wd)

    # Scheduler: Cosine for global, None/Constant for ctrl
    if config.stage == "global":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    no_improve = 0
    best_metric = float("inf")
    last_epoch = 0
    start_epoch = 0
    best_fid = float("inf")
    
    # Resume state if provided
    if resume_ckpt is not None:
        if resume_ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scheduler is not None and resume_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        if use_amp and resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        if "best_fid" in resume_ckpt and resume_ckpt["best_fid"] is not None:
            best_fid = float(resume_ckpt["best_fid"])
        print(f"✓ Resuming training from epoch {start_epoch} | Best FID so far: {best_fid:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Training Unconditional Model (Stage: {config.stage})")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Use EMA: {config.use_ema}")
    print(f"{'='*60}\n")
    
    # Generate initial samples (before training starts) if starting from scratch
    if start_epoch == 0:
        print("  Generating initial samples (before training)...")
        with torch.inference_mode():
            initial_samples = model.sample(batch_size=16, num_inference_steps=config.fid_inference_steps)
            save_image(initial_samples, f"{output_dir}/plots/samples_epoch_0_initial.png", 
                      nrow=4, normalize=True, value_range=(-1, 1))
            print(f"  ✓ Initial samples saved to {output_dir}/plots/samples_epoch_0_initial.png\n")
    
    for epoch in range(start_epoch, config.epochs):
        model.model.train()
        losses = []
        is_best = False
        fid_val = None
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            
            # Robust batch loading
            if isinstance(batch, dict):
                images = batch["image"].to(config.device)
            else:
                images = batch.to(config.device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(images)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if config.use_ema and model.ema_model is not None:
                ema_update(model.ema_model, model.model, config.ema_decay)
                
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        if scheduler is not None:
             scheduler.step()
             current_lr = scheduler.get_last_lr()[0]
        else:
             current_lr = config.lr
        
        last_epoch = epoch + 1
        print(f"Epoch {last_epoch}/{config.epochs} | Loss: {avg_loss:.5f} | LR: {current_lr:.2e}")
        
        # Metrics logic
        metrics = None
        
        # optional: save a real preview each epoch
        if (epoch + 1) % config.preview_real_every_epochs == 0:
            save_real_batch_preview(preview_loader, f"{output_dir}/plots/real_preview_epoch{epoch+1}.png")

        # generate samples
        if (epoch + 1) % config.sample_every_epochs == 0:
            samples = model.sample(batch_size=16, num_inference_steps=config.fid_inference_steps)
            save_image(samples, f"{output_dir}/plots/samples_epoch_{epoch+1}.png",
                       nrow=4, normalize=True, value_range=(-1, 1))

        # FID on train subset only
        if (epoch + 1) % config.fid_every_epochs == 0:
            net_for_eval = model  # model.sample handles EMA
            fid_val = compute_fid_uncond(
                net_for_eval, fid_loader, config.device,
                num_real=config.fid_num_real,
                num_gen=config.fid_num_gen,
                gen_bs=config.batch_size,
                steps=config.fid_inference_steps
            )
            metrics = {"fid": fid_val} if fid_val is not None else None
            print(f"  TRAIN-FID (n={config.fid_num_real}): {fid_val:.3f}")
            
            if fid_val < best_fid:
                best_fid = fid_val
                is_best = True

            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
            
        logger.update(epoch+1, avg_loss, metrics=metrics, lr=current_lr)
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
        save_dict = {
            'model': model.model.state_dict(),
            'ema_model': model.ema_model.state_dict() if (config.use_ema and model.ema_model is not None) else None,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if use_amp else None,
            'epoch': epoch+1,
            'best_fid': best_fid
        }
        torch.save(save_dict, checkpoint_path)
        torch.save(save_dict, f"{output_dir}/checkpoints/latest.pt")
        
        # Snapshot anchor EMA
        if config.use_ema and model.ema_model is not None:
            anchor_name = "theta_ctrl_ema.pt" if config.stage == "ctrl" else "theta_ref_ema.pt"
            torch.save(model.ema_model.state_dict(), f"{output_dir}/{anchor_name}")
            
        if is_best:
            torch.save(save_dict, f"{output_dir}/checkpoints/best.pt")
            if config.use_ema and model.ema_model is not None:
                best_name = "theta_ctrl_ema_best.pt" if config.stage == "ctrl" else "theta_ref_ema_best.pt"
                torch.save(model.ema_model.state_dict(), f"{output_dir}/{best_name}")
            print(f"  ★ New Best FID! Saved best checkpoints.")

        # Early stopping logic (only for ctrl stage)
        if config.stage == "ctrl" and fid_val is not None:
            if fid_val < best_metric - config.ctrl_min_delta:
                best_metric = fid_val
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= config.ctrl_patience:
                print(f"⏹ Early stop: no FID improvement for {config.ctrl_patience} evals")
                break
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path}\n")
    
    # Generate final samples
    print(f"  Generating final samples (after training)...")
    with torch.no_grad():
        final_samples = model.sample(batch_size=16, num_inference_steps=config.fid_inference_steps)
        final_path = f"{output_dir}/plots/samples_epoch_{last_epoch}_final.png"
        save_image(final_samples, final_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"  ✓ Final samples saved to {final_path}")
    
    print(f"\n✅ Training complete!")
    print(f"   Final checkpoint: {output_dir}/checkpoints/checkpoint_epoch_{last_epoch}.pt\n")

def main():
    parser = argparse.ArgumentParser(description="Train unconditional DDPM on BBBC021 dataset (single model)")
    parser.add_argument("--paths_csv", type=str, default="paths.csv", help="Path to paths.csv file")
    parser.add_argument("--stage", type=str, default="global", choices=["global", "ctrl"], help="Training stage")
    parser.add_argument("--global_ema_path", type=str, default=None, help="Path to global EMA weights")
    parser.add_argument("--global_ckpt", type=str, default=None, help="Path to global checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    # CUDA Optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    config = Config()
    config.stage = args.stage

    if args.global_ema_path is not None:
        config.global_ema_path = args.global_ema_path
    if args.global_ckpt is not None:
        config.global_ckpt = args.global_ckpt

    if config.stage == "ctrl":
        config.output_dir = config.ctrl_output_dir
        config.epochs = config.ctrl_epochs
        config.lr = config.ctrl_lr
        config.fid_every_epochs = 1
        config.sample_every_epochs = 1
        config.preview_real_every_epochs = 1
        config.fid_inference_steps = 50
        config.fid_num_gen = 1000
        config.fid_num_real = 1000
        config.ctrl_min_delta = 0.5
        
    assert config.cross_attention_dim % config.attention_head_dim == 0, \
           f"cross_attention_dim ({config.cross_attention_dim}) should be divisible by attention_head_dim ({config.attention_head_dim})"
    
    print("\n" + "="*60)
    print(f"UNCONDITIONAL DDPM PRETRAINING (STAGE: {config.stage})")
    print("="*60 + "\n")
    
    # Initialize Model
    model = UnconditionalDiffusionModel(config)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    if config.stage == "ctrl":
        load_global_ema_into_model(model, config)
    
    # Resume logic
    resume_ckpt = None
    if args.resume:
        latest = os.path.join(config.output_dir, "checkpoints/latest.pt")
        if os.path.exists(latest):
            resume_ckpt = torch.load(latest, map_location="cpu")
            model.model.load_state_dict(resume_ckpt["model"], strict=True)
            if config.use_ema and resume_ckpt.get("ema_model") is not None and model.ema_model is not None:
                model.ema_model.load_state_dict(resume_ckpt["ema_model"], strict=True)
            print(f"✓ Resumed weights from {latest}")
    
    # Load base dataset
    print("Loading Dataset...")
    # encoder = MorganFingerprintEncoder()  # Still needed for dataset initialization
    base_ds = BBBC021Dataset(
        config.data_dir, config.metadata_file, 
        image_size=config.image_size, split='train', 
        encoder=None,  # No fingerprint needed for unconditional
        paths_csv=args.paths_csv
    )
    if len(base_ds) == 0:
        base_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='', encoder=None, paths_csv=args.paths_csv)
    
    # Select dataset based on stage
    if config.stage == "global":
        train_ds = AllImagesDataset(base_ds)
    else:
        print("Filtering for Control (DMSO) images only...")
        train_ds = ControlOnlyImagesDataset(base_ds)
        
    # Create DataLoaders
    # num_workers=0 to avoid multiprocessing issues on Windows/Jupyter
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4 if os.name != 'nt' else 0, # Use workers on Linux
        pin_memory=config.device.startswith("cuda")
    )

    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(train_ds)}")
    
    # Create output directories
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    
    # Save preview of real images
    print("Saving real batch preview...")
    # quick sanity: save first 16 real images deterministically from a non-shuffled loader
    preview_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    save_real_batch_preview(preview_loader, f"{config.output_dir}/plots/real_preview_epoch0.png", max_images=16)

    # sanity: show noisy images at a fixed timestep
    print("Generating noisy preview at t=500...")
    batch = next(iter(preview_loader))
    if isinstance(batch, dict):
        batch = batch["image"]
    batch = batch.to(config.device)
    
    t = torch.full((batch.size(0),), 500, device=config.device, dtype=torch.long)
    noise = torch.randn_like(batch)
    scheduler_tmp = DDPMScheduler(
        num_train_timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    noisy = scheduler_tmp.add_noise(batch, noise, t).clamp(-1, 1)
    save_image(noisy, f"{config.output_dir}/plots/noisy_t500_preview.png", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"  ✓ Noisy preview saved to {config.output_dir}/plots/noisy_t500_preview.png")

    # Fixed train subset for FID
    fid_subset = make_fixed_subset(
        train_ds, n=config.fid_num_real, seed=42,
        save_path=f"{config.output_dir}/fid_subset_indices.npy"
    )
    fid_loader = DataLoader(fid_subset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)
    
    # Run training
    train_unconditional_model(model, train_loader, fid_loader, config, config.output_dir, resume_ckpt=resume_ckpt)

if __name__ == "__main__":
    main()
