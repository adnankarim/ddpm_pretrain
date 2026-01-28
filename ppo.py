
import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# Check for diffusers
try:
    from diffusers import UNet2DModel, DDPMScheduler
except ImportError:
    print("CRITICAL: 'diffusers' library not found. Install with: pip install diffusers")
    sys.exit(1)

# Check for RDKit
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
    metadata_file = "bbbc021_df_all.csv" # Fixed path: file is in root
    image_size = 96
    
    # Architecture
    base_model_id = "google/ddpm-cifar10-32"
    perturbation_emb_dim = 128 
    fingerprint_dim = 1024
    
    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training / PPO
    lr = 5e-6             # Lower LR for fine-tuning stability
    batch_size = 32       # Batch size for initial sampling (rollouts will be broken into minibatches)
    ppo_minibatch_size = 64 # Minibatch size for PPO updates
    
    # PPO Specifics
    ppo_epochs = 4        # K optimization epochs per rollout
    ppo_clip_range = 0.1  # Clipping parameter epsilon
    kl_beta = 0.5         # Weight for KL anchor to pretrained model (Increased from 0.05)
    rollout_steps = 1000  # Full rollout (temporarily disable striding)
    cond_drop_prob = 0.1  # Probability of dropping conditioning for CFG training
    guidance_scale = 1.0  # Classifier-free guidance scale (Init to 1.0 as pretrained model is not CFG-trained)
    
    # Logging
    log_file = "training_log.csv"
    
    # Reward Estimation (DDMEC Eq. 8)
    reward_n_terms = 32  # Terms in sum
    reward_mc = 3        # Monte Carlo repetitions
    
    # Paths
    theta_checkpoint = "./ddpm_diffusers_results/checkpoints/checkpoint_epoch_60.pt"     # Pretrained Theta
    phi_checkpoint = "./results_phi_phi/checkpoints/checkpoint_epoch_100.pt"   # Pretrained Phi
    output_dir = "ddpm_ddmec_results"
    
    # Evaluation
    eval_max_samples = 5000
    eval_steps = 50
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATASET & ENCODER (Copied from train2.py)
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
        
        # Load paths.csv for robust file lookup (same as train2.py)
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
        Robust file path finding using paths.csv lookup (same logic as train2.py).
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
                            # Handle path resolution (same as train2.py)
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
        
        # Use robust path finding (same as train2.py)
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
            # Also handle [C, H, W] case (if shape[0] == 3, it's already CHW)
            elif img.ndim == 3 and img.shape[0] == 3:
                # Already in CHW format, no transpose needed
                pass
            img = torch.from_numpy(img).float()
            
            # Normalize [0, 255] or [0, 1] -> [-1, 1]
            if img.max() > 1.0: 
                img = (img / 127.5) - 1.0
            else:
                img = (img * 2.0) - 1.0
                
            img = torch.clamp(img, -1, 1)
            
            # Debug: Check for constant images (Grey/Black)
            if img.min() == img.max():
                print(f"WARNING: Loade image {full_path} is constant value {img.min()}! (Original range: [{original_min}, {original_max}])")
            
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
    
    def __len__(self): return (len(self.indices) + self.bs - 1) // self.bs

# ============================================================================
# MODELS (Copied from train2.py)
# ============================================================================

class ModifiedDiffusersUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        base_model_id = Config.base_model_id
        
        try:
            unet_pre = UNet2DModel.from_pretrained(base_model_id)
        except:
            unet_pre = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")
        
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=6,
            out_channels=unet_pre.config.out_channels,
            layers_per_block=unet_pre.config.layers_per_block,
            block_out_channels=unet_pre.config.block_out_channels,
            down_block_types=unet_pre.config.down_block_types,
            up_block_types=unet_pre.config.up_block_types,
            dropout=unet_pre.config.dropout,
            attention_head_dim=getattr(unet_pre.config, "attention_head_dim", None),
            norm_num_groups=unet_pre.config.norm_num_groups,
            class_embed_type="identity"
        )
        
        # Conv in surgery
        old_conv = unet_pre.conv_in
        new_conv = nn.Conv2d(6, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
            new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        
        pretrained_state = unet_pre.state_dict()
        filtered_state = {k: v for k, v in pretrained_state.items() if not k.startswith('conv_in.')}
        self.unet.load_state_dict(filtered_state, strict=False)

        target_dim = self.unet.time_embedding.linear_1.out_features
        self.target_dim = target_dim
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 512), nn.SiLU(), nn.Linear(512, target_dim)
        )

    def forward(self, x, t, control, fingerprint, drop_cond=False):
        if drop_cond:
            control = torch.zeros_like(control)
            # Explicitly zero the embedding to avoid bias from projection layer
            emb = torch.zeros((x.shape[0], self.target_dim), device=x.device, dtype=x.dtype)
        else:
            emb = self.fingerprint_proj(fingerprint)
            
        x_in = torch.cat([x, control], dim=1)
        return self.unet(x_in, t, class_labels=emb).sample

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = ModifiedDiffusersUNet(config.image_size, config.fingerprint_dim).to(config.device)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule="linear",
            prediction_type="epsilon",
            variance_type="fixed_small",
            clip_sample=True
        )
        self.timesteps = config.timesteps

    def forward(self, x0, control, fingerprint, drop_cond=False):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.cfg.device).long()
        noise = torch.randn_like(x0)
        xt = self.noise_scheduler.add_noise(x0, noise, t)
        noise_pred = self.model(xt, t, control, fingerprint, drop_cond=drop_cond)
        return F.mse_loss(noise_pred, noise)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"Warning: Checkpoint {path} not found.")
            return
        print(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.cfg.device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        # Handle 'model.' prefix if present (e.g. from DDP or DiffusionModel wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            elif k.startswith('unet.'):
                 # Case where we are loading directly into unet but keys start with unet.
                 # This matches ModifiedDiffusersUNet structure if we load into it directly?
                 # No, ModifiedDiffusersUNet has 'unet' as a member.
                 # Wait, if we load into self.model (ModifiedDiffusersUNet), it expects:
                 # 'unet.conv_in...', 'fingerprint_proj...'
                 # Check if keys are 'unet...' or 'model.unet...'
                 new_state_dict[k] = v
            else:
                new_state_dict[k] = v
                
        # Load with strict=False to be safe, or check keys
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"Warning: Missing keys in checkpoint: {missing[:5]} ...")
        if len(unexpected) > 0:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected[:5]} ...")

# ============================================================================
# TRAJECTORY / PROBABILITY HELPERS 
# ============================================================================

def get_posterior_mean_variance(scheduler, x_t, eps_pred, t, t_prev, clip_sample=True):
    """
    Compute posterior mean and variance for p(x_{t_prev} | x_t, x_0) 
    CONSISTENT with DDPMScheduler (assuming epsilon prediction).
    Handles strided steps by using t and t_prev explicitly.
    """
    device = x_t.device
    
    # 1. Get alphas/betas for the specific integer timesteps
    tensor_t = t.to(device)
    tensor_t_prev = t_prev.to(device)
    
    alpha_prod_t = scheduler.alphas_cumprod.to(device)[tensor_t]
    
    # Handle t_prev < 0 case (final step)
    alpha_prod_t_prev = torch.ones_like(alpha_prod_t)
    mask_prev = (tensor_t_prev >= 0)
    if mask_prev.any():
        alpha_prod_t_prev[mask_prev] = scheduler.alphas_cumprod.to(device)[tensor_t_prev[mask_prev]]
        
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    
    # 2. Compute predicted x_0
    # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
    alpha_prod_t_reshaped = alpha_prod_t[:, None, None, None]
    beta_prod_t_reshaped = beta_prod_t[:, None, None, None]
    pred_original_sample = (x_t - beta_prod_t_reshaped ** 0.5 * eps_pred) / alpha_prod_t_reshaped ** 0.5
    
    # Clip x0 (only if scheduler is configured to do so)
    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    
    # 3. Compute Posterior Mean
    alpha_t_step = alpha_prod_t / alpha_prod_t_prev
    beta_t_step = 1 - alpha_t_step
    
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * beta_t_step) / beta_prod_t
    current_sample_coeff = (alpha_t_step ** 0.5 * beta_prod_t_prev) / beta_prod_t
    
    pred_prev_sample_mean = pred_original_sample_coeff[:, None, None, None] * pred_original_sample + \
                            current_sample_coeff[:, None, None, None] * x_t
                            
    # 4. Compute Posterior Variance
    # sigma^2 = beta_t_step * (1-alpha_bar_prev) / (1-alpha_bar_t)
    # This matches 'fixed_small' variance type which is standard for DDPM
    pred_prev_sample_var = (beta_t_step * beta_prod_t_prev) / beta_prod_t
    
    # Handle variance=0 at typically the last step.
    # We allow it to be computed as is, but clamp minimum for log_prob stability
    pred_prev_sample_var = torch.clamp(pred_prev_sample_var, min=1e-20)
    
    return pred_prev_sample_mean, pred_prev_sample_var[:, None, None, None], pred_original_sample

@torch.no_grad()
def rollout_with_logprobs(model: DiffusionModel, cond_img, fingerprint, steps=None):
    """
    Rollout trajectory matching DDPM scheduler physics and storing correct logprobs.
    """
    model.model.eval()
    scheduler = model.noise_scheduler
    device = model.cfg.device
    b, c, h, w = cond_img.shape
    
    # Start from random noise
    x = torch.randn((b, 3, h, w), device=device)
    
    # Set timesteps (strided or full)
    inference_steps = steps if steps else model.timesteps
    scheduler.set_timesteps(inference_steps, device=device)
    timesteps = scheduler.timesteps # Ordered high to low, e.g. [990, 980, ..., 0]
    
    # Get config for sample clipping
    clip_sample = getattr(scheduler.config, 'clip_sample', True)  # Default True for DDPM usually
    
    traj = [] # (x_t, x_prev, t, t_prev, old_logprob)
    
    for i, t in enumerate(timesteps):
        # Determine t and t_prev (cast to int for safety)
        t_int = int(t)
        t_batch = torch.full((b,), t_int, device=device, dtype=torch.long)
        
        # Get next timestep (t_prev)
        if i < len(timesteps) - 1:
            prev_t_int = int(timesteps[i + 1])
        else:
            prev_t_int = -1
            
        prev_t_batch = torch.full((b,), prev_t_int, device=device, dtype=torch.long)
        
        # 1. Model Prediction (CFG)
        eps_cond = model.model(x, t_batch, cond_img, fingerprint, drop_cond=False)
        w = model.cfg.guidance_scale
        
        if abs(w - 1.0) > 1e-6:
            eps_uncond = model.model(x, t_batch, torch.zeros_like(cond_img), torch.zeros_like(fingerprint), drop_cond=True)
            eps_pred = eps_uncond + w * (eps_cond - eps_uncond)
        else:
            eps_pred = eps_cond
        
        # 2. Get Posterior Distribution
        mu, var, x0_pred = get_posterior_mean_variance(scheduler, x, eps_pred, t_batch, prev_t_batch, clip_sample=clip_sample)
        
        # Micro-fix 2: Numerical safety
        sigma = (var + 1e-20).sqrt()
        
        # 3. Sample x_prev
        if prev_t_int >= 0:
            noise = torch.randn_like(x)
            x_prev = mu + sigma * noise
            
            # 4. Compute Log Prob (only for non-terminal steps)
            # Use float32 for stability
            dist = torch.distributions.Normal(mu.float(), sigma.float())
            log_prob = dist.log_prob(x_prev.float()).sum(dim=(1, 2, 3))
            
            # Store transition for PPO
            traj.append({
                'x_t': x.detach().cpu(),
                'x_prev': x_prev.detach().cpu(),
                't': t_batch.detach().cpu(),
                'prev_t': prev_t_batch.detach().cpu(),
                'old_logprob': log_prob.detach().cpu(),
            })
            
        else:
            # Final step: standard DDPM sets x_{t-1} to the predicted x_0 (clipped) directly
            # This avoids singularity at sigma=0
            x_prev = x0_pred
            
        x = x_prev
        
    return x, traj # x is final generated image x0

@torch.no_grad()
def reward_negloglik_ddpm(other_model: DiffusionModel,
                          target_img,      # y (the thing whose likelihood we score)
                          cond_img,        # x (conditioning)
                          fingerprint,
                          n_terms=32,       # number of timesteps sampled per reward estimate
                          mc=3):            # paper uses MC steps (e.g., 3)
    """
    Approximate -log p_other(target | cond) up to a constant using Eq.(8):
      const + 1/2 * sum_t E||eps - eps_phi(y_t, x, t)||^2
    We Monte-Carlo it by sampling timesteps and noises.
    Return REWARD = -negloglik (higher is better), like the paper’s RL objective.
    """
    other_model.model.eval()
    scheduler = other_model.noise_scheduler
    device = other_model.cfg.device
    b = target_img.shape[0]


    
    # Pre-fetch scale
    w_cfg = other_model.cfg.guidance_scale
    
    acc = torch.zeros((b,), device=device)

    for _ in range(mc):
        # sample a set of timesteps per batch element
        t_batch_mc = torch.randint(0, scheduler.config.num_train_timesteps, (n_terms, b), device=device).long()
        
        # Precompute weights for this batch of timesteps
        alpha_prod_t_mc = scheduler.alphas_cumprod.to(device)[t_batch_mc]
        weights_mc = 1.0 / (1 - alpha_prod_t_mc + 1e-5)
        
        for k in range(n_terms):
            tk = t_batch_mc[k]
            noise = torch.randn_like(target_img)
            y_t = scheduler.add_noise(target_img, noise, tk)
            
            # Use CFG for reward likelihood proxy
            eps_cond = other_model.model(y_t, tk, cond_img, fingerprint, drop_cond=False)
            
            if abs(w_cfg - 1.0) > 1e-6:
                eps_uncond = other_model.model(y_t, tk, torch.zeros_like(cond_img), torch.zeros_like(fingerprint), drop_cond=True)
                eps_pred = eps_uncond + w_cfg * (eps_cond - eps_uncond)
            else:
                eps_pred = eps_cond
            
            # Weighted MSE for this term
            mse_term = ((noise - eps_pred) ** 2).mean(dim=(1,2,3))
            
            # Apply weight
            weight_k = weights_mc[k]
            acc += mse_term * weight_k

    acc = acc / (mc * n_terms)

    # Eq(8) has 1/2 factor
    negloglik = 0.5 * acc
    
    # Safety: Clamp reward to avoid extreme values
    # negloglik is L2 error, so always positive. We return -negloglik.
    # If error is huge, reward is very negative.
    negloglik = torch.nan_to_num(negloglik, nan=1000.0, posinf=1000.0, neginf=0.0)
    negloglik = torch.clamp(negloglik, 0, 1000.0)

    # reward = - negloglik
    # Scale reward to be roughly N(0,1) friendly? Or just clamp.
    # Taking a soft limit might be better but hard clamp is robust.
    return -negloglik

def ppo_update_from_batch(model, ref_model, optimizer, batch, config):
    """
    Single PPO update step on a minibatch of transitions.
    batch: dict of tensors (flat across time*batch)
    """
    model.model.train()
    device = model.cfg.device

    x_t = batch['x_t'].to(device)
    x_prev = batch['x_prev'].to(device)
    t = batch['t'].to(device)
    prev_t = batch['prev_t'].to(device)
    old_logprob = batch['old_logprob'].to(device)
    adv = batch['adv'].to(device)
    cond_img = batch['cond_img'].to(device)
    fp = batch['fingerprint'].to(device)
    
    # Get config for sample clipping
    clip_sample = getattr(model.noise_scheduler.config, 'clip_sample', True)

    # 1. New Logprob under the SAME guided policy as rollout (CFG)
    eps_cond = model.model(x_t, t, cond_img, fp, drop_cond=False)
    eps_uncond = model.model(x_t, t, torch.zeros_like(cond_img), torch.zeros_like(fp), drop_cond=True)
    
    # Micro-fix 1: Explicit config access
    w = model.cfg.guidance_scale
    eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
    
    mu, var, _ = get_posterior_mean_variance(model.noise_scheduler, x_t, eps_guided, t, prev_t, clip_sample=clip_sample)
    
    # Micro-fix 2: Numerical safety
    sigma = (var + 1e-20).sqrt()
    # Use float32 for stability to match rollout
    dist = torch.distributions.Normal(mu.float(), sigma.float())
    new_logprob = dist.log_prob(x_prev.float()).sum(dim=(1, 2, 3))
    
    # 2. Ratio
    # log(a/b) = log(a) - log(b) -> a/b = exp(log(a) - log(b))
    log_ratio = new_logprob - old_logprob
    # Clamp log_ratio to prevent explosion/underflow
    log_ratio = torch.clamp(log_ratio, -20, 20)
    ratio = torch.exp(log_ratio)
    
    # Extra safety for NaNs
    ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.1)
    
    # 3. PPO Loss
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip_range, 1.0 + config.ppo_clip_range) * adv
    ppo_loss = -torch.min(surr1, surr2).mean()
    
    # 4. Marginal Constraint (Eq. 11: ||eps_theta(x,y,t) - eps_theta*(x,t)||^2)
    # We use the reference model in UNCONDITIONAL mode to approximate eps_theta*
    # Explicitly pass zero conditioning for safety
    with torch.no_grad():
        zero_cond = torch.zeros_like(cond_img)
        zero_fp = torch.zeros_like(fp)
        eps_ref = ref_model.model(x_t, t, zero_cond, zero_fp, drop_cond=True)
        
    # Soft marginal constraint: cond vs unconditional-ref (Strict Eq. 11 interpretation)
    # Penalize conditional model deviation from marginal prior, not just guided policy
    kl_loss = F.mse_loss(eps_cond, eps_ref)
    
    loss = ppo_loss + config.kl_beta * kl_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), ppo_loss.item(), kl_loss.item()

def flatten_trajectory(traj, reward, cond_img, fingerprint):
    """
    Flatten trajectory list into a single batch of tensors.
    Broadcasts reward, cond_img, and fingerprint to all timesteps.
    """
    if not traj:
        return None

    keys = traj[0].keys()
    flat_data = {k: [] for k in keys}
    adv_list = []
    
    # Calculate Advantage (Reward - Baseline? Or just Reward here since baseline is usually Value function)
    # DDMEC often uses just the final reward broadcasted.
    
    T = len(traj)
    
    # Create normalized advantage across the batch (simple baseline subtraction)
    # Note: Normalizing per-batch inside the loop is good practice
    adv_mean = reward.mean()
    adv_std = reward.std() + 1e-8
    norm_reward = (reward - adv_mean) / adv_std
    
    for step_data in traj:
        for k in keys:
            flat_data[k].append(step_data[k])
        
        # Advantage is the same for all time steps of a sample (episodic)
        adv_list.append(norm_reward.detach()) 
        
    # Stack
    batch = {}
    for k in keys:
        batch[k] = torch.cat(flat_data[k], dim=0)
        
    batch['adv'] = torch.cat(adv_list, dim=0)
    
    # Efficiently broadcast conditioning
    # Must use .repeat() (time-major: [Step0_Batch, Step1_Batch...]) to match torch.cat below
    # NOT repeat_interleave (which would be sample-major: [Sample0_Time, Sample1_Time...])
    batch['cond_img'] = cond_img.detach().cpu().repeat(T, 1, 1, 1)
    batch['fingerprint'] = fingerprint.detach().cpu().repeat(T, 1)
    
    return batch

# ============================================================================
# EVALUATION
# ============================================================================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance with stability features."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    
    # Product might be almost singular
    from scipy.linalg import sqrtm
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

@torch.no_grad()
def evaluate_metrics(theta_model, phi_model, dataloader, config):
    print("Running Evaluation (FID/KID)...")
    theta_model.model.eval()
    phi_model.model.eval()
    
    fid_metric_control = FrechetInceptionDistance(normalize=True).to(config.device) # FIDc (Real Ctrl vs Fake Ctrl from Trt)
    kid_metric_control = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device)
    
    fid_metric_treated = FrechetInceptionDistance(normalize=True).to(config.device) # FIDo (Real Trt vs Fake Trt from Ctrl)
    kid_metric_treated = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device)
    
    # Storage for CFID
    real_ctrl_feats = []
    fake_ctrl_feats = []
    real_trt_feats = []
    fake_trt_feats = []
    fingerprints = []
    
    # Storage for conditional FID (per compound)
    generated_treated_per_compound = {}  # compound -> list of generated treated images
    target_treated_per_compound = {}     # compound -> list of real treated images
    generated_control_per_compound = {}  # compound -> list of generated control images
    target_control_per_compound = {}     # compound -> list of real control images

    
    samples_count = 0
    
    # Need to iterate dataloader again without disrupting the main loop iterator
    # We'll just create a new iterator for a few steps
    # Note: 'dataloader' argument here is actually the 'loader' object which is an iterable PairedDataLoader
    # We can't easily re-iterate it if it's a generator. 
    # But PairedDataLoader in this script seems to be based on a Dataset, so we can make a new one or just assume we can iterate.
    # The main loop calls iter(loader), so we can just make a fresh iterator.
    
    eval_iter = iter(dataloader)
    
    for _ in tqdm(range(config.eval_max_samples // config.batch_size + 1), desc="Eval Batches"):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break
            
        ctrl = batch['control'].to(config.device) # Real Control
        trt = batch['perturbed'].to(config.device) # Real Treated
        fp = batch['fingerprint'].to(config.device)
        compounds = batch.get('compound', [])  # Get compound names
        
        # 1. Evaluate Theta (Ctrl -> Fake Trt) => Compare with Real Trt
        # Use specific eval steps
        fake_trt, _ = rollout_with_logprobs(theta_model, ctrl, fp, steps=config.eval_steps)
        
        # Normalize to [0, 1] matching train2.py logic (quantized to 255 levels)
        
        # Real
        real_trt_norm = torch.clamp((trt + 1) / 2, 0, 1)
        real_trt_norm = torch.floor(real_trt_norm * 255).to(torch.float32) / 255.0
        
        # Fake
        fake_trt_norm = torch.clamp((fake_trt + 1) / 2, 0, 1)
        fake_trt_norm = torch.floor(fake_trt_norm * 255).to(torch.float32) / 255.0
        
        fid_metric_treated.update(real_trt_norm, real=True)
        fid_metric_treated.update(fake_trt_norm, real=False)
        kid_metric_treated.update(real_trt_norm, real=True)
        kid_metric_treated.update(fake_trt_norm, real=False)
        
        # Store per compound for conditional FID (Treated)
        if compounds:
            for i, compound in enumerate(compounds):
                if compound not in generated_treated_per_compound:
                    generated_treated_per_compound[compound] = []
                    target_treated_per_compound[compound] = []
                generated_treated_per_compound[compound].append(fake_trt_norm[i:i+1].cpu())
                target_treated_per_compound[compound].append(real_trt_norm[i:i+1].cpu())
        else:
            # Debug: warn if compounds not found
            if samples_count == 0:
                print("  Warning: No compounds found in batch. Conditional FID may not be computed.")
        
        # 2. Evaluate Phi (Trt -> Fake Ctrl) => Compare with Real Ctrl
        fake_ctrl, _ = rollout_with_logprobs(phi_model, trt, fp, steps=config.eval_steps)
        
        # Real
        real_ctrl_norm = torch.clamp((ctrl + 1) / 2, 0, 1)
        real_ctrl_norm = torch.floor(real_ctrl_norm * 255).to(torch.float32) / 255.0
        
        # Fake
        fake_ctrl_norm = torch.clamp((fake_ctrl + 1) / 2, 0, 1)
        fake_ctrl_norm = torch.floor(fake_ctrl_norm * 255).to(torch.float32) / 255.0
        
        fid_metric_control.update(real_ctrl_norm, real=True)
        fid_metric_control.update(fake_ctrl_norm, real=False)
        kid_metric_control.update(real_ctrl_norm, real=True)
        kid_metric_control.update(fake_ctrl_norm, real=False)
        
        # Store per compound for conditional FID (Control)
        if compounds:
            for i, compound in enumerate(compounds):
                if compound not in generated_control_per_compound:
                    generated_control_per_compound[compound] = []
                    target_control_per_compound[compound] = []
                generated_control_per_compound[compound].append(fake_ctrl_norm[i:i+1].cpu())
                target_control_per_compound[compound].append(real_ctrl_norm[i:i+1].cpu())
        
        # --- CFID Feature Extraction ---
        # Reuse Inception from FID metric if available
        if hasattr(fid_metric_control, 'inception'):
            try:
                with torch.no_grad():
                    # Treated (Theta): Real Trt vs Fake Trt
                    r_trt_f = fid_metric_treated.inception(real_trt_norm.to(dtype=torch.uint8))
                    f_trt_f = fid_metric_treated.inception(fake_trt_norm.to(dtype=torch.uint8))
                    
                    # Control (Phi): Real Ctrl vs Fake Ctrl
                    r_ctrl_f = fid_metric_control.inception(real_ctrl_norm.to(dtype=torch.uint8))
                    f_ctrl_f = fid_metric_control.inception(fake_ctrl_norm.to(dtype=torch.uint8))
                    
                    if r_trt_f.dim() > 2: r_trt_f = r_trt_f.view(r_trt_f.size(0), -1)
                    if f_trt_f.dim() > 2: f_trt_f = f_trt_f.view(f_trt_f.size(0), -1)
                    if r_ctrl_f.dim() > 2: r_ctrl_f = r_ctrl_f.view(r_ctrl_f.size(0), -1)
                    if f_ctrl_f.dim() > 2: f_ctrl_f = f_ctrl_f.view(f_ctrl_f.size(0), -1)
                    
                    real_trt_feats.append(r_trt_f.cpu().numpy())
                    fake_trt_feats.append(f_trt_f.cpu().numpy())
                    real_ctrl_feats.append(r_ctrl_f.cpu().numpy())
                    fake_ctrl_feats.append(f_ctrl_f.cpu().numpy())
                    
                    fingerprints.append(fp.cpu().numpy())
            except Exception as e:
                # print(f"Warning: CFID feat extraction error: {e}")
                pass

        
        samples_count += ctrl.shape[0]
        if samples_count >= config.eval_max_samples:
            break
            
    # Compute
    try:
        fid_c = fid_metric_control.compute().item()
        kid_c_mean, kid_c_std = kid_metric_control.compute()
        kid_c = kid_c_mean.item()
    except Exception as e:
        print(f"Error computing Control metrics: {e}")
        fid_c, kid_c = -1, -1

    try:
        fid_t = fid_metric_treated.compute().item()
        kid_t_mean, kid_t_std = kid_metric_treated.compute()
        kid_t = kid_t_mean.item()
    except Exception as e:
        print(f"Error computing Treated metrics: {e}")
        fid_t, kid_t = -1, -1

    # Compute CFID
    cfid_c = -1.0
    cfid_t = -1.0
    if len(fingerprints) > 0:
        try:
            # Common Fingerprints
            fps_all = np.concatenate(fingerprints, axis=0) # [N, 1024]
            
            # --- CFID Control (Phi) ---
            r_c_all = np.concatenate(real_ctrl_feats, axis=0)
            f_c_all = np.concatenate(fake_ctrl_feats, axis=0)
            
            real_full_c = np.concatenate([r_c_all, fps_all], axis=1) # [N, 3072]
            fake_full_c = np.concatenate([f_c_all, fps_all], axis=1)
            
            mu1, sigma1 = np.mean(real_full_c, axis=0), np.cov(real_full_c, rowvar=False)
            mu2, sigma2 = np.mean(fake_full_c, axis=0), np.cov(fake_full_c, rowvar=False)
            cfid_c = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            
            # --- CFID Treated (Theta) ---
            r_t_all = np.concatenate(real_trt_feats, axis=0)
            f_t_all = np.concatenate(fake_trt_feats, axis=0)
            
            real_full_t = np.concatenate([r_t_all, fps_all], axis=1)
            fake_full_t = np.concatenate([f_t_all, fps_all], axis=1)
            
            mu1_t, sigma1_t = np.mean(real_full_t, axis=0), np.cov(real_full_t, rowvar=False)
            mu2_t, sigma2_t = np.mean(fake_full_t, axis=0), np.cov(fake_full_t, rowvar=False)
            cfid_t = calculate_frechet_distance(mu1_t, sigma1_t, mu2_t, sigma2_t)
            
        except Exception as e:
            print(f"Error computing CFID: {e}")

    # Compute Conditional FID per compound
    fid_per_compound_treated = {}
    fid_per_compound_control = {}
    
    print("\nComputing Conditional FID per compound...")
    for compound in generated_treated_per_compound.keys():
        if len(generated_treated_per_compound[compound]) < 2:
            continue  # Need at least 2 samples for FID
        
        try:
            # Treated (Theta) - per compound
            gen_t = torch.cat(generated_treated_per_compound[compound], dim=0).to(config.device)
            tgt_t = torch.cat(target_treated_per_compound[compound], dim=0).to(config.device)
            
            fid_metric_compound_t = FrechetInceptionDistance(normalize=True).to(config.device)
            fid_metric_compound_t.update(tgt_t, real=True)
            fid_metric_compound_t.update(gen_t, real=False)
            fid_per_compound_treated[compound] = fid_metric_compound_t.compute().item()
            print(f"  {compound} (Treated, {len(generated_treated_per_compound[compound])} samples): FID={fid_per_compound_treated[compound]:.4f}")
        except Exception as e:
            print(f"  Warning: Failed to compute FID for {compound} (Treated): {e}")
            fid_per_compound_treated[compound] = -1.0
    
    for compound in generated_control_per_compound.keys():
        if len(generated_control_per_compound[compound]) < 2:
            continue  # Need at least 2 samples for FID
        
        try:
            # Control (Phi) - per compound
            gen_c = torch.cat(generated_control_per_compound[compound], dim=0).to(config.device)
            tgt_c = torch.cat(target_control_per_compound[compound], dim=0).to(config.device)
            
            fid_metric_compound_c = FrechetInceptionDistance(normalize=True).to(config.device)
            fid_metric_compound_c.update(tgt_c, real=True)
            fid_metric_compound_c.update(gen_c, real=False)
            fid_per_compound_control[compound] = fid_metric_compound_c.compute().item()
            print(f"  {compound} (Control, {len(generated_control_per_compound[compound])} samples): FID={fid_per_compound_control[compound]:.4f}")
        except Exception as e:
            print(f"  Warning: Failed to compute FID for {compound} (Control): {e}")
            fid_per_compound_control[compound] = -1.0
    
    # Calculate average FID per compound
    avg_fid_treated = np.mean([v for v in fid_per_compound_treated.values() if v > 0]) if fid_per_compound_treated else -1.0
    avg_fid_control = np.mean([v for v in fid_per_compound_control.values() if v > 0]) if fid_per_compound_control else -1.0
    
    print(f"\nAverage Conditional FID - Treated: {avg_fid_treated:.4f}, Control: {avg_fid_control:.4f}")
        
    return {
        "FID_Control": fid_c,
        "KID_Control": kid_c,
        "CFID_Control": cfid_c,
        "FID_Treated": fid_t,
        "KID_Treated": kid_t,
        "CFID_Treated": cfid_t,
        "FID_per_compound_Treated": fid_per_compound_treated,
        "FID_per_compound_Control": fid_per_compound_control,
        "Average_FID_per_compound_Treated": avg_fid_treated,
        "Average_FID_per_compound_Control": avg_fid_control
    }

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    import os
    parser = argparse.ArgumentParser(description='DDMEC-PPO Training')
    parser.add_argument('--iters', type=int, default=100, help='Total number of training iterations')
    parser.add_argument('--eval_samples', type=int, default=5000, help='Number of samples for evaluation')
    parser.add_argument('--eval_steps', type=int, default=50, help='Number of inference steps for evaluation')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint in output_dir')
    parser.add_argument('--theta_checkpoint', type=str, default='./ddpm_diffusers_results/checkpoints/checkpoint_epoch_60.pt', 
                        help='Path to theta checkpoint (default: pretrained from config)')
    parser.add_argument('--phi_checkpoint', type=str, default='./results_phi_phi/checkpoints/checkpoint_epoch_100.pt',
                        help='Path to phi checkpoint (default: pretrained from config)')
    parser.add_argument('--output_dir', type=str, default='ddpm_ddmec_results', help='Output directory for checkpoints and logs')
    parser.add_argument('--paths_csv', type=str, default=None, help='Path to paths.csv file for robust file lookup (auto-detected if not specified)')
    parser.add_argument('--skip_initial_eval', action='store_true', help='Skip the initial evaluation before training starts')
    args = parser.parse_args()
    
    config = Config()
    
    # Override config with args
    config.eval_max_samples = args.eval_samples
    config.eval_steps = args.eval_steps
    config.output_dir = args.output_dir
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("Initializing Models...")
    
    # Determine starting iteration and checkpoint paths
    start_iter = 0
    if args.resume:
        # Find latest checkpoints in output directory
        import glob
        theta_checkpoints = glob.glob(f"{config.output_dir}/theta_*.pt")
        phi_checkpoints = glob.glob(f"{config.output_dir}/phi_*.pt")
        
        if theta_checkpoints and phi_checkpoints:
            # Extract iteration numbers and find the latest
            theta_iters = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in theta_checkpoints]
            phi_iters = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in phi_checkpoints]
            
            # Use the minimum of the two to ensure both models are at the same iteration
            start_iter = min(max(theta_iters), max(phi_iters))
            
            theta_checkpoint_path = f"{config.output_dir}/theta_{start_iter}.pt"
            phi_checkpoint_path = f"{config.output_dir}/phi_{start_iter}.pt"
            
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING FROM ITERATION {start_iter}")
            print(f"{'='*60}")
            print(f"  Theta checkpoint: {theta_checkpoint_path}")
            print(f"  Phi checkpoint: {phi_checkpoint_path}")
            print(f"  Will train for {args.iters - start_iter} more iterations")
            print(f"{'='*60}\n")
        else:
            print("Warning: --resume specified but no checkpoints found. Starting from pretrained models.")
            theta_checkpoint_path = config.theta_checkpoint
            phi_checkpoint_path = config.phi_checkpoint
    else:
        # Use command-line specified checkpoints or defaults from config
        theta_checkpoint_path = args.theta_checkpoint if args.theta_checkpoint else config.theta_checkpoint
        phi_checkpoint_path = args.phi_checkpoint if args.phi_checkpoint else config.phi_checkpoint
    
    # 1. Load Theta (Forward)
    theta_model = DiffusionModel(config)
    theta_model.load_checkpoint(theta_checkpoint_path)
    theta_ref = copy.deepcopy(theta_model)
    theta_ref.model.requires_grad_(False)
    theta_ref.model.eval()
    theta_opt = torch.optim.AdamW(theta_model.parameters(), lr=config.lr)
    
    print(f"Scheduler Config: Variance Type={theta_model.noise_scheduler.config.variance_type}, Clip Sample={theta_model.noise_scheduler.config.clip_sample}")
    
    # 2. Load Phi (Reverse)
    phi_model = DiffusionModel(config)
    phi_model.load_checkpoint(phi_checkpoint_path)
    phi_ref = copy.deepcopy(phi_model)
    phi_ref.model.requires_grad_(False)
    phi_ref.model.eval()
    phi_opt = torch.optim.AdamW(phi_model.parameters(), lr=config.lr)
    
    # 3. Data
    encoder = MorganFingerprintEncoder()
    ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder, paths_csv=args.paths_csv)
    loader = PairedDataLoader(ds, config.batch_size, shuffle=True)
    
    # Test Data for Evaluation (FIX: Use test split as requested)
    test_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='test', encoder=encoder, paths_csv=args.paths_csv)
    test_loader = PairedDataLoader(test_ds, config.batch_size, shuffle=False)
    
    # Print dataset details
    print(f"\n{'='*60}")
    print(f"Dataset Details:")
    print(f"{'='*60}")
    
    try:
        train_count = len(ds)
        test_count = len(test_ds)
        print(f"Train split: {train_count} samples")
        print(f"Test split: {test_count} samples")
        print(f"Total samples: {train_count + test_count}")
        
        # Count compounds
        if hasattr(ds, 'metadata') and ds.metadata:
            train_compounds = len(set([m.get('CPD_NAME', '') for m in ds.metadata]))
            test_compounds = len(set([m.get('CPD_NAME', '') for m in test_ds.metadata]))
            print(f"Train compounds: {train_compounds}")
            print(f"Test compounds: {test_compounds}")
            
            # Count batches
            train_batches = len(set([m.get('BATCH', '') for m in ds.metadata]))
            test_batches = len(set([m.get('BATCH', '') for m in test_ds.metadata]))
            print(f"Train batches: {train_batches}")
            print(f"Test batches: {test_batches}")
            
            # Count DMSO vs perturbed
            train_dmso = sum([1 for m in ds.metadata if str(m.get('CPD_NAME', '')).upper() == 'DMSO'])
            train_perturbed = len(ds.metadata) - train_dmso
            test_dmso = sum([1 for m in test_ds.metadata if str(m.get('CPD_NAME', '')).upper() == 'DMSO'])
            test_perturbed = len(test_ds.metadata) - test_dmso
            print(f"Train - DMSO: {train_dmso}, Perturbed: {train_perturbed}")
            print(f"Test - DMSO: {test_dmso}, Perturbed: {test_perturbed}")
        else:
            print("Warning: Could not access dataset metadata for detailed statistics")
        
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Error printing dataset details: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")

    # Save a random dataset sample image to verify loading is correct
    print("\nSaving random dataset sample images to verify loading...")
    try:
        import random
        from PIL import Image
        
        # Get a random sample from train dataset
        if len(ds) > 0:
            random_idx = random.randint(0, len(ds) - 1)
            sample = ds[random_idx]
            
            # Convert tensor to numpy image
            img_tensor = sample['image']  # Shape: [3, H, W], range [-1, 1]
            img_np = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            img_np = np.clip(img_np, 0, 255)
            
            # Save as JPG in current working directory
            img_pil = Image.fromarray(img_np)
            sample_filename = f"ppo_dataset_sample_{sample['compound'].replace('/', '_').replace(' ', '_')}.jpg"
            sample_path = os.path.abspath(sample_filename)
            img_pil.save(sample_path, "JPEG", quality=95)
            print(f"  ✓ Saved random train sample to: {sample_path}")
            print(f"    Compound: {sample['compound']}")
            print(f"    Image shape: {img_tensor.shape}")
            print(f"    Image range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
        
        # Also save a random test sample
        if len(test_ds) > 0:
            random_idx = random.randint(0, len(test_ds) - 1)
            sample = test_ds[random_idx]
            
            img_tensor = sample['image']
            img_np = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            img_np = np.clip(img_np, 0, 255)
            
            img_pil = Image.fromarray(img_np)
            sample_filename = f"ppo_test_sample_{sample['compound'].replace('/', '_').replace(' ', '_')}.jpg"
            sample_path = os.path.abspath(sample_filename)
            img_pil.save(sample_path, "JPEG", quality=95)
            print(f"  ✓ Saved random test sample to: {sample_path}")
            print(f"    Compound: {sample['compound']}")
            print(f"    Image shape: {img_tensor.shape}")
            print(f"    Image range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
            
    except Exception as e:
        print(f"  Warning: Could not save sample images: {e}")
        import traceback
        traceback.print_exc()
    
    print()  # Empty line for readability
    
    # Initialize CSV Log
    if not os.path.exists(config.log_file):
        with open(config.log_file, 'w') as f:
            f.write("iteration,theta_reward,phi_reward,theta_ppo_loss,phi_ppo_loss,theta_sup_loss,phi_sup_loss,fid_control,fid_treated,kid_control,kid_treated,cfid_control,cfid_treated\n")
            
    if start_iter > 0:
        print(f"Continuing DDMEC-PPO Loop from iteration {start_iter} to {args.iters}...")
    else:
        print(f"Starting DDMEC-PPO Loop for {args.iters} iterations...")
    iterator = iter(loader)
    
    # --- Initial Evaluation ---
    if not args.skip_initial_eval:
        print(f"\n{'='*60}")
        print(f"Running Initial Evaluation (Before Training)")
        print(f"{'='*60}")
        # Run a quick check with fewer samples if needed, or full eval as configured
        # Using config settings standardizes it
        try:
            metrics = evaluate_metrics(theta_model, phi_model, test_loader, config)
            print(f"Initial Evaluation Results:")
            print(f"  FID_Control (Phi quality): {metrics['FID_Control']:.2f}")
            print(f"  FID_Treated (Theta quality): {metrics['FID_Treated']:.2f}")
            print(f"  KID_Control: {metrics['KID_Control']:.4f}")
            print(f"  KID_Treated: {metrics['KID_Treated']:.4f}")
            print(f"  CFID_Control: {metrics['CFID_Control']:.4f}")
            print(f"  CFID_Treated: {metrics['CFID_Treated']:.4f}")
            
            # Print conditional FID per compound
            if 'FID_per_compound_Treated' in metrics and metrics['FID_per_compound_Treated']:
                print(f"\n  Conditional FID per Compound (Treated):")
                for compound, fid_val in sorted(metrics['FID_per_compound_Treated'].items()):
                    if fid_val > 0:
                        print(f"    {compound}: {fid_val:.4f}")
                if 'Average_FID_per_compound_Treated' in metrics:
                    print(f"  Average Conditional FID (Treated): {metrics['Average_FID_per_compound_Treated']:.4f}")
            
            if 'FID_per_compound_Control' in metrics and metrics['FID_per_compound_Control']:
                print(f"\n  Conditional FID per Compound (Control):")
                for compound, fid_val in sorted(metrics['FID_per_compound_Control'].items()):
                    if fid_val > 0:
                        print(f"    {compound}: {fid_val:.4f}")
                if 'Average_FID_per_compound_Control' in metrics:
                    print(f"  Average Conditional FID (Control): {metrics['Average_FID_per_compound_Control']:.4f}")
            
            # Save results to JSON
            import json
            import os
            os.makedirs("outputs/evaluation", exist_ok=True)
            output_name = f"ppo_initial_eval_{config.eval_max_samples}_{config.eval_steps}"
            json_path = f"outputs/evaluation/{output_name}.json"
            results = {
                "eval_split": "test",
                "num_samples": config.eval_max_samples,
                "inference_steps": config.eval_steps,
                "overall_fid_control": float(metrics['FID_Control']),
                "overall_fid_treated": float(metrics['FID_Treated']),
                "overall_kid_control": float(metrics['KID_Control']),
                "overall_kid_treated": float(metrics['KID_Treated']),
                "cfid_control": float(metrics['CFID_Control']),
                "cfid_treated": float(metrics['CFID_Treated']),
            }
            if 'FID_per_compound_Treated' in metrics:
                results["fid_per_compound_treated"] = {k: float(v) for k, v in metrics['FID_per_compound_Treated'].items()}
                results["fid_per_compound_control"] = {k: float(v) for k, v in metrics['FID_per_compound_Control'].items()}
                results["average_fid_per_compound_treated"] = float(metrics['Average_FID_per_compound_Treated'])
                results["average_fid_per_compound_control"] = float(metrics['Average_FID_per_compound_Control'])
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"  Results saved to: {json_path}")
            
            # Log initial values (iteration 0)
            with open(config.log_file, 'a') as f:
                f.write(f"0,0,0,0,0,0,0,"
                        f"{metrics['FID_Control']:.4f},{metrics['FID_Treated']:.4f},{metrics['KID_Control']:.6f},{metrics['KID_Treated']:.6f},{metrics['CFID_Control']:.4f},{metrics['CFID_Treated']:.4f}\n")
                        
        except Exception as e:
            print(f"Warning: Initial evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        print(f"{'='*60}\n")
    else:
        print("\nSkipping Initial Evaluation (--skip_initial_eval set)\n")
    
    theta_losses = []
    phi_losses = []

    for it in range(start_iter, args.iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            
        ctrl = batch['control'].to(config.device)
        trt = batch['perturbed'].to(config.device)
        fp = batch['fingerprint'].to(config.device)
        
        # ====================================================================
        # Phase A: Update Theta (PPO) using Phi Reward
        # ====================================================================
        
        # 1. Rollout
        x0_gen, traj_theta = rollout_with_logprobs(theta_model, ctrl, fp, steps=config.rollout_steps)
        r_theta = reward_negloglik_ddpm(phi_model, target_img=ctrl, cond_img=x0_gen, fingerprint=fp, 
                                        n_terms=config.reward_n_terms, mc=config.reward_mc)
        
        print(f"Iter {it} | Theta Reward: {r_theta.mean().item():.3f}")
        
        # 2. Flatten & PPO Update
        flat_batch = flatten_trajectory(traj_theta, r_theta, ctrl, fp)
        
        if flat_batch is not None:
            # Minibatch shuffling
            theta_batch_loss = 0
            dataset_size = flat_batch['x_t'].shape[0]
            
            for _ in range(config.ppo_epochs):
                 # Shuffle EVERY epoch
                indices = torch.randperm(dataset_size)
                
                for i in range(0, dataset_size, config.ppo_minibatch_size):
                    idx = indices[i:i+config.ppo_minibatch_size]
                    minibatch = {k: v[idx] for k, v in flat_batch.items()}
                    l, _, _ = ppo_update_from_batch(theta_model, theta_ref, theta_opt, minibatch, config)
                    theta_batch_loss += l
            theta_losses.append(theta_batch_loss / (config.ppo_epochs * ((dataset_size + config.ppo_minibatch_size - 1) // config.ppo_minibatch_size)))
        else:
            theta_losses.append(0.0)

        # ====================================================================
        # Phase B: Update Phi (Supervised) on Generated Data
        # ====================================================================
        
        # Input: x0_gen (fake treated), Target: ctrl (real control)
        phi_model.model.train()
        phi_opt.zero_grad()
        drop = (torch.rand(()) < config.cond_drop_prob)
        loss_su_phi = phi_model(ctrl, x0_gen.detach(), fp, drop_cond=drop)
        loss_su_phi.backward()
        torch.nn.utils.clip_grad_norm_(phi_model.parameters(), 1.0)
        phi_opt.step()
        
        # ====================================================================
        # Phase C: Update Phi (PPO) using Theta Reward
        # ====================================================================
        
        # 1. Rollout
        y0_gen, traj_phi = rollout_with_logprobs(phi_model, trt, fp, steps=config.rollout_steps)
        r_phi = reward_negloglik_ddpm(theta_model, target_img=trt, cond_img=y0_gen, fingerprint=fp,
                                      n_terms=config.reward_n_terms, mc=config.reward_mc)
        
        print(f"Iter {it} | Phi Reward:   {r_phi.mean().item():.3f}")
        
        flat_batch_phi = flatten_trajectory(traj_phi, r_phi, trt, fp)
        
        if flat_batch_phi is not None:
            dataset_size_phi = flat_batch_phi['x_t'].shape[0]
            
            phi_batch_loss = 0
            for _ in range(config.ppo_epochs):
                 # Shuffle EVERY epoch
                 indices_phi = torch.randperm(dataset_size_phi)
                 
                 for i in range(0, dataset_size_phi, config.ppo_minibatch_size):
                    idx = indices_phi[i:i+config.ppo_minibatch_size]
                    minibatch = {k: v[idx] for k, v in flat_batch_phi.items()}
                    l, _, _ = ppo_update_from_batch(phi_model, phi_ref, phi_opt, minibatch, config)
                    phi_batch_loss += l
            phi_losses.append(phi_batch_loss / (config.ppo_epochs * ((dataset_size_phi + config.ppo_minibatch_size - 1) // config.ppo_minibatch_size)))
        else:
            phi_losses.append(0.0)

        # ====================================================================
        # Phase D: Update Theta (Supervised)
        # ====================================================================
        
        theta_model.model.train()
        theta_opt.zero_grad()
        drop = (torch.rand(()) < config.cond_drop_prob)
        loss_su_theta = theta_model(trt, y0_gen.detach(), fp, drop_cond=drop)
        loss_su_theta.backward()
        torch.nn.utils.clip_grad_norm_(theta_model.parameters(), 1.0)
        theta_opt.step()
        
        # Save & Vis & Evaluate every 15 iterations
        if (it + 1) % 15 == 0:
            torch.save({'model': theta_model.model.state_dict()}, f"{config.output_dir}/theta_{it+1}.pt")
            torch.save({'model': phi_model.model.state_dict()}, f"{config.output_dir}/phi_{it+1}.pt")
            
            with torch.no_grad():
                # Vis: Ctrl -> Theta -> Phi -> ? should match Ctrl
                #      Trt -> Phi -> Theta -> ? should match Trt
                
                # Check cycle
                x0 = x0_gen[:4]
                y_recon_from_x0, _ = rollout_with_logprobs(phi_model, x0, fp[:4], steps=50)
                
                y0 = y0_gen[:4]
                x_recon_from_y0, _ = rollout_with_logprobs(theta_model, y0, fp[:4], steps=50)

                # Row 1: Ctrl, Theta(Ctrl), Phi(Theta(Ctrl)), Trt 
                # Row 2: Trt, Phi(Trt), Theta(Phi(Trt)), Ctrl
                
                row1 = torch.cat([ctrl[:4], x0_gen[:4], y_recon_from_x0, trt[:4]], dim=0) # 16 images
                save_image(row1, f"{config.output_dir}/vis_cycle_{it+1}.png", nrow=4, normalize=True, value_range=(-1, 1))
                print(f"Saved visualization to {config.output_dir}")
                
            # Evaluation on TEST Set
            print(f"\n{'='*60}")
            print(f"Running Evaluation at Iteration {it+1}")
            print(f"{'='*60}")
            metrics = evaluate_metrics(theta_model, phi_model, test_loader, config)
            print(f"Iter {it+1} Evaluation:")
            print(f"  FID_Control (Phi quality): {metrics['FID_Control']:.2f}")
            print(f"  FID_Treated (Theta quality): {metrics['FID_Treated']:.2f}")
            print(f"  CFID_Control: {metrics['CFID_Control']:.4f}")
            print(f"  CFID_Treated: {metrics['CFID_Treated']:.4f}")
            print(f"  KID_Control: {metrics['KID_Control']:.4f}")
            print(f"  KID_Treated: {metrics['KID_Treated']:.4f}")
            
            # Print conditional FID per compound
            if 'FID_per_compound_Treated' in metrics and metrics['FID_per_compound_Treated']:
                print(f"\n  Conditional FID per Compound (Treated):")
                for compound, fid_val in sorted(metrics['FID_per_compound_Treated'].items()):
                    if fid_val > 0:
                        print(f"    {compound}: {fid_val:.4f}")
                if 'Average_FID_per_compound_Treated' in metrics:
                    print(f"  Average Conditional FID (Treated): {metrics['Average_FID_per_compound_Treated']:.4f}")
            
            if 'FID_per_compound_Control' in metrics and metrics['FID_per_compound_Control']:
                print(f"\n  Conditional FID per Compound (Control):")
                for compound, fid_val in sorted(metrics['FID_per_compound_Control'].items()):
                    if fid_val > 0:
                        print(f"    {compound}: {fid_val:.4f}")
                if 'Average_FID_per_compound_Control' in metrics:
                    print(f"  Average Conditional FID (Control): {metrics['Average_FID_per_compound_Control']:.4f}")
            print(f"{'='*60}\n")
            
            # Save results to JSON
            import json
            import os
            os.makedirs("outputs/evaluation", exist_ok=True)
            output_name = f"ppo_iter_{it+1}_eval_{config.eval_max_samples}_{config.eval_steps}"
            json_path = f"outputs/evaluation/{output_name}.json"
            results = {
                "iteration": it+1,
                "eval_split": "test",
                "num_samples": config.eval_max_samples,
                "inference_steps": config.eval_steps,
                "overall_fid_control": float(metrics['FID_Control']),
                "overall_fid_treated": float(metrics['FID_Treated']),
                "overall_kid_control": float(metrics['KID_Control']),
                "overall_kid_treated": float(metrics['KID_Treated']),
                "cfid_control": float(metrics['CFID_Control']),
                "cfid_treated": float(metrics['CFID_Treated']),
            }
            if 'FID_per_compound_Treated' in metrics:
                results["fid_per_compound_treated"] = {k: float(v) for k, v in metrics['FID_per_compound_Treated'].items()}
                results["fid_per_compound_control"] = {k: float(v) for k, v in metrics['FID_per_compound_Control'].items()}
                results["average_fid_per_compound_treated"] = float(metrics['Average_FID_per_compound_Treated'])
                results["average_fid_per_compound_control"] = float(metrics['Average_FID_per_compound_Control'])
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"  Results saved to: {json_path}")
            
            # Log to CSV
            # Get latest losses (defaults to 0 if list empty or index error, though loop ensures append)
            t_rew = r_theta.mean().item() if 'r_theta' in locals() else 0.0
            p_rew = r_phi.mean().item() if 'r_phi' in locals() else 0.0
            t_ppo = theta_losses[-1] if theta_losses else 0.0
            p_ppo = phi_losses[-1] if phi_losses else 0.0
            t_sup = loss_su_theta.item() if 'loss_su_theta' in locals() else 0.0
            p_sup = loss_su_phi.item() if 'loss_su_phi' in locals() else 0.0
            
            with open(config.log_file, 'a') as f:
                f.write(f"{it+1},{t_rew:.4f},{p_rew:.4f},{t_ppo:.4f},{p_ppo:.4f},{t_sup:.4f},{p_sup:.4f},"
                        f"{metrics['FID_Control']:.4f},{metrics['FID_Treated']:.4f},{metrics['KID_Control']:.6f},{metrics['KID_Treated']:.6f},{metrics['CFID_Control']:.4f},{metrics['CFID_Treated']:.4f}\n")

if __name__ == "__main__":
    main()
