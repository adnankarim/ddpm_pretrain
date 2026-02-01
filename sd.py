"""
================================================================================
BBBC021 STABLE DIFFUSION: PARTIAL FINE-TUNING (Conditional)
================================================================================
Strategy: Sparse Fine-Tuning
--------------------------------------------------------------------------------
This script transforms Stable Diffusion into a Drug-Conditioned Generator.
It freezes most of the pre-trained weights to preserve visual knowledge, 
but unfreezes specific layers to learn biological drug effects.

Architecture:
1. VAE: Frozen (Encodes pixels -> latents)
2. U-Net: Partially Unfrozen
   - conv_in: Trained (Adapts to 8-channel input)
   - attn2 (Cross-Attn): Trained (Learns "Drug -> Visual" mapping)
   - conv_out: Trained (Refines biological texture)
   - ResNets/Self-Attn: FROZEN (Keeps pre-trained knowledge)
   
Input: [Noisy Latents (4ch) + Control Latents (4ch)] + Drug Fingerprint
Output: Denoised Latents (Target Cell)
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
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

# --- Plotting Backend ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Dependencies ---
try:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: Install dependencies: pip install diffusers transformers accelerate")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

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
    image_size = 512  # SD Native Resolution
    
    # Model
    model_id = "runwayml/stable-diffusion-v1-5"
    fingerprint_dim = 1024
    
    # Training
    epochs = 200
    batch_size = 64  # Lower batch size due to 512x512 resolution
    lr = 1e-5       # Lower LR for fine-tuning pre-trained weights (was 1e-4, too high)
    save_freq = 5
    eval_freq = 1
    
    output_dir = "sd_partial_finetune_results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = "fp16"

# ============================================================================
# LOGGING UTILS
# ============================================================================

class TrainingLogger:
    """
    Logs training metrics to CSV and generates plots every epoch.
    Now tracks KL, MSE, PSNR, SSIM.
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        # Main training log
        self.history = {'epoch': [], 'train_loss': [], 'learning_rate': []}
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        
        # Detailed metrics log
        self.metrics_csv_path = os.path.join(save_dir, "evaluation_metrics.csv")
        # Initialize metrics file with headers if it doesn't exist
        if not os.path.exists(self.metrics_csv_path):
            pd.DataFrame(columns=['epoch', 'kl_div', 'mse', 'psnr', 'ssim']).to_csv(self.metrics_csv_path, index=False)
        
    def update(self, epoch, loss, lr=None):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(loss)
        self.history['learning_rate'].append(lr if lr is not None else 0)
        
        # Save Training History
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Plot with dual y-axis for loss and learning rate
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss on left y-axis
        color = '#1f77b4'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss', color=color)
        line1 = ax1.plot(self.history['epoch'], self.history['train_loss'], 
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
        
        plt.title(f'Stable Diffusion Partial Fine-Tuning (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def log_metrics(self, epoch, metrics_dict):
        """Appends evaluation metrics to a separate CSV"""
        new_row = {'epoch': epoch}
        new_row.update(metrics_dict)
        df = pd.DataFrame([new_row])
        df.to_csv(self.metrics_csv_path, mode='a', header=False, index=False)
        print(f"  📊 Metrics logged to {self.metrics_csv_path}")

    def log_metrics(self, epoch, metrics_dict):
        """Appends evaluation metrics to a separate CSV"""
        new_row = {'epoch': epoch}
        new_row.update(metrics_dict)
        df = pd.DataFrame([new_row])
        df.to_csv(self.metrics_csv_path, mode='a', header=False, index=False)
        print(f"  📊 Metrics logged to {self.metrics_csv_path}")

# ============================================================================
# ARCHITECTURE: CONDITIONAL STABLE DIFFUSION
# ============================================================================

class DrugConditionedStableDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"Loading Stable Diffusion U-Net from {config.model_id}...")
        
        # 1. Load Pre-trained U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            config.model_id, subfolder="unet"
        )
        
        # 2. INPUT SURGERY (4 channels -> 8 channels)
        # We need to accept (Latent Noise + Latent Control)
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            8, old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            padding=old_conv.padding
        )
        
        # Initialize weights: Copy old weights to first 4 channels, zero rest
        with torch.no_grad():
            new_conv.weight[:, :4, :, :] = old_conv.weight
            new_conv.weight[:, 4:, :, :] = 0.0  # Zero-init control
            new_conv.bias = old_conv.bias
            
        self.unet.conv_in = new_conv
        print("  ✓ Input layer modified for 8 channels.")

        # 3. DRUG PROJECTION (Fingerprint -> CLIP Embedding Dim)
        # SD v1.5 uses 768-dim embeddings
        self.drug_proj = nn.Sequential(
            nn.Linear(config.fingerprint_dim, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
        )
        print("  ✓ Drug Projection layer added.")

        # 4. PARTIAL FREEZING STRATEGY
        self._apply_partial_freezing()

    def _apply_partial_freezing(self):
        print("\nConfiguring Partial Fine-Tuning...")
        
        # A. Freeze EVERYTHING first
        for param in self.unet.parameters():
            param.requires_grad = False
            
        trainable_params = []
        
        # B. Unfreeze Input Layer (Must adapt to 8 channels)
        for param in self.unet.conv_in.parameters():
            param.requires_grad = True
            trainable_params.append(param)
            
        # C. Unfreeze Output Layer (Refine pixel texture)
        for param in self.unet.conv_out.parameters():
            param.requires_grad = True
            trainable_params.append(param)
            
        # D. Unfreeze Cross-Attention Layers ('attn2')
        # These are the layers that "listen" to the drug embedding
        unfrozen_blocks = 0
        for name, module in self.unet.named_modules():
            if name.endswith("attn2"):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                unfrozen_blocks += 1
        
        # E. Unfreeze Drug Projection (Must train from scratch)
        for param in self.drug_proj.parameters():
            param.requires_grad = True
            trainable_params.append(param)

        # Statistics
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        frozen = total - trainable
        
        print(f"  Total Params: {total:,}")
        print(f"  Trainable:    {trainable:,} ({(trainable/total)*100:.2f}%)")
        print(f"  Frozen:       {frozen:,} ({(frozen/total)*100:.2f}%)")
        print(f"  Unfrozen Attention Blocks: {unfrozen_blocks}")
        
        # Store for later access
        self.total_params = total
        self.trainable_params = trainable
        self.frozen_params = frozen
        
        # Store for later access
        self.total_params = total
        self.trainable_params = trainable
        self.frozen_params = frozen

    def forward(self, noisy_latents, timesteps, control_latents, fingerprint):
        # 1. Project Drug Fingerprint
        # Shape: [B, 1024] -> [B, 1, 768] (Mimics 1 text token)
        drug_emb = self.drug_proj(fingerprint).unsqueeze(1)
        
        # 2. Concat Inputs (Noise + Control)
        # Shape: [B, 8, 64, 64]
        unet_input = torch.cat([noisy_latents, control_latents], dim=1)
        
        # 3. Forward Pass
        # We pass drug_emb as 'encoder_hidden_states'
        noise_pred = self.unet(
            unet_input, 
            timesteps, 
            encoder_hidden_states=drug_emb
        ).sample
        
        return noise_pred

# ============================================================================
# DATASET & ENCODER
# ============================================================================

class MorganEncoder:
    def __init__(self, n_bits=1024):
        self.n_bits = n_bits
        self.cache = {}
        
    def encode(self, smiles):
        if isinstance(smiles, list): 
            return np.array([self.encode(s) for s in smiles])
        if smiles in self.cache: 
            return self.cache[smiles]
        if RDKIT_AVAILABLE and smiles and smiles not in ['DMSO', '']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.float32)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                self.cache[smiles] = arr
                return arr
            except: 
                pass
        np.random.seed(hash(str(smiles)) % 2**32)
        arr = (np.random.rand(self.n_bits) > 0.5).astype(np.float32)
        self.cache[smiles] = arr
        return arr

class PairedBBBC021Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, size=512, split='train', paths_csv=None):
        self.data_dir = Path(data_dir).resolve()
        self.size = size
        self.encoder = MorganEncoder()
        self._first_load_logged = False
        
        # Robust CSV Loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file
            
        df = pd.read_csv(csv_full_path)
        if 'SPLIT' in df.columns:
            df = df[df['SPLIT'].str.lower() == split.lower()]
            
        self.metadata = df.to_dict('records')
        
        # Group by Batch to find controls
        self.controls = {}  # Batch -> [List of Control Indices]
        self.treated = []   # List of Treated Indices
        
        for idx, row in enumerate(self.metadata):
            batch = row.get('BATCH', 'unk')
            cpd = str(row.get('CPD_NAME', '')).upper()
            
            if cpd == 'DMSO':
                if batch not in self.controls: 
                    self.controls[batch] = []
                self.controls[batch].append(idx)
            else:
                self.treated.append(idx)
        
        # Pre-encode fingerprints
        self.fingerprints = {}
        if 'CPD_NAME' in df.columns:
            for cpd in df['CPD_NAME'].unique():
                row = df[df['CPD_NAME'] == cpd].iloc[0]
                smiles = row.get('SMILES', '')
                self.fingerprints[cpd] = self.encoder.encode(smiles)
                
        print(f"Dataset ({split}): {len(self.treated)} Treated, {sum(len(v) for v in self.controls.values())} Controls")

        # Load paths.csv for robust file lookup
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

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # SD expects [-1, 1]
        ])

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
        
        # Strategy 2: Search paths.csv by SAMPLE_KEY
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

    def _load_img(self, idx):
        meta = self.metadata[idx]
        path = meta.get('image_path') or meta.get('SAMPLE_KEY')
        
        if not path:
            raise ValueError(f"CRITICAL: No image path found in metadata! Index: {idx}")
        
        full_path = self._find_file_path(path)
        
        if full_path is None or not full_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Image file not found!\n"
                f"  Index: {idx}\n"
                f"  Path from metadata: {path}\n"
                f"  Data directory: {self.data_dir}\n"
                f"  paths.csv loaded: {len(self.paths_lookup) > 0}"
            )
        
        try:
            img = np.load(str(full_path))
            original_shape = img.shape
            
            # Handle shapes
            if img.ndim == 3 and img.shape[0] == 3: 
                img = img.transpose(1, 2, 0)  # CHW -> HWC for PIL
            
            # Normalize to 0-255 uint8 for PIL
            if img.max() > 1.0:
                img = img.astype(np.uint8)
            else:
                if img.min() < 0:
                    img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                img = (img * 255).astype(np.uint8)
            
            # If grayscale, make RGB
            if img.ndim == 2: 
                img = np.stack([img]*3, axis=-1)
            
            if not self._first_load_logged or idx < 3:
                print(f"\n{'='*60}", flush=True)
                print(f"✓ Successfully loaded image #{idx}", flush=True)
                print(f"  File path: {full_path}", flush=True)
                print(f"  Original shape: {original_shape}", flush=True)
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

        # Convert to PIL for transforms
        image_pil = Image.fromarray(img)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
            
        return self.transform(image_pil)

    def __len__(self): 
        return len(self.treated)

    def __getitem__(self, idx):
        # 1. Get Treated Sample
        trt_idx = self.treated[idx]
        trt_meta = self.metadata[trt_idx]
        batch = trt_meta.get('BATCH', 'unk')
        
        # 2. Get Random Control from SAME Batch
        if batch in self.controls and len(self.controls[batch]) > 0:
            ctrl_idx = np.random.choice(self.controls[batch])
        else:
            ctrl_idx = trt_idx  # Fallback
            
        # 3. Load Images
        trt_img = self._load_img(trt_idx)
        ctrl_img = self._load_img(ctrl_idx)
        
        # 4. Get Drug Fingerprint
        cpd = trt_meta.get('CPD_NAME', 'DMSO')
        fp = self.fingerprints.get(cpd, np.zeros(1024))

        return {
            'control': ctrl_img,
            'target': trt_img,
            'fingerprint': torch.from_numpy(fp).float()
        }

# ============================================================================
# METRICS & UTILITIES
# ============================================================================

def calculate_metrics(real_imgs, gen_imgs, noise_pred=None, noise_real=None):
    """Calculates PSNR, SSIM, MSE, and estimated KL"""
    metrics = {}
    
    # 1. MSE (Pixel Space)
    metrics['mse'] = F.mse_loss(gen_imgs, real_imgs).item()
    
    # 2. KL Divergence Estimate (Latent Space)
    # KL is proportional to MSE in latent space for DDPMs
    if noise_pred is not None and noise_real is not None:
        metrics['kl_div'] = F.mse_loss(noise_pred, noise_real).item()
    else:
        metrics['kl_div'] = None
    
    # 3. PSNR & SSIM (Requires CPU/Numpy)
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to numpy [0, 255] range
        real_np = ((real_imgs[0].cpu().permute(1,2,0).numpy() + 1) * 127.5).astype(np.uint8)
        gen_np = ((gen_imgs[0].cpu().permute(1,2,0).numpy() + 1) * 127.5).astype(np.uint8)
        
        # Convert to grayscale for SSIM
        real_gray = np.mean(real_np, axis=2)
        gen_gray = np.mean(gen_np, axis=2)
        
        metrics['psnr'] = psnr(real_np, gen_np, data_range=255)
        metrics['ssim'] = ssim(real_gray, gen_gray, data_range=255)
    except ImportError:
        print("  Warning: scikit-image not available. Install with: pip install scikit-image")
        metrics['psnr'] = None
        metrics['ssim'] = None
        
    return metrics

def generate_video(model, vae, noise_scheduler, control, fingerprint, save_path, num_frames=40):
    """Generates video of denoising process"""
    if not IMAGEIO_AVAILABLE: 
        print("  Warning: imageio not available. Skipping video generation.")
        return
    
    model.eval()
    
    # 1. Prepare Inputs
    with torch.no_grad():
        # Ensure correct dtype and device
        dtype = model.unet.dtype if hasattr(model, 'unet') else torch.float32
        
        ctrl_latents = vae.encode(control.unsqueeze(0).to(dtype=dtype)).latent_dist.mode() * vae.config.scaling_factor
        fp = fingerprint.unsqueeze(0)
        
        # 2. Setup Noise
        latents = torch.randn_like(ctrl_latents)
        frames = []
        
        # 3. Setup Scheduler
        noise_scheduler.set_timesteps(1000)
        save_steps = np.linspace(0, 999, num_frames, dtype=int)
        
        # FIX: Iterate directly. Do NOT use reversed() - timesteps are already [999, 998, ..., 0]
        for t in tqdm(noise_scheduler.timesteps, desc="  Generating Video", leave=False):
            timestep = torch.full((1,), t, device=latents.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(latents.float(), timestep, ctrl_latents.float(), fp)
            
            # Scheduler step - pass scalar timestep to avoid tensor boolean ambiguity
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save frame if at capture point
            if t.item() in save_steps or t.item() == noise_scheduler.timesteps[-1].item():
                # Decode to image
                decoded = vae.decode(latents / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(img_np)
        
        # Also decode control for side-by-side comparison
        ctrl_decoded = vae.decode(ctrl_latents / vae.config.scaling_factor).sample
        ctrl_decoded = (ctrl_decoded / 2 + 0.5).clamp(0, 1)
        ctrl_np = (ctrl_decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Create separator
        separator = np.zeros((ctrl_np.shape[0], 2, 3), dtype=np.uint8)
        
        # Stitch frames: [Generated (changing)] | [Control (static)]
        final_frames = []
        for frame in frames:
            combined = np.hstack([frame, separator, ctrl_np])
            final_frames.append(combined)
        
        # 4. Save Video
        if IMAGEIO_AVAILABLE and final_frames:
            import imageio
            imageio.mimsave(save_path, final_frames, fps=10)
            print(f"  ✓ Video saved to: {save_path}")
        else:
            print(f"  ⚠ Skipping video save (imageio not available or no frames)")

def load_checkpoint(model, optimizer, path, scheduler=None):
    if not os.path.exists(path): 
        return 0
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=model.cfg.device if hasattr(model, 'cfg') else 'cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0)

# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(model, vae, noise_scheduler, dataset, config, logger, checkpoint_epoch=None, eval_split="train"):
    """
    Run evaluation: generate video, grid, and metrics without training.
    
    Args:
        model: The trained model
        vae: VAE encoder/decoder
        noise_scheduler: Diffusion noise scheduler
        dataset: Dataset to evaluate on
        config: Configuration object
        logger: TrainingLogger instance
        checkpoint_epoch: Epoch number from checkpoint (for logging)
        eval_split: Data split name (for logging and file naming)
    """
    print("\n" + "="*60, flush=True)
    epoch_label = f"Epoch {checkpoint_epoch}" if checkpoint_epoch else "Evaluation"
    print(f"EVALUATION ({epoch_label}) - Split: {eval_split}", flush=True)
    print("="*60, flush=True)
    
    model.eval()
    
    # Create dataloader for evaluation
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Get a sample batch
    sample_batch = next(iter(eval_loader))
    weight_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
    
    ctrl_img = sample_batch['control'][:4].to(config.device, dtype=weight_dtype)
    target_img = sample_batch['target'][:4].to(config.device, dtype=weight_dtype)
    fp = sample_batch['fingerprint'][:4].to(config.device)
    
    with torch.no_grad():
        # 1. Encode Control and Target
        ctrl_latents = vae.encode(ctrl_img.float()).latent_dist.mode() * vae.config.scaling_factor
        target_latents = vae.encode(target_img.float()).latent_dist.mode() * vae.config.scaling_factor
        
        # 2. Run Reverse Diffusion Process (Full Sampling Loop)
        print("  Running reverse diffusion sampling...", flush=True)
        latents = torch.randn_like(ctrl_latents)
        
        # Use scheduler for proper sampling
        for t in tqdm(noise_scheduler.timesteps, desc="  Sampling", leave=False):
            # Create timestep tensor for model (batch_size,)
            timestep = torch.full((latents.shape[0],), t, device=config.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(latents.float(), timestep, ctrl_latents.float(), fp)
            
            # Scheduler step - pass scalar timestep to avoid tensor boolean ambiguity
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # 3. Decode Generated Images
        fake_imgs = vae.decode(latents / vae.config.scaling_factor).sample
        fake_imgs = (fake_imgs / 2 + 0.5).clamp(0, 1)
        
        # 4. Calculate Metrics
        real_imgs_norm = (target_img / 2 + 0.5).clamp(0, 1)
        
        # Also compute noise prediction metrics during forward pass
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                (target_latents.shape[0],), device=config.device).long()
        noisy_target = noise_scheduler.add_noise(target_latents, noise, timesteps)
        noise_pred_forward = model(noisy_target.float(), timesteps, ctrl_latents.float(), fp)
        
        metrics = calculate_metrics(real_imgs_norm, fake_imgs, noise_pred_forward, noise)
        
        # Print metrics
        print(f"\n  📊 EVALUATION METRICS:", flush=True)
        print(f"  {'-'*58}", flush=True)
        if metrics['kl_div'] is not None:
            print(f"    KL Divergence:     {metrics['kl_div']:.6f}", flush=True)
        if metrics['mse'] is not None:
            print(f"    MSE (gen vs real): {metrics['mse']:.6f}", flush=True)
        if metrics['psnr'] is not None:
            print(f"    PSNR:              {metrics['psnr']:.2f} dB", flush=True)
        if metrics['ssim'] is not None:
            print(f"    SSIM:              {metrics['ssim']:.4f}", flush=True)
        print(f"  {'-'*58}", flush=True)
        
        # Log metrics to CSV
        if checkpoint_epoch:
            logger.log_metrics(checkpoint_epoch, metrics)
        else:
            logger.log_metrics(0, metrics)  # Use epoch 0 for standalone evaluation
        
        # 5. Save Image Grid
        grid = torch.cat([
            (ctrl_img / 2 + 0.5).clamp(0, 1)[:4],
            fake_imgs[:4],
            real_imgs_norm[:4]  # Show target again for comparison
        ], dim=0)
        
        # Create filename with split info
        split_suffix = f"_{eval_split}" if eval_split != "train" else ""
        
        if checkpoint_epoch:
            grid_path = f"{config.output_dir}/plots/eval_epoch_{checkpoint_epoch}{split_suffix}.png"
        else:
            grid_path = f"{config.output_dir}/plots/eval_latest{split_suffix}.png"
        
        save_image(grid, grid_path, nrow=4, normalize=False)
        print(f"  ✓ Sample grid saved to: {grid_path}", flush=True)
        
        # 6. Generate Video
        if checkpoint_epoch:
            video_path = f"{config.output_dir}/plots/video_eval_epoch_{checkpoint_epoch}{split_suffix}.mp4"
        else:
            video_path = f"{config.output_dir}/plots/video_eval_latest{split_suffix}.mp4"
        
        generate_video(model, vae, noise_scheduler, ctrl_img[0], fp[0], video_path)
        
        print("="*60 + "\n", flush=True)
        
        # Log to wandb
        if WANDB_AVAILABLE:
            wandb_metrics = {"eval_epoch": checkpoint_epoch if checkpoint_epoch else 0}
            if metrics['kl_div'] is not None:
                wandb_metrics['eval_kl_div'] = metrics['kl_div']
            if metrics['mse'] is not None:
                wandb_metrics['eval_mse_gen_real'] = metrics['mse']
            if metrics['psnr'] is not None:
                wandb_metrics['eval_psnr'] = metrics['psnr']
            if metrics['ssim'] is not None:
                wandb_metrics['eval_ssim'] = metrics['ssim']
            wandb.log(wandb_metrics)

def main():
    parser = argparse.ArgumentParser(description="Partial Fine-Tuning of Stable Diffusion for Drug-Conditioned Generation")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only (generate video, grid, and metrics)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file for evaluation (default: uses latest.pt from output_dir)")
    parser.add_argument("--eval_split", type=str, default="train", choices=["train", "test", "val"], help="Data split to use for evaluation (default: train)")
    args = parser.parse_args()
    
    config = Config()
    if args.output_dir:
    config.output_dir = args.output_dir
    
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: 
        wandb.init(project="bbbc021-sd-partial", config=config.__dict__)

    # 1. Load VAE (Frozen)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae").to(config.device)
    vae.requires_grad_(False)
    
    # 2. Load Model (Partial Tuning)
    model = DrugConditionedStableDiffusion(config).to(config.device)
    
    # Store config in model for checkpoint loading
    model.cfg = config
    
    # Model Summary
    print(f"\n{'='*60}", flush=True)
    print(f"Model Summary:", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Use stored values from model if available, otherwise calculate
    if hasattr(model, 'total_params'):
        total_params = model.total_params
        trainable_params = model.trainable_params
        frozen_params = model.frozen_params
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
    
    print(f"  Total Parameters:     {total_params:,}", flush=True)
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)", flush=True)
    print(f"  Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)", flush=True)
    
    # Print model architecture summary
    print(f"\n  Model Architecture:", flush=True)
    print(f"    - U-Net: UNet2DConditionModel (Stable Diffusion v1.5)", flush=True)
    print(f"    - Input Channels: 8 (4 noise + 4 control latents)", flush=True)
    print(f"    - Drug Projection: {config.fingerprint_dim} -> 768", flush=True)
    print(f"    - Partial Fine-Tuning: Input, Cross-Attention, Output layers", flush=True)
    
    # PyTorch Model Summary
    print(f"\n  PyTorch Model Structure:", flush=True)
    print(f"    Model Type: {type(model).__name__}", flush=True)
    print(f"    UNet Type: {type(model.unet).__name__}", flush=True)
    print(f"    Drug Projection: {type(model.drug_proj).__name__}", flush=True)
    
    # Try to use torchsummary if available
    try:
        from torchsummary import summary
        print(f"\n  Note: torchsummary available (model too complex for detailed summary)", flush=True)
    except ImportError:
        print(f"\n  Note: Install torchsummary for detailed layer info: pip install torchsummary", flush=True)
    
    print(f"{'='*60}\n", flush=True)
    
    # 3. Optimizer (Only Trainable Params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.lr,
        weight_decay=0.01
    )
    
    # 4. Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,
        eta_min=1e-6
    )
    
    # 5. Scheduler (Noise)
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    
    # 6. Data
    print("\nLoading Dataset...")
    dataset = PairedBBBC021Dataset(
        config.data_dir, 
        config.metadata_file, 
        size=config.image_size,
        split='train',
        paths_csv=args.paths_csv
    )
    
    # Dataset Summary
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset Summary:", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total Treated Samples: {len(dataset.treated):,}", flush=True)
    print(f"  Total Control Samples: {sum(len(v) for v in dataset.controls.values()):,}", flush=True)
    print(f"  Unique Batches: {len(dataset.controls):,}", flush=True)
    print(f"  Image Size: {config.image_size}x{config.image_size}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    # Check if evaluation-only mode
    if args.eval_only:
        # Determine checkpoint path
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = f"{config.output_dir}/checkpoints/latest.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print(f"Please provide a valid checkpoint path with --checkpoint or ensure latest.pt exists")
            return
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(ckpt['model'], strict=False)
        checkpoint_epoch = ckpt.get('epoch', 0)
        print(f"  Loaded checkpoint from epoch {checkpoint_epoch}")
        
        # Load dataset for evaluation
        eval_split = args.eval_split.lower()
        print(f"\nLoading Dataset for Evaluation (split: {eval_split})...")
        dataset = PairedBBBC021Dataset(
            config.data_dir, 
            config.metadata_file, 
            size=config.image_size,
            split=eval_split,
            paths_csv=args.paths_csv
        )
        print(f"  Dataset loaded: {len(dataset)} samples from '{eval_split}' split")
        
        # Run evaluation
        run_evaluation(model, vae, noise_scheduler, dataset, config, logger, checkpoint_epoch, eval_split)
        print("Evaluation complete!")
        return
    
    # Load checkpoint for training
    checkpoint_path = f"{config.output_dir}/checkpoints/latest.pt"
    start_epoch = 0
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
    
    print("Starting Training...")
    
    weight_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            ctrl_img = batch['control'].to(config.device, dtype=weight_dtype)
            target_img = batch['target'].to(config.device, dtype=weight_dtype)
            fp = batch['fingerprint'].to(config.device)
            
            optimizer.zero_grad()
            
            # A. Encode Images to Latents (VAE)
            # VAE requires float32, so convert images if they're in float16
            with torch.no_grad():
                # Control: Use Mode (Deterministic)
                ctrl_latents = vae.encode(ctrl_img.float()).latent_dist.mode() * vae.config.scaling_factor
                # Target: Use Sample (Stochastic)
                target_latents = vae.encode(target_img.float()).latent_dist.sample() * vae.config.scaling_factor
            
            # B. Add Noise
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (target_latents.shape[0],), device=config.device
            ).long()
            noisy_target = noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            # C. Forward Pass
            if config.mixed_precision == "fp16":
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    noise_pred = model(noisy_target.float(), timesteps, ctrl_latents.float(), fp)
                    loss = F.mse_loss(noise_pred.float(), noise.float())
            else:
                noise_pred = model(noisy_target.float(), timesteps, ctrl_latents.float(), fp)
                loss = F.mse_loss(noise_pred, noise)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/Inf loss detected, skipping this batch", flush=True)
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                optimizer.step()
            
            epoch_losses.append(loss.item())
            progress.set_postfix({"loss": loss.item()})
            
            if WANDB_AVAILABLE: 
                wandb.log({"loss": loss.item(), "step": epoch * len(loader) + len(epoch_losses)})
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Filter out NaN/Inf losses
        valid_losses = [l for l in epoch_losses if not (np.isnan(l) or np.isinf(l))]
        if valid_losses:
            avg_loss = np.mean(valid_losses)
        else:
            avg_loss = float('nan')
            print(f"\n⚠️  WARNING: All losses in epoch {epoch+1} were NaN/Inf!", flush=True)
        
        print(f"\nEpoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.5f} | LR: {current_lr:.2e}", flush=True)
        
        # Stop training if loss is NaN
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            print(f"\n❌ Training stopped due to NaN/Inf loss.", flush=True)
            break
        
        logger.update(epoch+1, avg_loss, current_lr)
        if WANDB_AVAILABLE: 
            wandb.log({
                "epoch": epoch+1,
                "epoch_loss": avg_loss,
                "learning_rate": current_lr
            })

        # Checkpointing
        if (epoch + 1) % config.save_freq == 0:
            # Save specific epoch checkpoint (unique file for every evaluated epoch)
            epoch_path = f"{config.output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1,
                'config': config.__dict__
            }, epoch_path)
            
            # Update 'latest' for resuming
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1,
                'config': config.__dict__
            }, f"{config.output_dir}/checkpoints/latest.pt")
            
            print(f"  ✓ Saved checkpoint: {epoch_path} (LR: {current_lr:.2e})", flush=True)
            
        # --- EVALUATION ---
        if (epoch + 1) % config.eval_freq == 0:
            # Use the reusable evaluation function (always use train split during training)
            run_evaluation(model, vae, noise_scheduler, dataset, config, logger, epoch+1, "train")

if __name__ == "__main__":
    main()
