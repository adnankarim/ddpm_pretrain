"""
================================================================================
BBBC021 CONTROLNET + LoRA + DRUG CONDITIONING (Gold Standard Architecture)
================================================================================
Strategy: ControlNet + LoRA (No Input Surgery - Preserves Pretrained Weights)
--------------------------------------------------------------------------------
This is the "Gold Standard" architecture that fixes the "Static/Garbage" problem:

1. **Frozen Brain:** Main U-Net is 100% frozen. Never breaks pretrained weights.
2. **ControlNet:** Separate trainable encoder that guides U-Net via residuals.
3. **Zero-Convolution:** ControlNet initializes to zero, so Epoch 0 = Standard SD.
4. **LoRA:** Injected into U-Net for style learning (~1% trainable).
5. **Dual Conditioning:** CLIP text + Multi-token drug fingerprint.

Architecture:
1. VAE: Frozen (Encodes pixels -> latents)
2. Text Encoder: Frozen (CLIP embeddings)
3. U-Net: Frozen + LoRA adapters (Trainable ~1%)
4. ControlNet: Trainable (Learns structure from control images)
5. Drug Projector: Trainable (Multi-token projection for voice balance)

Input: [Noisy Latents] + [Control Pixel Image] + [CLIP Text + Drug Tokens]
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
    from diffusers import (
        AutoencoderKL, 
        DDPMScheduler, 
        UNet2DConditionModel,
        ControlNetModel
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: Install dependencies: pip install diffusers transformers accelerate")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    print("CRITICAL: Install peft: pip install peft")
    PEFT_AVAILABLE = False
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
    batch_size = 32  # Lower batch size due to 512x512 resolution
    lr = 1e-5       # 1e-5 is perfect for ControlNet + LoRA
    save_freq = 1
    eval_freq = 1
    
    # Token Strategy (Fixes Flaw #1: Voice Imbalance)
    num_drug_tokens = 4  # Give drug 4 "words" of attention instead of 1
    
    output_dir = "controlnet_lora_results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = "fp32"  # Use FP32 for stability (GH200 has enough VRAM)

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
        
        plt.title(f'ControlNet + LoRA Training (Epoch {epoch})')
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

# ============================================================================
# ARCHITECTURE: DRUG PROJECTOR
# ============================================================================

class DrugProjector(nn.Module):
    """
    Drug Projector: Projects fingerprint to multi-token embeddings
    Fixes Flaw #1: Voice Imbalance by giving drug 4 tokens instead of 1
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Projects 1024 -> (4 * 768)
        self.net = nn.Sequential(
            nn.Linear(config.fingerprint_dim, 768 * config.num_drug_tokens),
            nn.SiLU(),
            nn.Linear(768 * config.num_drug_tokens, 768 * config.num_drug_tokens)
        )
    
    def forward(self, fingerprint):
        x = self.net(fingerprint)
        # Reshape to [Batch, Num_Tokens, 768]
        return x.view(-1, self.config.num_drug_tokens, 768)

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
            'control': ctrl_img,  # Pixel space for ControlNet
            'target': trt_img,    # Pixel space (will be encoded to latents)
            'fingerprint': torch.from_numpy(fp).float(),
            'prompt': "fluorescence microscopy image of a cell"  # Static prompt for style
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

def generate_video(unet, controlnet, vae, scheduler, drug_proj, tokenizer, text_encoder, control, fingerprint, prompt, save_path, device, num_frames=40):
    """
    Generates video of denoising process (Fixed: robust to timestep index errors)
    """
    if not IMAGEIO_AVAILABLE: 
        print("  Warning: imageio not available. Skipping video generation.")
        return
    
    unet.eval()
    controlnet.eval()
    
    with torch.no_grad():
        # 1. Inputs
        ctrl_pixel = control.unsqueeze(0).to(device)
        dtype = next(unet.parameters()).dtype
        latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)
        
        # 2. Context
        tokens = tokenizer([prompt], padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        text_emb = text_encoder(tokens)[0].to(dtype=dtype)
        drug_emb = drug_proj(fingerprint.unsqueeze(0).to(device, dtype=dtype))
        context = torch.cat([text_emb, drug_emb], dim=1)
        
        # 3. Scheduler Setup
        scheduler.set_timesteps(1000)
        frames = []
        save_steps = np.linspace(0, 999, num_frames, dtype=int)
        
        # 4. Denoising Loop
        for t in tqdm(scheduler.timesteps, desc="  Generating Video", leave=False):
            # FIX: Force t to be a valid index (0-999)
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            t_val = min(t_val, scheduler.config.num_train_timesteps - 1) 
            t_tensor = torch.full((1,), t_val, device=device, dtype=torch.long)
            
            # A. ControlNet
            down, mid = controlnet(
                latents, t_tensor, encoder_hidden_states=context, 
                controlnet_cond=ctrl_pixel, return_dict=False
            )
            
            # B. U-Net
            if hasattr(unet, 'base_model'):
                noise_pred = unet.base_model.model(
                    latents, t_tensor, encoder_hidden_states=context,
                    down_block_additional_residuals=down, mid_block_additional_residual=mid
                ).sample
            else:
                noise_pred = unet(
                    latents, t_tensor, encoder_hidden_states=context,
                    down_block_additional_residuals=down, mid_block_additional_residual=mid
                ).sample
            
            # C. Step (Use t_val to avoid index error)
            latents = scheduler.step(noise_pred, t_val, latents).prev_sample
            
            # D. Save Frame
            if t_val in save_steps or t_val == 0:
                decoded = vae.decode((latents / vae.config.scaling_factor).float()).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(img_np)
        
        # 5. Save
        if frames:
            # Helper to create side-by-side
            ctrl_decoded = vae.decode(vae.encode(ctrl_pixel.float()).latent_dist.mode() / vae.config.scaling_factor).sample
            ctrl_decoded = (ctrl_decoded / 2 + 0.5).clamp(0, 1)
            ctrl_np = (ctrl_decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            separator = np.zeros((ctrl_np.shape[0], 2, 3), dtype=np.uint8)
            
            final_frames = [np.hstack([f, separator, ctrl_np]) for f in frames]
            imageio.mimsave(save_path, final_frames, fps=10)
            print(f"  ✓ Video saved to: {save_path}")

def load_checkpoint(unet, controlnet, drug_proj, optimizer, path, scheduler=None):
    if not os.path.exists(path): 
        return 0
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')
    
    # Load separate components
    if 'unet' in ckpt:
        unet.load_state_dict(ckpt['unet'], strict=False)
    if 'controlnet' in ckpt:
        controlnet.load_state_dict(ckpt['controlnet'], strict=False)
    if 'drug_proj' in ckpt:
        drug_proj.load_state_dict(ckpt['drug_proj'], strict=False)
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0)

# ============================================================================
# MAIN
# ============================================================================

def calculate_metrics_torchmetrics_sd(unet, controlnet, vae, noise_scheduler, drug_proj, tokenizer, text_encoder, dataset, config, num_samples=20480, num_inference_steps=50):
    """
    Calculate evaluation metrics using torchmetrics (FID/KID) following reference pattern.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        TORCHMETRICS_AVAILABLE = True
    except ImportError:
        print("  Warning: torchmetrics not available. Install with: pip install torchmetrics")
        return None
    
    unet.eval()
    controlnet.eval()
    
    # Initialize metrics
    fid_metric = FrechetInceptionDistance(normalize=True).to(config.device, non_blocking=True)
    kid_metric = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device, non_blocking=True)
    
    # Group samples by compound for per-class metrics
    generated_samples = {}
    target_samples = {}
    
    # Calculate expected number of batches
    batch_size = 4
    total_dataset_samples = len(dataset)
    expected_batches = min((num_samples + batch_size - 1) // batch_size, (total_dataset_samples + batch_size - 1) // batch_size)
    print(f"  Processing {num_samples} samples from {total_dataset_samples} available (will process ~{expected_batches} batches)", flush=True)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    weight_dtype = torch.float32
    
    sample_count = 0
    with torch.no_grad():
        from tqdm.auto import tqdm
        pbar = tqdm(eval_loader, desc=f"  Evaluating ({num_samples} samples)", total=expected_batches, leave=False)
        for batch in pbar:
            if sample_count >= num_samples:
                break
            
            # Handle last batch that might exceed num_samples
            remaining_samples = num_samples - sample_count
            if remaining_samples < batch_size:
                # Only process the first remaining_samples from this batch
                ctrl_img = batch['control'][:remaining_samples].to(config.device, dtype=weight_dtype)
                target_img = batch['target'][:remaining_samples].to(config.device, dtype=weight_dtype)
                fp = batch['fingerprint'][:remaining_samples].to(config.device)
                prompts = batch['prompt'][:remaining_samples] if isinstance(batch['prompt'], list) else batch['prompt']
            else:
                ctrl_img = batch['control'].to(config.device, dtype=weight_dtype)
                target_img = batch['target'].to(config.device, dtype=weight_dtype)
                fp = batch['fingerprint'].to(config.device)
                prompts = batch['prompt']
            
            # Prepare context
            tokens = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
            text_emb = text_encoder(tokens)[0].to(dtype=weight_dtype)
            drug_emb = drug_proj(fp.to(dtype=weight_dtype))
            context = torch.cat([text_emb, drug_emb], dim=1)
            
            # Generate samples
            latents = torch.randn_like(vae.encode(target_img.to(dtype=torch.float32)).latent_dist.mode() * vae.config.scaling_factor)
            noise_scheduler.set_timesteps(num_inference_steps, device=config.device)
            
            for t in noise_scheduler.timesteps:
                t_val = t.item() if isinstance(t, torch.Tensor) else t
                t_val = min(t_val, noise_scheduler.config.num_train_timesteps - 1)
                timestep = torch.full((latents.shape[0],), t_val, device=config.device, dtype=torch.long)
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latents, timestep, encoder_hidden_states=context,
                    controlnet_cond=ctrl_img, return_dict=False,
                )
                
                if hasattr(unet, 'base_model'):
                    noise_pred = unet.base_model.model(
                        latents, timestep, encoder_hidden_states=context,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                else:
                    noise_pred = unet(
                        latents, timestep, encoder_hidden_states=context,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                
                latents = noise_scheduler.step(noise_pred, t_val, latents).prev_sample
            
            # Decode
            generated = vae.decode((latents / vae.config.scaling_factor).float()).sample
            generated = (generated / 2 + 0.5).clamp(0, 1)
            real_norm = (target_img / 2 + 0.5).clamp(0, 1)
            
            # Convert to [0, 255] for torchmetrics
            real_uint8 = torch.floor(real_norm * 255.0).to(torch.uint8)
            gen_uint8 = torch.floor(generated * 255.0).to(torch.uint8)
            
            # Update metrics
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(gen_uint8, real=False)
            kid_metric.update(real_uint8, real=True)
            kid_metric.update(gen_uint8, real=False)
            
            # Group by compound (extract from metadata if available)
            actual_batch_size = target_img.shape[0]
            for i in range(actual_batch_size):
                # Try to get compound name from dataset metadata
                compound = "unknown"
                if hasattr(dataset, 'metadata') and len(dataset.metadata) > sample_count + i:
                    try:
                        meta = dataset.metadata[dataset.treated[sample_count + i]]
                        compound = meta.get('CPD_NAME', 'unknown')
                    except:
                        pass
                
                if compound not in generated_samples:
                    generated_samples[compound] = []
                    target_samples[compound] = []
                generated_samples[compound].append(gen_uint8[i])
                target_samples[compound].append(real_uint8[i])
            
            sample_count += actual_batch_size
            pbar.set_postfix({"samples": sample_count, "target": num_samples})
    
    # Compute overall metrics
    fid = fid_metric.compute()
    kid_mean, kid_std = kid_metric.compute()
    
    # Compute per-class metrics
    fid_per_class = {}
    kid_per_class = {}
    
    for compound in tqdm(generated_samples.keys(), desc="  Computing per-class metrics", leave=False):
        if len(generated_samples[compound]) == 0:
            continue
        
        try:
            gen_stack = torch.stack(generated_samples[compound]).to(config.device)
            target_stack = torch.stack(target_samples[compound]).to(config.device)
            
            fid_metric_class = FrechetInceptionDistance(normalize=True).to(config.device, non_blocking=True)
            fid_metric_class.update(target_stack, real=True)
            fid_metric_class.update(gen_stack, real=False)
            fid_per_class[compound] = float(fid_metric_class.compute().cpu().item())
            
            dynamic_subset_size = min(len(generated_samples[compound]), 100)
            kid_metric_class = KernelInceptionDistance(subset_size=dynamic_subset_size, normalize=True).to(config.device, non_blocking=True)
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

def run_evaluation(unet, controlnet, vae, noise_scheduler, drug_proj, tokenizer, text_encoder, dataset, config, logger, checkpoint_epoch=None, eval_split="train"):
    """
    Run evaluation: generate video, grid, and metrics without training.
    """
    print("\n" + "="*60, flush=True)
    epoch_label = f"Epoch {checkpoint_epoch}" if checkpoint_epoch else "Evaluation"
    print(f"EVALUATION ({epoch_label}) - Split: {eval_split}", flush=True)
    print("="*60, flush=True)
    
    unet.eval()
    controlnet.eval()
    
    # Create dataloader for evaluation
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Get a sample batch
    sample_batch = next(iter(eval_loader))
    weight_dtype = torch.float32  # Use FP32 for stability
    
    ctrl_img = sample_batch['control'][:4].to(config.device, dtype=weight_dtype)
    target_img = sample_batch['target'][:4].to(config.device, dtype=weight_dtype)
    fp = sample_batch['fingerprint'][:4].to(config.device)
    prompts = sample_batch['prompt']
    
    with torch.no_grad():
        # 1. Encode Target to Latents
        target_latents = vae.encode(target_img.to(dtype=torch.float32)).latent_dist.mode() * vae.config.scaling_factor
        
        # 2. Prepare Context (Text + Drug)
        tokens = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
        text_emb = text_encoder(tokens)[0]  # [B, 77, 768]
        # Convert fingerprint to match drug_proj dtype
        weight_dtype = torch.float32  # Use FP32 for stability
        # Convert text_emb to match weight_dtype (float16 if mixed precision)
        text_emb = text_emb.to(dtype=weight_dtype)
        drug_emb = drug_proj(fp.to(dtype=weight_dtype))  # [B, N, 768]
        context = torch.cat([text_emb, drug_emb], dim=1)  # [B, 77+N, 768]
        
        # 3. Run Reverse Diffusion Process (Full Sampling Loop)
        print("  Running reverse diffusion sampling...", flush=True)
        latents = torch.randn_like(target_latents)
        
        for t in tqdm(noise_scheduler.timesteps, desc="  Sampling", leave=False):
            # FIX: Force t to be a valid index (0-999)
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            t_val = min(t_val, noise_scheduler.config.num_train_timesteps - 1)
            timestep = torch.full((latents.shape[0],), t_val, device=config.device, dtype=torch.long)
            
            # A. ControlNet Step
            down_block_res_samples, mid_block_res_sample = controlnet(
                latents,
                timestep,
                encoder_hidden_states=context,
                controlnet_cond=ctrl_img,  # Pixel image
                return_dict=False,
            )
            
            # B. U-Net Step
            if hasattr(unet, 'base_model'):
                noise_pred = unet.base_model.model(
                    latents,
                    timestep,
                    encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                noise_pred = unet(
                    latents,
                    timestep,
                    encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            
            # C. Scheduler Step (use t_val to avoid index error)
            latents = noise_scheduler.step(noise_pred, t_val, latents).prev_sample
        
        # 4. Decode Generated Images
        fake_imgs = vae.decode((latents / vae.config.scaling_factor).float()).sample
        fake_imgs = (fake_imgs / 2 + 0.5).clamp(0, 1)
        
        # 5. Calculate Metrics
        real_imgs_norm = (target_img / 2 + 0.5).clamp(0, 1)
        
        # Also compute noise prediction metrics during forward pass
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                (target_latents.shape[0],), device=config.device).long()
        noisy_target = noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # Forward pass for metrics
        down_res, mid_res = controlnet(
            noisy_target,
            timesteps,
            encoder_hidden_states=context,
            controlnet_cond=ctrl_img,
            return_dict=False,
        )
        if hasattr(unet, 'base_model'):
            noise_pred_forward = unet.base_model.model(
                noisy_target,
                timesteps,
                encoder_hidden_states=context,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample
        else:
            noise_pred_forward = unet(
                noisy_target,
                timesteps,
                encoder_hidden_states=context,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample
        
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
            logger.log_metrics(0, metrics)
        
        # 6. Save Image Grid
        grid = torch.cat([
            (ctrl_img / 2 + 0.5).clamp(0, 1)[:4],
            fake_imgs[:4],
            real_imgs_norm[:4]
        ], dim=0)
        
        split_suffix = f"_{eval_split}" if eval_split != "train" else ""
        if checkpoint_epoch:
            grid_path = f"{config.output_dir}/plots/eval_epoch_{checkpoint_epoch}{split_suffix}.png"
        else:
            grid_path = f"{config.output_dir}/plots/eval_latest{split_suffix}.png"
        
        save_image(grid, grid_path, nrow=4, normalize=False)
        print(f"  ✓ Sample grid saved to: {grid_path}", flush=True)
        
        # 7. Generate Video
        if checkpoint_epoch:
            video_path = f"{config.output_dir}/plots/video_eval_epoch_{checkpoint_epoch}{split_suffix}.mp4"
        else:
            video_path = f"{config.output_dir}/plots/video_eval_latest{split_suffix}.mp4"
        
        generate_video(unet, controlnet, vae, noise_scheduler, drug_proj, tokenizer, text_encoder, 
                      ctrl_img[0], fp[0], prompts[0], video_path, config.device)
        
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
    parser = argparse.ArgumentParser(
        description="ControlNet + LoRA for Drug-Conditioned Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train normally
  python sdlora.py --output_dir ./results

  # Evaluate latest checkpoint (from output_dir/checkpoints/latest.pt)
  python sdlora.py --eval_only --output_dir ./results

  # Evaluate specific checkpoint on test split
  python sdlora.py --eval_only --checkpoint ./results/checkpoints/checkpoint_epoch_10.pt --eval_split test

  # Resume training from latest checkpoint
  python sdlora.py --resume --output_dir ./results
        """
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: controlnet_lora_results)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file for robust file lookup")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only: generate samples, plot grid, video, and metrics (no training)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file for evaluation (default: uses latest.pt from output_dir)")
    parser.add_argument("--num_samples", type=int, default=20480, help="Number of samples to calculate FID and KID (default: 20480)")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of inference steps for generation (default: 50)")
    parser.add_argument("--eval_split", type=str, default="train", choices=["train", "test", "val"], help="Data split to use for evaluation (default: train)")
    args = parser.parse_args()
    
    config = Config()
    if args.output_dir:
        config.output_dir = args.output_dir
    
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: 
        wandb.init(project="bbbc021-controlnet-lora", config=config.__dict__)

    # 1. Load Standard Components (All Frozen)
    print("Loading Components...")
    # Use FP32 for stability (GH200 has enough VRAM, no need for FP16)
    weight_dtype = torch.float32
    
    # VAE must always be float32 (it doesn't work well with float16)
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae").to(config.device, dtype=torch.float32)
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder").to(config.device)
    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet").to(config.device, dtype=weight_dtype)
    
    # Freeze everything initially
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    print("  ✓ VAE, Text Encoder, and U-Net loaded and frozen")
    
    # 2. Setup ControlNet (Initialized from U-Net weights for fast convergence)
    print("Creating ControlNet from U-Net...")
    controlnet = ControlNetModel.from_unet(unet)
    controlnet.to(config.device, dtype=weight_dtype)
    controlnet.requires_grad_(True)  # ControlNet is TRAINABLE
    print("  ✓ ControlNet created and trainable")
    
    # 3. Inject LoRA into U-Net (Trainable)
    print("Injecting LoRA into U-Net...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Cross-attention layers
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    print("  ✓ LoRA adapters injected")
    
    # 4. Drug Projector (Trainable)
    print("Creating Drug Projector...")
    drug_proj = DrugProjector(config).to(config.device, dtype=weight_dtype)
    print(f"  ✓ Drug Projector: {config.fingerprint_dim} -> {config.num_drug_tokens} tokens (768 dim each)")
    
    # Model Summary
    print(f"\n{'='*60}", flush=True)
    print(f"Model Summary:", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Count parameters correctly (iterate over parameters, not models)
    total_params = sum(p.numel() for model in [unet, controlnet, drug_proj] for p in model.parameters())
    trainable_params = sum(p.numel() for model in [unet, controlnet, drug_proj] for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  Total Parameters:     {total_params:,}", flush=True)
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)", flush=True)
    print(f"  Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)", flush=True)
    print(f"\n  Architecture:", flush=True)
    print(f"    - U-Net: Frozen + LoRA adapters", flush=True)
    print(f"    - ControlNet: Trainable (from U-Net weights)", flush=True)
    print(f"    - Drug Projector: Trainable (multi-token)", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 5. Optimizer (Train ControlNet + LoRA + DrugProj)
    params = list(controlnet.parameters()) + list(unet.parameters()) + list(drug_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.01)
    
    # 6. Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,
        eta_min=1e-6
    )
    
    # 7. Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    
    # 8. Data
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
            return
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        start_epoch = load_checkpoint(unet, controlnet, drug_proj, optimizer, checkpoint_path, scheduler)
        checkpoint_epoch = start_epoch
        print(f"  Loaded checkpoint from epoch {checkpoint_epoch}")
        
        # Load dataset for evaluation
        eval_split = args.eval_split.lower()
        dataset = PairedBBBC021Dataset(
            config.data_dir, 
            config.metadata_file, 
            size=config.image_size,
            split=eval_split,
            paths_csv=args.paths_csv
        )
        print(f"  Dataset loaded: {len(dataset)} samples from '{eval_split}' split")
        
        # Run evaluation using torchmetrics
        print("Running evaluation with torchmetrics...", flush=True)
        import json
        metrics = calculate_metrics_torchmetrics_sd(unet, controlnet, vae, noise_scheduler, drug_proj, 
                                                     tokenizer, text_encoder, dataset, config,
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
        output_name = f"sdlora_eval_{eval_split}_{args.num_samples}_{args.inference_steps}"
        os.makedirs("outputs/evaluation", exist_ok=True)
        json_path = f"outputs/evaluation/{output_name}.json"
        
        results = {
            "model": "sdlora.py",
            "checkpoint": checkpoint_path,
            "eval_split": eval_split,
            "num_samples": args.num_samples,
            "inference_steps": args.inference_steps,
            **metrics
        }
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Evaluation complete! Results saved to {json_path}", flush=True)
        return
    
    # Load checkpoint for training
    checkpoint_path = f"{config.output_dir}/checkpoints/latest.pt"
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(unet, controlnet, drug_proj, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch+1}...")
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Starting training from epoch 1...")
    else:
        start_epoch = load_checkpoint(unet, controlnet, drug_proj, optimizer, checkpoint_path, scheduler)
        if start_epoch > 0:
            print(f"Found checkpoint, resuming from epoch {start_epoch+1}...")
            print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Starting training from epoch 1...")
    
    # ---------------------------------------------------------
    # Epoch 0 Sanity Check (Before Training Starts)
    # ---------------------------------------------------------
    if start_epoch == 0:
        print("\n🔎 Running Epoch 0 Sanity Check (Untrained Model)...")
        print("="*60, flush=True)
        # Force a video and grid generation immediately to verify base model works
        run_evaluation(
            unet, controlnet, vae, noise_scheduler, drug_proj, 
            tokenizer, text_encoder, dataset, config, logger, 
            checkpoint_epoch=0,  # Label as Epoch 0
            eval_split="train"
        )
        print("✅ Epoch 0 Check Complete. Check ./controlnet_lora_results/plots/eval_epoch_0.png")
        print("="*60 + "\n", flush=True)
    
    # ---------------------------------------------------------
    # Start Training Loop
    # ---------------------------------------------------------
    print("Starting Training (ControlNet + LoRA)...")
    
    for epoch in range(start_epoch, config.epochs):
        unet.train()
        controlnet.train()
        drug_proj.train()
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            # Inputs (all in pixel space)
            ctrl_pixel = batch['control'].to(config.device, dtype=weight_dtype)
            target_pixel = batch['target'].to(config.device, dtype=weight_dtype)
            fp = batch['fingerprint'].to(config.device)
            prompts = batch['prompt']
            
            optimizer.zero_grad()
            
            # A. Prepare Latents (Target)
            with torch.no_grad():
                target_latents = vae.encode(target_pixel.to(dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                target_latents = target_latents.to(dtype=weight_dtype)
            
            # B. Prepare Context (Text + Drug)
            with torch.no_grad():
                tokens = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(config.device)
                text_emb = text_encoder(tokens)[0]  # [B, 77, 768]
                # Convert text_emb to match weight_dtype (float16 if mixed precision)
                text_emb = text_emb.to(dtype=weight_dtype)
            
            # Convert fingerprint to match drug_proj dtype
            drug_emb = drug_proj(fp.to(dtype=weight_dtype))  # [B, N, 768]
            context = torch.cat([text_emb, drug_emb], dim=1)  # [B, 77+N, 768]
            
            # C. Add Noise
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                    (target_latents.shape[0],), device=config.device).long()
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            # D. Forward Pass
            # 1. ControlNet calculates residuals from Pixel Image
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=context,
                controlnet_cond=ctrl_pixel,  # Pass Pixel Image directly!
                return_dict=False,
            )
            
            # 2. U-Net uses residuals (FP32 - no autocast needed)
            if hasattr(unet, 'base_model'):
                noise_pred = unet.base_model.model(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            
            loss = F.mse_loss(noise_pred, noise)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/Inf loss detected, skipping this batch", flush=True)
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
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
        
        # CHECKPOINTING (Save every epoch)
        # Save epoch-specific checkpoint
        epoch_path = f"{config.output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'unet': unet.state_dict(),
            'controlnet': controlnet.state_dict(),
            'drug_proj': drug_proj.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1,
            'config': config.__dict__
        }, epoch_path)
        
        # Also update latest.pt for easy resuming
        torch.save({
            'unet': unet.state_dict(),
            'controlnet': controlnet.state_dict(),
            'drug_proj': drug_proj.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1,
            'config': config.__dict__
        }, f"{config.output_dir}/checkpoints/latest.pt")
        
        print(f"  ✓ Checkpoint saved: {epoch_path} (LR: {current_lr:.2e})", flush=True)
            
        # --- EVALUATION ---
        if (epoch + 1) % config.eval_freq == 0:
            run_evaluation(unet, controlnet, vae, noise_scheduler, drug_proj, tokenizer, text_encoder, 
                          dataset, config, logger, epoch+1, "train")

if __name__ == "__main__":
    main()
