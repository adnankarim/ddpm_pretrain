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
import json
from datetime import datetime

# Check for diffusers
try:
    from diffusers import UNet2DModel, DDPMScheduler
except ImportError:
    print("CRITICAL: 'diffusers' library not found. Install with: pip install diffusers")
    sys.exit(1)

# Check for RDKit
try:
    from rdkit import Chem
    from rdkit import DataStructs
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
    metadata_file = "bbbc021_df_all.csv"
    image_size = 96
    
    # Architecture
    base_model_id = "google/ddpm-cifar10-32"
    fingerprint_dim = 1024  # Size of Drug Vector (Morgan Fingerprint)
    
    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training / PPO
    lr = 1e-5             
    batch_size = 16       
    ppo_minibatch_size = 8 
    
    # PPO Specifics
    ppo_epochs = 4        
    ppo_clip_range = 0.1  
    kl_beta = 0.1         # Constraint to keep model close to marginals
    rollout_steps = 50    # Steps for PPO rollout (faster than full timesteps)
    ppo_traj_stride = 5   # Store every k-th timestep to reduce memory (1=all, 5=every 5th)
    cond_drop_prob = 0.1  # For Classifier-Free Guidance training
    cond_drop_prob_img = None  # If None, uses cond_drop_prob for both. If set, separate prob for image
    cond_drop_prob_drug = None  # If None, uses cond_drop_prob for both. If set, separate prob for drug
    guidance_scale = 1.0
    
    # Reward Estimation (DDMEC)
    reward_n_terms = 16   # Timesteps to sample for likelihood est
    reward_mc = 2         # Monte Carlo samples
    
    # Paths (Unconditional Checkpoints)
    theta_pretrained = "./ddpm_uncond_perturbed/checkpoints/latest.pt" # Trained on Treated
    phi_pretrained = "./ddpm_uncond_control/checkpoints/latest.pt"     # Trained on Control
    output_dir = "ddpm_ddmec_drug_results"
    
    # Evaluation & Logging
    eval_every = 10           # Evaluate FID/KID every N iterations
    eval_steps = 50           # Number of diffusion steps for evaluation
    eval_samples = 5000       # Number of samples for FID/KID
    save_checkpoint_every = 50  # Save checkpoint every N iterations
    log_every = 1             # Log metrics every N iterations
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATASET & ENCODER (Drug Signal Logic)
# ============================================================================

class MorganFingerprintEncoder:
    """Encodes SMILES strings into Fixed Size Bit Vectors"""
    def __init__(self, n_bits=1024):
        self.n_bits = n_bits
        self.cache = {}

    def encode(self, smiles, cpd_name=None):
        """
        Encode SMILES to fingerprint. Use cpd_name to identify null drugs (DMSO).
        """
        cache_key = f"{smiles}_{cpd_name}" if cpd_name else smiles
        if cache_key in self.cache: 
            return self.cache[cache_key]
        
        # Default empty/zero for missing or DMSO (check CPD_NAME, not SMILES)
        # DMSO can have SMILES like "CS(=O)C", so we check the compound name
        is_null_drug = (cpd_name == 'DMSO' if cpd_name else False) or not smiles or not RDKIT_AVAILABLE
        
        if is_null_drug:
            # DMSO is the "Null" drug, represented as zero vector
            arr = np.zeros((self.n_bits,), dtype=np.float32)
            self.cache[cache_key] = arr
            return arr

        try:
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
            arr = np.zeros((self.n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            self.cache[cache_key] = arr
            return arr
        except:
            # Fallback for errors
            print(f"Warning: Failed to encode {smiles} (CPD: {cpd_name})")
            return np.zeros((self.n_bits,), dtype=np.float32)

class BBBC021PairedDataset(Dataset):
    """
    Returns pairs: (Control Image, Treated Image, Drug Vector)
    """
    def __init__(self, data_dir, metadata_file, encoder):
        self.data_dir = Path(data_dir)
        df = pd.read_csv(self.data_dir / metadata_file)
        
        # Filter for treated samples
        self.treated_df = df[df['CPD_NAME'] != 'DMSO'].reset_index(drop=True)
        
        # Optimization: Pre-group controls by batch to avoid pandas operations in __getitem__
        # Create a dict mapping BATCH -> list of image paths (or SAMPLE_KEYs if path not present)
        # We assume image_path is the better key if available
        self.image_col = 'image_path' if 'image_path' in df.columns else 'SAMPLE_KEY'
        
        self.controls_by_batch = (
            df[df['CPD_NAME'] == 'DMSO']
            .groupby('BATCH')[self.image_col]
            .apply(list)
            .to_dict()
        )
        
        self.encoder = encoder
        self.fingerprints = {}
        
        # Pre-compute fingerprints (pass CPD_NAME to correctly identify null drugs)
        unique_cpds = df[['CPD_NAME', 'SMILES']].drop_duplicates()
        for _, row in unique_cpds.iterrows():
            self.fingerprints[row['CPD_NAME']] = self.encoder.encode(
                row['SMILES'], 
                cpd_name=row['CPD_NAME']
            )

    def __len__(self):
        return len(self.treated_df)

    def load_image(self, path):
        full_path = self.data_dir / path
        img = np.load(full_path)
        img = torch.from_numpy(img).float()

        # Ensure channel-first
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)

        # Normalize robustly to [-1, 1]
        if img.max() <= 1.0:
            img = img * 2.0 - 1.0
        else:
            img = (img / 127.5) - 1.0

        return img

    def __getitem__(self, idx):
        # 1. Get Treated Sample
        row = self.treated_df.iloc[idx]
        trt_img = self.load_image(row[self.image_col])
        
        # 2. Get Paired Control (Same Batch) - Optimized
        batch = row['BATCH']
        ctrl_paths = self.controls_by_batch.get(batch, [])
        
        if len(ctrl_paths) > 0:
            # Fast random selection from list
            ctrl_path = np.random.choice(ctrl_paths)
            ctrl_img = self.load_image(ctrl_path)
        else:
            # Fallback if no control in batch
            ctrl_img = torch.zeros_like(trt_img)

        # 3. Get Drug Vector
        fp = self.fingerprints.get(row['CPD_NAME'])
        
        return {
            'control': ctrl_img,
            'perturbed': trt_img,
            'fingerprint': torch.from_numpy(fp).float(),
            'cpd_name': row['CPD_NAME']
        }

# ============================================================================
# CONDITIONAL MODEL (Drug + Image)
# ============================================================================

class ConditionalUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        
        # Load base architecture
        # We use a standard UNet but modified for 6 channels input
        unet_pre = UNet2DModel.from_pretrained(Config.base_model_id)
        
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=6, # 3 for Noisy Input + 3 for Conditioning Image
            out_channels=3,
            layers_per_block=unet_pre.config.layers_per_block,
            block_out_channels=unet_pre.config.block_out_channels,
            down_block_types=unet_pre.config.down_block_types,
            up_block_types=unet_pre.config.up_block_types,
            attention_head_dim=unet_pre.config.attention_head_dim,
            class_embed_type="identity" # Allows passing embeddings directly
        )

        # Drug Embedding Projection
        # Projects 1024-bit fingerprint to the Model's Time Embedding Dimension
        time_embed_dim = self.unet.time_embedding.linear_1.out_features
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.SiLU(),
            nn.Linear(512, time_embed_dim)
        )

    def forward(self, x, t, condition_img, fingerprint, drop_cond=False, drop_img=None, drop_drug=None):
        """
        x: Noisy Image [B, 3, H, W]
        t: Timestep
        condition_img: The source domain image [B, 3, H, W]
        fingerprint: Drug vector [B, 1024]
        drop_cond: bool or Tensor [B] - if Tensor, per-sample dropout mask (drops both)
        drop_img: Optional Tensor [B] - separate mask for image conditioning
        drop_drug: Optional Tensor [B] - separate mask for drug conditioning
        """
        # 1. Conditioning Dropout (for Classifier Free Guidance)
        # Support separate dropout for image and drug, or combined dropout
        
        # If separate masks provided, use them
        if drop_img is not None and drop_drug is not None:
            # Separate dropout for image and drug
            drop_mask_img = drop_img.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
            condition_img = condition_img * (~drop_mask_img).float()
            
            emb = self.fingerprint_proj(fingerprint)
            drop_mask_drug_1d = drop_drug.view(-1, 1)  # [B, 1] for embedding
            emb = emb * (~drop_mask_drug_1d).float()
        elif isinstance(drop_cond, torch.Tensor) and drop_cond.dtype == torch.bool:
            # Per-sample dropout: drop conditioning for specific samples (both together)
            drop_mask = drop_cond.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
            condition_img = condition_img * (~drop_mask).float()
            
            # Project drug vector, then zero out dropped samples
            emb = self.fingerprint_proj(fingerprint)
            drop_mask_1d = drop_cond.view(-1, 1)  # [B, 1] for embedding
            emb = emb * (~drop_mask_1d).float()
        elif drop_cond:
            # Batch-level: drop all conditioning
            condition_img = torch.zeros_like(condition_img)
            emb = torch.zeros((x.shape[0], self.fingerprint_proj[2].out_features), 
                              device=x.device, dtype=x.dtype)
        else:
            # No dropout: use all conditioning
            emb = self.fingerprint_proj(fingerprint)

        # 2. Channel Concatenation (Spatial Conditioning)
        x_in = torch.cat([x, condition_img], dim=1) # [B, 6, H, W]

        # 3. U-Net Forward (Drug passed as class_labels)
        # Diffusers adds class_labels output to the time embedding internally
        return self.unet(x_in, t, class_labels=emb).sample

class DDMECModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.model = ConditionalUNet(config.image_size, config.fingerprint_dim).to(config.device)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            prediction_type="epsilon"
        )
        self.timesteps = config.timesteps

    def load_pretrained_unconditional(self, path):
        """
        Smart loading: Loads 3-channel unconditional weights into 6-channel conditional model.
        Initializes new channels (conditioning image) to zero and randomizes drug projection.
        """
        print(f"Loading unconditional weights from {path}...")
        ckpt = torch.load(path, map_location=self.cfg.device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt

        # Filter out keys that don't match (specifically conv_in)
        model_dict = self.model.unet.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'conv_in' not in k}
        
        # Load compatible weights
        self.model.unet.load_state_dict(filtered_dict, strict=False)

        # Handle conv_in manually (Expand 3ch -> 6ch)
        if 'conv_in.weight' in state_dict:
            old_w = state_dict['conv_in.weight'] # [32, 3, 3, 3]
            new_w = self.model.unet.conv_in.weight # [32, 6, 3, 3]
            with torch.no_grad():
                new_w[:, :3, :, :] = old_w # Copy weights for noisy input
                new_w[:, 3:, :, :] = 0.0   # Zero init for conditioning input
                if 'conv_in.bias' in state_dict:
                    self.model.unet.conv_in.bias.copy_(state_dict['conv_in.bias'])
            print("  - Expanded conv_in weights 3->6 channels (zero-initialized new channels).")
        
        print("  - Fingerprint projection layer initialized randomly.")

    def forward(self, x0, condition_img, fingerprint, drop_cond=None):
        # Training Loss Calculation
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.cfg.device).long()
        noise = torch.randn_like(x0)
        
        xt = self.noise_scheduler.add_noise(x0, noise, t)
        
        # Classifier-free guidance: per-sample dropout during training
        # Support separate dropout probabilities for image and drug
        drop_img = None
        drop_drug = None
        
        if drop_cond is None:
            # Independent dropout logic (robust fallback)
            p_img = self.cfg.cond_drop_prob_img if self.cfg.cond_drop_prob_img is not None else self.cfg.cond_drop_prob
            p_drug = self.cfg.cond_drop_prob_drug if self.cfg.cond_drop_prob_drug is not None else self.cfg.cond_drop_prob
            
            drop_img = torch.rand(b, device=self.cfg.device) < p_img
            drop_drug = torch.rand(b, device=self.cfg.device) < p_drug
        
        # Pass per-sample dropout mask(s) to model
        noise_pred = self.model(xt, t, condition_img, fingerprint, 
                                drop_cond=drop_cond, drop_img=drop_img, drop_drug=drop_drug)
        
        return F.mse_loss(noise_pred, noise)

# ============================================================================
# DDMEC / PPO LOGIC
# ============================================================================

@torch.no_grad()
def rollout_with_logprobs(model, cond_img, fingerprint, steps=None, record_traj=True):
    """
    Generates a sample and records trajectory data for PPO.
    """
    model.model.eval()
    scheduler = model.noise_scheduler
    device = model.cfg.device
    b, c, h, w = cond_img.shape

    x = torch.randn((b, 3, h, w), device=device)

    inference_steps = steps if steps else model.timesteps
    scheduler.set_timesteps(inference_steps, device=device)

    # Make sure scheduler buffers are on the same device for indexing/broadcasting
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    betas = scheduler.betas.to(device)
    alphas = scheduler.alphas.to(device)

    traj = []
    timesteps = scheduler.timesteps  # usually a 1D tensor, descending

    for i, t in enumerate(timesteps):
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        prev_t_int = int(timesteps[i + 1].item()) if (i + 1) < len(timesteps) else -1

        t_batch = torch.full((b,), t_int, device=device, dtype=torch.long)

        # Classifier-Free Guidance (CFG) Sampling
        guidance_scale = model.cfg.guidance_scale
        
        if guidance_scale != 1.0:
            # Conditional prediction
            eps_cond = model.model(x, t_batch, cond_img, fingerprint, drop_cond=False)
            
            # Unconditional prediction
            eps_uncond = model.model(x, t_batch, 
                                    torch.zeros_like(cond_img), 
                                    torch.zeros_like(fingerprint), 
                                    drop_cond=True)
            
            # CFG combination: eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            # No guidance: use conditional prediction only
            eps = model.model(x, t_batch, cond_img, fingerprint, drop_cond=False)

        # Scalars -> broadcastable [1,1,1,1]
        alpha_prod_t = alphas_cumprod[t_int].view(1, 1, 1, 1)
        alpha_prod_prev = (alphas_cumprod[prev_t_int].view(1, 1, 1, 1)
                           if prev_t_int >= 0 else torch.ones_like(alpha_prod_t))

        # Predict x0 from epsilon
        # pred_x0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_x0 = (x - (1.0 - alpha_prod_t).sqrt() * eps) / alpha_prod_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)

        # Correct Posterior for arbitrary stride (t -> prev_t)
        # Using Diffusers/DDPM formulation
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev = 1 - alpha_prod_prev
        
        # Effective beta for current step
        current_alpha_t = alpha_prod_t / alpha_prod_prev
        current_beta_t = 1 - current_alpha_t
        
        # Mean coefficients
        pred_original_sample_coeff = (alpha_prod_prev.sqrt() * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t.sqrt() * beta_prod_prev) / beta_prod_t
        
        mean = pred_original_sample_coeff * pred_x0 + current_sample_coeff * x
        
        # Variance
        var = (beta_prod_prev / beta_prod_t) * current_beta_t
        var = var.clamp(min=1e-20)
        std = var.sqrt()

        # Fix 1: Noise/Step consistency (only add noise if transitioning to non-zero step)
        # RE-FIX: Standard DDPM rule: add noise iff current timestep t_int > 0
        noise = torch.randn_like(x) if t_int > 0 else torch.zeros_like(x)
        x_prev = mean + std * noise

        # Memory optimization: store only every k-th timestep (reduce memory usage)
        # CRITICAL FIX: Skip the final deterministic step (prev_t_int == -1) to avoid PPO instability
        stride = model.cfg.ppo_traj_stride
        
        # Robust Logic: only compute log_prob when we actually store the step
        if record_traj and prev_t_int >= 0 and (i % stride == 0):
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(x_prev).sum(dim=(1, 2, 3))
            
            traj.append({
                # Optimization: Store as float16 on CPU to save memory
                "x_t": x.detach().to(dtype=torch.float16).cpu(),
                "x_prev": x_prev.detach().to(dtype=torch.float16).cpu(),
                "t": t_batch.detach().cpu(),
                "prev_t": torch.full((b,), prev_t_int, dtype=torch.long).cpu(),
                "old_logprob": log_prob.detach().cpu(), # keep fp32 for precision
            })

        x = x_prev

    return x, traj

@torch.no_grad()
def reward_negloglik(other_model, target_img, cond_img, fingerprint):
    other_model.model.eval()
    scheduler = other_model.noise_scheduler
    device = other_model.cfg.device
    b = target_img.shape[0]

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    n_terms = other_model.cfg.reward_n_terms
    total = torch.zeros((b,), device=device)

    for _ in range(other_model.cfg.reward_mc):
        t_samples = torch.randint(0, scheduler.config.num_train_timesteps, (n_terms, b), device=device)

        for k in range(n_terms):
            t = t_samples[k]                              # [b] on device
            noise = torch.randn_like(target_img)           # [b,3,H,W]
            y_t = scheduler.add_noise(target_img, noise, t)

            eps_pred = other_model.model(y_t, t, cond_img, fingerprint, drop_cond=False)

            mse = ((noise - eps_pred) ** 2).mean(dim=(1,2,3))   # [b]
            alpha = alphas_cumprod[t]                            # [b]
            snr = alpha / (1 - alpha + 1e-8)                     # [b]
            total += snr * mse

    return -total / (n_terms * other_model.cfg.reward_mc)

def ppo_update(model, ref_model, optimizer, batch, config):
    """
    Standard PPO update with KL penalty vs Reference Model
    """
    model.model.train()
    device = config.device
    scheduler = model.noise_scheduler

    # Cast back to float32 on device
    x_t = batch['x_t'].to(device, dtype=torch.float32)
    x_prev = batch['x_prev'].to(device, dtype=torch.float32)
    t = batch['t'].to(device).long()
    prev_t = batch['prev_t'].to(device).long()
    old_logprob = batch['old_logprob'].to(device)
    adv = batch['adv'].to(device)
    cond_img = batch['cond_img'].to(device)
    fp = batch['fingerprint'].to(device)

    # buffers on device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    eps = model.model(x_t, t, cond_img, fp, drop_cond=False)

    # Compute mean/std per-sample timestep (vectorized indexing)
    alpha_prod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_prod_prev = torch.where(
        (prev_t >= 0).view(-1, 1, 1, 1),
        alphas_cumprod[torch.clamp(prev_t, min=0)].view(-1, 1, 1, 1),
        torch.ones_like(alpha_prod_t),
    )

    pred_x0 = (x_t - (1.0 - alpha_prod_t).sqrt() * eps) / alpha_prod_t.sqrt()
    pred_x0 = pred_x0.clamp(-1, 1)

    # Correct Posterior for Strided Steps (Match rollout)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev = 1 - alpha_prod_prev
    current_alpha_t = alpha_prod_t / alpha_prod_prev
    current_beta_t = 1 - current_alpha_t
    
    pred_original_sample_coeff = (alpha_prod_prev.sqrt() * current_beta_t) / beta_prod_t
    current_sample_coeff = (current_alpha_t.sqrt() * beta_prod_prev) / beta_prod_t
    
    mean = pred_original_sample_coeff * pred_x0 + current_sample_coeff * x_t
    
    var = (beta_prod_prev / beta_prod_t) * current_beta_t
    var = var.clamp(min=1e-20)
    std = var.sqrt()

    dist = torch.distributions.Normal(mean, std)
    new_logprob = dist.log_prob(x_prev).sum(dim=(1, 2, 3))

    ratio = (new_logprob - old_logprob).exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip_range, 1.0 + config.ppo_clip_range) * adv
    ppo_loss = -torch.min(surr1, surr2).mean()

    # --- Reference Model Distribution (FIXED: Use CONDITIONAL reference, not unconditional) ---
    # This computes KL(π_θ || π_ref) where both are CONDITIONAL on the same inputs
    # The reference model has frozen parameters but uses the SAME conditioning
    with torch.no_grad():
        eps_ref = ref_model.model(
            x_t, t,
            cond_img,  # FIXED: Use same conditioning (was zeros)
            fp,        # FIXED: Use same drug vector (was zeros)
            drop_cond=False  # FIXED: Keep conditioning (was True)
        )
        
        pred_x0_ref = (x_t - (1.0 - alpha_prod_t).sqrt() * eps_ref) / alpha_prod_t.sqrt()
        pred_x0_ref = pred_x0_ref.clamp(-1, 1)
        
        mean_ref = pred_original_sample_coeff * pred_x0_ref + current_sample_coeff * x_t

    # --- KL Divergence (Gaussians with same variance) ---
    # KL(P || Q) = (μ_P - μ_Q)² / (2σ²) for Gaussians with same variance
    # This is now TRUE KL between conditional distributions (policy vs frozen reference)
    kl_terms = ((mean - mean_ref)**2) / (2 * var)
    kl_loss = kl_terms.sum(dim=(1, 2, 3)).mean()

    loss = ppo_loss + config.kl_beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
    
    # Log PPO Grad Norm (Debug)
    if np.random.rand() < 0.01:
        if hasattr(model.model, 'fingerprint_proj') and model.model.fingerprint_proj[0].weight.grad is not None:
              norm = model.model.fingerprint_proj[0].weight.grad.norm().item()
              print(f"  [PPO] Drug grad norm: {norm:.6f}")

    optimizer.step()
    
    return {
        "loss": loss.item(),
        "ppo_loss": ppo_loss.item(),
        "kl": kl_loss.item(),
        "ratio_mean": ratio.mean().item(),
    }

# ============================================================================
# PPO BATCH BUILDING & ADVANTAGE COMPUTATION
# ============================================================================

def build_ppo_batch(traj, reward, cond_img, fingerprint):
    """
    Flatten trajectory into a batch and compute advantages.
    
    Args:
        traj: List of dicts from rollout_with_logprobs, each dict has tensors [B, ...]
        reward: Tensor [B] - reward per sample (assigned to all timesteps)
        cond_img: Tensor [B, C, H, W] - conditioning image
        fingerprint: Tensor [B, F] - drug fingerprint
    
    Returns:
        batch: Dict with flattened tensors ready for ppo_update
    """
    if len(traj) == 0:
        return None

    storage_device = traj[0]["x_t"].device  # should be CPU with your rollout
    reward = reward.detach().to(storage_device)
    cond_img = cond_img.detach().to(storage_device)
    fingerprint = fingerprint.detach().to(storage_device)

    # Normalize reward -> advantages (constant per timestep)
    normalized_reward = (reward - reward.mean()) / (reward.std() + 1e-8)

    x_t_list, x_prev_list, t_list, prev_t_list = [], [], [], []
    old_logprob_list, adv_list, cond_img_list, fp_list = [], [], [], []

    for step_dict in traj:
        x_t_list.append(step_dict["x_t"])
        x_prev_list.append(step_dict["x_prev"])
        t_list.append(step_dict["t"])
        prev_t_list.append(step_dict["prev_t"])
        old_logprob_list.append(step_dict["old_logprob"])

        adv_list.append(normalized_reward)   # [B]
        cond_img_list.append(cond_img)       # [B,3,H,W]
        fp_list.append(fingerprint)          # [B,F]

    batch = {
        "x_t": torch.cat(x_t_list, dim=0),
        "x_prev": torch.cat(x_prev_list, dim=0),
        "t": torch.cat(t_list, dim=0),
        "prev_t": torch.cat(prev_t_list, dim=0),
        "old_logprob": torch.cat(old_logprob_list, dim=0),
        "adv": torch.cat(adv_list, dim=0),
        "cond_img": torch.cat(cond_img_list, dim=0),
        "fingerprint": torch.cat(fp_list, dim=0),
    }
    return batch

# ============================================================================
# EVALUATION & CHECKPOINTING
# ============================================================================

@torch.no_grad()
def evaluate_fid_kid(theta, phi, dataset, config, num_samples=5000, steps=50):
    """
    Evaluate FID and KID metrics for both theta and phi models.
    Returns per-class and overall metrics.
    """
    theta.model.eval()
    phi.model.eval()
    
    # Initialize metrics
    fid_theta = FrechetInceptionDistance(normalize=True).to(config.device)
    kid_theta = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device)
    fid_phi = FrechetInceptionDistance(normalize=True).to(config.device)
    kid_phi = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device)
    
    # Per-class storage
    theta_samples_per_class = {}
    phi_samples_per_class = {}
    real_ctrl_per_class = {}
    real_trt_per_class = {}
    
    # Sample generation
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    total_generated = 0
    
    print(f"\n[EVAL] Generating {num_samples} samples for FID/KID evaluation...")
    
    for batch in tqdm(loader, desc="Evaluating"):
        if total_generated >= num_samples:
            break
            
        ctrl = batch['control'].to(config.device)
        trt = batch['perturbed'].to(config.device)
        fp = batch['fingerprint'].to(config.device)
        cpd_names = batch['cpd_name']
        
        batch_size = ctrl.shape[0]
        
        # Generate samples
        # Fix 3: Disable trajectory recording during eval for speed/memory
        x_gen, _ = rollout_with_logprobs(theta, ctrl, fp, steps=steps, record_traj=False)  # ctrl -> trt
        y_gen, _ = rollout_with_logprobs(phi, trt, fp, steps=steps, record_traj=False)     # trt -> ctrl
        
        # Normalize to [0, 1] for FID/KID
        ctrl_norm = (ctrl * 0.5 + 0.5).clamp(0, 1)
        trt_norm = (trt * 0.5 + 0.5).clamp(0, 1)
        x_gen_norm = (x_gen * 0.5 + 0.5).clamp(0, 1)
        y_gen_norm = (y_gen * 0.5 + 0.5).clamp(0, 1)
        
        # Fix 3: Convert to uint8 [0, 255] for torchmetrics
        ctrl_u8 = (ctrl_norm * 255).to(torch.uint8)
        trt_u8 = (trt_norm * 255).to(torch.uint8)
        x_gen_u8 = (x_gen_norm * 255).to(torch.uint8)
        y_gen_u8 = (y_gen_norm * 255).to(torch.uint8)
        
        # Update overall metrics
        fid_theta.update(trt_u8, real=True)
        fid_theta.update(x_gen_u8, real=False)
        kid_theta.update(trt_u8, real=True)
        kid_theta.update(x_gen_u8, real=False)
        
        fid_phi.update(ctrl_u8, real=True)
        fid_phi.update(y_gen_u8, real=False)
        kid_phi.update(ctrl_u8, real=True)
        kid_phi.update(y_gen_u8, real=False)
        
        # Store per-class samples
        for i in range(batch_size):
            cpd = cpd_names[i]
            if cpd not in theta_samples_per_class:
                theta_samples_per_class[cpd] = []
                phi_samples_per_class[cpd] = []
                real_ctrl_per_class[cpd] = []
                real_trt_per_class[cpd] = []
            
            theta_samples_per_class[cpd].append(x_gen_norm[i].cpu())
            phi_samples_per_class[cpd].append(y_gen_norm[i].cpu())
            real_ctrl_per_class[cpd].append(ctrl_norm[i].cpu())
            real_trt_per_class[cpd].append(trt_norm[i].cpu())
        
        total_generated += batch_size
    
    # Compute overall metrics
    fid_theta_val = fid_theta.compute().item()
    kid_theta_mean, kid_theta_std = kid_theta.compute()
    fid_phi_val = fid_phi.compute().item()
    kid_phi_mean, kid_phi_std = kid_phi.compute()
    
    # Compute per-class metrics
    fid_theta_per_class = {}
    kid_theta_per_class = {}
    fid_phi_per_class = {}
    kid_phi_per_class = {}
    
    print("\n[EVAL] Computing per-class metrics...")
    for cpd in tqdm(theta_samples_per_class.keys(), desc="Per-class FID/KID"):
        if len(theta_samples_per_class[cpd]) < 10:  # Skip if too few samples
            continue
        
        # Theta (ctrl -> trt)
        try:
            fid_metric_class = FrechetInceptionDistance(normalize=True).to(config.device)
            real_trt_class = torch.stack(real_trt_per_class[cpd]).to(config.device)
            gen_trt_class = torch.stack(theta_samples_per_class[cpd]).to(config.device)
            
            # Fix 2: Convert per-class batches to uint8 [0, 255]
            real_trt_u8 = (real_trt_class.clamp(0, 1) * 255).to(torch.uint8)
            gen_trt_u8 = (gen_trt_class.clamp(0, 1) * 255).to(torch.uint8)
            
            fid_metric_class.update(real_trt_u8, real=True)
            fid_metric_class.update(gen_trt_u8, real=False)
            fid_theta_per_class[cpd] = fid_metric_class.compute().item()
            
            subset_size = min(len(theta_samples_per_class[cpd]), 100)
            kid_metric_class = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(config.device)
            kid_metric_class.update(real_trt_u8, real=True)
            kid_metric_class.update(gen_trt_u8, real=False)
            kid_mean, kid_std = kid_metric_class.compute()
            kid_theta_per_class[cpd] = {"mean": kid_mean.item(), "std": kid_std.item()}
        except Exception as e:
            print(f"  [WARN] Failed to compute theta metrics for {cpd}: {e}")
        
        # Phi (trt -> ctrl)
        try:
            fid_metric_class = FrechetInceptionDistance(normalize=True).to(config.device)
            real_ctrl_class = torch.stack(real_ctrl_per_class[cpd]).to(config.device)
            gen_ctrl_class = torch.stack(phi_samples_per_class[cpd]).to(config.device)
            
            # Fix 2: Convert per-class batches to uint8 [0, 255]
            real_ctrl_u8 = (real_ctrl_class.clamp(0, 1) * 255).to(torch.uint8)
            gen_ctrl_u8 = (gen_ctrl_class.clamp(0, 1) * 255).to(torch.uint8)
            
            fid_metric_class.update(real_ctrl_u8, real=True)
            fid_metric_class.update(gen_ctrl_u8, real=False)
            fid_phi_per_class[cpd] = fid_metric_class.compute().item()
            
            subset_size = min(len(phi_samples_per_class[cpd]), 100)
            kid_metric_class = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(config.device)
            kid_metric_class.update(real_ctrl_u8, real=True)
            kid_metric_class.update(gen_ctrl_u8, real=False)
            kid_mean, kid_std = kid_metric_class.compute()
            kid_phi_per_class[cpd] = {"mean": kid_mean.item(), "std": kid_std.item()}
        except Exception as e:
            print(f"  [WARN] Failed to compute phi metrics for {cpd}: {e}")
    
    # Compute averages
    avg_fid_theta = np.mean(list(fid_theta_per_class.values())) if fid_theta_per_class else 0.0
    avg_fid_phi = np.mean(list(fid_phi_per_class.values())) if fid_phi_per_class else 0.0
    avg_kid_theta_mean = np.mean([v["mean"] for v in kid_theta_per_class.values()]) if kid_theta_per_class else 0.0
    avg_kid_phi_mean = np.mean([v["mean"] for v in kid_phi_per_class.values()]) if kid_phi_per_class else 0.0
    
    results = {
        "theta": {
            "overall_fid": fid_theta_val,
            "overall_kid_mean": kid_theta_mean.item(),
            "overall_kid_std": kid_theta_std.item(),
            "avg_fid": avg_fid_theta,
            "avg_kid_mean": avg_kid_theta_mean,
            "fid_per_class": fid_theta_per_class,
            "kid_per_class": kid_theta_per_class,
        },
        "phi": {
            "overall_fid": fid_phi_val,
            "overall_kid_mean": kid_phi_mean.item(),
            "overall_kid_std": kid_phi_std.item(),
            "avg_fid": avg_fid_phi,
            "avg_kid_mean": avg_kid_phi_mean,
            "fid_per_class": fid_phi_per_class,
            "kid_per_class": kid_phi_per_class,
        }
    }
    
    print(f"\n[EVAL] Theta (ctrl->trt): FID={fid_theta_val:.2f}, KID={kid_theta_mean.item():.4f}±{kid_theta_std.item():.4f}")
    print(f"[EVAL] Phi (trt->ctrl): FID={fid_phi_val:.2f}, KID={kid_phi_mean.item():.4f}±{kid_phi_std.item():.4f}")
    
    return results

def save_checkpoint(theta, phi, theta_opt, phi_opt, iteration, metrics_history, config):
    """Save training checkpoint"""
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Fix 4: Truncate metrics history in checkpoint to avoid blow-up
    # Keep only the last 1000 entries (full history is in log file/json)
    if len(metrics_history) > 1000:
        sorted_keys = sorted(metrics_history.keys())[-1000:]
        metrics_small = {k: metrics_history[k] for k in sorted_keys}
    else:
        metrics_small = metrics_history

    checkpoint = {
        "iteration": iteration,
        "theta_state_dict": theta.model.state_dict(),
        "phi_state_dict": phi.model.state_dict(),
        "theta_opt_state_dict": theta_opt.state_dict(),
        "phi_opt_state_dict": phi_opt.state_dict(),
        "metrics_history": metrics_small,
        # Fix 3: Correctly serialize Config class attributes
        "config": {k: getattr(config, k) for k in dir(config)
                   if not k.startswith("__") and not callable(getattr(config, k))},
    }
    
    # Save latest
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    
    # Save numbered checkpoint
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_iter_{iteration}.pt")
    
    print(f"[CHECKPOINT] Saved at iteration {iteration}")

def load_checkpoint(checkpoint_path, theta, phi, theta_opt, phi_opt, config):
    """Load training checkpoint"""
    print(f"[CHECKPOINT] Loading from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    theta.model.load_state_dict(checkpoint["theta_state_dict"])
    phi.model.load_state_dict(checkpoint["phi_state_dict"])
    theta_opt.load_state_dict(checkpoint["theta_opt_state_dict"])
    phi_opt.load_state_dict(checkpoint["phi_opt_state_dict"])
    
    iteration = checkpoint["iteration"]
    metrics_history = checkpoint.get("metrics_history", {})
    
    print(f"[CHECKPOINT] Resumed from iteration {iteration}")
    return iteration, metrics_history

def plot_metrics(metrics_history, output_dir):
    """Plot training metrics"""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    iterations = sorted(metrics_history.keys())
    
    # Rewards
    r_theta = [metrics_history[i].get("r_theta_mean", 0) for i in iterations]
    r_phi = [metrics_history[i].get("r_phi_mean", 0) for i in iterations]
    
    # Losses
    loss_theta_ppo = [metrics_history[i].get("loss_theta_ppo", 0) for i in iterations]
    loss_phi_ppo = [metrics_history[i].get("loss_phi_ppo", 0) for i in iterations]
    loss_theta_joint = [metrics_history[i].get("loss_theta_joint", 0) for i in iterations]
    loss_phi_joint = [metrics_history[i].get("loss_phi_joint", 0) for i in iterations]
    
    # KL
    kl_theta = [metrics_history[i].get("kl_theta", 0) for i in iterations]
    kl_phi = [metrics_history[i].get("kl_phi", 0) for i in iterations]
    
    # FID/KID (sparse)
    fid_theta = [(i, metrics_history[i]["fid_theta"]) for i in iterations if "fid_theta" in metrics_history[i]]
    fid_phi = [(i, metrics_history[i]["fid_phi"]) for i in iterations if "fid_phi" in metrics_history[i]]
    kid_theta = [(i, metrics_history[i]["kid_theta_mean"]) for i in iterations if "kid_theta_mean" in metrics_history[i]]
    kid_phi = [(i, metrics_history[i]["kid_phi_mean"]) for i in iterations if "kid_phi_mean" in metrics_history[i]]
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Rewards
    axes[0, 0].plot(iterations, r_theta, label="Theta", alpha=0.7)
    axes[0, 0].plot(iterations, r_phi, label="Phi", alpha=0.7)
    axes[0, 0].set_title("Rewards")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PPO Losses
    axes[0, 1].plot(iterations, loss_theta_ppo, label="Theta PPO", alpha=0.7)
    axes[0, 1].plot(iterations, loss_phi_ppo, label="Phi PPO", alpha=0.7)
    axes[0, 1].set_title("PPO Losses")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Joint Losses
    axes[1, 0].plot(iterations, loss_theta_joint, label="Theta Joint", alpha=0.7)
    axes[1, 0].plot(iterations, loss_phi_joint, label="Phi Joint", alpha=0.7)
    axes[1, 0].set_title("Joint Constraint Losses")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[1, 1].plot(iterations, kl_theta, label="Theta KL", alpha=0.7)
    axes[1, 1].plot(iterations, kl_phi, label="Phi KL", alpha=0.7)
    axes[1, 1].set_title("KL Divergence")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("KL")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # FID
    if fid_theta:
        fid_theta_iters, fid_theta_vals = zip(*fid_theta)
        axes[2, 0].plot(fid_theta_iters, fid_theta_vals, 'o-', label="Theta", alpha=0.7)
    if fid_phi:
        fid_phi_iters, fid_phi_vals = zip(*fid_phi)
        axes[2, 0].plot(fid_phi_iters, fid_phi_vals, 's-', label="Phi", alpha=0.7)
    axes[2, 0].set_title("FID Score")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("FID")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # KID
    if kid_theta:
        kid_theta_iters, kid_theta_vals = zip(*kid_theta)
        axes[2, 1].plot(kid_theta_iters, kid_theta_vals, 'o-', label="Theta", alpha=0.7)
    if kid_phi:
        kid_phi_iters, kid_phi_vals = zip(*kid_phi)
        axes[2, 1].plot(kid_phi_iters, kid_phi_vals, 's-', label="Phi", alpha=0.7)
    axes[2, 1].set_title("KID Score")
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].set_ylabel("KID")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "training_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[PLOT] Saved training metrics to {plots_dir / 'training_metrics.png'}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Fix 3: Enforce guidance_scale == 1.0 for PPO
    assert config.guidance_scale == 1.0, "Set guidance_scale=1.0 during PPO training; use CFG only at eval."

    # 1. Initialize Models
    # Theta: Control -> Perturbed (using Drug)
    theta = DDMECModel(config)
    theta.load_pretrained_unconditional(config.theta_pretrained) # Load weights trained on Perturbed
    theta_ref = copy.deepcopy(theta) # Reference for KL constraint
    
    # Phi: Perturbed -> Control (using Drug)
    phi = DDMECModel(config)
    phi.load_pretrained_unconditional(config.phi_pretrained) # Load weights trained on Control
    phi_ref = copy.deepcopy(phi)
    
    # Fix 2: Freeze + eval() reference models
    theta_ref.model.eval()
    phi_ref.model.eval()
    for p in theta_ref.model.parameters():
        p.requires_grad_(False)
    for p in phi_ref.model.parameters():
        p.requires_grad_(False)

    theta_opt = torch.optim.AdamW(theta.model.parameters(), lr=config.lr)
    phi_opt = torch.optim.AdamW(phi.model.parameters(), lr=config.lr)
    
    # 2. Data
    encoder = MorganFingerprintEncoder()
    dataset = BBBC021PairedDataset(config.data_dir, config.metadata_file, encoder)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    print("Starting DDMEC-PPO with Drug Conditioning...")
    
    # Fix 5: Iteration counter and eval/checkpointing
    iteration = 0
    metrics_history = {}

    for epoch in range(100): 
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            iteration += 1
            
            ctrl = batch['control'].to(config.device)
            trt = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            # --- PHASE 1: Update Theta with PPO (Algorithm 1) ---
            # 1. Generate Fake Perturbed (x_gen) from Control + Drug
            x_gen, traj_theta = rollout_with_logprobs(theta, ctrl, fp, steps=config.rollout_steps)
            
            # 2. Calculate Reward using Phi
            # Phi estimates: P(Control | x_gen, Drug)
            # High prob = x_gen looks like a perturbed state that corresponds to this control/drug pair
            # Fix 2: Detach generated images to prevent accidental gradients/memory leak
            r_theta = reward_negloglik(phi, target_img=ctrl, cond_img=x_gen.detach(), fingerprint=fp)
            
            # 3. PPO Update Theta (reward + marginal constraint)
            loss_theta_ppo = 0
            kl_theta_last = 0
            if len(traj_theta) > 0:
                ppo_batch_theta = build_ppo_batch(traj_theta, r_theta, ctrl, fp)
                
                if ppo_batch_theta is not None:
                    # PPO epochs with minibatches
                    total_samples = ppo_batch_theta['x_t'].shape[0]
                    theta_stats = []
                    
                    for ppo_epoch in range(config.ppo_epochs):
                        # Reshuffle for each PPO epoch
                        indices = torch.randperm(total_samples, device=ppo_batch_theta['x_t'].device)  # CPU
                        
                        for start_idx in range(0, total_samples, config.ppo_minibatch_size):
                            end_idx = min(start_idx + config.ppo_minibatch_size, total_samples)
                            batch_indices = indices[start_idx:end_idx]
                            
                            minibatch = {
                                k: v[batch_indices] for k, v in ppo_batch_theta.items()
                            }
                            
                            stats = ppo_update(theta, theta_ref, theta_opt, minibatch, config)
                            theta_stats.append(stats)
                    
                    if theta_stats:
                        loss_theta_ppo = np.mean([s["loss"] for s in theta_stats])
                        kl_theta_last = np.mean([s["kl"] for s in theta_stats])
            
            # --- PHASE 2: Update Phi with Joint Constraint (Algorithm 2) ---
            # After theta generates x_gen, update phi to agree: phi should map x_gen -> ctrl
            phi_opt.zero_grad(set_to_none=True)
            # Fix 1: Force conditioning ON during joint training check
            loss_phi_joint = phi(ctrl, x_gen.detach(), fp, drop_cond=False)  # phi: learn to reconstruct ctrl given condition x_gen
            
            # Optimization: Add small unconditional loss (5%) allow CFG at eval
            loss_phi_uncond = phi(ctrl, torch.zeros_like(x_gen), torch.zeros_like(fp), drop_cond=True)
            loss_phi_joint_total = loss_phi_joint + 0.05 * loss_phi_uncond
            
            loss_phi_joint_total.backward()
            
            # Fix 4: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(phi.model.parameters(), 1.0)
            
            # Log Grad Norm immediately (sampled)
            if np.random.rand() < 0.05:
                if phi.model.fingerprint_proj[0].weight.grad is not None:
                     norm = phi.model.fingerprint_proj[0].weight.grad.norm().item()
                     print(f"  [JOINT] Phi drug grad norm: {norm:.6f}")
                     
            phi_opt.step()
            
            # --- PHASE 3: Update Phi with PPO (Algorithm 1) ---
            # 1. Generate Fake Control (y_gen) from Perturbed + Drug
            y_gen, traj_phi = rollout_with_logprobs(phi, trt, fp, steps=config.rollout_steps)
            
            # 2. Calculate Reward using Theta
            # Theta estimates: P(Perturbed | y_gen, Drug)
            # Fix 2: Detach generated images
            r_phi = reward_negloglik(theta, target_img=trt, cond_img=y_gen.detach(), fingerprint=fp)
            
            # 3. PPO Update Phi (reward + marginal constraint)
            loss_phi_ppo = 0
            kl_phi_last = 0
            if len(traj_phi) > 0:
                ppo_batch_phi = build_ppo_batch(traj_phi, r_phi, trt, fp)
                
                if ppo_batch_phi is not None:
                    # PPO epochs with minibatches
                    total_samples = ppo_batch_phi['x_t'].shape[0]
                    phi_stats = []
                    
                    for ppo_epoch in range(config.ppo_epochs):
                        # Reshuffle for each PPO epoch
                        indices = torch.randperm(total_samples, device=ppo_batch_phi['x_t'].device)  # CPU
                        
                        for start_idx in range(0, total_samples, config.ppo_minibatch_size):
                            end_idx = min(start_idx + config.ppo_minibatch_size, total_samples)
                            batch_indices = indices[start_idx:end_idx]
                            
                            minibatch = {
                                k: v[batch_indices] for k, v in ppo_batch_phi.items()
                            }
                            
                            stats = ppo_update(phi, phi_ref, phi_opt, minibatch, config)
                            phi_stats.append(stats)
                    
                    if phi_stats:
                        loss_phi_ppo = np.mean([s["loss"] for s in phi_stats])
                        kl_phi_last = np.mean([s["kl"] for s in phi_stats])
            
            # --- PHASE 4: Update Theta with Joint Constraint (Algorithm 2) ---
            # After phi generates y_gen, update theta to agree: theta should map y_gen -> trt
            theta_opt.zero_grad(set_to_none=True)
            # Fix 1: Force conditioning ON during joint training check
            loss_theta_joint = theta(trt, y_gen.detach(), fp, drop_cond=False)
            
            # Optimization: Add small unconditional loss (5%) allow CFG at eval
            loss_theta_uncond = theta(trt, torch.zeros_like(y_gen), torch.zeros_like(fp), drop_cond=True)
            loss_theta_joint_total = loss_theta_joint + 0.05 * loss_theta_uncond
            
            loss_theta_joint_total.backward()
            
            # Fix 4: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(theta.model.parameters(), 1.0)
            
            # Log Grad Norm immediately (sampled)
            if np.random.rand() < 0.05:
                if theta.model.fingerprint_proj[0].weight.grad is not None:
                     norm = theta.model.fingerprint_proj[0].weight.grad.norm().item()
                     print(f"  [JOINT] Theta drug grad norm: {norm:.6f}")
                     
            theta_opt.step()
            
            # Logging
            if iteration % config.log_every == 0:
                print(f"Iter {iteration} | R_Theta: {r_theta.mean().item():.3f} | R_Phi: {r_phi.mean().item():.3f}")
                metrics_history[iteration] = {
                    "r_theta_mean": r_theta.mean().item(),
                    "r_phi_mean": r_phi.mean().item(),
                    "r_phi_mean": r_phi.mean().item(),
                    "loss_theta_ppo": loss_theta_ppo,
                    "loss_phi_ppo": loss_phi_ppo,
                    "loss_theta_joint": loss_theta_joint.item(),
                    "loss_phi_joint": loss_phi_joint.item(),
                    "kl_theta": kl_theta_last if "kl_theta_last" in locals() else 0.0,
                    "kl_phi": kl_phi_last if "kl_phi_last" in locals() else 0.0,
                }

            # Checkpointing
            if iteration % config.save_checkpoint_every == 0:
                save_checkpoint(theta, phi, theta_opt, phi_opt, iteration, metrics_history, config)
                try:
                    plot_metrics(metrics_history, config.output_dir)
                except Exception as e:
                    print(f"Plotting failed: {e}")

            # Evaluation
            if iteration % config.eval_every == 0:
                results = evaluate_fid_kid(theta, phi, dataset, config,
                                           num_samples=config.eval_samples,
                                           steps=config.eval_steps)
                                           
                # Store full metrics history
                metrics_history[iteration]["fid_theta"] = results["theta"]["overall_fid"]
                metrics_history[iteration]["kid_theta_mean"] = results["theta"]["overall_kid_mean"]
                metrics_history[iteration]["fid_phi"] = results["phi"]["overall_fid"]
                metrics_history[iteration]["kid_phi_mean"] = results["phi"]["overall_kid_mean"]
                
                # Also save detailed results to file
                with open(f"{config.output_dir}/eval_{iteration}.json", 'w') as f:
                    # Convert tensors/numpy to native types for JSON
                    json.dump(results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

if __name__ == "__main__":
    main()