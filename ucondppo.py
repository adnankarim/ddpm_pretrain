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
    rollout_steps = 1000  
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
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
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
        
        # Group controls by batch for pairing
        self.controls_by_batch = df[df['CPD_NAME'] == 'DMSO'].groupby('BATCH')
        
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
        img_key = 'image_path' if 'image_path' in row.index else 'SAMPLE_KEY'
        trt_img = self.load_image(row[img_key])
        
        # 2. Get Paired Control (Same Batch)
        batch = row['BATCH']
        if batch in self.controls_by_batch.groups:
            ctrl_row = self.controls_by_batch.get_group(batch).sample(1).iloc[0]
            ctrl_img = self.load_image(ctrl_row[img_key] if img_key in ctrl_row.index else ctrl_row['SAMPLE_KEY'])
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
def rollout_with_logprobs(model, cond_img, fingerprint, steps=None):
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

        eps = model.model(x, t_batch, cond_img, fingerprint, drop_cond=False)

        # Scalars -> broadcastable [1,1,1,1]
        alpha_prod_t = alphas_cumprod[t_int].view(1, 1, 1, 1)
        alpha_prod_prev = (alphas_cumprod[prev_t_int].view(1, 1, 1, 1)
                           if prev_t_int >= 0 else torch.ones_like(alpha_prod_t))
        beta_t = betas[t_int].view(1, 1, 1, 1)
        alpha_t = alphas[t_int].view(1, 1, 1, 1)

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

        noise = torch.randn_like(x) if t_int > 0 else torch.zeros_like(x)
        x_prev = mean + std * noise

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(x_prev).sum(dim=(1, 2, 3))

        # Memory optimization: store only every k-th timestep (reduce memory usage)
        # CRITICAL FIX: Skip the final deterministic step (prev_t_int == -1) to avoid PPO instability
        stride = model.cfg.ppo_traj_stride
        if prev_t_int >= 0 and (i % stride == 0):
            traj.append({
                "x_t": x.detach().cpu(),
                "x_prev": x_prev.detach().cpu(),
                "t": t_batch.detach().cpu(),
                "prev_t": torch.full((b,), prev_t_int, dtype=torch.long).cpu(),
                "old_logprob": log_prob.detach().cpu(),
            })

        x = x_prev

    return x, traj

@torch.no_grad()
def reward_negloglik(other_model, target_img, cond_img, fingerprint):
    """
    DDMEC Reward: -log P_phi(Target | Generated_Condition, Drug)
    Used to train Theta.
    target_img: Real Control Image
    cond_img: Generated Perturbed Image
    fingerprint: Drug Vector
    """
    other_model.model.eval()
    scheduler = other_model.noise_scheduler
    device = other_model.cfg.device
    b = target_img.shape[0]
    
    # Monte Carlo estimation of likelihood (Eq 8 in paper)
    # We estimate likelihood by calculating MSE at random timesteps
    n_samples = other_model.cfg.reward_n_terms
    total_mse = torch.zeros((b,), device=device)
    
    for _ in range(other_model.cfg.reward_mc):
        t_samples = torch.randint(0, scheduler.config.num_train_timesteps, (n_samples, b), device=device)
        
        for k in range(n_samples):
            t = t_samples[k]
            noise = torch.randn_like(target_img)
            
            # Add noise to the Target (Control)
            y_t = scheduler.add_noise(target_img, noise, t)
            
            # Predict noise using Other Model (Phi)
            # Phi inputs: (Noisy Control, Time, Condition=Perturbed, Class=Drug)
            eps_pred = other_model.model(y_t, t, cond_img, fingerprint, drop_cond=False)
            
            total_mse += ((noise - eps_pred)**2).mean(dim=(1,2,3))
            
    reward = -1.0 * (total_mse / (n_samples * other_model.cfg.reward_mc))
    return reward

def ppo_update(model, ref_model, optimizer, batch, config):
    """
    Standard PPO update with KL penalty vs Reference Model
    """
    model.model.train()
    device = config.device
    scheduler = model.noise_scheduler

    x_t = batch['x_t'].to(device)
    x_prev = batch['x_prev'].to(device)
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

    with torch.no_grad():
        eps_ref = ref_model.model(
            x_t, t,
            torch.zeros_like(cond_img),
            torch.zeros_like(fp),
            drop_cond=True
        )

    kl_loss = F.mse_loss(eps, eps_ref)
    loss = ppo_loss + config.kl_beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    
    # Log PPO Grad Norm (Debug)
    if np.random.rand() < 0.01:
        if hasattr(model.model, 'fingerprint_proj') and model.model.fingerprint_proj[0].weight.grad is not None:
              norm = model.model.fingerprint_proj[0].weight.grad.norm().item()
              print(f"  [PPO] Drug grad norm: {norm:.6f}")

    optimizer.step()
    return loss.item()

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
# MAIN TRAINING LOOP
# ============================================================================

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Initialize Models
    # Theta: Control -> Perturbed (using Drug)
    theta = DDMECModel(config)
    theta.load_pretrained_unconditional(config.theta_pretrained) # Load weights trained on Perturbed
    theta_ref = copy.deepcopy(theta) # Reference for KL constraint
    
    # Phi: Perturbed -> Control (using Drug)
    phi = DDMECModel(config)
    phi.load_pretrained_unconditional(config.phi_pretrained) # Load weights trained on Control
    phi_ref = copy.deepcopy(phi)
    
    theta_opt = torch.optim.AdamW(theta.model.parameters(), lr=config.lr)
    phi_opt = torch.optim.AdamW(phi.model.parameters(), lr=config.lr)
    
    # 2. Data
    encoder = MorganFingerprintEncoder()
    dataset = BBBC021PairedDataset(config.data_dir, config.metadata_file, encoder)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    print("Starting DDMEC-PPO with Drug Conditioning...")
    
    for epoch in range(10): # Example epochs
        for batch in tqdm(loader):
            ctrl = batch['control'].to(config.device)
            trt = batch['perturbed'].to(config.device)
            fp = batch['fingerprint'].to(config.device)
            
            # --- PHASE 1: Update Theta with PPO (Algorithm 1) ---
            # 1. Generate Fake Perturbed (x_gen) from Control + Drug
            x_gen, traj_theta = rollout_with_logprobs(theta, ctrl, fp, steps=50) # Reduced steps for speed
            
            # 2. Calculate Reward using Phi
            # Phi estimates: P(Control | x_gen, Drug)
            # High prob = x_gen looks like a perturbed state that corresponds to this control/drug pair
            r_theta = reward_negloglik(phi, target_img=ctrl, cond_img=x_gen, fingerprint=fp)
            
            # 3. PPO Update Theta (reward + marginal constraint)
            if len(traj_theta) > 0:
                ppo_batch_theta = build_ppo_batch(traj_theta, r_theta, ctrl, fp)
                
                if ppo_batch_theta is not None:
                    # PPO epochs with minibatches
                    total_samples = ppo_batch_theta['x_t'].shape[0]
                    indices = torch.randperm(total_samples, device=ppo_batch_theta['x_t'].device)  # CPU
                    
                    for ppo_epoch in range(config.ppo_epochs):
                        for start_idx in range(0, total_samples, config.ppo_minibatch_size):
                            end_idx = min(start_idx + config.ppo_minibatch_size, total_samples)
                            batch_indices = indices[start_idx:end_idx]
                            
                            minibatch = {
                                k: v[batch_indices] for k, v in ppo_batch_theta.items()
                            }
                            
                            loss_val = ppo_update(theta, theta_ref, theta_opt, minibatch, config)
            
            # --- PHASE 2: Update Phi with Joint Constraint (Algorithm 2) ---
            # After theta generates x_gen, update phi to agree: phi should map x_gen -> ctrl
            # This enforces the joint constraint: p_theta and p_phi should agree on generated samples
            phi_opt.zero_grad(set_to_none=True)
            loss_phi_joint = phi(ctrl, x_gen, fp)  # phi: learn to reconstruct ctrl given condition x_gen
            loss_phi_joint.backward()
            
            # Log Grad Norm immediately
            if np.random.rand() < 0.05:
                if phi.model.fingerprint_proj[0].weight.grad is not None:
                     norm = phi.model.fingerprint_proj[0].weight.grad.norm().item()
                     print(f"  [JOINT] Phi drug grad norm: {norm:.6f}")
                     
            phi_opt.step()
            
            # --- PHASE 3: Update Phi with PPO (Algorithm 1) ---
            # 1. Generate Fake Control (y_gen) from Perturbed + Drug
            y_gen, traj_phi = rollout_with_logprobs(phi, trt, fp, steps=50)
            
            # 2. Calculate Reward using Theta
            # Theta estimates: P(Perturbed | y_gen, Drug)
            r_phi = reward_negloglik(theta, target_img=trt, cond_img=y_gen, fingerprint=fp)
            
            # 3. PPO Update Phi (reward + marginal constraint)
            if len(traj_phi) > 0:
                ppo_batch_phi = build_ppo_batch(traj_phi, r_phi, trt, fp)
                
                if ppo_batch_phi is not None:
                    # PPO epochs with minibatches
                    total_samples = ppo_batch_phi['x_t'].shape[0]
                    indices = torch.randperm(total_samples, device=ppo_batch_phi['x_t'].device)  # CPU
                    
                    for ppo_epoch in range(config.ppo_epochs):
                        for start_idx in range(0, total_samples, config.ppo_minibatch_size):
                            end_idx = min(start_idx + config.ppo_minibatch_size, total_samples)
                            batch_indices = indices[start_idx:end_idx]
                            
                            minibatch = {
                                k: v[batch_indices] for k, v in ppo_batch_phi.items()
                            }
                            
                            loss_val = ppo_update(phi, phi_ref, phi_opt, minibatch, config)
            
            # --- PHASE 4: Update Theta with Joint Constraint (Algorithm 2) ---
            # After phi generates y_gen, update theta to agree: theta should map y_gen -> trt
            # This enforces the joint constraint: p_theta and p_phi should agree on generated samples
            theta_opt.zero_grad(set_to_none=True)
            loss_theta_joint = theta(trt, y_gen, fp)  # theta: learn to reconstruct trt given condition y_gen
            loss_theta_joint.backward()
            
            # Log Grad Norm immediately
            if np.random.rand() < 0.05:
                if theta.model.fingerprint_proj[0].weight.grad is not None:
                     norm = theta.model.fingerprint_proj[0].weight.grad.norm().item()
                     print(f"  [JOINT] Theta drug grad norm: {norm:.6f}")
                     
            theta_opt.step()
            
            # NOTE: For a hybrid approach (more stable but not pure unpaired DDMEC), you can add
            # supervised losses on real pairs after the joint constraint updates:
            #   theta_opt.zero_grad(set_to_none=True)
            #   phi_opt.zero_grad(set_to_none=True)
            #   loss_theta_sup = theta(trt, ctrl, fp)  # Learn on real pairs
            #   loss_phi_sup = phi(ctrl, trt, fp)       # Learn on real pairs
            #   loss_theta_sup.backward()
            #   loss_phi_sup.backward()
            #   theta_opt.step()
            #   phi_opt.step()
            
            # Logging and drug conditioning verification
            if np.random.rand() < 0.01:
                print(f"R_Theta: {r_theta.mean().item():.3f} | R_Phi: {r_phi.mean().item():.3f}")
                
                # Removed late gradient logging (moved to immediately after backward)
                
                # Ablation test: generate with real drug vs zero drug
                with torch.no_grad():
                    theta.model.eval()
                    # Generate with real drug
                    x_gen_real_fp, _ = rollout_with_logprobs(theta, ctrl[:1], fp[:1], steps=10)
                    # Generate with zero drug
                    fp_zero = torch.zeros_like(fp[:1])
                    x_gen_zero_fp, _ = rollout_with_logprobs(theta, ctrl[:1], fp_zero, steps=10)
                    # Compute difference
                    drug_effect = F.mse_loss(x_gen_real_fp, x_gen_zero_fp).item()
                    print(f"  Drug conditioning effect (MSE between real vs zero drug): {drug_effect:.6f}")
                    if drug_effect < 1e-4:
                        print("  WARNING: Drug conditioning appears to have minimal effect!")
                
        # Save visualization
        with torch.no_grad():
            vis = torch.cat([ctrl[:4], x_gen[:4], trt[:4], y_gen[:4]], dim=0)
            save_image(vis, f"{config.output_dir}/epoch_{epoch}.png", nrow=4, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()