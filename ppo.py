
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
    kl_beta = 0.05        # Weight for KL anchor to pretrained model
    rollout_steps = 100   # Strided rollout (faster RL)
    cond_drop_prob = 0.1  # Probability of dropping conditioning for CFG training
    cond_drop_prob = 0.1  # Probability of dropping conditioning for CFG training
    guidance_scale = 2.0  # Classifier-free guidance scale for RL rollout & reward
    
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
    eval_max_samples = 500
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
        
        # Robust CSV loading
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file
            
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
        
        # Paths CSV handling (Simplified for brevity, assuming standard paths or paths.csv exists)
        self.paths_lookup = {}
        if paths_csv and os.path.exists(paths_csv):
             paths_df = pd.read_csv(paths_csv)
             for _, row in paths_df.iterrows():
                 self.paths_lookup[row['filename']] = row['relative_path']

    def _group_by_batch(self):
        groups = {}
        for idx, row in enumerate(self.metadata):
            b = row.get('BATCH', 'unknown')
            if b not in groups: groups[b] = {'ctrl': [], 'trt': []}
            cpd = str(row.get('CPD_NAME', '')).upper()
            if cpd == 'DMSO': groups[b]['ctrl'].append(idx)
            else: groups[b]['trt'].append(idx)
        return groups

    def get_perturbed_indices(self):
        return [i for i, m in enumerate(self.metadata) if str(m.get('CPD_NAME', '')).upper() != 'DMSO']

    def get_paired_sample(self, trt_idx):
        batch = self.metadata[trt_idx].get('BATCH', 'unknown')
        if batch in self.batch_map and self.batch_map[batch]['ctrl']:
            ctrls = self.batch_map[batch]['ctrl']
            return (np.random.choice(ctrls), trt_idx)
        return (trt_idx, trt_idx)

    def __len__(self): return len(self.metadata)

    def _find_file_path(self, path):
        # Simplified path finding
        if not path: return None
        path_obj = Path(path)
        
        # Try lookup
        if path_obj.name in self.paths_lookup:
            cand = self.data_dir / self.paths_lookup[path_obj.name]
            if cand.exists(): return cand
            cand = self.data_dir.parent / self.paths_lookup[path_obj.name]
            if cand.exists(): return cand

        # Try direct
        cand = self.data_dir / path
        if cand.exists(): return cand
        
        # Try glob
        matches = list(self.data_dir.rglob(path_obj.name))
        if matches: return matches[0]
        
        return None

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        path = meta.get('image_path') or meta.get('SAMPLE_KEY')
        full_path = self._find_file_path(path)
        
        if full_path is None or not full_path.exists():
            # Create dummy data if file missing to avoid crash during dev
            # print(f"Warning: File not found {path}, returning zeros")
            return {
                'image': torch.zeros((3, self.image_size, self.image_size)),
                'fingerprint': torch.zeros((1024,)),
                'compound': 'Unknown'
            }

        img = np.load(full_path)
        if img.ndim == 3 and img.shape[-1] == 3: img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if img.max() > 1.0: img = (img / 127.5) - 1.0
        else: img = (img * 2.0) - 1.0
        img = torch.clamp(img, -1, 1)
        
        cpd = meta.get('CPD_NAME', 'DMSO')
        fp = self.fingerprints.get(cpd, np.zeros(1024))
        
        return {'image': img, 'fingerprint': torch.from_numpy(fp).float(), 'compound': cpd}

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
        eps_uncond = model.model(x, t_batch, torch.zeros_like(cond_img), torch.zeros_like(fingerprint), drop_cond=True)
        
        # Micro-fix 1: Explicit config access
        w = model.cfg.guidance_scale
        eps_pred = eps_uncond + w * (eps_cond - eps_uncond)
        
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

    # Accumulate MSE terms
    acc = torch.zeros((b,), device=device)

    for _ in range(mc):
        # sample a set of timesteps per batch element
        t_batch = torch.randint(0, scheduler.config.num_train_timesteps, (n_terms, b), device=device).long()
        for k in range(n_terms):
            tk = t_batch[k]
            noise = torch.randn_like(target_img)
            y_t = scheduler.add_noise(target_img, noise, tk)
            
            # Use CFG for reward likelihood proxy
            eps_cond = other_model.model(y_t, tk, cond_img, fingerprint, drop_cond=False)
            eps_uncond = other_model.model(y_t, tk, torch.zeros_like(cond_img), torch.zeros_like(fingerprint), drop_cond=True)
            
            # Micro-fix 1: Explicit config access
            w = other_model.cfg.guidance_scale
            eps_pred = eps_uncond + w * (eps_cond - eps_uncond)
            
            mse = ((noise - eps_pred) ** 2).mean(dim=(1,2,3))
            acc += mse

    acc = acc / (mc * n_terms)

    # Eq(8) has 1/2 factor; constant irrelevant
    # Note: DDMEC paper likely weights this by 1/sigma_t^2 or similar. 
    # This unweighted sum is a simplified surrogate.
    negloglik = 0.5 * acc

    # reward = - negloglik
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
    ratio = torch.exp(new_logprob - old_logprob)
    
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

@torch.no_grad()
def evaluate_metrics(theta_model, phi_model, dataloader, config):
    print("Running Evaluation (FID/KID)...")
    theta_model.model.eval()
    phi_model.model.eval()
    
    fid_metric_control = FrechetInceptionDistance(normalize=True).to(config.device) # FIDc (Real Ctrl vs Fake Ctrl from Trt)
    kid_metric_control = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device)
    
    fid_metric_treated = FrechetInceptionDistance(normalize=True).to(config.device) # FIDo (Real Trt vs Fake Trt from Ctrl)
    kid_metric_treated = KernelInceptionDistance(subset_size=100, normalize=True).to(config.device)
    
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
        
        # 1. Evaluate Theta (Ctrl -> Fake Trt) => Compare with Real Trt
        # Use specific eval steps
        fake_trt, _ = rollout_with_logprobs(theta_model, ctrl, fp, steps=config.eval_steps)
        
        # Normalize to [0, 1] for torchmetrics if not already? 
        # rollout output is [-1, 1]. Torchmetrics expects [0, 1] or [0, 255].
        # "normalize=True" argument in FID means input is [0, 1] or [0, 255].
        # So we convert [-1, 1] -> [0, 1].
        
        real_trt_norm = (trt + 1) / 2
        fake_trt_norm = (fake_trt + 1) / 2
        
        real_trt_norm = torch.clamp(real_trt_norm, 0, 1)
        fake_trt_norm = torch.clamp(fake_trt_norm, 0, 1)
        
        fid_metric_treated.update(real_trt_norm, real=True)
        fid_metric_treated.update(fake_trt_norm, real=False)
        kid_metric_treated.update(real_trt_norm, real=True)
        kid_metric_treated.update(fake_trt_norm, real=False)
        
        # 2. Evaluate Phi (Trt -> Fake Ctrl) => Compare with Real Ctrl
        fake_ctrl, _ = rollout_with_logprobs(phi_model, trt, fp, steps=config.eval_steps)
        
        real_ctrl_norm = (ctrl + 1) / 2
        fake_ctrl_norm = (fake_ctrl + 1) / 2
        
        real_ctrl_norm = torch.clamp(real_ctrl_norm, 0, 1)
        fake_ctrl_norm = torch.clamp(fake_ctrl_norm, 0, 1)
        
        fid_metric_control.update(real_ctrl_norm, real=True)
        fid_metric_control.update(fake_ctrl_norm, real=False)
        kid_metric_control.update(real_ctrl_norm, real=True)
        kid_metric_control.update(fake_ctrl_norm, real=False)
        
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
        
    return {
        "FID_Control": fid_c,
        "KID_Control": kid_c,
        "FID_Treated": fid_t,
        "KID_Treated": kid_t
    }

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DDMEC-PPO Training')
    parser.add_argument('--iters', type=int, default=100, help='Total number of training iterations')
    parser.add_argument('--eval_samples', type=int, default=1000, help='Number of samples for evaluation')
    parser.add_argument('--eval_steps', type=int, default=50, help='Number of inference steps for evaluation')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint in output_dir')
    parser.add_argument('--theta_checkpoint', type=str, default='./ddpm_diffusers_results/checkpoints/checkpoint_epoch_60.pt', 
                        help='Path to theta checkpoint (default: pretrained from config)')
    parser.add_argument('--phi_checkpoint', type=str, default='./results_phi_phi/checkpoints/checkpoint_epoch_100.pt',
                        help='Path to phi checkpoint (default: pretrained from config)')
    parser.add_argument('--output_dir', type=str, default='ddpm_ddmec_results', help='Output directory for checkpoints and logs')
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
    ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='train', encoder=encoder)
    loader = PairedDataLoader(ds, config.batch_size, shuffle=True)
    
    # Test Data for Evaluation (FIX: Use test split as requested)
    test_ds = BBBC021Dataset(config.data_dir, config.metadata_file, split='test', encoder=encoder)
    test_loader = PairedDataLoader(test_ds, config.batch_size, shuffle=False)
    
    # Initialize CSV Log
    if not os.path.exists(config.log_file):
        with open(config.log_file, 'w') as f:
            f.write("iteration,theta_reward,phi_reward,theta_ppo_loss,phi_ppo_loss,theta_sup_loss,phi_sup_loss,fid_control,fid_treated,kid_control,kid_treated\n")
            
    if start_iter > 0:
        print(f"Continuing DDMEC-PPO Loop from iteration {start_iter} to {args.iters}...")
    else:
        print(f"Starting DDMEC-PPO Loop for {args.iters} iterations...")
    iterator = iter(loader)
    
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
            print(f"  KID_Control: {metrics['KID_Control']:.4f}")
            print(f"  KID_Treated: {metrics['KID_Treated']:.4f}")
            print(f"{'='*60}\n")
            
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
                        f"{metrics['FID_Control']:.4f},{metrics['FID_Treated']:.4f},{metrics['KID_Control']:.6f},{metrics['KID_Treated']:.6f}\n")

if __name__ == "__main__":
    main()
