"""
================================================================================
BBBC021 INFERENCE & VIDEO GENERATION
================================================================================
Generates a side-by-side video comparing:
[Control Image] -> [Generated Perturbed Cell] -> [Real Ground Truth Cell]

Usage:
    python generate_inference_video.py --checkpoint_path "ddpm_diffusers_results/checkpoints/latest.pt"
================================================================================
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import imageio
from pathlib import Path
from diffusers import UNet2DModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --- Dependencies Check ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not found. Chemical encoding may fail.")

# ============================================================================
# ARCHITECTURE (Must match training script exactly)
# ============================================================================

class ModifiedDiffusersUNet(nn.Module):
    def __init__(self, image_size=96, fingerprint_dim=1024):
        super().__init__()
        # Load base with default architecture (matches training script)
        # Use ignore_mismatched_sizes since we'll load our own checkpoint anyway
        self.unet = UNet2DModel.from_pretrained(
            "google/ddpm-cifar10-32",
            sample_size=image_size,
            class_embed_type="identity",
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )
        
        # Re-apply 3 -> 6 channel surgery
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0
            new_conv.bias = old_conv.bias
        self.unet.conv_in = new_conv

        # Re-apply Projection surgery - get dimension from model
        target_dim = self.unet.time_embedding.linear_1.out_features
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.SiLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, t, control, fingerprint):
        x_in = torch.cat([x, control], dim=1)
        emb = self.fingerprint_proj(fingerprint)
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
        
        self.timesteps = config.timesteps
        beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)
        alpha = 1. - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

    @torch.no_grad()
    def sample_trajectory(self, control, fingerprint, num_frames=50):
        """
        Runs reverse diffusion and returns a list of frames showing the process.
        """
        self.model.eval()
        b = control.shape[0]
        # Start from random noise
        xt = torch.randn_like(control)
        frames = []
        
        # Steps to capture for video
        save_steps = np.linspace(0, self.timesteps-1, num_frames, dtype=int)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.cfg.device, dtype=torch.long)
            noise_pred = self.model(xt, t, control, fingerprint)
            
            # Coefficients
            alpha = 1 - torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.timesteps).to(self.cfg.device)[i]
            alpha_bar = self.alpha_bar[i]
            beta = 1 - alpha
            
            z = torch.randn_like(xt) if i > 0 else 0
            
            # Denoising Step: x_{t-1} = 1/sqrt(alpha) * (x_t - ...) + sigma * z
            xt = (1 / torch.sqrt(alpha)) * (xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * z
            xt = torch.clamp(xt, -1, 1)
            
            if i in save_steps or i == 0:
                # Convert to numpy image [0, 255]
                img_np = ((xt[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                frames.append(img_np)
                
        return frames

# ============================================================================
# UTILITIES
# ============================================================================

class Config:
    # Minimal config for inference
    image_size = 96
    fingerprint_dim = 1024
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    device = "cuda" if torch.cuda.is_available() else "cpu"

class MorganFingerprintEncoder:
    def __init__(self, n_bits=1024):
        self.n_bits = n_bits
    def encode(self, smiles):
        if RDKIT_AVAILABLE and smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.float32)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                return arr
            except: pass
        # Fallback
        np.random.seed(hash(str(smiles)) % 2**32)
        arr = (np.random.rand(self.n_bits) > 0.5).astype(np.float32)
        return arr

def load_data_sample(data_dir, metadata_file, encoder, paths_csv=None, split='test'):
    """
    Finds a random Perturbed sample and its paired Control sample from the CSV.
    Uses paths.csv if available for efficient file lookup.
    
    Args:
        data_dir: Directory containing data
        metadata_file: Path to metadata CSV file
        encoder: Fingerprint encoder
        paths_csv: Optional path to paths.csv file (auto-detected if None)
        split: Data split to use ('train', 'val', 'test', or None for all)
    """
    df = pd.read_csv(os.path.join(data_dir, metadata_file)) if os.path.exists(os.path.join(data_dir, metadata_file)) else pd.read_csv(metadata_file)
    
    # Filter by split if SPLIT column exists
    if split and 'SPLIT' in df.columns:
        split_df = df[df['SPLIT'].str.upper() == split.upper()].copy()
        if len(split_df) > 0:
            df = split_df
        else:
            # Try lowercase
            split_df = df[df['SPLIT'].str.lower() == split.lower()].copy()
            if len(split_df) > 0:
                df = split_df
    
    # Load paths.csv if available for efficient file lookup
    paths_lookup = {}  # filename -> list of relative_paths
    paths_by_rel = {}  # relative_path -> full info
    paths_by_basename = {}  # basename (without extension) -> list of paths
    
    if paths_csv:
        paths_csv_path = Path(paths_csv)
    else:
        paths_csv_path = Path("paths.csv")
        if not paths_csv_path.exists():
            # Try in data_dir
            paths_csv_path = Path(data_dir) / "paths.csv"
    
    if paths_csv_path.exists():
        print(f"Loading file paths from {paths_csv_path}...")
        paths_df = pd.read_csv(paths_csv_path)
        
        # Create multiple lookup strategies
        for _, row in paths_df.iterrows():
            filename = row['filename']
            rel_path = row['relative_path']
            basename = Path(filename).stem  # filename without extension
            
            # Lookup by exact filename
            if filename not in paths_lookup:
                paths_lookup[filename] = []
            paths_lookup[filename].append(rel_path)
            
            # Lookup by relative path
            paths_by_rel[rel_path] = row.to_dict()
            
            # Lookup by basename (for matching without extension)
            if basename not in paths_by_basename:
                paths_by_basename[basename] = []
            paths_by_basename[basename].append(rel_path)
        
        print(f"  Loaded {len(paths_lookup)} unique filenames from paths.csv")
        print(f"  Total files in paths.csv: {len(paths_df)}")
    else:
        print("  Note: paths.csv not found, will use fallback path resolution")
    
    # 1. Find a perturbed sample (Non-DMSO)
    perturbed_df = df[df['CPD_NAME'].str.upper() != 'DMSO']
    if len(perturbed_df) == 0:
        print("Error: No perturbed samples found in CSV.")
        return None
    
    target_row = perturbed_df.sample(1).iloc[0]
    target_batch = target_row['BATCH']
    target_compound = target_row['CPD_NAME']
    
    # 2. Find a control sample (DMSO) from SAME BATCH
    control_candidates = df[(df['BATCH'] == target_batch) & (df['CPD_NAME'].str.upper() == 'DMSO')]
    
    if len(control_candidates) == 0:
        print(f"Warning: No control found for batch {target_batch}. Using target as dummy control.")
        control_row = target_row
    else:
        control_row = control_candidates.sample(1).iloc[0]
        
    print(f"Selected Pair from Batch {target_batch}:")
    print(f"  Control: DMSO")
    print(f"  Target:  {target_compound}")

    # 3. Load Images
    def load_img(row):
        path = row.get('image_path') or row.get('SAMPLE_KEY')
        if not path:
            raise ValueError(f"No image path found in row: {row.get('CPD_NAME', 'unknown')}")
        
        print(f"  Looking for image with path: {path}")
        
        data_dir_path = Path(data_dir).resolve()
        full_path = None
        
        # First, try using paths.csv lookup (multiple strategies)
        if paths_lookup:
            path_obj = Path(path)
            filename = path_obj.name
            basename = path_obj.stem  # filename without extension
            filename_no_ext = filename.replace('.npy', '').replace('.NPY', '')
            
            # Strategy 1: Exact filename match
            if filename in paths_lookup:
                for rel_path in paths_lookup[filename]:
                    candidate = data_dir_path / rel_path
                    if candidate.exists():
                        full_path = candidate
                        print(f"  Found image via paths.csv (exact filename): {full_path}")
                        break
            
            # Strategy 2: Match by basename (without extension)
            if full_path is None and basename in paths_by_basename:
                for rel_path in paths_by_basename[basename]:
                    candidate = data_dir_path / rel_path
                    if candidate.exists():
                        full_path = candidate
                        print(f"  Found image via paths.csv (basename match): {full_path}")
                        break
            
            # Strategy 3: Match filename without extension
            if full_path is None:
                for key in paths_lookup.keys():
                    key_basename = Path(key).stem
                    if key_basename == filename_no_ext or key_basename == basename:
                        for rel_path in paths_lookup[key]:
                            candidate = data_dir_path / rel_path
                            if candidate.exists():
                                full_path = candidate
                                print(f"  Found image via paths.csv (no-ext match): {full_path}")
                                break
                        if full_path:
                            break
            
            # Strategy 4: Check if path from metadata matches any relative_path
            if full_path is None:
                # Try direct relative path match (path from metadata is a rel_path key)
                if path in paths_by_rel:
                    rel_path = path  # path is already the rel_path key
                    candidate = data_dir_path / rel_path
                    if candidate.exists():
                        full_path = candidate
                        print(f"  Found image via paths.csv (direct path match): {full_path}")
                
                # Try partial path matching (path contains or is contained in rel_path)
                if full_path is None:
                    for rel_path_key in paths_by_rel.keys():
                        # Check if metadata path is part of relative path or vice versa
                        if path in rel_path_key or rel_path_key.endswith(path) or path.endswith(rel_path_key):
                            candidate = data_dir_path / rel_path_key
                            if candidate.exists():
                                full_path = candidate
                                print(f"  Found image via paths.csv (partial path match): {full_path}")
                                break
        
        # Fallback: Try multiple path variations
        if full_path is None:
            possible_paths = [
                data_dir_path / path,  # Direct path
                data_dir_path / (path + '.npy'),  # With .npy extension
                Path(path).resolve() if Path(path).is_absolute() else None,  # Absolute path
                data_dir_path / Path(path).name,  # Just filename
                data_dir_path / (Path(path).name + '.npy'),  # Just filename with .npy
            ]
            
            # Remove None values
            possible_paths = [p for p in possible_paths if p is not None]
            
            for p in possible_paths:
                if p.exists():
                    full_path = p
                    print(f"  Found image at: {full_path}")
                    break
        
        # If still not found, try recursive search
        if full_path is None:
            search_pattern = Path(path).name + '.npy' if not path.endswith('.npy') else Path(path).name
            matches = list(data_dir_path.rglob(search_pattern))
            if matches:
                full_path = matches[0]
                print(f"  Found image via recursive search at: {full_path}")
        
        if full_path is None:
            print(f"Error: Could not find image file for path: {path}")
            raise FileNotFoundError(f"Image file not found for: {row.get('CPD_NAME', 'unknown')} (path: {path})")
        
        img = np.load(full_path)
        if img.ndim == 3 and img.shape[-1] == 3: img = img.transpose(2, 0, 1) # HWC to CHW
        img = torch.from_numpy(img).float()
        if img.max() > 1.0: img = (img / 127.5) - 1.0 # Normalize
        return torch.clamp(img, -1, 1).unsqueeze(0) # Add batch dim

    ctrl_tensor = load_img(control_row)
    real_tensor = load_img(target_row)
    
    # 4. Encode Fingerprint
    smiles = target_row.get('SMILES', '')
    fp = encoder.encode(smiles)
    fp_tensor = torch.from_numpy(fp).float().unsqueeze(0)
    
    return ctrl_tensor, real_tensor, fp_tensor, target_compound

# ============================================================================
# BATCH EVALUATION
# ============================================================================

def calculate_fid(real_images, generated_images, device='cuda'):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images: Tensor of real images [N, 3, H, W] in range [-1, 1]
        generated_images: Tensor of generated images [N, 3, H, W] in range [-1, 1]
        device: Device to run computation on
    
    Returns:
        FID score
    """
    try:
        from torchvision.models import inception_v3
        from scipy import linalg
        TORCHVISION_AVAILABLE = True
    except ImportError:
        return None
    
    # Load Inception v3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.fc = nn.Identity()  # Remove final classification layer
    
    def get_inception_features(images):
        """Extract features from Inception network"""
        # Resize to 299x299 for Inception v3
        images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Convert from [-1, 1] to [0, 1] then to [0, 255] for Inception
        images_resized = (images_resized + 1) / 2.0  # [-1, 1] -> [0, 1]
        images_resized = images_resized * 255.0  # [0, 1] -> [0, 255]
        images_resized = images_resized.clamp(0, 255)
        
        with torch.no_grad():
            features = inception(images_resized)
        return features.cpu().numpy()
    
    # Extract features
    print("  Computing FID: Extracting Inception features for real images...")
    real_features = get_inception_features(real_images)
    
    print("  Computing FID: Extracting Inception features for generated images...")
    gen_features = get_inception_features(generated_images)
    
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

def evaluate_batch(model, data_dir, metadata_file, encoder, paths_csv, num_samples=1000, output_dir="evaluation_results", split='test', generate_videos=True, num_videos=10):
    """
    Evaluate model on multiple samples and compute metrics.
    """
    import time
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        SKIMAGE_AVAILABLE = True
    except ImportError:
        print("Warning: scikit-image not available. Install with: pip install scikit-image")
        print("SSIM and PSNR metrics will not be computed.")
        SKIMAGE_AVAILABLE = False
    
    os.makedirs(output_dir, exist_ok=True)
    videos_dir = os.path.join(output_dir, "videos")
    if generate_videos:
        os.makedirs(videos_dir, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(os.path.join(data_dir, metadata_file)) if os.path.exists(os.path.join(data_dir, metadata_file)) else pd.read_csv(metadata_file)
    
    # Filter by split if specified
    if split and 'SPLIT' in df.columns:
        split_df = df[df['SPLIT'].str.upper() == split.upper()].copy()
        if len(split_df) > 0:
            print(f"Found {len(split_df)} samples in {split.upper()} split")
            df = split_df
        else:
            # Try lowercase
            split_df = df[df['SPLIT'].str.lower() == split.lower()].copy()
            if len(split_df) > 0:
                print(f"Found {len(split_df)} samples in {split.lower()} split")
                df = split_df
            else:
                print(f"Warning: No {split} split found, using all data")
    elif 'SPLIT' in df.columns:
        print("Note: SPLIT column found but no split specified, using all data")
    else:
        print("Note: No SPLIT column found, using all data")
    
    # Get perturbed samples (non-DMSO)
    perturbed_df = df[df['CPD_NAME'].str.upper() != 'DMSO'].copy()
    if len(perturbed_df) == 0:
        print("Error: No perturbed samples found in CSV.")
        return None
    
    # Sample up to num_samples
    num_samples = min(num_samples, len(perturbed_df))
    eval_df = perturbed_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    print(f"Evaluating on {num_samples} samples from test split...")
    
    # Load paths.csv if available
    paths_lookup = {}
    paths_by_rel = {}
    paths_by_basename = {}
    
    if paths_csv:
        paths_csv_path = Path(paths_csv)
    else:
        paths_csv_path = Path("paths.csv")
        if not paths_csv_path.exists():
            paths_csv_path = Path(data_dir) / "paths.csv"
    
    if paths_csv_path.exists():
        print(f"Loading paths.csv...")
        paths_df = pd.read_csv(paths_csv_path)
        for _, row in paths_df.iterrows():
            filename = row['filename']
            rel_path = row['relative_path']
            basename = Path(filename).stem
            if filename not in paths_lookup:
                paths_lookup[filename] = []
            paths_lookup[filename].append(rel_path)
            paths_by_rel[rel_path] = row.to_dict()
            if basename not in paths_by_basename:
                paths_by_basename[basename] = []
            paths_by_basename[basename].append(rel_path)
    
    results = []
    real_images_list = []  # For FID calculation
    generated_images_list = []  # For FID calculation
    video_samples = []  # Store samples for video generation
    data_dir_path = Path(data_dir).resolve()
    config = Config()
    
    # Select samples for video generation (first num_videos or random)
    video_indices = set()
    if generate_videos and num_videos > 0:
        num_videos = min(num_videos, num_samples)
        # Select evenly spaced samples for videos
        video_indices = set(np.linspace(0, num_samples - 1, num_videos, dtype=int))
        print(f"Will generate videos for {num_videos} samples (indices: {sorted(video_indices)})")
    
    def find_file_path(row, debug=False):
        """Find file path using paths.csv lookup"""
        path = row.get('image_path') or row.get('SAMPLE_KEY')
        if not path:
            if debug:
                print(f"      No path in row: {row.get('CPD_NAME', 'unknown')}")
            return None
        
        path_obj = Path(path)
        filename = path_obj.name
        basename = path_obj.stem
        
        if debug:
            print(f"      Looking for: path={path}, filename={filename}, basename={basename}")
        
        # Try paths.csv lookup
        if paths_lookup:
            if filename in paths_lookup:
                for rel_path in paths_lookup[filename]:
                    # Handle relative_path - paths.csv has paths like "bbbc021_all/Week9/..."
                    # If data_dir is "./data/bbbc021_all", we need to handle this correctly
                    rel_path_str = str(rel_path)
                    
                    # Try multiple path combinations
                    candidates = []
                    
                    # 1. If rel_path starts with data_dir name, use it relative to parent
                    if data_dir_path.name in rel_path_str:
                        # Remove data_dir name from start of rel_path
                        if rel_path_str.startswith(data_dir_path.name + '/'):
                            rel_path_clean = rel_path_str[len(data_dir_path.name) + 1:]
                            candidates.append(data_dir_path / rel_path_clean)
                        # Or use parent directory
                        candidates.append(data_dir_path.parent / rel_path)
                    
                    # 2. Try as absolute path from current directory
                    candidates.append(Path(rel_path).resolve())
                    
                    # 3. Try relative to data_dir
                    candidates.append(data_dir_path / rel_path)
                    
                    # 4. Try relative to data_dir parent
                    candidates.append(data_dir_path.parent / rel_path)
                    
                    # Remove duplicates and None
                    candidates = list(dict.fromkeys([c for c in candidates if c is not None]))
                    
                    for candidate in candidates:
                        if candidate.exists():
                            if debug:
                                print(f"      Found via filename match: {candidate}")
                            return candidate
                    if debug:
                        print(f"      Tried {len(candidates)} candidates for filename {filename}, none found")
                        if len(candidates) <= 3:
                            for c in candidates:
                                print(f"        - {c} (exists: {c.exists()})")
            if basename in paths_by_basename:
                for rel_path in paths_by_basename[basename]:
                    rel_path_str = str(rel_path)
                    candidates = []
                    
                    if data_dir_path.name in rel_path_str:
                        if rel_path_str.startswith(data_dir_path.name + '/'):
                            rel_path_clean = rel_path_str[len(data_dir_path.name) + 1:]
                            candidates.append(data_dir_path / rel_path_clean)
                        candidates.append(data_dir_path.parent / rel_path)
                    
                    candidates.append(Path(rel_path).resolve())
                    candidates.append(data_dir_path / rel_path)
                    candidates.append(data_dir_path.parent / rel_path)
                    
                    candidates = list(dict.fromkeys([c for c in candidates if c is not None]))
                    
                    for candidate in candidates:
                        if candidate.exists():
                            if debug:
                                print(f"      Found via basename match: {candidate}")
                            return candidate
                    if debug:
                        print(f"      Tried {len(candidates)} candidates for basename {basename}, none found")
        
        # Fallback - try direct path matching
        for candidate in [data_dir_path / path, data_dir_path / (path + '.npy')]:
            if candidate.exists():
                if debug:
                    print(f"      Found via direct path: {candidate}")
                return candidate
        
        # Last resort: recursive search by filename
        if paths_lookup:
            search_pattern = filename if filename.endswith('.npy') else filename + '.npy'
            matches = list(data_dir_path.rglob(search_pattern))
            if matches:
                if debug:
                    print(f"      Found via recursive search: {matches[0]}")
                return matches[0]
        
        if debug:
            print(f"      No file found for: {path}")
        return None
    
    def load_image_file(file_path):
        """Load and normalize image"""
        try:
            img = np.load(file_path)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            if img.max() > 1.0:
                img = (img / 127.5) - 1.0
            return torch.clamp(img, -1, 1)
        except:
            return None
    
    successful = 0
    failed = 0
    
    for idx, row in eval_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{num_samples} samples... (Successful: {successful}, Failed: {failed})")
        
        try:
            # Find target file
            target_path = find_file_path(row, debug=(failed < 3))  # Debug first 3 failures
            if target_path is None:
                if failed < 3:  # Print first few failures for debugging
                    print(f"    Failed to find target file for sample {idx}: {row.get('CPD_NAME', 'unknown')}")
                    print(f"      Row path fields: image_path={row.get('image_path', 'N/A')}, SAMPLE_KEY={row.get('SAMPLE_KEY', 'N/A')}")
                failed += 1
                continue
            
            # Load target image
            target_img = load_image_file(target_path)
            if target_img is None:
                failed += 1
                continue
            
            # Find control (DMSO) from same batch
            batch = row['BATCH']
            control_candidates = df[(df['BATCH'] == batch) & (df['CPD_NAME'].str.upper() == 'DMSO')]
            if len(control_candidates) == 0:
                if failed < 5:  # Print first few failures for debugging
                    print(f"    No control (DMSO) found for batch {batch} (sample {idx})")
                failed += 1
                continue
            
            control_row = control_candidates.sample(1).iloc[0]
            control_path = find_file_path(control_row)
            if control_path is None:
                if failed < 5:  # Print first few failures for debugging
                    print(f"    Failed to find control file for batch {batch} (sample {idx})")
                failed += 1
                continue
            
            control_img = load_image_file(control_path)
            if control_img is None:
                if failed < 5:  # Print first few failures for debugging
                    print(f"    Failed to load control image from {control_path} (sample {idx})")
                failed += 1
                continue
            
            # Get fingerprint
            smiles = row.get('SMILES', '')
            fp = encoder.encode(smiles)
            fp_tensor = torch.from_numpy(fp).float().unsqueeze(0)
            
            # Generate prediction
            ctrl_tensor = control_img.unsqueeze(0).to(config.device)
            fp_tensor = fp_tensor.to(config.device)
            
            with torch.no_grad():
                generated = model.sample(ctrl_tensor, fp_tensor)
            
            # Compute metrics
            real_tensor = target_img.unsqueeze(0).to(config.device)
            generated_np = ((generated[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
            real_np = ((real_tensor[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
            
            # MSE
            mse = F.mse_loss(generated, real_tensor).item()
            
            # SSIM and PSNR (if available)
            if SKIMAGE_AVAILABLE:
                gen_gray = np.mean(generated_np, axis=2)
                real_gray = np.mean(real_np, axis=2)
                ssim_val = ssim(real_gray, gen_gray, data_range=255)
                psnr_val = psnr(real_np, generated_np, data_range=255)
            else:
                ssim_val = 0.0
                psnr_val = 0.0
            
            # Store images for FID calculation (keep in [-1, 1] range)
            real_images_list.append(real_tensor.cpu())
            generated_images_list.append(generated.cpu())
            
            # Store sample info for video generation if selected
            if generate_videos and idx in video_indices:
                video_samples.append({
                    'idx': idx,
                    'compound': row['CPD_NAME'],
                    'batch': batch,
                    'ctrl_tensor': ctrl_tensor,
                    'fp_tensor': fp_tensor,
                    'real_tensor': real_tensor,
                    'generated': generated
                })
            
            results.append({
                'sample_idx': idx,
                'compound': row['CPD_NAME'],
                'batch': batch,
                'mse': mse,
                'ssim': ssim_val,
                'psnr': psnr_val,
                'smiles': smiles,
                'video_generated': 'yes' if (generate_videos and idx in video_indices) else 'no'
            })
            
            successful += 1
            
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            failed += 1
            continue
    
    # Check if we have any successful results
    if len(results) == 0:
        print(f"\n{'='*60}")
        print(f"ERROR: No samples were successfully processed!")
        print(f"{'='*60}")
        print(f"Total samples attempted: {num_samples}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\nThis usually means:")
        print(f"  - Files not found (check paths.csv and data directory)")
        print(f"  - Control samples missing for batches")
        print(f"  - Image loading errors")
        return None, None
    
    # Save results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    # Generate videos for selected samples
    if generate_videos and len(video_samples) > 0:
        print(f"\nGenerating videos for {len(video_samples)} samples...")
        for i, sample in enumerate(video_samples):
            try:
                print(f"  Generating video {i+1}/{len(video_samples)}: {sample['compound']} (idx {sample['idx']})...")
                # Generate trajectory frames
                gen_frames = model.sample_trajectory(sample['ctrl_tensor'], sample['fp_tensor'], num_frames=60)
                
                # Prepare static images
                ctrl_img = ((sample['ctrl_tensor'][0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                real_img = ((sample['real_tensor'][0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
                
                # Stitch frames together
                final_video = []
                separator = np.zeros((96, 2, 3), dtype=np.uint8)
                
                for frame in gen_frames:
                    combined = np.hstack([ctrl_img, separator, frame, separator, real_img])
                    final_video.append(combined)
                
                # Save video
                video_filename = f"sample_{sample['idx']:04d}_{sample['compound'].replace('/', '_')}.mp4"
                video_path = os.path.join(videos_dir, video_filename)
                imageio.mimsave(video_path, final_video, fps=10)
                print(f"    Saved: {video_path}")
            except Exception as e:
                print(f"    Error generating video for sample {sample['idx']}: {e}")
    
    # Calculate FID if we have enough samples
    fid_score = None
    if len(real_images_list) > 0 and len(generated_images_list) > 0:
        print(f"\nComputing FID on {len(real_images_list)} image pairs...")
        real_images_tensor = torch.cat(real_images_list, dim=0)
        generated_images_tensor = torch.cat(generated_images_list, dim=0)
        fid_score = calculate_fid(real_images_tensor, generated_images_tensor, device=config.device)
        if fid_score is not None:
            print(f"  FID Score: {fid_score:.4f}")
        else:
            print("  FID calculation failed (missing dependencies)")
    
    # Compute summary statistics
    summary = {
        'total_samples': num_samples,
        'successful': successful,
        'failed': failed,
        'mean_mse': results_df['mse'].mean(),
        'std_mse': results_df['mse'].std(),
        'mean_ssim': results_df['ssim'].mean(),
        'std_ssim': results_df['ssim'].std(),
        'mean_psnr': results_df['psnr'].mean(),
        'std_psnr': results_df['psnr'].std(),
        'fid': fid_score if fid_score is not None else 'N/A',
    }
    
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nMetrics:")
    print(f"  MSE:  {summary['mean_mse']:.6f} ± {summary['std_mse']:.6f}")
    print(f"  SSIM: {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f"  PSNR: {summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f}")
    if fid_score is not None:
        print(f"  FID:  {fid_score:.4f}")
    print(f"\nResults saved to: {results_csv}")
    print(f"Summary saved to: {summary_csv}")
    if generate_videos and len(video_samples) > 0:
        print(f"Videos saved to: {videos_dir}")
    
    return results_df, summary_df

# ============================================================================
# MAIN INFERENCE
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="ddpm_diffusers_results/checkpoints/latest.pt", help="Path to checkpoint file (default: ddpm_diffusers_results/checkpoints/latest.pt)")
    parser.add_argument("--data_dir", type=str, default="./data/bbbc021_all")
    parser.add_argument("--metadata_file", type=str, default="metadata/bbbc021_df_all.csv")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file (auto-detected if not specified)")
    parser.add_argument("--output_path", type=str, default="inference_video.mp4")
    parser.add_argument("--batch_eval", action="store_true", help="Run batch evaluation on 1000 samples (default mode)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for batch evaluation (default: 1000)")
    parser.add_argument("--eval_output_dir", type=str, default="evaluation_results", help="Output directory for evaluation results")
    parser.add_argument("--split", type=str, default="test", help="Data split to use: 'train', 'val', 'test', or 'all' (default: 'test')")
    parser.add_argument("--no_videos", action="store_true", help="Disable video generation during batch evaluation")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of videos to generate during batch evaluation (default: 10)")
    args = parser.parse_args()

    config = Config()
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    model = DiffusionModel(config)
    checkpoint = torch.load(args.checkpoint_path, map_location=config.device)
    # Use strict=False to allow partial loading if architecture differs
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
    model.eval()
    
    # 2. Check if batch evaluation mode
    if args.batch_eval:
        print("Running batch evaluation mode...")
        encoder = MorganFingerprintEncoder()
        evaluate_batch(
            model, 
            args.data_dir, 
            args.metadata_file, 
            encoder, 
            args.paths_csv,
            num_samples=args.num_samples,
            output_dir=args.eval_output_dir,
            split=args.split if args.split.lower() != 'all' else None,
            generate_videos=not args.no_videos,
            num_videos=args.num_videos
        )
        return
    
    # 3. Single sample inference mode
    encoder = MorganFingerprintEncoder()
    data = load_data_sample(args.data_dir, args.metadata_file, encoder, paths_csv=args.paths_csv, split=args.split if args.split.lower() != 'all' else None)
    if data is None: return
    
    ctrl, real_target, fp, compound_name = data
    ctrl = ctrl.to(config.device)
    real_target = real_target.to(config.device)
    fp = fp.to(config.device)
    
    # 4. Generate Video Frames
    print(f"Generating diffusion trajectory for {compound_name}...")
    gen_frames = model.sample_trajectory(ctrl, fp, num_frames=60)
    
    # 5. Prepare Static Images for Comparison
    # Convert Control and Real Target to [0, 255] numpy images
    ctrl_img = ((ctrl[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
    real_img = ((real_target[0].cpu().permute(1,2,0) + 1) * 127.5).numpy().astype(np.uint8)
    
    # 6. Stitch Frames Together
    final_video = []
    separator = np.zeros((96, 2, 3), dtype=np.uint8) # Black line separator
    
    for frame in gen_frames:
        # Layout: [Control] | [Generated (Changing)] | [Real Ground Truth]
        combined = np.hstack([ctrl_img, separator, frame, separator, real_img])
        final_video.append(combined)
        
    # 7. Save
    imageio.mimsave(args.output_path, final_video, fps=10)
    print(f"Video saved to {args.output_path}")
    print("Left: Source (DMSO) | Middle: Generated Prediction | Right: Real Ground Truth")

if __name__ == "__main__":
    main()