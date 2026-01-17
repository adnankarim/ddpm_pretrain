"""
================================================================================
BBBC021 STABLE DIFFUSION LoRA FINE-TUNING
================================================================================
Stage 1: Learning the Cell Prior via LoRA
--------------------------------------------------------------------------------
This script fine-tunes Stable Diffusion v1-5 on the 'train' split of the dataset.
It uses LoRA (Low-Rank Adaptation) to efficiently learn the style of the cells
without retraining the entire massive model.

Input: "microscopy image of a biological cell" + Random Noise
Output: Realistic Cell Image
Data: Train Split Only
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
from torch.utils.data import DataLoader, Dataset, Subset
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
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("CRITICAL: Libraries missing. Install: pip install diffusers transformers peft accelerate")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    data_dir = "./data/bbbc021_all"
    metadata_file = "metadata/bbbc021_df_all.csv"
    
    # Stable Diffusion requires 512x512 for best results. 
    # 96x96 is too small for the VAE (which compresses 8x -> 12x12 latent).
    image_size = 512 
    
    # Model
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    revision = None
    variant = None
    
    # LoRA Config
    lora_rank = 4          # Rank of the LoRA matrices
    lora_alpha = 4         # Scaling factor
    lora_dropout = 0.0     # Dropout probability
    lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"] # Apply LoRA to attention layers
    
    # Text Prompt (Trigger word)
    instance_prompt = "microscopy image of a biological cell"
    
    # Training
    epochs = 100
    batch_size = 1 # SD is heavy, batch size 1 is common on consumer GPUs
    gradient_accumulation_steps = 4
    lr = 1e-4
    save_freq = 5
    
    output_dir = "sd_lora_cell_results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = "fp16" # "no", "fp16", "bf16"

# ============================================================================
# DATASET (ADAPTED FOR SD)
# ============================================================================

class BBBC021LoRADataset(Dataset):
    def __init__(self, data_dir, metadata_file, tokenizer, size=512, split='train', paths_csv=None):
        self.data_dir = Path(data_dir).resolve()
        self.size = size
        self.tokenizer = tokenizer
        self._first_load_logged = False  # Track if we've logged the first successful load
        
        # Robust CSV Loading (same as train.py)
        csv_full_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(csv_full_path):
            csv_full_path = metadata_file  # Try relative path
        
        if not os.path.exists(csv_full_path):
            raise FileNotFoundError(f"Cannot find metadata CSV at {csv_full_path}")
            
        df = pd.read_csv(csv_full_path)
        if 'SPLIT' in df.columns:
            df = df[df['SPLIT'].str.lower() == split.lower()]
            
        self.metadata = df.to_dict('records')
        
        # Load paths.csv for robust file lookup (same as train.py)
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
                filename = str(row['filename'])  # Ensure string
                rel_path = str(row['relative_path'])  # Ensure string
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

        # Transforms for Stable Diffusion
        self.train_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # SD expects [-1, 1]
        ])

    def _find_file_path(self, path):
        """
        Robust file path finding using paths.csv lookup (same logic as train.py).
        Returns Path object if found, None otherwise.
        """
        if not path:
            return None
        
        path_str = str(path)  # Ensure string
        path_obj = Path(path_str)
        filename = path_obj.name
        basename = path_obj.stem  # filename without extension
        
        # Strategy 1: Parse SAMPLE_KEY format (Week7_34681_7_3338_348.0 -> Week7/34681/7_3338_348.0.npy)
        if '_' in path_str and path_str.startswith('Week'):
            parts = path_str.replace('.0', '').split('_')
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
                            # Handle path resolution
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
        for candidate in [self.data_dir / path_str, self.data_dir / (path_str + '.npy')]:
            if candidate.exists():
                return candidate
        
        # Last resort: Recursive search
        search_pattern = filename if filename.endswith('.npy') else filename + '.npy'
        matches = list(self.data_dir.rglob(search_pattern))
        if matches:
            return matches[0]
        
        # Also try recursive search for SAMPLE_KEY in directory structure
        if '_' in path_str:
            parts = path_str.split('_')
            if len(parts) >= 2:
                week_part = parts[0]  # Week7
                batch_part = parts[1] if len(parts) > 1 else None  # 34681
                
                if batch_part:
                    search_dir = self.data_dir / week_part / batch_part
                    if search_dir.exists():
                        search_pattern = path_str if path_str.endswith('.npy') else path_str + '.npy'
                        matches = list(search_dir.rglob(search_pattern))
                        if matches:
                            return matches[0]
        
        return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        path = meta.get('image_path') or meta.get('SAMPLE_KEY')
        
        if not path:
            raise ValueError(
                f"CRITICAL: No image path found in metadata!\n"
                f"  Index: {idx}\n"
                f"  Metadata keys: {list(meta.keys())}"
            )
        
        # Use robust path finding (same as train.py)
        full_path = self._find_file_path(path)
        
        # CRITICAL: Check if file exists before attempting to load
        if full_path is None or not full_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Image file not found!\n"
                f"  Index: {idx}\n"
                f"  Path from metadata: {path}\n"
                f"  Data directory: {self.data_dir}\n"
                f"  Data directory exists: {self.data_dir.exists()}\n"
                f"  paths.csv loaded: {len(self.paths_lookup) > 0}"
            )
        
        try:
            # Get file size before loading
            file_size_bytes = full_path.stat().st_size if full_path.exists() else 0
            
            img = np.load(str(full_path))
            original_shape = img.shape
            original_dtype = img.dtype
            original_min = float(img.min())
            original_max = float(img.max())
            
            # Handle shapes
            if img.ndim == 3 and img.shape[0] == 3: 
                img = img.transpose(1, 2, 0)  # CHW -> HWC for PIL
            
            # Normalize to 0-255 uint8 for PIL
            if img.max() > 1.0:
                # Already in 0-255 range
                img = img.astype(np.uint8)
            else:
                # Normalize from [0, 1] or [-1, 1] to [0, 255]
                if img.min() < 0:
                    img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                img = (img * 255).astype(np.uint8)
            
            # If grayscale, make RGB
            if img.ndim == 2: 
                img = np.stack([img]*3, axis=-1)
            
            # Log details for first successful load (or first few)
            if not self._first_load_logged or idx < 3:
                print(f"\n{'='*60}", flush=True)
                print(f"✓ Successfully loaded image #{idx}", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"  File path: {full_path}", flush=True)
                print(f"  File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024:.2f} KB)", flush=True)
                print(f"  Original shape: {original_shape} (dtype: {original_dtype})", flush=True)
                print(f"  Original range: [{original_min:.2f}, {original_max:.2f}]", flush=True)
                print(f"  Processed shape: {img.shape} (dtype: {img.dtype})", flush=True)
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
            
        pixel_values = self.train_transforms(image_pil)

        # Tokenize Text Prompt
        # We use the same prompt for all images to teach the concept
        text_inputs = self.tokenizer(
            Config.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids[0],
        }

# ============================================================================
# LOGGING UTILS
# ============================================================================

class TrainingLogger:
    """
    Logs training metrics to CSV and generates plots every epoch.
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = {
            'epoch': [],
            'train_loss': [],
            'learning_rate': []
        }
        self.csv_path = os.path.join(save_dir, "training_history.csv")
        self.plot_path = os.path.join(save_dir, "training_loss.png")
        
    def update(self, epoch, loss, lr=None):
        """
        Update logger with training loss and learning rate.
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(loss)
        self.history['learning_rate'].append(lr if lr is not None else 0)
        
        # Save to CSV immediately
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # Generate Plot
        self._plot_loss()
        
    def _plot_loss(self):
        """Plot training loss curve with learning rate"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Loss on left y-axis
        color = '#1f77b4'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss', color=color)
        line1 = ax1.plot(self.history['epoch'], self.history['train_loss'], 
                        label='Training Loss', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')  # Log scale for diffusion loss
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
        
        plt.title(f'Stable Diffusion LoRA Training (Epoch {self.history["epoch"][-1]})')
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA on BBBC021")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--paths_csv", type=str, default=None, help="Path to paths.csv file (auto-detected if not specified)")
    args = parser.parse_args()
    
    config = Config()
    config.output_dir = args.output_dir
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{config.output_dir}/validation", exist_ok=True)
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    
    # Initialize Logger
    logger = TrainingLogger(config.output_dir)
    
    if WANDB_AVAILABLE: wandb.init(project="bbbc021-sd-lora", config=config.__dict__)

    print(f"Loading Stable Diffusion components from {config.pretrained_model_name_or_path}...")

    # 1. Load Tokenizer & Encoder
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
    
    # 2. Load VAE (Autoencoder)
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
    
    # 3. Load UNet
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")
    
    # 4. Freeze Frozen Components
    # We only train LoRA on UNet. VAE and Text Encoder stay frozen.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # Freeze base UNet

    # 5. Inject LoRA Adapters
    print(f"Injecting LoRA adapters (Rank: {config.lora_rank})...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    # This wraps the UNet and adds the trainable low-rank matrices
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters() # Verify we are training only a small %

    # Move to device
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16": weight_dtype = torch.float16
    elif config.mixed_precision == "bf16": weight_dtype = torch.bfloat16
    
    vae.to(config.device, dtype=weight_dtype)
    text_encoder.to(config.device, dtype=weight_dtype)
    unet.to(config.device) # UNet usually kept in float32 for training stability, or mixed precision context

    # 6. Dataset & Loader
    print("\nLoading Dataset...", flush=True)
    # Load 'train' split only, then sample 10% from all weeks
    dataset_full = BBBC021LoRADataset(config.data_dir, config.metadata_file, tokenizer, size=config.image_size, split='train', paths_csv=args.paths_csv)
    print(f"Total Train Images Available: {len(dataset_full):,}")
    
    # Sample 10% of the train dataset randomly (from all weeks)
    import random
    random.seed(42)  # For reproducibility
    total_size = len(dataset_full)
    sample_size = max(1, int(total_size * 0.10))  # At least 1 sample
    indices = random.sample(range(total_size), sample_size)
    dataset = Subset(dataset_full, indices)
    print(f"Using 10% of train dataset: {len(dataset):,} images ({sample_size/total_size*100:.1f}%)", flush=True)
    
    # Log paths.csv status (access through dataset_full since Subset doesn't have these attributes)
    print(f"\n{'='*60}", flush=True)
    print(f"File Path Resolution Status:", flush=True)
    print(f"{'='*60}", flush=True)
    if len(dataset_full.paths_lookup) > 0:
        print(f"  ✓ paths.csv loaded successfully", flush=True)
        print(f"  - Unique filenames in lookup: {len(dataset_full.paths_lookup):,}", flush=True)
        print(f"  - Total paths indexed: {len(dataset_full.paths_by_rel):,}", flush=True)
        print(f"  - Basename lookups: {len(dataset_full.paths_by_basename):,}", flush=True)
    else:
        print(f"  ⚠ paths.csv not found - using fallback path resolution", flush=True)
    print(f"  - Data directory: {dataset_full.data_dir}", flush=True)
    print(f"  - Data directory exists: {dataset_full.data_dir.exists()}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Save a random dataset sample image
    print("Saving random dataset sample image...", flush=True)
    try:
        import random
        random_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[random_idx]
        
        # Get pixel values and convert to image
        pixel_values = sample['pixel_values']  # Already normalized to [-1, 1]
        img_np = ((pixel_values.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        img_np = np.clip(img_np, 0, 255)
        
        img_pil = Image.fromarray(img_np)
        sample_filename = f"lora_dataset_sample.jpg"
        absolute_path = os.path.abspath(sample_filename)
        img_pil.save(absolute_path, "JPEG", quality=95)
        print(f"  ✓ Saved random sample to: {absolute_path}", flush=True)
        print(f"  (Current working directory: {os.getcwd()})", flush=True)
        print(f"  Image shape: {pixel_values.shape}", flush=True)
    except Exception as e:
        print(f"  ⚠ Warning: Could not save sample image: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4
    )

    # 7. Optimizer & Scheduler
    # Only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Learning Rate Scheduler - Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,  # Full training cycle
        eta_min=1e-6  # Minimum learning rate
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Resume training if requested
    start_epoch = 0
    ckpt_path = f"{config.output_dir}/checkpoints/latest.pt"
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location=config.device)
            # Load LoRA weights
            if 'lora_state_dict' in ckpt:
                unet.load_state_dict(ckpt['lora_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0)
            print(f"  Resumed from epoch {start_epoch}, current LR: {scheduler.get_last_lr()[0]:.2e}")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            print(f"  Starting from epoch 1...")
    
    print("\nStarting LoRA Fine-Tuning...")
    
    global_step = 0
    
    for epoch in range(start_epoch, config.epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        epoch_losses = []
        
        for step, batch in enumerate(train_dataloader):
            # A. Prepare Inputs
            pixel_values = batch["pixel_values"].to(config.device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(config.device)
            
            # B. Convert Images to Latents (VAE)
            # Map [-1, 1] images to Latent Space
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor # Scaling required for SD
            
            # C. Get Text Embeddings (CLIP)
            encoder_hidden_states = text_encoder(input_ids)[0]
            
            # D. Sample Noise & Timesteps
            bsz = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            
            # Add noise to latents (Forward Diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # E. Predict Noise (UNet + LoRA)
            # The UNet takes noisy latents, timestep, and text embeddings
            # Note: We cast input to float32 if unet is float32
            if config.mixed_precision == "fp16":
                with torch.cuda.amp.autocast():
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            else:
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # F. Loss & Backprop
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/Inf loss detected at step {step}, skipping this batch", flush=True)
                optimizer.zero_grad()
                continue
            
            loss.backward()
            epoch_losses.append(loss.item())
            
            # Gradient Accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            if WANDB_AVAILABLE: wandb.log({"loss": loss.item(), "step": global_step})
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = np.mean(epoch_losses)
        
        print(f"\nEpoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.5f} | LR: {current_lr:.2e}", flush=True)
        
        # Logging & Plotting
        logger.update(epoch+1, avg_loss, current_lr)
        if WANDB_AVAILABLE: 
            wandb.log({
                "epoch": epoch+1,
                "epoch_loss": avg_loss,
                "learning_rate": current_lr
            })

        # Save Checkpoint (Only LoRA weights)
        if (epoch + 1) % config.save_freq == 0:
            save_path = os.path.join(config.output_dir, "checkpoints", f"lora_epoch_{epoch+1}")
            # This method only saves the adapter weights, keeping file size small (~few MBs)
            unet.save_pretrained(save_path)
            
            # Also save full checkpoint with optimizer and scheduler
            torch.save({
                'lora_state_dict': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1,
                'config': config.__dict__
            }, f"{config.output_dir}/checkpoints/latest.pt")
            
            print(f"Saved LoRA weights to {save_path}")
            print(f"Saved full checkpoint to {config.output_dir}/checkpoints/latest.pt (LR: {current_lr:.2e})")
            
            # Validation generation
            run_validation(epoch, unet, vae, tokenizer, text_encoder, noise_scheduler, config, weight_dtype)

def run_validation(epoch, unet, vae, tokenizer, text_encoder, noise_scheduler, config, dtype):
    """
    Runs a quick inference to see how the model is learning the cell prior.
    """
    print("Running validation sampling...")
    unet.eval()
    
    # Use float32 for validation to avoid dtype mismatches with LoRA
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet, # This UNet has the LoRA weights loaded
        safety_checker=None,
        torch_dtype=torch.float32  # Use float32 for stability during validation
    )
    pipeline.to(config.device)
    
    # Ensure UNet is in float32 to match pipeline expectations
    pipeline.unet = pipeline.unet.float()
    
    with torch.no_grad():
        # Generate 4 images
        images = pipeline(
            prompt=Config.instance_prompt, 
            num_inference_steps=30, 
            num_images_per_prompt=4
        ).images
        
    # Save grid
    os.makedirs(f"{config.output_dir}/validation", exist_ok=True)
    
    # Create grid
    w, h = images[0].size
    grid = Image.new('RGB', (w*2, h*2))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (w, 0))
    grid.paste(images[2], (0, h))
    grid.paste(images[3], (w, h))
    
    grid.save(f"{config.output_dir}/validation/epoch_{epoch+1}.png")
    
    # Clean up to save memory
    del pipeline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()