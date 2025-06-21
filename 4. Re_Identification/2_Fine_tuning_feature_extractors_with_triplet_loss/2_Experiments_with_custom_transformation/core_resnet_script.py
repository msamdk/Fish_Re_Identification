#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- SCRIPT TO FINE-TUNE RESNET50 USING METRIC LEARNING (TRIPLET LOSS) ---
# --- Includes epoch-wise loss/accuracy plotting & saving embeddings ---
# --- NOW CORRECTLY CONFIGURED FOR ID-DISJOINT TRAIN/VALIDATION SETS ---
# --- Includes fixes for AttributeError and ValueError for NaN/Inf formatting ---

import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict, namedtuple
import time
import random
import copy # For saving best model state
import warnings # To handle warnings
import faiss # Ensure faiss is available
import traceback
import faiss

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F # Often used, good to import
import torchvision.transforms.functional as TF # For ResizeAndPadToSquare
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# --- Plotting Import ---
import matplotlib.pyplot as plt


# --- Check and Import Metric Learning Library ---
try:
    import pytorch_metric_learning
    from pytorch_metric_learning import distances, losses, miners, reducers
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    print(f"PyTorch Metric Learning Version: {pytorch_metric_learning.__version__}")
except ImportError:
    print("\n--- ERROR ---")
    print("pytorch-metric-learning library not found.")
    print("Please install it: pip install pytorch-metric-learning")
    print("-------------")
    exit()
except AttributeError:
    print("Could not retrieve pytorch-metric-learning version (likely older version).")


print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Faiss Version: {faiss.__version__ if hasattr(faiss, '__version__') else 'N/A'}")

# --- Configuration ---
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)
print(f"\n--- Configuration ---")
print(f"Using device: {DEVICE}")

# --- Paths ---
BASE_PATH = "/work3/msam/Thesis/autofish/"
OUTPUT_TRAIN_CROP_DIR = os.path.join(BASE_PATH, "metric_learning_gt_crops/train")
OUTPUT_VAL_CROP_DIR = os.path.join(BASE_PATH, "metric_learning_gt_crops/val")

TRAIN_METADATA_PATH = os.path.join(OUTPUT_TRAIN_CROP_DIR, "train_crop_metadata.json")
VAL_METADATA_PATH = os.path.join(OUTPUT_VAL_CROP_DIR, "val_crop_metadata.json")

EXPERIMENT_NAME = "Resnet_exp_8_batch_256_hard_margin_0.5_euclidean"
EXPERIMENT_BASE_DIR = os.path.join(BASE_PATH, "Re_ID_Experiments/new_order", EXPERIMENT_NAME)

MODEL_SAVE_DIR = os.path.join(EXPERIMENT_BASE_DIR, "finetuned_metric_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_resnet50_triplet_finetuned.pth")

PLOTS_DIR = os.path.join(EXPERIMENT_BASE_DIR, "plots_output")
os.makedirs(PLOTS_DIR, exist_ok=True)
EMBEDDINGS_SAVE_PATH = os.path.join(PLOTS_DIR, "validation_embeddings_for_tsne.npz")
ID_TO_LABEL_MAP_TRAIN_SAVE_PATH = os.path.join(PLOTS_DIR, "id_to_label_map_train.json")
ID_TO_LABEL_MAP_VAL_SAVE_PATH = os.path.join(PLOTS_DIR, "id_to_label_map_val.json")
LOSS_ACCURACY_PLOT_SAVE_PATH = os.path.join(PLOTS_DIR, "training_curves_plot.png")


print(f"Train Crop Metadata: {TRAIN_METADATA_PATH}")
print(f"Val Crop Metadata:   {VAL_METADATA_PATH}")
print(f"Model Save Directory: {MODEL_SAVE_DIR}")
print(f"Plots Directory: {PLOTS_DIR}")
print("-" * 20)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-5
NUM_EPOCHS = 300
P_IDENTITIES = 32
K_INSTANCES = 8
MARGIN = 0.5
EMBEDDING_DIM = 512
WEIGHT_DECAY = 1e-4
BATCH_SIZE = P_IDENTITIES * K_INSTANCES
IMG_SIZE = 224
BACKBONE_OUTPUT_DIM = 2048
NUM_WORKERS = 4

print("--- Hyperparameters ---")
print(f"LEARNING_RATE: {LEARNING_RATE}, NUM_EPOCHS: {NUM_EPOCHS}, MARGIN: {MARGIN}")
print(f"P_IDENTITIES: {P_IDENTITIES}, K_INSTANCES: {K_INSTANCES}, BATCH_SIZE: {BATCH_SIZE}")
print(f"EMBEDDING_DIM: {EMBEDDING_DIM}, WEIGHT_DECAY: {WEIGHT_DECAY}, IMG_SIZE: {IMG_SIZE}")
print("-" * 20)

# --- Model Definition ---
class FishReIDNet(nn.Module):
    def __init__(self, backbone_out_dim, embedding_dim, pretrained=True):
        super().__init__()
        print(f"Initializing FishReIDNet with ResNet50 backbone (pretrained={pretrained})")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim)
        )
        print(f"  Embedding head maps {backbone_out_dim} -> {embedding_dim}")

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return embeddings

# --- Custom Dataset (Corrected for ID-Disjoint Validation) ---
class FishCropDataset(Dataset):
    def __init__(self, metadata_path, transform=None, id_to_label_map=None, is_train=True, image_dir_override=None, build_map_if_not_train_and_no_map=False):
        self.metadata_path = metadata_path
        self.transform = transform
        self.is_train = is_train
        self.image_dir_override = image_dir_override
        print(f"Loading dataset metadata from: {metadata_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"  Loaded {len(self.metadata)} samples.")
            if not self.metadata: raise ValueError("Metadata is empty.")
            if not all('image_path' in item and 'fish_id' in item for item in self.metadata):
                raise ValueError("Metadata items require 'image_path' and 'fish_id'.")

            self.id_to_label_map = id_to_label_map if id_to_label_map is not None else {}
            self.next_available_label = 0
            if self.id_to_label_map:
                self.next_available_label = max(self.id_to_label_map.values()) + 1 if self.id_to_label_map else 0

            if self.is_train or (build_map_if_not_train_and_no_map and not id_to_label_map):
                self._build_label_map_internal()
            elif not self.is_train and not id_to_label_map and not build_map_if_not_train_and_no_map:
                print(f"  Warning: Validation dataset created without an initial id_to_label_map and 'build_map_if_not_train_and_no_map' is False. "
                      "IDs not in the provided map (if any was given) will get label -1.")
            elif not self.is_train and id_to_label_map:
                print(f"  Info: Validation dataset using a provided id_to_label_map. IDs not in this map will get label -1.")
        except Exception as e:
            raise IOError(f"Error reading or processing metadata from {metadata_path}: {e}\n{traceback.format_exc()}")

    def _build_label_map_internal(self):
        print(f"  Building Fish ID (string) to Integer Label Map for dataset: {os.path.basename(self.metadata_path)}...")
        unique_ids_str_in_metadata = sorted(list(set(str(item['fish_id']) for item in self.metadata)))
        newly_added_ids = 0
        for fish_id_str in unique_ids_str_in_metadata:
            if fish_id_str not in self.id_to_label_map:
                self.id_to_label_map[fish_id_str] = self.next_available_label
                self.next_available_label += 1
                newly_added_ids +=1
        print(f"  Label map building complete for {os.path.basename(self.metadata_path)}. Added {newly_added_ids} new IDs. Total unique IDs in this map now: {len(self.id_to_label_map)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]
        img_path_json = item['image_path']
        fish_id_str = str(item['fish_id'])
        if self.image_dir_override:
            img_path = os.path.join(self.image_dir_override, os.path.basename(img_path_json))
        else:
            img_path = img_path_json
        try:
            if not os.path.exists(img_path):
                # print(f"Warning: Image not found at {img_path}, returning zeros for item {index}")
                return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(-1, dtype=torch.long)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # print(f"Error loading image {img_path} for item {index}: {e}, returning zeros")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(-1, dtype=torch.long)
        
        label = self.id_to_label_map.get(fish_id_str, -1) # Use -1 for IDs not in map (esp. for val if map is from train)
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

#--- custom class to perform augmentation only for 20% of the 
class ProbabilisticAugmentor:
    """
    A custom transform that applies a set of augmentations with a given probability.
    The base transform (like resizing and normalization) is always applied.
    """
    def __init__(self, base_transform, augmentation_transform, p=0.2):
        self.base_transform = base_transform
        self.augmentation_transform = augmentation_transform
        self.p = p

    def __call__(self, img):
        # First, apply the base transformations that are always needed.
        img = self.base_transform(img)
        
        # With probability p, apply the additional augmentations.
        if random.random() < self.p:
            img = self.augmentation_transform(img)
            
        return img




# --- Custom PKSampler ---
class PKsampler(Sampler):
    def __init__(self, dataset, p, k):
        self.dataset = dataset; self.p = p; self.k = k
        self.labels_to_indices = defaultdict(list)
        print("PK Sampler: Populating labels_to_indices...")
        all_labels = []
        # Efficiently get labels if dataset object supports it directly (e.g. if labels are pre-calculated)
        # For FishCropDataset, we iterate as labels are determined in __getitem__ via id_to_label_map
        for idx in tqdm(range(len(dataset)), desc="  PK Sampler label collection"):
            try:
                _, label_tensor = dataset[idx] # This calls __getitem__
                label_item = label_tensor.item()
                all_labels.append(label_item)
            except Exception as e:
                # print(f"PK Sampler: Error getting label for idx {idx}: {e}. Appending -1.")
                all_labels.append(-1)

        for idx, label_item in enumerate(all_labels):
            if label_item != -1: # Valid label (not -1 from missing in map or error)
                self.labels_to_indices[label_item].append(idx) # Store index of instance in dataset

        self.labels_with_k_or_more = [lbl for lbl, indices in self.labels_to_indices.items() if len(indices) >= k]
        if not self.labels_with_k_or_more:
            print("PK Sampler WARNING: No labels have at least K instances. Sampler will be empty.")
            self.p_actual = 0
        else:
            self.p_actual = min(p, len(self.labels_with_k_or_more))
            if self.p_actual < p:
                print(f"PK Sampler WARNING: Requested P={p}, but only {len(self.labels_with_k_or_more)} labels have >= K instances. Using P_actual={self.p_actual}.")
        
        self.num_batches = len(self.labels_with_k_or_more) // self.p_actual if self.p_actual > 0 else 0
        print(f"PK Sampler: P_actual={self.p_actual}, K={k}, Total labels with >=K instances: {len(self.labels_with_k_or_more)}, Batches/Epoch={self.num_batches}")
        if self.num_batches == 0 and len(dataset) > 0 :
            print("PK Sampler CRITICAL WARNING: Sampler will produce 0 batches! Check P, K, and your dataset's label distribution.")

    def __len__(self):
        return self.num_batches # Number of batches per epoch

    def __iter__(self): # Returns list of indices for one batch
        if self.num_batches == 0:
            return iter([]) # Return an empty iterator if no batches

        available_labels = list(self.labels_with_k_or_more)
        random.shuffle(available_labels)
        
        batch_indices_list = [] # This will store lists of indices, one list per batch

        for i in range(self.num_batches):
            # Select P_actual labels for the current batch
            batch_labels = available_labels[i * self.p_actual : (i+1) * self.p_actual]
            
            current_batch_indices = []
            possible_batch = True
            for label in batch_labels:
                try:
                    indices_for_label = self.labels_to_indices[label]
                    sampled_indices = random.sample(indices_for_label, self.k)
                    current_batch_indices.extend(sampled_indices)
                except ValueError: # Not enough instances for this label (shouldn't happen if labels_with_k_or_more is correct)
                    possible_batch = False
                    # print(f"PK Sampler: Warning - label {label} had < K instances during batch creation. This shouldn't happen.")
                    break 
            
            if possible_batch and len(current_batch_indices) == self.p_actual * self.k:
                batch_indices_list.append(current_batch_indices)
            # Optional: Handle cases where the last few labels don't form a full p_actual set
            # For now, this strict // self.p_actual ensures full batches

        return iter(batch_indices_list) # This should return an iterator of lists of indices


# ---- custom function to prevent unwanted cropping of the fish with padding of black pixels----
class ResizeAndPadToSquare:
    def __init__(self, output_size_square, fill_color=(0, 0, 0)):
        """
        Args:
            output_size_square (int): The desired height and width of the square output image.
            fill_color (tuple): RGB tuple for the padding color. Default is black.
        """
        assert isinstance(output_size_square, int)
        self.output_size = output_size_square
        self.fill_color = fill_color

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to resize and pad.
        Returns:
            PIL Image: Resized and padded image.
        """
        original_w, original_h = img.size

        if original_w > original_h:
            new_w = self.output_size
            new_h = int(self.output_size * (original_h / original_w))
        elif original_h > original_w:
            new_h = self.output_size
            new_w = int(self.output_size * (original_w / original_h))
        else:
            new_w = self.output_size
            new_h = self.output_size

        resized_img = TF.resize(img, (new_h, new_w)) # TF.resize expects (h, w)

        pad_left = (self.output_size - new_w) // 2
        pad_right = self.output_size - new_w - pad_left
        pad_top = (self.output_size - new_h) // 2
        pad_bottom = self.output_size - new_h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        padded_img = TF.pad(resized_img, padding, fill=self.fill_color, padding_mode='constant')
        return padded_img


# --- Training Function (MODIFIED FOR ROBUST LOSS HANDLING/DISPLAY) ---
def train_one_epoch(model, train_loader, optimizer, loss_fn, miner, device, epoch_num):
    model.train()
    total_loss = 0.0
    num_valid_batches_processed = 0
    skipped_batches_miner = 0
    skipped_batches_invalid_samples = 0

    progress_bar = tqdm(train_loader, desc=f"  Epoch {epoch_num} - Training", leave=False, dynamic_ncols=True)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        valid_samples_idx = labels != -1
        if not torch.any(valid_samples_idx):
            skipped_batches_invalid_samples +=1
            continue

        images, labels = images[valid_samples_idx], labels[valid_samples_idx]

        if images.shape[0] < 2 or len(torch.unique(labels)) < 2 :
            # print(f"Skipping batch {batch_idx}: Not enough samples ({images.shape[0]}) or unique labels ({len(torch.unique(labels))}) for triplet mining.")
            skipped_batches_miner +=1
            continue

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images)

        try:
            hard_triplets = miner(embeddings, labels)
            if hard_triplets is None or (isinstance(hard_triplets, tuple) and (len(hard_triplets) < 3 or hard_triplets[0].numel() == 0)):
                # print(f"Skipping batch {batch_idx}: Miner returned no triplets.")
                skipped_batches_miner += 1
                continue
            loss = loss_fn(embeddings, labels, hard_triplets)
        except Exception as e:
            if "no triplets" in str(e).lower() or "at least 2 classes" in str(e).lower() or "must contain labels from at least 2 classes" in str(e).lower():
                # print(f"Skipping batch {batch_idx} due to miner error: {e}")
                skipped_batches_miner += 1
                continue
            else:
                print(f"Epoch {epoch_num} Training: Unexpected error in Batch {batch_idx} during mining/loss: {e}. Skipping batch.")
                # traceback.print_exc() # Optional: for more detailed error
                skipped_batches_miner += 1
                continue

        if loss is not None and torch.is_tensor(loss) and loss.numel() == 1 and torch.isfinite(loss):
            current_loss_item = loss.item()
            if not np.isnan(current_loss_item) and not np.isinf(current_loss_item) and current_loss_item >= 0: # Ensure item is finite and non-negative
                loss.backward()
                optimizer.step()
                total_loss += current_loss_item
                num_valid_batches_processed += 1

                loss_display_str = f"{current_loss_item:.4f}"
                avg_loss_calc_val = total_loss / num_valid_batches_processed if num_valid_batches_processed > 0 else 0.0

                if np.isnan(avg_loss_calc_val) or np.isinf(avg_loss_calc_val):
                    avg_loss_display_str = "N/A_avg"
                else:
                    avg_loss_display_str = f"{avg_loss_calc_val:.4f}"

                progress_bar.set_postfix_str(f"Loss: {loss_display_str}, Avg Loss: {avg_loss_display_str}, SkippedInv: {skipped_batches_invalid_samples}, SkippedMiner: {skipped_batches_miner}")
            else:
                # print(f"Skipping batch {batch_idx}: Invalid loss value {current_loss_item}")
                skipped_batches_miner +=1
        else:
            # print(f"Skipping batch {batch_idx}: Loss is None or not a valid tensor.")
            skipped_batches_miner +=1

    if skipped_batches_invalid_samples > 0:
        print(f"Epoch {epoch_num} Training: Skipped {skipped_batches_invalid_samples} batches due to all samples being invalid (label -1).")
    if skipped_batches_miner > 0:
        print(f"Epoch {epoch_num} Training: Skipped {skipped_batches_miner} batches due to miner/loss issues or invalid loss values.")

    if num_valid_batches_processed > 0:
        final_avg_loss = total_loss / num_valid_batches_processed
        if np.isnan(final_avg_loss) or np.isinf(final_avg_loss):
            print(f"Epoch {epoch_num} Training: Warning! Final total_loss is {total_loss} from {num_valid_batches_processed} batches, leading to non-finite average loss.")
            return float('nan')
        return final_avg_loss
    else:
        print(f"Epoch {epoch_num} Training: No valid batches were processed. Average loss is undefined.")
        return float('inf') # Or nan, to indicate undefined

# --- Validation Function (MODIFIED FOR ACCURACYCALCULATOR AND ROBUST LOSS HANDLING) ---
def validate_model(model, val_loader, device, accuracy_metrics_to_include, accuracy_k_value, loss_fn_val, miner_fn_val, epoch_num): # Pass direct values for accuracy calc
    model.eval()
    print(f"  Epoch {epoch_num} - Starting Validation...")
    all_val_embeddings_list = []
    all_val_labels_list = []
    total_val_loss = 0.0
    num_val_loss_batches = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"  Epoch {epoch_num} - Validating", leave=False, dynamic_ncols=True):
            valid_idx = labels != -1
            if not torch.any(valid_idx): continue
            images_batch, labels_batch = images[valid_idx], labels[valid_idx]
            if images_batch.shape[0] == 0: continue

            images_gpu = images_batch.to(device)
            labels_gpu = labels_batch.to(device)
            embeddings = model(images_gpu)

            all_val_embeddings_list.append(embeddings.cpu())
            all_val_labels_list.append(labels_batch.cpu())

            # Optional: Calculate validation loss (can be slow due to mining)
            try:
                if images_gpu.shape[0] >= 2 and len(torch.unique(labels_gpu)) >=2: # Need at least 2 samples and 2 unique classes for most miners
                    val_triplets = miner_fn_val(embeddings, labels_gpu) # Use passed validation miner
                    if val_triplets is not None and not (isinstance(val_triplets, tuple) and (len(val_triplets) < 3 or val_triplets[0].numel() == 0)):
                        val_loss = loss_fn_val(embeddings, labels_gpu, val_triplets) # Use passed validation loss_fn
                        if val_loss is not None and torch.is_tensor(val_loss) and val_loss.numel()==1 and torch.isfinite(val_loss): # Check tensor
                            current_val_loss_item = val_loss.item()
                            if not np.isnan(current_val_loss_item) and not np.isinf(current_val_loss_item) and current_val_loss_item >=0: # Check item
                                total_val_loss += current_val_loss_item
                                num_val_loss_batches += 1
            except Exception:
                # print(f"Epoch {epoch_num} Validation: Error during val loss calculation (Batch): {e_val_loss}. Skipping loss for this batch.")
                pass # Silently ignore validation loss calculation errors for a batch

    # Calculate average validation loss
    if num_val_loss_batches > 0:
        avg_val_loss_calc = total_val_loss / num_val_loss_batches
        if np.isnan(avg_val_loss_calc) or np.isinf(avg_val_loss_calc):
            print(f"  Epoch {epoch_num} - Warning: Validation total_loss became non-finite ({total_val_loss}) from {num_val_loss_batches} batches, resulting in non-finite avg_val_loss.")
            avg_val_loss = float('nan')
        else:
            avg_val_loss = avg_val_loss_calc
    else:
        # print(f"  Epoch {epoch_num} - Validation: No valid batches for loss calculation.")
        avg_val_loss = float('inf') # Or nan, if no loss calculated

    if not all_val_embeddings_list:
        print(f"  Epoch {epoch_num} - Warning: No valid validation embeddings collected. Cannot calculate accuracy.")
        return {'precision_at_1': 0.0, 'mean_average_precision_at_r': 0.0}, avg_val_loss

    all_val_embeddings = torch.cat(all_val_embeddings_list)
    all_val_labels = torch.cat(all_val_labels_list)

    print(f"  Epoch {epoch_num} - Validation samples for metrics: {all_val_embeddings.shape[0]}, Embeddings Shape: {all_val_embeddings.shape}, Labels Shape: {all_val_labels.shape}")

    if torch.isnan(all_val_embeddings).any() or torch.isinf(all_val_embeddings).any():
        print(f"  Epoch {epoch_num} - FATAL: NaNs/Infs detected in collected validation embeddings. Cannot calculate accuracy.");
        return {'precision_at_1': 0.0, 'mean_average_precision_at_r': 0.0}, avg_val_loss

    if len(all_val_labels) < 2 or len(torch.unique(all_val_labels)) < 2:
        print(f"  Epoch {epoch_num} - Warning: Validation set too small or too few unique labels ({len(torch.unique(all_val_labels))}) for meaningful metric calculation.");
        return {'precision_at_1': 0.0, 'mean_average_precision_at_r': 0.0}, avg_val_loss

    accuracies = {}
    # print(f"  Epoch {epoch_num} - DIAGNOSTIC: Forcing CPU for all accuracy calculations.")
    try:
        # Ensure embeddings and labels are on CPU for AccuracyCalculator if it's set to CPU
        cpu_embeddings = all_val_embeddings.cpu() # Already on CPU from list append
        cpu_labels = all_val_labels.cpu() # Already on CPU

        accuracy_calculator_cpu = AccuracyCalculator(
            include=accuracy_metrics_to_include,
            k=accuracy_k_value,
            device=torch.device("cpu") # Explicitly set for clarity, though PML might handle it
        )
        accuracies = accuracy_calculator_cpu.get_accuracy(
            query=cpu_embeddings, reference=cpu_embeddings,
            query_labels=cpu_labels, reference_labels=cpu_labels
            # exclude_and_ref_are_same=True # Already default
        )
    except Exception as e_cpu:
        print(f"  Epoch {epoch_num} - ERROR during FORCED CPU accuracy calculation: {e_cpu}\n{traceback.format_exc()}")
        accuracies = {'precision_at_1': 0.0, 'mean_average_precision_at_r': 0.0} # Default on error

    print(f"  Epoch {epoch_num} - Validation finished.");
    return accuracies, avg_val_loss

# --- Function to get and save embeddings ---
def extract_and_save_embeddings(model, dataloader, device, save_path, dataset_label_map): # dataset_label_map for potential future use with string labels
    model.eval()
    all_embeddings_list, all_original_labels_list = [], []
    print(f"Extracting embeddings from dataset for saving to '{save_path}'...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Embeddings"):
            valid_idx = labels != -1; # Filter out any invalid labels if they exist
            if not torch.any(valid_idx): continue
            images_b, labels_b = images[valid_idx], labels[valid_idx]
            if images_b.shape[0] == 0: continue

            embeddings = model(images_b.to(device))
            all_embeddings_list.append(embeddings.cpu().numpy())
            all_original_labels_list.append(labels_b.cpu().numpy()) # These are integer labels

    if not all_embeddings_list:
        print("No embeddings extracted. File not saved.")
        return False

    all_emb_np = np.vstack(all_embeddings_list)
    all_lbl_np = np.concatenate(all_original_labels_list)
    try:
        np.savez_compressed(save_path, embeddings=all_emb_np, labels=all_lbl_np)
        print(f"Embeddings and integer labels saved to '{save_path}'")
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False

# --- Function to plot losses and mAP@R accuracy ---
def plot_training_curves(train_losses, val_losses, val_map_rs, num_epochs_completed, save_path):
    epochs_range = range(1, num_epochs_completed + 1)
    plt.figure(figsize=(15, 5))

    # Subplot 1: Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')

    valid_val_loss_indices = [i for i, vl in enumerate(val_losses) if vl is not None and not np.isinf(vl) and not np.isnan(vl)]
    if valid_val_loss_indices:
        epochs_for_val_loss = np.array(epochs_range)[valid_val_loss_indices]
        plot_val_losses = np.array(val_losses)[valid_val_loss_indices]
        plt.plot(epochs_for_val_loss, plot_val_losses, 'ro-', label='Validation Loss')
    else:
        print("Plotting: No valid validation loss points to plot.")

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    # Subplot 2: Validation mAP@R Accuracy
    plt.subplot(1, 2, 2)
    if val_map_rs:
        valid_map_r_indices = [i for i, map_r in enumerate(val_map_rs) if map_r is not None]
        if valid_map_r_indices:
            epochs_for_map_r = np.array(epochs_range)[valid_map_r_indices]
            plot_map_r_accuracies = np.array([val_map_rs[i] for i in valid_map_r_indices]) * 100
            plt.plot(epochs_for_map_r, plot_map_r_accuracies, 'go-', label='Validation mAP@R Accuracy')
        else:
            print("Plotting: No valid mAP@R accuracy points to plot.")
    else:
        print("Plotting: val_map_rs list is empty.")

    plt.title('Validation mAP@R Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('mAP@R Accuracy (%)')
    plt.legend(); plt.grid(True)

    plt.suptitle(f'Training Curves after {num_epochs_completed} Epochs', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(save_path)
        print(f"Training curves plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()


# --- Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()
    print("\n--- Setting up Datasets and Loaders ---")
#transfornations to the training set
    train_transform = transforms.Compose([
        ResizeAndPadToSquare(IMG_SIZE, fill_color=(0,0,0)), #my custom resizing function other than the resizing and center cropping
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.35, saturation=0.38, hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0495, 0.0503, 0.0535], std=[0.1370, 0.1363, 0.1412]), #custom calculated mean and std for autofish
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    val_transform = transforms.Compose([
        ResizeAndPadToSquare(IMG_SIZE, fill_color=(0,0,0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0495, 0.0503, 0.0535], std=[0.1370, 0.1363, 0.1412]) #same custom normaliztion
    ])
    try:
        train_dataset = FishCropDataset(TRAIN_METADATA_PATH, transform=train_transform, is_train=True, image_dir_override=OUTPUT_TRAIN_CROP_DIR)
        if train_dataset.id_to_label_map:
            try:
                with open(ID_TO_LABEL_MAP_TRAIN_SAVE_PATH, 'w') as f: json.dump(train_dataset.id_to_label_map, f, indent=4)
                print(f"ID to Label map for TRAINING saved to: {ID_TO_LABEL_MAP_TRAIN_SAVE_PATH}")
            except Exception as e: print(f"Error saving training id_to_label_map: {e}")
        
        val_dataset = FishCropDataset(VAL_METADATA_PATH, transform=val_transform, id_to_label_map=None,
                                      is_train=False, build_map_if_not_train_and_no_map=True, image_dir_override=OUTPUT_VAL_CROP_DIR)

        if val_dataset.id_to_label_map:
            try:
                with open(ID_TO_LABEL_MAP_VAL_SAVE_PATH, 'w') as f: json.dump(val_dataset.id_to_label_map, f, indent=4)
                print(f"ID to Label map for VALIDATION saved to: {ID_TO_LABEL_MAP_VAL_SAVE_PATH}")
            except Exception as e: print(f"Error saving validation id_to_label_map: {e}")

    except Exception as e:
        print(f"FATAL Error initializing datasets: {e}\n{traceback.format_exc()}"); exit()

    if len(train_dataset) == 0: print("FATAL: Training dataset is empty."); exit()
    if len(val_dataset) == 0: print("Warning: Validation dataset is empty. Validation steps will be skipped.")

    try:
        train_sampler = PKsampler(train_dataset, p=P_IDENTITIES, k=K_INSTANCES)
        if len(train_sampler) == 0 and len(train_dataset) > 0:
            print("FATAL: PK Sampler created 0 batches for a non-empty training dataset. Check P, K, and data distribution."); exit()
        
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS,
                                  pin_memory=(DEVICE.type=="cuda"), persistent_workers=(NUM_WORKERS > 0))
        val_loader = None
        if len(val_dataset) > 0:
            val_bs = BATCH_SIZE * 2
            if len(val_dataset) < val_bs: val_bs = max(1, len(val_dataset))
            val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=NUM_WORKERS,
                                    pin_memory=(DEVICE.type=="cuda"), persistent_workers=(NUM_WORKERS > 0))
    except Exception as e: print(f"FATAL Error creating dataloaders: {e}\n{traceback.format_exc()}"); exit()

    print("\n--- Initializing Model and Training Components ---")
    model = FishReIDNet(BACKBONE_OUTPUT_DIM, EMBEDDING_DIM, pretrained=True); model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True, min_lr=1e-7)

    distance_fn = distances.LpDistance(p=2); reducer_fn = reducers.AvgNonZeroReducer()
    loss_func = losses.TripletMarginLoss(margin=MARGIN, distance=distance_fn, reducer=reducer_fn)
    miner_func = miners.TripletMarginMiner(margin=MARGIN, distance=distance_fn, type_of_triplets="hard")
    print(f"Using TripletMarginLoss with margin={MARGIN} and TripletMarginMiner (type: hard)")

    ACC_METRICS_TO_INCLUDE = ("precision_at_1", "mean_average_precision_at_r")
    ACC_K_VALUE = None

    print("\n--- Starting Training ---")
    best_val_metric = -1.0; best_epoch = -1
    # needed documenting # loss and validation accuracy (mAP@R)
    train_losses_history, val_losses_history, val_map_r_history = [], [], []

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time(); print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, miner_func, DEVICE, epoch + 1)
        train_losses_history.append(avg_train_loss)

        avg_loss_epoch_str = f"{avg_train_loss:.4f}" if not (np.isnan(avg_train_loss) or np.isinf(avg_train_loss)) else "N/A"
        print(f"  Epoch {epoch+1} Training - Avg Loss: {avg_loss_epoch_str}")

        current_val_map_r, current_avg_val_loss = 0.0, float('inf')

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_accuracies, avg_val_loss_epoch = validate_model(
                model, val_loader, DEVICE, ACC_METRICS_TO_INCLUDE, ACC_K_VALUE,
                loss_func, miner_func, epoch + 1
            )
            # We only need mAP@R, so we don't need to store P@1
            current_val_map_r = val_accuracies.get('mean_average_precision_at_r', 0.0)
            current_avg_val_loss = avg_val_loss_epoch

            avg_val_loss_display_str = f"{current_avg_val_loss:.4f}" if not (np.isnan(current_avg_val_loss) or np.isinf(current_avg_val_loss)) else "N/A"
            mapr_display_str = f"{current_val_map_r*100:.2f}%"
            
            print(f"  Epoch {epoch+1} Validation - Avg Loss: {avg_val_loss_display_str}, MAP@R: {mapr_display_str}")

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_metric = current_avg_val_loss
                if np.isnan(scheduler_metric) or np.isinf(scheduler_metric): scheduler_metric = 1e9
                scheduler.step(scheduler_metric)
            elif scheduler is not None:
                scheduler.step()

            if current_val_map_r > best_val_metric:
                print(f"  Val mAP@R improved ({best_val_metric*100:.2f}% -> {current_val_map_r*100:.2f}%). Saving model...")
                best_val_metric = current_val_map_r; best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                    print(f"  Best model saved: {BEST_MODEL_SAVE_PATH}")
                except Exception as e: print(f"  ERROR saving best model: {e}")
        else:
            print("  Skipping validation: validation dataloader not available or dataset empty.")
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        val_losses_history.append(current_avg_val_loss)
        val_map_r_history.append(current_val_map_r if val_loader else None)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            # CORRECTED: This now checks if the current model is the best based on mAP@R
            is_best_model_so_far = (val_loader is not None and current_val_map_r == best_val_metric and epoch + 1 == best_epoch)
            if not is_best_model_so_far:
                chkpt_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.pth")
                try:
                    torch.save(model.state_dict(), chkpt_path)
                    print(f"  Checkpoint model saved: {chkpt_path}")
                except Exception as e: print(f"  ERROR saving checkpoint model: {e}")

        print(f"  Epoch {epoch+1} finished in {time.time() - epoch_start_time:.2f}s. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("\n--- Training Finished ---")
    final_model_to_report = BEST_MODEL_SAVE_PATH if best_epoch != -1 and os.path.exists(BEST_MODEL_SAVE_PATH) else \
                             os.path.join(MODEL_SAVE_DIR, f"model_epoch_{NUM_EPOCHS}.pth")
    
    if os.path.exists(final_model_to_report):
        print(f"Best/Last model (MAP@R: {best_val_metric*100:.2f}% @ Epoch {best_epoch if best_epoch !=-1 else NUM_EPOCHS}) is at: {final_model_to_report}")
    else:
        print(f"No model checkpoint found at expected final path: {final_model_to_report}.")

    if NUM_EPOCHS > 0 :
        # CORRECTED: Pass the mAP@R history to the plotting function
        plot_training_curves(train_losses_history, val_losses_history, val_map_r_history, NUM_EPOCHS, save_path=LOSS_ACCURACY_PLOT_SAVE_PATH)

    model_for_extraction_path = final_model_to_report
    if os.path.exists(model_for_extraction_path) and val_loader is not None and len(val_loader.dataset) > 0:
        print(f"\n--- Extracting and Saving Fine-Tuned Embeddings using model: {os.path.basename(model_for_extraction_path)} ---")
        extraction_model = FishReIDNet(BACKBONE_OUTPUT_DIM, EMBEDDING_DIM, pretrained=False)
        extraction_model.to(DEVICE)
        try:
            extraction_model.load_state_dict(torch.load(model_for_extraction_path, map_location=DEVICE))
            if extract_and_save_embeddings(extraction_model, val_loader, DEVICE, EMBEDDINGS_SAVE_PATH, val_dataset.id_to_label_map):
                print(f"--- Embeddings from validation set saved to '{EMBEDDINGS_SAVE_PATH}' ---")
                print(f"--- Corresponding validation label map is at '{ID_TO_LABEL_MAP_VAL_SAVE_PATH}' ---")
        except Exception as e: print(f"Error during setup for embedding extraction: {e}\n{traceback.format_exc()}")

    print(f"\nTotal script run time: {time.time() - main_start_time:.2f}s ({(time.time() - main_start_time)/60:.2f}m)")
    print(f"\n*** Use saved model (e.g., '{os.path.basename(final_model_to_report)}') for EVALUATION script ***")
    print(f"*** Visualize saved embeddings using separate script with '{EMBEDDINGS_SAVE_PATH}' and '{ID_TO_LABEL_MAP_VAL_SAVE_PATH}' ***")

