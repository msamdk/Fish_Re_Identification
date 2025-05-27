#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- SCRIPT TO FINE-TUNE RESNET50 USING METRIC LEARNING (TRIPLET LOSS) ---
# --- Assumes pre-processed GT crops and JSON metadata files exist ---

import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict, namedtuple
import time
import random
import copy # For saving best model state

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# --- Check and Import Metric Learning Library ---
try:
    import pytorch_metric_learning # Import top-level
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
print(f"Torchvision Version: {torchvision.__version__}") # <--- CORRECTED LINE



# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n--- Configuration ---")
print(f"Using device: {DEVICE}")

# --- !!! IMPORTANT PATHS - VERIFY THESE !!! ---
BASE_PATH = "/work3/msam/Thesis/autofish/" # <--- Adjust if your base path is different

# Directories containing PRE-PROCESSED GT CROPS and METADATA JSON files
# These MUST point to the output of your 'prepare_gt_crops.py' script
OUTPUT_TRAIN_CROP_DIR = os.path.join(BASE_PATH, "metric_learning_instance_split/train_is") # Contains train_crop_metadata.json
OUTPUT_VAL_CROP_DIR   = os.path.join(BASE_PATH, "metric_learning_instance_split/val_is")   # Contains val_crop_metadata.json
TRAIN_METADATA_PATH = os.path.join(OUTPUT_TRAIN_CROP_DIR, "train_crop_metadata.json")
VAL_METADATA_PATH   = os.path.join(OUTPUT_VAL_CROP_DIR, "val_crop_metadata.json")

# --- Directory for Saving Fine-Tuned Models ---
MODEL_SAVE_DIR = os.path.join(BASE_PATH, "finetuned_metric_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create directory if it doesn't exist
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_resnet50_triplet_finetuned.pth") # Output file

print(f"Train Crop Metadata: {TRAIN_METADATA_PATH}")
print(f"Val Crop Metadata:   {VAL_METADATA_PATH}")
print(f"Model Save Directory: {MODEL_SAVE_DIR}")
print("-" * 20)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-5
NUM_EPOCHS    = 50
P_IDENTITIES  = 8
K_INSTANCES   = 4
MARGIN        = 0.3
EMBEDDING_DIM = 512
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = P_IDENTITIES * K_INSTANCES
IMG_SIZE      = 224
BACKBONE_OUTPUT_DIM = 2048 # ResNet50 specific
NUM_WORKERS   = 4

print("--- Hyperparameters ---")
# ... (print statements for hyperparameters from previous script) ...
print(f"  Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}, P: {P_IDENTITIES}, K: {K_INSTANCES}")
print(f"  Batch Size: {BATCH_SIZE}, Margin: {MARGIN}, Embedding Dim: {EMBEDDING_DIM}")
print("-" * 20)


# --- Model Definition (ResNet50 specific) ---
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
        embedding = self.embedding_head(features)
        return embedding

# --- Custom Dataset (Uses JSON Metadata) ---
class FishCropDataset(Dataset):
    # ... (Exactly the same as the version in the previous 'full script for training' response) ...
    def __init__(self, metadata_path, transform=None, id_to_label_map=None, is_train=True):
        self.metadata_path = metadata_path
        self.transform = transform
        self.is_train = is_train
        self.id_to_label_map = id_to_label_map if id_to_label_map is not None else {}
        self.next_available_label = 0
        if self.id_to_label_map:
            self.next_available_label = max(self.id_to_label_map.values()) + 1 if self.id_to_label_map else 0

        print(f"Loading dataset metadata from: {metadata_path}")
        if not os.path.exists(metadata_path):
             raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"  Loaded {len(self.metadata)} samples.")
            if not self.metadata: raise ValueError("Metadata file is empty or invalid format.")
            if self.is_train:
                self._build_label_map()
            elif not self.id_to_label_map:
                raise ValueError("Validation dataset requires an id_to_label_map from training.")
        except Exception as e:
             raise IOError(f"Error reading/parsing/processing metadata file {metadata_path}: {e}")

    def _build_label_map(self):
        print(f"  Building Fish ID (string) -> Integer Label Map...")
        unique_ids_str = sorted(list(set(item['fish_id'] for item in self.metadata)))
        start_label_count = len(self.id_to_label_map)
        for fish_id_str in unique_ids_str:
            if fish_id_str not in self.id_to_label_map:
                 self.id_to_label_map[fish_id_str] = self.next_available_label
                 self.next_available_label += 1
        print(f"  Label map complete. Total unique IDs mapped: {len(self.id_to_label_map)}")
        if len(self.id_to_label_map) == start_label_count and start_label_count == 0:
             print("  WARNING: No unique fish IDs found in training metadata to build map!")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]
        img_path = item['image_path']
        fish_id_str = item['fish_id']
        try:
            if not os.path.exists(img_path):
                 return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(-1, dtype=torch.long)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Error opening image {img_path} at index {index}: {e}")
             return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(-1, dtype=torch.long)
        label = self.id_to_label_map.get(fish_id_str, -1) # Default to -1 if ID not in map
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- Custom PKSampler (Same as before) ---
class PKsampler(Sampler):
    # ... (Exactly the same as the version in the previous 'full script for training' response) ...
    def __init__(self, dataset, p, k):
        self.dataset = dataset; self.p = p; self.k = k
        print(f"Initializing PK Sampler: P={p}, K={k}")
        self.labels_to_indices = defaultdict(list)
        print("  Grouping dataset indices by label for PK Sampler...")
        num_skipped = 0; all_labels = []
        for idx in range(len(dataset)):
             try: _, label_tensor = dataset[idx]; all_labels.append(label_tensor.item())
             except Exception: all_labels.append(-1)
        for idx, label_item in enumerate(all_labels):
             if label_item != -1: self.labels_to_indices[label_item].append(idx)
             else: num_skipped += 1
        if num_skipped > 0: print(f"  Skipped {num_skipped} indices with label -1 during sampler setup.")
        self.labels_with_k_or_more = [lbl for lbl, indices in self.labels_to_indices.items() if len(indices) >= k]
        num_labels_valid = len(self.labels_with_k_or_more)
        print(f"  Found {num_labels_valid} labels with >= {k} instances.")
        self.p_actual = min(p, num_labels_valid)
        if self.p_actual < p: print(f"  WARNING: Only {num_labels_valid} valid labels, P={p} requested. Using P={self.p_actual}.")
        elif self.p_actual == 0: print("  ERROR: No labels have >= K instances."); self.num_batches = 0; return
        self.num_batches = len(self.labels_with_k_or_more) // self.p_actual
        print(f"  Sampler ready. Estimated batches per epoch: {self.num_batches}")
    def __len__(self): return self.num_batches
    def __iter__(self):
        if self.p_actual == 0: return iter([])
        available_labels = self.labels_with_k_or_more[:]; random.shuffle(available_labels)
        for i in range(self.num_batches):
            batch_labels = available_labels[i * self.p_actual : (i + 1) * self.p_actual]
            batch_indices = []
            for label in batch_labels:
                try:
                    possible_indices = self.labels_to_indices[label]
                    chosen_indices = random.sample(possible_indices, self.k)
                    batch_indices.extend(chosen_indices)
                except ValueError: continue
            if len(batch_indices) == self.p_actual * self.k: yield batch_indices

# --- Training and Validation Functions (Same robust versions as before) ---
def train_one_epoch(model, train_loader, optimizer, loss_fn, miner, device):
    # ... (Exactly the same as the version in the previous 'full script for training' response) ...
    model.train(); total_loss = 0.0; num_valid_batches = 0; skipped_batches_miner=0; skipped_batches_loss=0
    progress_bar = tqdm(train_loader, desc="  Training", leave=False, dynamic_ncols=True)
    for images, labels in progress_bar:
        valid_idx = labels != -1
        if not torch.any(valid_idx): continue
        images, labels = images[valid_idx], labels[valid_idx]
        if images.shape[0] < P_IDENTITIES : continue # Need at least P identities for some miners or full batches
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); embeddings = model(images)
        try:
            hard_triplets = miner(embeddings, labels)
            if hard_triplets is None or (isinstance(hard_triplets, tuple) and (len(hard_triplets) < 3 or hard_triplets[0].numel() == 0)):
                skipped_batches_miner += 1; continue
            loss = loss_fn(embeddings, labels, hard_triplets)
        except Exception as e:
            if "applicable" in str(e) or "No triplets" in str(e) or "must contain triplets" in str(e) or "at least 2 classes" in str(e):
                skipped_batches_miner +=1; continue
            else: print(f"\nError during loss/mining: {e}"); continue
        if loss is not None and torch.isfinite(loss) and loss.item() > 0:
            loss.backward(); optimizer.step(); total_loss += loss.item(); num_valid_batches += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
        else: skipped_batches_loss += 1
    if skipped_batches_miner > 0: print(f"\n  Skipped {skipped_batches_miner} batches (no hard triplets/valid batch).")
    if skipped_batches_loss > 0: print(f"\n  Skipped {skipped_batches_loss} batches (invalid loss value).")
    return total_loss / num_valid_batches if num_valid_batches > 0 else float('inf')


def validate_model(model, val_loader, device, accuracy_calculator):
    # ... (Exactly the same as the version in the previous 'full script for training' response) ...
    model.eval(); print("  Starting Validation..."); all_val_embeddings_list=[]; all_val_labels_list=[]
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Validating", leave=False, dynamic_ncols=True):
            valid_idx = labels != -1
            if not torch.any(valid_idx): continue
            images, labels = images[valid_idx], labels[valid_idx]
            if images.shape[0] == 0 : continue
            images = images.to(device); embeddings = model(images)
            all_val_embeddings_list.append(embeddings.cpu()); all_val_labels_list.append(labels.cpu())
    if not all_val_embeddings_list: print("  Warning: No valid validation embeddings found."); return {}
    all_val_embeddings = torch.cat(all_val_embeddings_list); all_val_labels = torch.cat(all_val_labels_list)
    unique_labels_val = torch.unique(all_val_labels)
    if len(all_val_labels) < 2 or len(unique_labels_val) < 2:
         print(f"  Warning: Val set small/one class ({len(all_val_labels)} samples, {len(unique_labels_val)} classes)."); return {'precision_at_1': 0.0}
    print(f"  Calculating metrics on {len(all_val_labels)} validation samples from {len(unique_labels_val)} classes...")
        # Inside validate_model function
    try:
        # When query and reference embeddings/labels are the same,
        # AccuracyCalculator usually handles self-match exclusion correctly by default.
        accuracies = accuracy_calculator.get_accuracy(
            query=all_val_embeddings,
            reference=all_val_embeddings,
            query_labels=all_val_labels,
            reference_labels=all_val_labels
            # embeddings_come_from_same_source argument REMOVED
        )
    except Exception as e:
         print(f"  Error during accuracy calculation: {e}")
         # More detailed error printing
         import traceback
         print(traceback.format_exc())
         return {'precision_at_1': 0.0} # Return default on error
    print("  Validation finished."); return accuracies

# --- Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()

    # 1. Setup Datasets and Loaders
    print("\n--- Setting up Datasets and Loaders ---")
    # Define Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = FishCropDataset(TRAIN_METADATA_PATH, transform=train_transform, is_train=True)
        val_dataset = FishCropDataset(VAL_METADATA_PATH, transform=val_transform, id_to_label_map=train_dataset.id_to_label_map, is_train=False)
    except Exception as e:
        print(f"FATAL Error initializing datasets: {e}"); exit()

    if len(train_dataset) == 0: print("FATAL: Training dataset is empty."); exit()
    # Validation can be empty if VAL_METADATA_PATH is for an empty set

    try:
        train_sampler = PKsampler(train_dataset, p=P_IDENTITIES, k=K_INSTANCES)
        if len(train_sampler) == 0 and len(train_dataset) > 0 : # Check if sampler created batches if dataset is not empty
            print("FATAL: PK Sampler created 0 batches. Check P/K or training data distribution."); exit()
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=DEVICE=="cuda")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2 if len(val_dataset)>0 else 1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE=="cuda")
    except Exception as e:
        print(f"FATAL Error creating sampler or dataloaders: {e}"); exit()

    # 2. Initialize Model, Loss, Miner, Optimizer
    print("\n--- Initializing Model and Training Components ---")
    model = FishReIDNet(backbone_out_dim=BACKBONE_OUTPUT_DIM, embedding_dim=EMBEDDING_DIM, pretrained=True)
    model.to(DEVICE)

    distance = distances.LpDistance(p=2)
    reducer = reducers.AvgNonZeroReducer()
    loss_func = losses.TripletMarginLoss(margin=MARGIN, distance=distance, reducer=reducer)
    miner_func = miners.TripletMarginMiner(margin=MARGIN, distance=distance, type_of_triplets="hard")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r"), k=10)

    # 3. Training Loop
    print("\n--- Starting Training ---")
    best_val_p1 = -1.0
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, miner_func, DEVICE)
        print(f"  Epoch {epoch+1} Training - Average Loss: {avg_train_loss:.4f}")

        if len(val_loader.dataset) > 0: # Only validate if there's validation data
            val_accuracies = validate_model(model, val_loader, DEVICE, accuracy_calculator)
            val_p1 = val_accuracies.get('precision_at_1', 0.0)
            val_map_r = val_accuracies.get('mean_average_precision_at_r', 0.0)
            print(f"  Epoch {epoch+1} Validation - P@1 (Rank-1): {val_p1*100:.2f}% | MAP@R: {val_map_r*100:.2f}%")

            if val_p1 > best_val_p1:
                print(f"  Validation P@1 improved ({best_val_p1*100:.2f}% -> {val_p1*100:.2f}%). Saving model...")
                best_val_p1 = val_p1; best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                    print(f"  Best model saved to {BEST_MODEL_SAVE_PATH}")
                except Exception as save_e: print(f"  ERROR saving model: {save_e}")
            # if scheduler: scheduler.step(val_p1)
        else:
            print("  Skipping validation phase as validation dataset is empty or too small.")
            # Save model periodically if no validation, e.g., every 10 epochs or last epoch
            if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS :
                 current_save_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.pth")
                 torch.save(model.state_dict(), current_save_path)
                 print(f"  Model saved to {current_save_path} (no validation improvement check)")


        epoch_time = time.time() - epoch_start_time
        print(f"  Epoch finished in {epoch_time:.2f} seconds.")
        if DEVICE == "cuda": torch.cuda.empty_cache()

    print("\n--- Training Finished ---")
    if best_epoch != -1:
        print(f"Best model (validation P@1): {best_val_p1*100:.2f}% (Epoch {best_epoch})")
        print(f"Best model weights saved at: {BEST_MODEL_SAVE_PATH}")
    else:
        print(f"No improvement in validation P@1 observed over initial value. Last epoch model might be at '{current_save_path if 'current_save_path' in locals() else 'N/A'}' or only best model path '{BEST_MODEL_SAVE_PATH}' was used/overwritten if val_p1 started >0.")

    total_run_time = time.time() - main_start_time
    print(f"Total script run time: {total_run_time:.2f} seconds ({total_run_time/60:.2f} minutes)")
    print(f"\n*** Use the saved model ('{os.path.basename(BEST_MODEL_SAVE_PATH)}') in your EVALUATION script ***")


# In[ ]:




