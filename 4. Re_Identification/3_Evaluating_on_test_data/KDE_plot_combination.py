#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- SCRIPT TO EVALUATE & COMBINE 8 RE-ID MODELS INTO A SINGLE 2-COLUMN KDE PLOT ---

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import warnings
import random
import re # Imported to help with sorting

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, resnet50

# --- Visualization and ML Imports ---
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import seaborn as sns
import matplotlib.pyplot as plt

# --- SET SEED FOR REPRODUCIBILITY ---
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"--- Random seed set to {SEED} for reproducibility ---")

# --- !!! IMPORTANT CONFIGURATION - EDIT THIS SECTION !!! ---

# 1. Define all 8 of your experiments here.
#    The script will automatically sort them and place them in the correct columns.
EXPERIMENTS = [
    # --- Swin-T Models ---
    {'title': 'Swin-T: 16 batch size', 'model_type': 'swin', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp1_batch_16_hard_margin_0.5_euclidean/finetuned_metric_models/best_swin_triplet_finetuned_resize_pad.pth'},
    {'title': 'Swin-T: 32 batch size', 'model_type': 'swin', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp2_batch_32_hard_margin_0.5_euclidean/finetuned_metric_models/best_swin_triplet_finetuned_resize_pad.pth'},
    {'title': 'Swin-T: 64 batch size', 'model_type': 'swin', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp3_batch_64_hard_margin_0.5_euclidean/finetuned_metric_models/best_swin_triplet_finetuned_resize_pad.pth'},
    {'title': 'Swin-T: 256 batch size', 'model_type': 'swin', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp4_batch_256_hard_margin_0.5_euclidean/finetuned_metric_models/best_swin_triplet_finetuned_resize_pad.pth'},
    # --- ResNet-50 Models ---
    {'title': 'ResNet-50: 32 batch size', 'model_type': 'resnet', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/Resnet_exp_5_batch_32_hard_margin_0.5_euclidean/finetuned_metric_models/best_resnet50_triplet_finetuned.pth'},
    {'title': 'ResNet-50: 16 batch size', 'model_type': 'resnet', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/Resnet_exp_6_batch_16_hard_margin_0.5_euclidean/finetuned_metric_models/best_resnet50_triplet_finetuned.pth'},
    {'title': 'ResNet-50: 64 batch size', 'model_type': 'resnet', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/Resnet_exp_7_batch_64_hard_margin_0.5_euclidean/finetuned_metric_models/best_resnet50_triplet_finetuned.pth'},
    {'title': 'ResNet-50: 256 batch size', 'model_type': 'resnet', 'path': '/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/Resnet_exp_8_batch_256_hard_margin_0.5_euclidean/finetuned_metric_models/best_resnet50_triplet_finetuned.pth'},
]

# 2. Path to the test set metadata (should be the same for all)
BASE_PATH = "/work3/msam/Thesis/autofish/"
TEST_METADATA_PATH = os.path.join(BASE_PATH, "crop_dataset2/metric_learning_gt_crops", "test", "test_crop_metadata.json")

# 3. Path to save the final combined output plot
OUTPUT_DIR = "/work3/msam/Thesis/autofish/Re_ID_Experiments/evaluation_summary/"
COMBINED_KDE_PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, "combined_distance_kde_plots_v2.png")

# 4. Model & Data Hyperparameters (MUST match training scripts)
EMBEDDING_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32 # This is for evaluation dataloader, not training
NUM_WORKERS = 4
CUSTOM_MEAN = [0.0495, 0.0503, 0.0535]
CUSTOM_STD = [0.1370, 0.1363, 0.1412]


# --- Model Definitions ---
class FishReIDNetSwin(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.backbone = swin_t(weights=None)
        self.backbone.head = nn.Identity()
        self.embedding_head = nn.Sequential(nn.Linear(768, embedding_dim))
    def forward(self, x):
        return self.embedding_head(self.backbone(x))

class FishReIDNetResNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.embedding_head = nn.Sequential(nn.Linear(2048, embedding_dim))
    def forward(self, x):
        return self.embedding_head(self.backbone(x))


# --- Custom Dataset & Transforms ---
class ResizeAndPadToSquare:
    def __init__(self, output_size_square, fill_color=(0, 0, 0)):
        self.output_size = output_size_square
        self.fill_color = fill_color
    def __call__(self, img):
        original_w, original_h = img.size
        ratio = min(self.output_size / original_w, self.output_size / original_h)
        new_w, new_h = int(original_w * ratio), int(original_h * ratio)
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_img = Image.new("RGB", (self.output_size, self.output_size), self.fill_color)
        pad_left = (self.output_size - new_w) // 2
        pad_top = (self.output_size - new_h) // 2
        padded_img.paste(resized_img, (pad_left, pad_top))
        return padded_img

class FishCropDataset(Dataset):
    def __init__(self, metadata, transform, id_to_label_map):
        self.transform = transform
        self.metadata = metadata
        self.id_to_label_map = id_to_label_map
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, index):
        item = self.metadata[index]
        img_path = item['image_path']
        fish_id_str = str(item['fish_id'])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), -1, "-1"
        label = self.id_to_label_map.get(fish_id_str, -1)
        return image, label, fish_id_str


# --- MODIFIED KDE PLOTTING FUNCTION ---
def visualize_distance_distribution_kde(query_embeddings, query_labels, gallery_embeddings, gallery_labels, ax, device, title, show_legend=False):
    print(f"Calculating distances for '{title}'...")
    positive_distances, negative_distances = [], []
    query_embeddings, gallery_embeddings = query_embeddings.to(device), gallery_embeddings.to(device)

    for i in tqdm(range(len(query_embeddings)), desc="Calculating distances", leave=False):
        distances = torch.cdist(query_embeddings[i].unsqueeze(0), gallery_embeddings).squeeze(0)
        is_positive = (gallery_labels == query_labels[i])
        positive_distances.extend(distances[is_positive].cpu().numpy())
        negative_distances.extend(distances[~is_positive].cpu().numpy())
    
    if len(negative_distances) > 500000:
        negative_distances = np.random.choice(negative_distances, 500000, replace=False)

    # UPDATED: Larger font sizes for journal publication
    font_params = {
        'axes.labelsize': 20, 'axes.titlesize': 20,
        'xtick.labelsize': 18, 'ytick.labelsize': 18,
        'legend.fontsize': 24,
    }
    plt.rcParams.update(font_params)

    # UPDATED: New colors (blue, maroon) and less transparency (alpha=0.5)
    sns.kdeplot(positive_distances, fill=True, label='Positive Pairs (Same ID)', color='#800074', alpha=0.5, ax=ax)
    sns.kdeplot(negative_distances, fill=True, label='Negative Pairs (Different ID)', color='#70747c', alpha=0.5, ax=ax)

    # UPDATED: Only show legend if requested
    if show_legend:
        ax.legend()
    else:
        # Clear any automatically generated legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    ax.set_title(title)
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Density')
    ax.grid(True)
    ax.set_xlim(left=0)


# --- Helper function to extract batch size for sorting ---
def get_batch_size(exp_title):
    # Use regular expressions to find the number after "batch size"
    match = re.search(r'(\d+)\s+batch size', exp_title)
    if match:
        return int(match.group(1))
    return -1 # Return -1 if not found, so it sorts last

# --- Main Evaluation Logic ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Setup Dataset (done once) ---
    eval_transform = transforms.Compose([
        ResizeAndPadToSquare(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD)
    ])
    
    print("Loading test metadata...")
    with open(TEST_METADATA_PATH, 'r') as f:
        test_metadata = json.load(f)
    all_test_ids = sorted(list(set(str(item['fish_id']) for item in test_metadata)))
    id_to_label_map = {fish_id: i for i, fish_id in enumerate(all_test_ids)}
    
    dataset = FishCropDataset(test_metadata, eval_transform, id_to_label_map)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # --- UPDATED: Create master plot and sort experiments ---
    fig, axes = plt.subplots(4, 2, figsize=(20, 26)) # 4 rows, 2 columns

    swin_experiments = sorted([e for e in EXPERIMENTS if e['model_type'] == 'swin'], key=lambda x: get_batch_size(x['title']))
    resnet_experiments = sorted([e for e in EXPERIMENTS if e['model_type'] == 'resnet'], key=lambda x: get_batch_size(x['title']))
    
    # --- Loop through each column (Swin and ResNet) ---
    for col_idx, experiment_list in enumerate([swin_experiments, resnet_experiments]):
        for row_idx, exp_config in enumerate(experiment_list):
            print("\n" + "="*50)
            print(f"--- Processing: {exp_config['title']} ---")
            print("="*50)
            
            ax = axes[row_idx, col_idx] # Get the correct subplot axis

            # 1. Load Model
            MODEL_PATH = exp_config['path']
            MODEL_TYPE = exp_config['model_type']
            
            if not os.path.exists(MODEL_PATH):
                print(f"WARNING: Model path not found, skipping: {MODEL_PATH}")
                ax.text(0.5, 0.5, f"Model Not Found\n{os.path.basename(MODEL_PATH)}", 
                        ha='center', va='center', fontsize=12, wrap=True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            print(f"Loading {MODEL_TYPE} model from {MODEL_PATH}")
            model = FishReIDNetResNet(embedding_dim=EMBEDDING_DIM) if MODEL_TYPE == 'resnet' else FishReIDNetSwin(embedding_dim=EMBEDDING_DIM)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            # 2. Extract Embeddings
            print("Extracting embeddings from the test set...")
            all_embeddings, all_labels, all_fish_ids = [], [], []
            with torch.no_grad():
                for images, labels, fish_ids in tqdm(dataloader, desc="Extracting Embeddings"):
                    valid_idx = labels != -1
                    if not torch.any(valid_idx): continue
                    embeddings = model(images[valid_idx].to(DEVICE))
                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(labels[valid_idx])
                    all_fish_ids.extend(np.array(fish_ids)[valid_idx.numpy()])
            all_embeddings = torch.cat(all_embeddings)
            all_labels = torch.cat(all_labels)
            all_fish_ids = np.array(all_fish_ids)

            # 3. Perform Query/Gallery Split
            indices_by_id = defaultdict(list)
            for idx, fish_id in enumerate(all_fish_ids):
                indices_by_id[fish_id].append(idx)
            query_indices, gallery_indices = [], []
            for fish_id, indices in indices_by_id.items():
                if len(indices) < 2: continue
                random.shuffle(indices)
                query_indices.append(indices[0])
                gallery_indices.extend(indices[1:])
            query_embeddings = all_embeddings[query_indices]
            query_labels = all_labels[query_indices]
            gallery_embeddings = all_embeddings[gallery_indices]
            gallery_labels = all_labels[gallery_indices]

            # 4. Generate KDE plot on the correct subplot
            # UPDATED: Show legend only for the top-left plot (Swin, first in the list)
            show_legend_flag = (col_idx == 0 and row_idx == 0)
            visualize_distance_distribution_kde(
                query_embeddings, query_labels, gallery_embeddings, gallery_labels,
                ax=ax,
                device=DEVICE,
                title=exp_config['title'],
                show_legend=show_legend_flag
            )
            
            # (Optional) Print accuracy
            calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1"))
            accuracies = calculator.get_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
            r1 = accuracies.get('precision_at_1', 0.0) * 100
            map_r = accuracies.get('mean_average_precision_at_r', 0.0) * 100
            print(f"--- Performance for {exp_config['title']} ---")
            print(f"Rank-1 Accuracy (R1): {r1:.2f}%")
            print(f"Mean Average Precision @ R (mAP@R): {map_r:.2f}%")

    # --- Finalize and Save the Combined Plot ---
    fig.suptitle('Comparison of Pairwise Distance Distributions Across Batch Sizes', fontsize=28)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    print(f"\nSaving combined plot to: {COMBINED_KDE_PLOT_SAVE_PATH}")
    fig.savefig(COMBINED_KDE_PLOT_SAVE_PATH, dpi=300)
    plt.close()
    print("--- All experiments processed. Combined plot saved successfully! ---")

if __name__ == '__main__':
    main()

