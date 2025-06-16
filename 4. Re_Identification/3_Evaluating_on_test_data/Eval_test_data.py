#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- SCRIPT TO EVALUATE A TRAINED RE-ID MODEL ---
# This version dynamically creates the ID-to-Label map from the test metadata,
# removing the need for an external label map file during evaluation.

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import warnings
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, resnet50


# --- Visualization and ML Imports ---
import plotly.express as px
import plotly.graph_objects as go  # <--- ADD THIS LINE
import plotly.colors               # <--- This was also missing but needed by the new function
from sklearn.manifold import TSNE
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import seaborn as sns
import matplotlib.pyplot as plt

# --- SET SEED FOR REPRODUCIBILITY ---
SEED = 1234  # Or any number you like
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"--- Random seed set to {SEED} for reproducibility ---")

# --- !!! IMPORTANT CONFIGURATION - CHANGE THESE !!! ---

# 1. Path to the trained model weights (.pth file)
MODEL_PATH = "/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp3_batch_64_hard_margin_0.5_euclidean/finetuned_metric_models/best_swin_triplet_finetuned_resize_pad.pth"

# 2. Specify the model architecture: "swin" or "resnet"
MODEL_TYPE = "swin"

# 3. Path to the test set metadata
BASE_PATH = "/work3/msam/Thesis/autofish/"
TEST_METADATA_PATH = os.path.join(BASE_PATH, "crop_dataset2/metric_learning_gt_crops", "test", "test_crop_metadata.json")

# 4. Paths to save the output plots
OUTPUT_DIR = "/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp23_batch_64_hard_margin_0.5_euclidean/evaluation_outputs"
TSNE_PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, "test_set_tsne_3d.html")
KDE_PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, "test_set_distance_kde.png")

# 5. Model & Data Hyperparameters (MUST match the training script)
EMBEDDING_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
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
    def __init__(self, metadata, transform, id_to_label_map): # Now receives metadata directly
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
        # This lookup will now always succeed
        label = self.id_to_label_map.get(fish_id_str, -1)
        return image, label, fish_id_str

# --- NEW ADVANCED T-SNE VISUALIZATION FUNCTION ---
def visualize_tsne_3d_interactive(embeddings, labels, save_path):
    """
    Runs t-SNE and creates a highly customized, interactive 3D plot with Plotly.
    """
    print("\n--- Generating Advanced t-SNE Visualization ---")
    if embeddings.shape[0] == 0:
        print("Warning: No embeddings to visualize. Skipping t-SNE.")
        return

    embeddings_np = embeddings.cpu().numpy()
    
    # Handle perplexity for small datasets
    perplexity_value = min(30, embeddings_np.shape[0] - 1)
    if perplexity_value <= 0:
        print("Warning: Not enough samples for t-SNE. Skipping visualization.")
        return

    print(f"Running t-SNE with perplexity={perplexity_value} (this can take a few minutes)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'n_iter' was renamed to 'max_iter'", category=FutureWarning)
        tsne = TSNE(n_components=3, perplexity=perplexity_value, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
        X_3d = tsne.fit_transform(embeddings_np)
            
    print("t-SNE complete. Generating Plotly figure...")
    fig = go.Figure()
    
    # Define color and symbol cycles
    plot_colors = plotly.colors.qualitative.Plotly
    plot_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'circle-open', 'square-open', 'diamond-open']
    
    unique_ids_str = sorted(list(set(labels)))

    # Add a separate trace for each fish ID for custom styling
    for i, fish_id_name in enumerate(unique_ids_str):
        # Find the indices corresponding to the current fish ID
        idx = (labels == fish_id_name)
        
        if not np.any(idx): continue

        fig.add_trace(go.Scatter3d(
            x=X_3d[idx, 0], y=X_3d[idx, 1], z=X_3d[idx, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_colors[i % len(plot_colors)],
                symbol=plot_symbols[i % len(plot_symbols)],
                opacity=0.8
            ),
            name=str(fish_id_name), # Legend uses actual fish ID strings
            text=[str(fish_id_name)] * np.sum(idx), # Hover text also uses actual fish ID
            hoverinfo='text+name'
        ))

    # Apply the advanced dark-theme layout
    fig.update_layout(
        title=dict(text=f'<b>3D t-SNE of Fine-Tuned {embeddings.shape[1]}-D Embeddings</b>', y=0.98, x=0.5, xanchor='center', yanchor='top', font=dict(size=20, color='white')),
        scene=dict(
            xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Dimension 3',
            xaxis=dict(backgroundcolor="rgb(50,50,50)", gridcolor="rgb(100,100,100)", showbackground=True, zerolinecolor="rgb(150,150,150)", title_font=dict(color='white'), tickfont=dict(color='white')),
            yaxis=dict(backgroundcolor="rgb(50,50,50)", gridcolor="rgb(100,100,100)", showbackground=True, zerolinecolor="rgb(150,150,150)", title_font=dict(color='white'), tickfont=dict(color='white')),
            zaxis=dict(backgroundcolor="rgb(50,50,50)", gridcolor="rgb(100,100,100)", showbackground=True, zerolinecolor="rgb(150,150,150)", title_font=dict(color='white'), tickfont=dict(color='white')),
            bgcolor="rgb(17, 17, 17)"
        ),
        legend=dict(title='<b>Fish IDs</b>', orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, bgcolor='rgba(60,60,60,0.7)', font=dict(color='white'), title_font=dict(color='white'), bordercolor="Gray", borderwidth=1),
        paper_bgcolor='rgb(0, 0, 0)',
        width=1200, height=900,
        margin=dict(r=200, b=10, l=10, t=60)
    )
    
    try:
        fig.write_html(save_path)
        print(f"--- Advanced visualization saved to '{save_path}' ---")
    except Exception as e:
        print(f"--- ERROR: Could not save interactive plot. {e} ---")


def visualize_distance_distribution_kde(query_embeddings, query_labels, gallery_embeddings, gallery_labels, save_path, device):
    print("\nCalculating distances for KDE plot...")
    positive_distances, negative_distances = [], []
    query_embeddings, gallery_embeddings = query_embeddings.to(device), gallery_embeddings.to(device)
    for i in tqdm(range(len(query_embeddings)), desc="Calculating distances"):
        distances = torch.cdist(query_embeddings[i].unsqueeze(0), gallery_embeddings).squeeze(0)
        is_positive = (gallery_labels == query_labels[i])
        positive_distances.extend(distances[is_positive].cpu().numpy())
        negative_distances.extend(distances[~is_positive].cpu().numpy())
    if len(negative_distances) > 500000:
        negative_distances = np.random.choice(negative_distances, 500000, replace=False)
    print("Creating KDE plot...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(positive_distances, fill=True, label='Positive Pairs (Same ID)', color='g')
    sns.kdeplot(negative_distances, fill=True, label='Negative Pairs (Different ID)', color='r')
    plt.legend()
    plt.title('Distribution of Pairwise Distances')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"--- Distance KDE plot saved to: {save_path} ---")


# --- Main Evaluation Logic ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    print(f"Loading {MODEL_TYPE} model architecture...")
    model = FishReIDNetResNet(embedding_dim=EMBEDDING_DIM) if MODEL_TYPE == 'resnet' else FishReIDNetSwin(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # 2. Setup Dataset and DataLoader
    eval_transform = transforms.Compose([
        ResizeAndPadToSquare(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD)
    ])
    
    print("Loading test metadata and creating label map dynamically...")
    with open(TEST_METADATA_PATH, 'r') as f:
        test_metadata = json.load(f)
    all_test_ids = sorted(list(set(str(item['fish_id']) for item in test_metadata)))
    id_to_label_map = {fish_id: i for i, fish_id in enumerate(all_test_ids)}
    print(f"Created map for {len(id_to_label_map)} unique test IDs.")

    dataset = FishCropDataset(test_metadata, eval_transform, id_to_label_map)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Extract All Embeddings
    print("\nExtracting embeddings from the test set...")
    all_embeddings, all_labels, all_fish_ids = [], [], []
    with torch.no_grad():
        for images, labels, fish_ids in tqdm(dataloader, desc="Extracting"):
            valid_idx = labels != -1
            if not torch.any(valid_idx): continue
            embeddings = model(images[valid_idx].to(DEVICE))
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels[valid_idx])
            all_fish_ids.extend(np.array(fish_ids)[valid_idx.numpy()])

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    all_fish_ids = np.array(all_fish_ids)
    print(f"Extracted {len(all_embeddings)} valid embeddings.")

    # 4. Perform Query/Gallery Split
    print("\nPreparing query and gallery sets according to standard protocol...")
    indices_by_id = defaultdict(list)
    for i, fish_id in enumerate(all_fish_ids):
        indices_by_id[fish_id].append(i)

    query_indices, gallery_indices = [], []
    for fish_id, indices in indices_by_id.items():
        if len(indices) < 2: continue
        np.random.shuffle(indices)
        query_indices.append(indices[0])
        gallery_indices.extend(indices[1:])

    query_embeddings = all_embeddings[query_indices]
    query_labels = all_labels[query_indices]
    gallery_embeddings = all_embeddings[gallery_indices]
    gallery_labels = all_labels[gallery_indices]
    gallery_fish_ids = all_fish_ids[gallery_indices]
    print(f"Split complete. Query set size: {len(query_embeddings)}, Gallery set size: {len(gallery_embeddings)}")

    # 5. Calculate Accuracy
    print("\nCalculating performance metrics...")
    calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1"))
    accuracies = calculator.get_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
    r1 = accuracies.get('precision_at_1', 0.0) * 100
    map_r = accuracies.get('mean_average_precision_at_r', 0.0) * 100
    print("\n--- Test Set Performance ---")
    print(f"Rank-1 Accuracy (R1): {r1:.2f}%")
    print(f"Mean Average Precision @ R (mAP@R): {map_r:.2f}%")
    print("----------------------------")
    
    # 6. Generate Visualizations
    visualize_tsne_3d_interactive(
        embeddings=gallery_embeddings,
        labels=gallery_fish_ids,
        save_path=TSNE_PLOT_SAVE_PATH
    )
    
    visualize_distance_distribution_kde(
        query_embeddings, query_labels, gallery_embeddings, gallery_labels,
        save_path=KDE_PLOT_SAVE_PATH,
        device=DEVICE
    )

if __name__ == '__main__':
    main()

