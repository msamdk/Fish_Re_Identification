#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- FINAL SCRIPT FOR SINGLE SUBSET ANALYSIS AND VISUALIZATION ---
# This robust script uses a manual metric calculation to avoid library bugs
# and includes advanced t-SNE and KDE plotting functions.

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import warnings
import re
import random

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_t

# --- Visualization and ML Imports (All re-included) ---
import plotly.graph_objects as go
import plotly.colors
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# --- SET SEED FOR REPRODUCIBILITY ---
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"--- Random seed set to {SEED} for reproducibility ---")


# --- !!! IMPORTANT CONFIGURATION - CHANGE THESE !!! ---

# 1. Path to the trained model weights (.pth file)
MODEL_PATH = "/work3/msam/Thesis/autofish/Re_ID_Experiments/new_order/swin_exp1_batch_16_hard_margin_0.5_euclidean/finetuned_metric_models/best_swin_triplet_finetuned_resize_pad.pth"

# 2. Specify the model architecture
MODEL_TYPE = "swin"

# 3. Path to the METADATA FILE of the SUBSET you want to test
SUBSET_METADATA_PATH = "/work3/msam/Thesis/autofish/Re_ID_Experiments/reid_subsets/separated_initial/metadata.json"

# 4. Directory to save the output plots
OUTPUT_DIR = "/work3/msam/Thesis/autofish/Re_ID_Experiments/detailed_analysis/sep_init_sep_init"

# 5. Visualization Sample Size (to prevent memory crashes with t-SNE)
VIZ_SAMPLE_SIZE = 1500 # t-SNE will run on a random sample of this size if the gallery is larger

# 6. Model & Data Hyperparameters
EMBEDDING_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
CUSTOM_MEAN = [0.0495, 0.0503, 0.0535]
CUSTOM_STD = [0.1370, 0.1363, 0.1412]


# --- Model and Dataset Class Definitions ---
class FishReIDNetSwin(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.backbone = swin_t(weights=None)
        self.backbone.head = nn.Identity()
        self.embedding_head = nn.Sequential(nn.Linear(768, embedding_dim))
    def forward(self, x):
        return self.embedding_head(self.backbone(x))

class ResizeAndPadToSquare:
    def __init__(self, s, f=(0,0,0)): self.s, self.f = s, f
    def __call__(self, img):
        w, h = img.size; r = min(self.s/w, self.s/h); nw, nh = int(w*r), int(h*r)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        p = Image.new("RGB", (self.s, self.s), self.f); p.paste(img, ((self.s-nw)//2, (self.s-nh)//2)); return p

# --- Analysis and Visualization Functions ---

def calculate_metrics_manually(query_embeddings, query_labels, gallery_embeddings, gallery_labels, device):
    # ... (The manual calculation function from the previous response) ...
    query_embeddings = query_embeddings.to(device); gallery_embeddings = gallery_embeddings.to(device)
    query_labels = query_labels.to(device); gallery_labels = gallery_labels.to(device)
    dist_mat = torch.cdist(query_embeddings, gallery_embeddings)
    sorted_indices = torch.argsort(dist_mat, dim=1)
    top_1_labels = gallery_labels[sorted_indices[:, 0]]
    matches = (top_1_labels == query_labels).float()
    r1_score = (matches.sum() / query_labels.shape[0]) * 100
    total_ap = 0.0
    for i in range(query_labels.shape[0]):
        is_match = (gallery_labels[sorted_indices[i]] == query_labels[i])
        match_indices = torch.where(is_match)[0]
        if len(match_indices) == 0: continue
        num_correct_at_k = torch.arange(1, len(match_indices) + 1, device=device)
        precision_at_k = num_correct_at_k / (match_indices + 1)
        total_ap += precision_at_k.mean()
    map_score = (total_ap / query_labels.shape[0]) * 100
    return r1_score.item(), map_score.item()

def visualize_tsne_3d_interactive(embeddings, labels, save_path):
    # ... (The advanced t-SNE function from our previous responses) ...
    print("\n--- Generating Advanced t-SNE Visualization ---")
    if embeddings.shape[0] == 0: return
    perplexity_value = min(30, embeddings.shape[0] - 1)
    if perplexity_value <= 0: return
    print(f"Running t-SNE with perplexity={perplexity_value}...")
    tsne = TSNE(n_components=3, perplexity=perplexity_value, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    X_3d = tsne.fit_transform(embeddings.cpu().numpy())
    fig = go.Figure()
    plot_colors = plotly.colors.qualitative.Plotly
    plot_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'circle-open', 'square-open', 'diamond-open']
    unique_ids_str = sorted(list(set(labels)))
    for i, fish_id_name in enumerate(unique_ids_str):
        idx = (labels == fish_id_name)
        if not np.any(idx): continue
        fig.add_trace(go.Scatter3d(x=X_3d[idx, 0], y=X_3d[idx, 1], z=X_3d[idx, 2], mode='markers', marker=dict(size=5, color=plot_colors[i % len(plot_colors)], symbol=plot_symbols[i % len(plot_symbols)], opacity=0.8), name=str(fish_id_name), text=[str(fish_id_name)]*np.sum(idx), hoverinfo='text+name'))
    fig.update_layout(title=dict(text=f'<b>3D t-SNE of {embeddings.shape[1]}-D Embeddings</b>', y=0.98, x=0.5, font=dict(size=20, color='white')), scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3', xaxis=dict(backgroundcolor="rgb(50,50,50)", gridcolor="rgb(100,100,100)"), yaxis=dict(backgroundcolor="rgb(50,50,50)", gridcolor="rgb(100,100,100)"), zaxis=dict(backgroundcolor="rgb(50,50,50)", gridcolor="rgb(100,100,100)"), bgcolor="rgb(17, 17, 17)"), legend=dict(title='<b>Fish IDs</b>', font=dict(color='white'), title_font=dict(color='white')), paper_bgcolor='rgb(0, 0, 0)', width=1200, height=900)
    fig.write_html(save_path)
    print(f"--- Advanced visualization saved to '{save_path}' ---")

def visualize_distance_distribution_kde(query_embeddings, query_labels, gallery_embeddings, gallery_labels, save_path, device):
    # ... (The KDE function from our previous responses) ...
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
    plt.figure(figsize=(10, 6)); sns.kdeplot(positive_distances, fill=True, label='Positive Pairs (Same ID)', color='g'); sns.kdeplot(negative_distances, fill=True, label='Negative Pairs (Different ID)', color='r')
    plt.legend(); plt.title('Distribution of Pairwise Distances'); plt.xlabel('Euclidean Distance'); plt.ylabel('Density'); plt.grid(True); plt.savefig(save_path, dpi=300); plt.close()
    print(f"--- Distance KDE plot saved to '{save_path}' ---")

# --- Main Logic ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model, Metadata, and Extract Embeddings
    # ... (This section is the same as the previous minimal script) ...
    print("--- Loading Model ---")
    model = FishReIDNetSwin(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"--- Loading Subset Metadata From: {SUBSET_METADATA_PATH} ---")
    with open(SUBSET_METADATA_PATH, 'r') as f:
        subset_metadata = json.load(f)

    print("--- Extracting Embeddings ---")
    all_embeddings, all_fish_ids = [], []
    eval_transform = transforms.Compose([ResizeAndPadToSquare(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD)])
    with torch.no_grad():
        for item in tqdm(subset_metadata, desc="Extracting embeddings"):
            try:
                image = Image.open(item['image_path']).convert('RGB')
                embedding = model(eval_transform(image).unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
                all_embeddings.append(embedding); all_fish_ids.append(str(item['fish_id']))
            except Exception as e:
                print(f"Warning: Could not process {item['image_path']}. Error: {e}")
    all_embeddings = torch.stack(all_embeddings); all_fish_ids = np.array(all_fish_ids)

    # 2. Create label map and split data for evaluation
    unique_ids = sorted(list(set(all_fish_ids)))
    id_to_label = {fish_id: i for i, fish_id in enumerate(unique_ids)}
    all_labels = torch.tensor([id_to_label[fid] for fid in all_fish_ids])
    indices_by_id = defaultdict(list)
    for i, fish_id in enumerate(all_fish_ids): indices_by_id[fish_id].append(i)
    query_indices, gallery_indices = [], []
    for fish_id, indices in indices_by_id.items():
        if len(indices) < 2: continue
        np.random.shuffle(indices)
        query_indices.append(indices[0]); gallery_indices.extend(indices[1:])

    # 3. Calculate Accuracy
    query_embeddings = all_embeddings[query_indices]; query_labels = all_labels[query_indices]
    gallery_embeddings = all_embeddings[gallery_indices]; gallery_labels = all_labels[gallery_indices]
    
    print("\n--- Running Accuracy Calculation (Manual Method) ---")
    r1, map_r = calculate_metrics_manually(query_embeddings, query_labels, gallery_embeddings, gallery_labels, DEVICE)

    print("\n" + "#"*15 + " FINAL RESULTS " + "#"*15)
    print(f"Subset Tested: {os.path.basename(os.path.dirname(SUBSET_METADATA_PATH))}")
    print(f"  - Rank-1 Accuracy (R1): {r1:.2f}%")
    print(f"  - Mean Average Precision @ R (mAP@R): {map_r:.2f}%")
    print("#"*47)

    # 4. Generate Visualizations
    print("\n--- Generating Visualizations ---")
    gallery_fish_ids = all_fish_ids[gallery_indices]
    
    # Take a random sample for t-SNE if the gallery is too large
    if len(gallery_embeddings) > VIZ_SAMPLE_SIZE:
        print(f"Gallery size ({len(gallery_embeddings)}) is larger than VIZ_SAMPLE_SIZE ({VIZ_SAMPLE_SIZE}).")
        print("Taking a random sample for t-SNE to prevent memory issues.")
        sample_indices = np.random.choice(len(gallery_embeddings), VIZ_SAMPLE_SIZE, replace=False)
        embeddings_for_viz = gallery_embeddings[sample_indices]
        labels_for_viz = gallery_fish_ids[sample_indices]
    else:
        embeddings_for_viz = gallery_embeddings
        labels_for_viz = gallery_fish_ids

    visualize_tsne_3d_interactive(
        embeddings=embeddings_for_viz,
        labels=labels_for_viz,
        save_path=os.path.join(OUTPUT_DIR, "tsne_plot.html")
    )
    
    visualize_distance_distribution_kde(
        query_embeddings, query_labels, gallery_embeddings, gallery_labels,
        save_path=os.path.join(OUTPUT_DIR, "kde_plot.png"),
        device=DEVICE
    )
    
    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()

