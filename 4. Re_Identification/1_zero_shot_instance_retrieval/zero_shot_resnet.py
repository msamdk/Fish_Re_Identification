#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- SCRIPT FOR ZERO-SHOT RE-ID EVALUATION (SINGLE-QUERY PROTOCOL with mAP@R) ---

import os
import json
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict, namedtuple
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import faiss
import timm
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


# Set a seed for reproducibility of the random query/gallery split
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Zero-Shot Re-ID Evaluation (mAP@R Protocol) ---")
print(f"Using device: {DEVICE}")
print(f"Random seed set to {SEED} for reproducibility.")


# paths
BASE_PATH        = "/work3/msam/Thesis/autofish/"
COCO_ANN_PATH    = os.path.join(BASE_PATH, "annotations.json")
EXTRACTOR_TYPE   = 'resnet50'  # Options: 'resnet50', 'swin_t', 'dinov2_vits14'
TEST_GROUP_NAMES = ["group_10", "group_14", "group_20", "group_21", "group_22"]
IMG_SIZE         = 224
CROP_PADDING     = 2



# --- Determine Feature Dimension based on Extractor Type ---
if EXTRACTOR_TYPE == 'resnet50':      FEAT_DIM = 2048
elif EXTRACTOR_TYPE == 'swin_t':      FEAT_DIM = 768
elif EXTRACTOR_TYPE == 'dinov2_vits14': FEAT_DIM = 384
else: raise ValueError(f"Unsupported extractor type: {EXTRACTOR_TYPE}")
print(f"Using Extractor: {EXTRACTOR_TYPE}, Feature Dim: {FEAT_DIM}")


# --- Model Loading ---
print(f"\nLoading Feature Extractor: {EXTRACTOR_TYPE}...")
if EXTRACTOR_TYPE == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor.fc = nn.Identity()
elif EXTRACTOR_TYPE == 'swin_t':
    model_name = 'swin_tiny_patch4_window7_224'
    feature_extractor = timm.create_model(model_name, pretrained=True)
    feature_extractor.head = nn.Identity()
    print(f"  Loaded Swin Transformer: {model_name}")
elif EXTRACTOR_TYPE == 'dinov2_vits14':
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
    print("  Loaded DINOv2 ViT-Small")

feature_extractor.eval()
feature_extractor.to(DEVICE)
print("Feature extractor loaded successfully.")


# --- Data Structures & Preprocessing ---
AnnotationInfo = namedtuple("AnnotationInfo", ["image_path", "annotation_id", "gt_fish_id", "img_h", "img_w", "gt_mask_data"])


class ResizeAndPadToSquare:
    """
    Resizes an image to fit within a square of output_size,
    maintaining aspect ratio and padding with a fill color.
    """
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


transform = transforms.Compose([
    ResizeAndPadToSquare(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Helper Functions for Data Processing ---
def get_gt_mask(annotation_seg_data, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if isinstance(annotation_seg_data, list) and annotation_seg_data:
        for poly in annotation_seg_data:
            if not poly: continue
            try:
                poly_np = np.array(poly, dtype=np.int32).reshape(-1, 2)
                if poly_np.shape[0] >= 3: cv2.fillPoly(mask, [poly_np], 1)
            except (ValueError, TypeError): continue
    if mask.sum() == 0: return None
    return mask

def crop_image_from_gt_mask(img_np, gt_mask_np, padding=0):
    if gt_mask_np is None: return None
    y_coords, x_coords = np.where(gt_mask_np > 0)
    if len(y_coords) == 0: return None
    y_min, y_max, x_min, x_max = y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()
    img_h, img_w = img_np.shape[:2]
    y_min_p, y_max_p = max(0, y_min - padding), min(img_h - 1, y_max + padding)
    x_min_p, x_max_p = max(0, x_min - padding), min(img_w - 1, x_max + padding)
    return img_np[y_min_p:y_max_p+1, x_min_p:x_max_p+1]

def extract_feature_from_gt_crop(full_img_np, gt_mask_for_crop, model, transform_fn, device, crop_padding):
    fish_crop_np = crop_image_from_gt_mask(full_img_np, gt_mask_for_crop, padding=crop_padding)
    if fish_crop_np is None or fish_crop_np.size == 0: return None
    
    # FIX 3: Convert color channels from BGR (OpenCV) to RGB (PIL/PyTorch)
    fish_crop_rgb = cv2.cvtColor(fish_crop_np, cv2.COLOR_BGR2RGB)
    
    fish_pil = Image.fromarray(fish_crop_rgb).convert("RGB")
    tensor_in = transform_fn(fish_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat_tensor = model(tensor_in)

    if feat_tensor.ndim == 4: # Handle CNN-style output (B, C, H, W)
        return F.adaptive_avg_pool2d(feat_tensor, (1, 1)).flatten(start_dim=1)
    elif feat_tensor.ndim == 3: # Handle Transformer-style output (B, L, C)
        return torch.mean(feat_tensor, dim=1)
    return feat_tensor # Handle models that already output (B, C)


# --- Main Workflow ---
if __name__ == "__main__":
    eval_start_time = time.time()

    # 1. Load and filter annotations from COCO JSON
    print("\nLoading and pre-processing annotations for Test Groups...")
    with open(COCO_ANN_PATH, 'r') as f: coco_data = json.load(f)
    image_id_to_meta = {img['id']: {'path': os.path.join(BASE_PATH, img['file_name']),'height': img['height'],'width': img['width']} for img in coco_data.get('images', [])}
    
    all_annotations_in_test_set = []
    test_group_set = set(TEST_GROUP_NAMES)
    for ann in tqdm(coco_data.get('annotations', []), desc="Filtering Test Annotations"):
        img_meta = image_id_to_meta.get(ann['image_id'])
        if not img_meta or ann.get("fish_id") is None: continue
        current_group = os.path.basename(os.path.dirname(img_meta['path']))
        if current_group not in test_group_set: continue
        all_annotations_in_test_set.append(AnnotationInfo(image_path=img_meta['path'], annotation_id=str(ann['id']),
                                                          gt_fish_id=str(ann["fish_id"]), img_h=img_meta['height'], img_w=img_meta['width'],
                                                          gt_mask_data=ann['segmentation']))
    print(f"Found {len(all_annotations_in_test_set)} total annotations in test groups.")

    # 2. Extract features for ALL valid instances
    print("\nExtracting features for all test instances...")
    all_embeddings_list, all_fish_ids_str, loaded_images_cache = [], [], {}
    for ann_info in tqdm(all_annotations_in_test_set, desc="Extracting All Features"):
        if ann_info.image_path not in loaded_images_cache:
            try: loaded_images_cache[ann_info.image_path] = cv2.imread(ann_info.image_path)
            except: continue
        img_np = loaded_images_cache[ann_info.image_path]
        gt_mask_np = get_gt_mask(ann_info.gt_mask_data, ann_info.img_h, ann_info.img_w)
        if gt_mask_np is None: continue
        feature = extract_feature_from_gt_crop(img_np, gt_mask_np, feature_extractor, transform, DEVICE, CROP_PADDING)
        if feature is not None:
            all_embeddings_list.append(feature.cpu())
            all_fish_ids_str.append(ann_info.gt_fish_id)

    if not all_embeddings_list:
        print("FATAL: No features were extracted. Exiting."); exit()
        
    all_embeddings = torch.cat(all_embeddings_list)
    all_fish_ids_str = np.array(all_fish_ids_str)
    print(f"\nSuccessfully extracted {len(all_embeddings)} total features.")
    del loaded_images_cache; torch.cuda.empty_cache()

    # 3. Perform Label Encoding
    unique_ids = sorted(list(set(all_fish_ids_str)))
    id_to_label_map = {fish_id: i for i, fish_id in enumerate(unique_ids)}
    all_labels = torch.tensor([id_to_label_map[fish_id] for fish_id in all_fish_ids_str])
    
    # 4. Perform Single-Query/Gallery Split
    print("\nPreparing query and gallery sets using single-query protocol...")
    indices_by_id = defaultdict(list)
    for i, fish_id in enumerate(all_fish_ids_str):
        indices_by_id[fish_id].append(i)

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
    print(f"Split complete. Query set size: {len(query_embeddings)}, Gallery set size: {len(gallery_embeddings)}")

    # 5. Calculate Accuracy using AccuracyCalculator
    if len(query_embeddings) == 0 or len(gallery_embeddings) == 0:
        print("FATAL: Query or Gallery set is empty after split. Cannot evaluate."); exit()
        
    print("\nCalculating performance metrics using AccuracyCalculator...")
    calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1"), k=None)
    accuracies = calculator.get_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
    
    r1 = accuracies.get('precision_at_1', 0.0) * 100
    map_r = accuracies.get('mean_average_precision_at_r', 0.0) * 100

    print("\n--- Zero-Shot Test Set Performance ---")
    print(f"Model: {EXTRACTOR_TYPE}")
    print(f"Rank-1 Accuracy (R1): {r1:.2f}%")
    print(f"Mean Average Precision @ R (mAP@R): {map_r:.2f}%")
    print("----------------------------------------")

    total_eval_time = time.time() - eval_start_time
    print(f"\nTotal script time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")

