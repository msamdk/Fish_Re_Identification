#!/usr/bin/env python
# coding: utf-8

# ## zero shot instance retrieval/ zero shot re-identification method for off-the-shelf- models (ResNet50, Swin and DINOv2)

# Core idea to create a sparse gallery (few instances of a single fish_id and use others as query to evaluate the re-id performance
# 
# ###Method
# 
# 1. Isolate TEST_GROUPS
#    group 10,14, 20,21,22
# 
# 2. indentify images subsets (Set 1, set 2 and all: initial, flipped). so categoriz 60 images from the image_ids
# 
# - set_1_initial; 00001 to 00010 (separated)
# - set_1_flipped: 00010 to 00020 (separated)
# - set_2_intial: 00021 to 00030 (separated)
# - set_2_flipped: 00031 to 00040 (separated)
# - All_set_initial: 00041 to 00050 (touched)
# - All_set_flipped: 00051 to 00060 (touched)
# 
# so then we have to do an enrollment set with the few shot strategy
# - Pool "Separated-Initial": Instances of the current fish_id that appear in Set1-initial (images 001-010) OR Set2-initial (images 021-030).
# - Pool "Separated-Flipped": Instances of the current fish_id that appear in Set1-flipped (images 011-020) OR Set2-flipped (images 031-040).
# - Pool "Occluded-Initial": Instances of the current fish_id that appear in All-initial (images 041-050).
# - Pool "Occluded-Flipped": Instances of the current fish_id that appear in All-flipped (images 051-060).
# 
# Now for each fish_id, we'll try to pick from these pools:
# 
# Gallery goal
# - N_GALLERY_SEP_INIT = 2 (from Pool "Separated-Initial")
# - N_GALLERY_SEP_FLIP = 2 (from Pool "Separated-Flipped")
# - N_GALLERY_OCC_INIT = 5 (from Pool "Occluded-Initial")
# - N_GALLERY_OCC_FLIP = 5 (from Pool "Occluded-Flipped")
# 

# In[2]:


#!/usr/bin/env python
# coding: utf-8

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
# from torchvision.models import resnet50, ResNet50_Weights # Import based on EXTRACTOR_TYPE
import faiss
import timm # For Swin, DINOv2 might be torch.hub
from tqdm import tqdm # For progress bars

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Few-Shot Re-ID Evaluation (GT Masks) ---")
print(f"Using device: {DEVICE}")

# --- Paths ---
BASE_PATH     = "/work3/msam/Thesis/autofish/"
COCO_ANN_PATH = os.path.join(BASE_PATH, "annotations.json")

# --- Parameters ---
TEST_GROUP_NAMES = ["group_10", "group_14", "group_20", "group_21", "group_22"]
print(f"Test Groups: {TEST_GROUP_NAMES}")

K_NEIGHBORS   = 40
EVAL_K        = 30
IMG_SIZE      = 224
CROP_PADDING  = 2

N_GALLERY_SEP_INIT   = 2
N_GALLERY_SEP_FLIP   = 2
N_GALLERY_OCC_INIT   = 5
N_GALLERY_OCC_FLIP   = 5

# --- Feature Extractor Selection ---
EXTRACTOR_TYPE = 'swin_t' # Options: 'resnet50', 'swin_t', 'dinov2_vits14'

if EXTRACTOR_TYPE == 'resnet50':
    FEAT_DIM = 2048
elif EXTRACTOR_TYPE == 'swin_t':
    FEAT_DIM = 768
elif EXTRACTOR_TYPE == 'dinov2_vits14':
    FEAT_DIM = 384
else:
    raise ValueError(f"Unsupported extractor type: {EXTRACTOR_TYPE}")
print(f"Using Extractor: {EXTRACTOR_TYPE}, Feature Dim: {FEAT_DIM}")


# --- Data Structures ---
AnnotationInfo = namedtuple("AnnotationInfo", [
    "image_id", "image_path", "annotation_id", "gt_fish_id",
    "image_subset_type",
    "img_h", "img_w", "gt_mask_data"
])
GalleryEntry = namedtuple("GalleryEntry", ["source_ann_info", "feature_index"])

# --- Model Loading ---
print(f"\nLoading Feature Extractor: {EXTRACTOR_TYPE}...")
if EXTRACTOR_TYPE == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor.fc = nn.Identity()
elif EXTRACTOR_TYPE == 'swin_t':
    model_name = 'swin_tiny_patch4_window7_224'
    try:
        feature_extractor = timm.create_model(model_name, pretrained=True)
        feature_extractor.head = nn.Identity()
        print(f"  Loaded Swin Transformer: {model_name}")
    except Exception as e: print(f"Error loading Swin: {e}"); exit()
elif EXTRACTOR_TYPE == 'dinov2_vits14':
    try:
       feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
       print("  Loaded DINOv2 ViT-Small")
    except Exception as e: print(f"Could not load DINOv2: {e}"); exit()
else:
     raise ValueError(f"Unsupported extractor type for loading: {EXTRACTOR_TYPE}")

feature_extractor.eval()
feature_extractor.to(DEVICE)
print("Feature extractor loaded successfully.")

# Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def get_gt_mask(annotation_seg_data, img_h, img_w, ann_id_for_debug="N/A"): # Added ann_id for debug
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    seg = annotation_seg_data
    if isinstance(seg, list) and seg:
        for poly_idx, poly in enumerate(seg):
            if not poly: continue
            try:
                poly_np = np.array(poly, dtype=np.int32).reshape(-1, 2)
                if poly_np.shape[0] < 3:
                    # print(f"Debug: Ann {ann_id_for_debug}, skipping invalid polygon (less than 3 points): {poly}")
                    continue
                cv2.fillPoly(mask, [poly_np], 1)
            except (ValueError, TypeError) as e:
                 # print(f"Debug: Ann {ann_id_for_debug}, skipping malformed polygon {poly_idx}: {e}")
                 continue
    elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
        try:
            from pycocotools import mask as mask_utils
            rle = mask_utils.frPyObjects(seg, img_h, img_w)
            decoded_mask = mask_utils.decode(rle)
            if len(decoded_mask.shape) > 2: decoded_mask = decoded_mask[..., 0]
            mask = decoded_mask.astype(np.uint8)
        except ImportError: print("ERROR: pycocotools needed for RLE masks."); return None
        except Exception as e: print(f"Error decoding RLE for ann {ann_id_for_debug}: {e}"); return None
    else:
        # print(f"Debug: Ann {ann_id_for_debug}, unsupported/empty segmentation data: {type(seg)}")
        return None
    if mask.sum() == 0:
        # print(f"Debug: Ann {ann_id_for_debug}, generated mask is all zeros (empty).")
        return None
    return mask

def crop_image_from_gt_mask(img_np, gt_mask_np, padding=0, ann_id_for_debug="N/A"): # Added ann_id for debug
    if gt_mask_np is None or gt_mask_np.sum() == 0:
        # print(f"Debug: Ann {ann_id_for_debug}, crop_image_from_gt_mask received None or empty mask.")
        return None
    mask_bin = (gt_mask_np > 0).astype(np.uint8) * 255
    img_h, img_w = img_np.shape[:2]
    if mask_bin.shape[:2] != (img_h, img_w):
        # print(f"Debug: Ann {ann_id_for_debug}, mask shape mismatch for cropping. Mask: {mask_bin.shape[:2]}, Img: {(img_h, img_w)}")
        return None
    if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    masked_img = cv2.bitwise_and(img_np, mask_3ch)
    y_coords, x_coords = np.where(mask_bin > 0)
    if len(y_coords) == 0:
        # print(f"Debug: Ann {ann_id_for_debug}, no foreground pixels in mask_bin after bitwise_and.")
        return None
    y_min,y_max=y_coords.min(),y_coords.max(); x_min,x_max=x_coords.min(),x_coords.max()
    y_min_p=max(0,y_min-padding); y_max_p=min(img_h-1,y_max+padding)
    x_min_p=max(0,x_min-padding); x_max_p=min(img_w-1,x_max+padding)
    fish_crop = masked_img[y_min_p:y_max_p+1, x_min_p:x_max_p+1]
    if fish_crop.size == 0:
        # print(f"Debug: Ann {ann_id_for_debug}, final crop has zero size.")
        return None
    return fish_crop

def extract_feature_from_gt_crop(full_img_np, gt_mask_for_crop, model, transform_fn, device, crop_padding, ann_id_for_debug="N/A"):
    fish_crop_np = crop_image_from_gt_mask(full_img_np, gt_mask_for_crop, padding=crop_padding, ann_id_for_debug=ann_id_for_debug)
    if fish_crop_np is None:
        return None
    try:
        fish_pil = Image.fromarray(cv2.cvtColor(fish_crop_np, cv2.COLOR_BGR2RGB)).convert("RGB")
    except Exception as e:
        return None

    tensor_in = transform_fn(fish_pil).unsqueeze(0).to(device) # Shape: [1, 3, IMG_SIZE, IMG_SIZE]
    with torch.no_grad():
        feat_tensor = model(tensor_in) # For Swin-T, this now outputs [1, H, W, C] or [1, L, C] after self.norm

    # --- ADD GLOBAL AVERAGE POOLING ---
    # Swin-T output from timm after norm (before head) is often (B, L, C)
    # e.g., (1, 49, 768) for a 7x7 grid flattened.
    # Or if it's (B, H, W, C), we need to permute and then pool.
    # Let's assume output after `model(tensor_in)` where `head=Identity` is (B, L, C)
    # L is num_patches (e.g., 7*7=49 for Swin-T with 224 input and patch_size 4, 4 stages)
    # C is feature_dim (768 for Swin-T)

    if feat_tensor.ndim == 3 and feat_tensor.shape[0] == 1: # (1, L, C) e.g. (1, 49, 768)
        # Global average pool across the sequence length L
        pooled_feat = torch.mean(feat_tensor, dim=1) # Result will be (1, C)
    elif feat_tensor.ndim == 4 and feat_tensor.shape[0] == 1: # (1, C, H, W) or (1, H, W, C)
        # This is less common for Swin after norm, but handle it.
        # If (1, H, W, C), permute to (1, C, H, W) for PyTorch pooling
        if feat_tensor.shape[3] == FEAT_DIM: # Likely (1, H, W, C)
            feat_tensor = feat_tensor.permute(0, 3, 1, 2) # -> (1, C, H, W)

        if feat_tensor.shape[1] == FEAT_DIM: # Now it's (1, C, H, W)
             # Apply Global Average Pooling across H and W
            pooled_feat = F.adaptive_avg_pool2d(feat_tensor, (1, 1)).squeeze(-1).squeeze(-1) # -> (1, C)
        else:
            print(f"Debug: Ann {ann_id_for_debug}, unhandled feature tensor shape for pooling: {feat_tensor.shape}")
            return None
    elif feat_tensor.ndim == 2 and feat_tensor.shape == (1, FEAT_DIM): # Already correct
        pooled_feat = feat_tensor
    else:
        print(f"Debug: Ann {ann_id_for_debug}, unexpected feature tensor shape: {feat_tensor.shape}")
        return None

    feature_np = pooled_feat.cpu().numpy()
    # print(f"Debug: Ann {ann_id_for_debug}, POOLED feature shape: {feature_np.shape}") # Should be (1, FEAT_DIM)
    return feature_np


def categorize_image_by_filename_num(image_fname_base_int):
    if 1 <= image_fname_base_int <= 10: return "set1_init"
    elif 11 <= image_fname_base_int <= 20: return "set1_flip"
    elif 21 <= image_fname_base_int <= 30: return "set2_init"
    elif 31 <= image_fname_base_int <= 40: return "set2_flip"
    elif 41 <= image_fname_base_int <= 50: return "all_init"
    elif 51 <= image_fname_base_int <= 60: return "all_flip"
    return "unknown_subset"

def categorize_instance_to_pool_type(ann_info: AnnotationInfo):
    subset = ann_info.image_subset_type
    if subset in ["set1_init", "set2_init"]: return "separated_initial"
    elif subset in ["set1_flip", "set2_flip"]: return "separated_flipped"
    elif subset == "all_init": return "occluded_initial"
    elif subset == "all_flip": return "occluded_flipped"
    return "unknown_pool"

# --- Main Workflow (Revised for Single-Query Protocol) ---
if __name__ == "__main__":
    eval_start_time = time.time()

    # 1. Load and process annotations (This part remains the same)
    print("\nLoading and pre-processing annotations for Test Groups...")
    # ... (Keep your existing annotation loading and grouping logic here)
    with open(COCO_ANN_PATH, 'r') as f: coco_data = json.load(f)
    image_id_to_meta = {img['id']: {'path': os.path.join(BASE_PATH, img['file_name']),'height': img['height'],'width': img['width']} for img in coco_data.get('images', [])}
    all_annotations_in_test_set = []
    test_group_set = set(TEST_GROUP_NAMES)
    for ann in tqdm(coco_data.get('annotations', []), desc="Filtering Test Annotations"):
        img_meta = image_id_to_meta.get(ann['image_id'])
        if not img_meta: continue
        current_group = os.path.basename(os.path.dirname(img_meta['path']))
        if current_group not in test_group_set: continue
        
        ann_info = AnnotationInfo(image_id=ann['image_id'], image_path=img_meta['path'], annotation_id=str(ann['id']),
                                  gt_fish_id=str(ann.get("fish_id")), image_subset_type=None, # subset_type not needed now
                                  img_h=img_meta['height'], img_w=img_meta['width'],
                                  gt_mask_data=ann['segmentation'])
        all_annotations_in_test_set.append(ann_info)
    print(f"Found {len(all_annotations_in_test_set)} total annotations in the specified test groups.")

    # 2. Extract features for ALL valid instances in the test set
    print("\nExtracting features for all test instances...")
    all_features = []
    all_labels = [] # We'll store the fish IDs here
    loaded_images_cache = {}

    for ann_info in tqdm(all_annotations_in_test_set, desc="Extracting All Features"):
        if ann_info.image_path not in loaded_images_cache:
            try: loaded_images_cache[ann_info.image_path] = np.array(Image.open(ann_info.image_path).convert("RGB"))
            except: continue
        img_np = loaded_images_cache[ann_info.image_path]
        
        gt_mask_np = get_gt_mask(ann_info.gt_mask_data, ann_info.img_h, ann_info.img_w, ann_info.annotation_id)
        if gt_mask_np is None: continue

        feature = extract_feature_from_gt_crop(img_np, gt_mask_np, feature_extractor, transform, DEVICE, CROP_PADDING, ann_info.annotation_id)
        if feature is not None and feature.shape == (1, FEAT_DIM):
            all_features.append(feature)
            all_labels.append(ann_info.gt_fish_id)

    if not all_features:
        print("FATAL: No features were extracted. Exiting.")
        exit()
        
    all_features_np = np.concatenate(all_features, axis=0).astype('float32')
    all_labels_np = np.array(all_labels)
    print(f"\nSuccessfully extracted {len(all_features_np)} total features.")
    del loaded_images_cache
    torch.cuda.empty_cache()

    # 3. Perform Single-Query/Gallery Split
    print("\nPreparing query and gallery sets using single-query protocol...")
    indices_by_id = defaultdict(list)
    for i, fish_id in enumerate(all_labels_np):
        indices_by_id[fish_id].append(i)

    query_indices, gallery_indices = [], []
    for fish_id, indices in indices_by_id.items():
        if len(indices) < 2: continue # Can't create a query if there's no match in the gallery
        random.shuffle(indices)
        query_indices.append(indices[0]) # One for query
        gallery_indices.extend(indices[1:]) # The rest for gallery
    
    query_features = all_features_np[query_indices]
    query_labels = all_labels_np[query_indices]
    gallery_features = all_features_np[gallery_indices]
    gallery_labels = all_labels_np[gallery_indices]
    print(f"Split complete. Query set size: {len(query_features)}, Gallery set size: {len(gallery_features)}")

    # 4. Build Faiss Index from the Gallery
    if len(gallery_features) == 0:
        print("FATAL: Gallery is empty after splitting. Cannot evaluate.")
        exit()

    print("\nBuilding Faiss index for the gallery...")
    faiss_index = faiss.IndexFlatL2(FEAT_DIM)
    faiss.normalize_L2(gallery_features)
    faiss_index.add(gallery_features)
    print(f"Faiss index built with {faiss_index.ntotal} gallery features.")

    # 5. Process Queries and Evaluate
    print("\nProcessing queries and calculating Rank-k accuracy...")
    faiss.normalize_L2(query_features)
    distances, indices = faiss_index.search(query_features, k=EVAL_K)

    # Convert gallery indices back to fish IDs for evaluation
    retrieved_ids_ranked = gallery_labels[indices]
    
    def calculate_rank_k_eval(true_ids, retrieved_ids, k):
        hits = 0
        for i in range(len(true_ids)):
            if true_ids[i] in retrieved_ids[i, :k]:
                hits += 1
        return hits / len(true_ids) if len(true_ids) > 0 else 0.0

    print(f"\n--- Zero-Shot Re-ID Results ({EXTRACTOR_TYPE}) ---")
    ranks_to_print = [r for r in [1, 5, 10, 20, 30] if r <= EVAL_K and r <= faiss_index.ntotal]
    for k_val in ranks_to_print:
        rank_k_acc = calculate_rank_k_eval(query_labels, retrieved_ids_ranked, k_val)
        print(f"  Rank-{k_val:<2} Accuracy: {rank_k_acc * 100:.2f}%")

    total_eval_time = time.time() - eval_start_time
    print(f"\nTotal script time: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")


# # New evaluation with R1 and mAP@R

# In[7]:


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

# --- Configuration ---
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


# --- !!! IMPORTANT: PLEASE VERIFY THESE PATHS AND PARAMETERS !!! ---
BASE_PATH        = "/work3/msam/Thesis/autofish/"
COCO_ANN_PATH    = os.path.join(BASE_PATH, "annotations.json")
EXTRACTOR_TYPE   = 'swin_t'  # Options: 'resnet50', 'swin_t', 'dinov2_vits14'
TEST_GROUP_NAMES = ["group_10", "group_14", "group_20", "group_21", "group_22"]
IMG_SIZE         = 224
CROP_PADDING     = 2
# --- End Configuration ---


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

# FIX 2: Added the missing class definition
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

# FIX 1: Corrected the normalization to use ImageNet stats for zero-shot evaluation
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


# # resnet baseline

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

# --- Configuration ---
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


# --- !!! IMPORTANT: PLEASE VERIFY THESE PATHS AND PARAMETERS !!! ---
BASE_PATH        = "/work3/msam/Thesis/autofish/"
COCO_ANN_PATH    = os.path.join(BASE_PATH, "annotations.json")
EXTRACTOR_TYPE   = 'resnet50'  # Options: 'resnet50', 'swin_t', 'dinov2_vits14'
TEST_GROUP_NAMES = ["group_10", "group_14", "group_20", "group_21", "group_22"]
IMG_SIZE         = 224
CROP_PADDING     = 2
# --- End Configuration ---


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

# FIX 2: Added the missing class definition
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

# FIX 1: Corrected the normalization to use ImageNet stats for zero-shot evaluation
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


# In[ ]:




