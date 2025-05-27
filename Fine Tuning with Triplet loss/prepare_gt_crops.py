#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# --- SCRIPT TO PREPARE GT-CROPPED FISH IMAGES FOR METRIC LEARNING ---

import os
import json
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import shutil # For creating directories
from tqdm import tqdm # For progress bar
import time

# --- !!! IMPORTANT CONFIGURATION - CHANGE THESE !!! ---

# 1. Path to your main AutoFish dataset directory
BASE_PATH = "/work3/msam/Thesis/autofish/" # <--- CHANGE THIS

# 2. Path to your main annotations file
COCO_ANN_PATH = os.path.join(BASE_PATH, "annotations.json") # <--- CHANGE THIS if different

# 3. Define your Training and Validation Group Names

print("Defining Train/Val groups according to the AutoFish paper split...")
TRAIN_IDS = [2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24]
VAL_IDS   = [1, 6, 11, 17, 25]

TEST_IDS  = [10, 14, 20, 21, 22]

# Generate the list of group name strings (e.g., "group_01", "group_02")
TRAIN_GROUP_NAMES = [f"group_{i:02d}" for i in TRAIN_IDS]
VAL_GROUP_NAMES   = [f"group_{i:02d}" for i in VAL_IDS]

# 4. Define WHERE to save the output crops and metadata
#    These directories will be CREATED by this script.
OUTPUT_BASE_DIR       = os.path.join(BASE_PATH, "metric_learning_gt_crops") # Main output folder
OUTPUT_TRAIN_CROP_DIR = os.path.join(OUTPUT_BASE_DIR, "train")
OUTPUT_VAL_CROP_DIR   = os.path.join(OUTPUT_BASE_DIR, "val")
TRAIN_METADATA_PATH = os.path.join(OUTPUT_TRAIN_CROP_DIR, "train_crop_metadata.json")
VAL_METADATA_PATH   = os.path.join(OUTPUT_VAL_CROP_DIR, "val_crop_metadata.json")

# 5. Optional: Padding around the tight crop (in pixels)
"""
adding a small padding will help to add a safety margin
against minor segmentation errors or edge effects during resizing without significantly adding irrelavent information
"""
CROP_PADDING = 2

print("--- Data Preparation Configuration ---")
print(f"Base Dataset Path: {BASE_PATH}")
print(f"Annotations Path: {COCO_ANN_PATH}")
print(f"Training Groups ({len(TRAIN_GROUP_NAMES)}): {TRAIN_GROUP_NAMES}")
print(f"Validation Groups ({len(VAL_GROUP_NAMES)}): {VAL_GROUP_NAMES}")
print(f"Output Train Crops: {OUTPUT_TRAIN_CROP_DIR}")
print(f"Output Val Crops: {OUTPUT_VAL_CROP_DIR}")
print("-" * 30)

# --- Helper Functions ---

def load_and_filter_annotations(coco_path, base_img_path, target_group_names):
    """Loads COCO annotations and filters for specific image groups."""
    print(f"Loading annotations from {coco_path}...")
    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"Annotation file not found: {coco_path}")
    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        raise IOError(f"Error reading annotations file: {e}")

    print(f"Filtering annotations for {len(target_group_names)} target groups...")
    imgid_to_info = {}
    target_image_ids = set()
    target_group_set = set(target_group_names)
    all_image_files_in_groups = defaultdict(list) # Store actual files found

    # First, find all image files physically present in the target group folders
    for group_name in target_group_names:
         group_path = os.path.join(base_img_path, group_name)
         if os.path.isdir(group_path):
              for fname in os.listdir(group_path):
                  if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                      all_image_files_in_groups[group_name].append(fname)
         else:
              print(f"Warning: Target group folder not found: {group_path}")


    # Now, cross-reference with the JSON data
    num_json_imgs_matched = 0
    for img_meta in coco_data.get('images', []):
        try:
            file_name = img_meta.get('file_name')
            img_id = img_meta.get('id')
            if not file_name or img_id is None: continue

            # Check if the image group is in our target list
            group_name = os.path.dirname(file_name)
            base_fname = os.path.basename(file_name)

            if group_name in target_group_set:
                # Crucially, check if this specific file exists in the group folder found earlier
                if base_fname in all_image_files_in_groups.get(group_name, []):
                    img_full_path = os.path.join(base_img_path, file_name)
                    imgid_to_info[img_id] = {
                        'path': img_full_path,
                        'height': img_meta['height'],
                        'width': img_meta['width'],
                        'group': group_name,
                        'fname': base_fname
                    }
                    target_image_ids.add(img_id)
                    num_json_imgs_matched += 1
                # else:
                #     print(f"Debug: Image {file_name} in JSON but not found in filesystem scan of {group_name}")

        except Exception as e:
             print(f"  Warning: Could not process image entry {img_meta.get('id', 'N/A')} in JSON: {e}")

    # Filter annotations based on the identified target image IDs and presence of fish_id
    target_annotations = []
    for ann in coco_data.get('annotations', []):
        if ann['image_id'] in target_image_ids and ann.get("fish_id") is not None:
            target_annotations.append(ann)

    print(f"Found {len(target_image_ids)} relevant images listed in JSON and present on disk.")
    print(f"Found {len(target_annotations)} relevant annotations with fish_id.")
    if len(target_image_ids) == 0 or len(target_annotations) == 0:
         print("WARNING: No relevant images or annotations found for the specified groups. Check paths and group names.")

    return imgid_to_info, target_annotations

def get_gt_mask(annotation, img_h, img_w):
    """Creates a binary mask from a COCO annotation (polygon or RLE)."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    seg = annotation['segmentation']
    if isinstance(seg, list) and seg: # Polygon (ensure not empty list)
        for poly in seg:
            if not poly: continue # Skip empty polygon lists within the main list
            try:
                poly_np = np.array(poly, dtype=np.int32).reshape(-1, 2)
                if poly_np.shape[0] < 3: continue # Need at least 3 points for a polygon
                cv2.fillPoly(mask, [poly_np], 1)
            except (ValueError, TypeError) as e:
                 print(f"Warning: Skipping malformed polygon {poly} in ann {annotation['id']}: {e}")
                 continue
    elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg: # RLE
        try:
            # Important: Ensure pycocotools is installed: pip install pycocotools
            from pycocotools import mask as mask_utils
            # Ensure counts are bytes if needed by your version
            if isinstance(seg['counts'], list):
                seg['counts'] = str(seg['counts']) # Or handle appropriately if it's truly list of ints
            
            rle = mask_utils.frPyObjects(seg, img_h, img_w)
            decoded_mask = mask_utils.decode(rle)
            if len(decoded_mask.shape) > 2: decoded_mask = decoded_mask[..., 0]
            mask = decoded_mask.astype(np.uint8)
        except ImportError:
            print("\nFATAL ERROR: pycocotools not installed. Cannot decode RLE masks.")
            print("Please install it: pip install pycocotools")
            exit() # Exit because RLE decoding is essential if used
        except Exception as e:
            print(f"Error decoding RLE for ann {annotation['id']}: {e}")
            return None
    else: # Neither polygon nor valid RLE found
        # print(f"Warning: Unsupported or empty segmentation format for ann {annotation['id']}")
        return None # Indicate failure
    
    # Return mask only if something was drawn (check sum > 0)
    return mask if mask.sum() > 0 else None


def crop_image_from_gt_mask(img_np, gt_mask_np, padding=0):
    """Crops the image using the bounding box of the GT mask, applying the mask."""
    if gt_mask_np is None or gt_mask_np.sum() == 0: return None
    
    mask_bin = (gt_mask_np > 0).astype(np.uint8) * 255
    img_h, img_w = img_np.shape[:2]

    if mask_bin.shape[:2] != (img_h, img_w):
         print(f"  Warning: Mask shape {mask_bin.shape[:2]} mismatch with image shape {(img_h, img_w)}. Cannot crop.")
         return None

    # Ensure image is 3 channels for bitwise_and
    if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Apply the mask (make background black)
    mask_3ch = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    masked_img = cv2.bitwise_and(img_np, mask_3ch)

    # Find bounding box coordinates of the mask
    y_coords, x_coords = np.where(mask_bin > 0)
    if len(y_coords) == 0: return None # Should not happen if mask.sum() > 0, but safety check

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # Apply padding, ensuring bounds stay within image dimensions
    y_min = max(0, y_min - padding)
    y_max = min(img_h - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(img_w - 1, x_max + padding)

    # Crop the *masked* image using the padded bounding box
    fish_crop = masked_img[y_min:y_max+1, x_min:x_max+1]

    return fish_crop if fish_crop.size > 0 else None


# --- Main Data Preparation Function ---
def prepare_split_data(split_name, group_names, output_crop_dir, metadata_path):
    print(f"\n--- Preparing {split_name} Set ---")
    prep_start_time = time.time()

    # 1. Load and filter annotations for the specific groups
    imgid_to_info, annotations = load_and_filter_annotations(
        COCO_ANN_PATH, BASE_PATH, group_names
    )

    if not annotations:
         print(f"No annotations found for {split_name} groups. Skipping preparation.")
         return

    # 2. Create Output Directory
    print(f"Creating output directory for {split_name} crops: {output_crop_dir}")
    os.makedirs(output_crop_dir, exist_ok=True) # Create parent dirs too if needed

    # 3. Process annotations and save crops
    print(f"Processing {len(annotations)} {split_name} annotations...")
    split_crop_metadata = []
    processed_count = 0
    skipped_no_mask = 0
    skipped_no_crop = 0
    skipped_img_error = 0
    loaded_images_cache = {} # Cache images per split

    for annotation in tqdm(annotations, desc=f"Cropping {split_name} Fish"):
        image_id = annotation['image_id']
        fish_id = annotation['fish_id']
        annotation_id = annotation['id']

        img_info = imgid_to_info.get(image_id)
        if not img_info: continue # Skip if image info wasn't loaded/matched

        img_path = img_info['path']
        img_h, img_w = img_info['height'], img_info['width']
        group_name = img_info['group']
        fname_base = os.path.splitext(img_info['fname'])[0]

        try:
            # Load image (use cache)
            if img_path not in loaded_images_cache:
                if not os.path.exists(img_path):
                    # print(f"Warning: Image file not found: {img_path}")
                    skipped_img_error += 1
                    continue
                img_pil = Image.open(img_path).convert("RGB")
                loaded_images_cache[img_path] = np.array(img_pil) # Store as numpy
            img_np = loaded_images_cache[img_path]

            # Get GT Mask
            gt_mask = get_gt_mask(annotation, img_h, img_w)
            if gt_mask is None:
                skipped_no_mask += 1
                continue

            # Crop Image using GT Mask
            fish_crop_np = crop_image_from_gt_mask(img_np, gt_mask, padding=CROP_PADDING)
            if fish_crop_np is None:
                skipped_no_crop += 1
                continue

            # Save Cropped Image
            # Format: <group>_img<basename>_ann<annID>_id<fishID>.png
            crop_filename = f"{group_name}_{fname_base}_ann{annotation_id}_id{fish_id}.png"
            crop_save_path = os.path.join(output_crop_dir, crop_filename)

            crop_pil = Image.fromarray(cv2.cvtColor(fish_crop_np, cv2.COLOR_BGR2RGB))
            crop_pil.save(crop_save_path)

            # Store Metadata (path to the saved CROP)
            split_crop_metadata.append({
                'image_path': crop_save_path,
                'fish_id': str(fish_id) # Use string fish ID
            })
            processed_count += 1

        except Exception as e:
            print(f"  Error processing annotation {annotation_id} from image {img_path}: {e}")
            skipped_img_error += 1


    print(f"\n--- {split_name} Cropping Summary ---")
    print(f"Successfully saved crops: {processed_count}")
    print(f"Skipped (no GT mask generated): {skipped_no_mask}")
    print(f"Skipped (cropping failed):    {skipped_no_crop}")
    print(f"Skipped (image load/other errors): {skipped_img_error}")

    # 4. Save Metadata
    if split_crop_metadata: # Only save if crops were generated
        print(f"Saving {split_name} crop metadata to: {metadata_path}")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(split_crop_metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    else:
         print(f"No metadata to save for {split_name} set.")

    prep_time = time.time() - prep_start_time
    print(f"{split_name} set preparation finished in {prep_time:.2f} seconds.")


# --- Main Execution ---
if __name__ == "__main__":
    overall_start_time = time.time()

    # Prepare Training Data
    prepare_split_data(
        split_name="Training",
        group_names=TRAIN_GROUP_NAMES,
        output_crop_dir=OUTPUT_TRAIN_CROP_DIR,
        metadata_path=TRAIN_METADATA_PATH
    )

    # Prepare Validation Data
    prepare_split_data(
        split_name="Validation",
        group_names=VAL_GROUP_NAMES,
        output_crop_dir=OUTPUT_VAL_CROP_DIR,
        metadata_path=VAL_METADATA_PATH
    )

    overall_time = time.time() - overall_start_time
    print(f"\n--- Total Data Preparation Finished ---")
    print(f"Total time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Check output folders:\n  Train: {OUTPUT_TRAIN_CROP_DIR}\n  Val:   {OUTPUT_VAL_CROP_DIR}")
    print("\nYou can now run the training script ('train_resnet50_metric_learning.py').")

