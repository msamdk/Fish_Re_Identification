# Training YOLO-segementation models for the segmentation masks in the annotations

<img src="images/COCOtoYOLO.png" alt="Alt text" width="1500">


## COCO format provides 
- segmentation masks as polygons using absolute pixel coordinates  
- uses JSON format with a list of (x,y) points that form the contour
- "segmentation": [[x1,y1, x2,y2, x3,y3, ...]]

## YOLO format
- stores segmentation masks as normalized coordinates (0-1)
- use plain txt format (.txt) with space separated values
- format per line class_id x1 y1 x2 y2 ... xn yn
- coordinates are normalized by image width and height
  


## steps for the trainins

1. Making a separate YOLO-seg model compatible dataset for the training

```python
import json
import os
import shutil
from pathlib import Path

# Paths
coco_path = f"/work3/msam/Thesis/autofish/annotations.json"
image_groups_dir = f"/work3/msam/Thesis/autofish"
output_base_dir = f"/work3/msam/Thesis/segmentation/yolodataset_seg"  # Changed name for segmentation
os.makedirs(output_base_dir, exist_ok=True)

# Keeping your original split as paper
train_groups = {2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24}
val_groups   = {1, 6, 11, 17, 25}
test_groups  = {10, 14, 20, 21, 22}

# Creating output directories for each split
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, "labels"), exist_ok=True)

def convert_segmentation_to_yolo(segmentation, img_width, img_height):
   
    # Get the polygon from segmentation
    polygon = segmentation[0]
    normalized_coords = []
    
    # Convert each pair of coordinates
    for i in range(0, len(polygon), 2):
        # Normalize x coordinate
        x = polygon[i] / img_width
        # Normalize y coordinate
        y = polygon[i + 1] / img_height
        
        # Ensure coordinates are within [0, 1]
        x = min(max(x, 0), 1)
        y = min(max(y, 0), 1)
        
        normalized_coords.extend([x, y])
    
    return normalized_coords

def process_dataset(coco_path, image_groups_dir, output_base_dir):
    # Load COCO annotations
    with open(coco_path, "r") as f:
        data = json.load(f)
    
    # Build a mapping from image_id to its annotations for quick lookup
    annotations_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)
    
    # Process each image defined in the original annotations
    for img in data["images"]:
        img_id = img["id"]
        file_name = img["file_name"]  # e.g., "group_03/00001.jpg"
        parts = file_name.split('/')
        if len(parts) < 2:
            print(f"Warning: File name format is unexpected: {file_name}")
            continue
        
        # Extract the group folder and original file name
        group_folder = parts[0]         # e.g., "group_03"
        original_filename = parts[-1]   # e.g., "00001.jpg"
        
        # Parse the group number
        try:
            group_num = int(group_folder.split('_')[-1])
        except ValueError:
            print(f"Warning: Could not parse group number from {group_folder}")
            continue
        
        # Determine the dataset split based on the group number
        if group_num in train_groups:
            split = "train"
        elif group_num in val_groups:
            split = "val"
        elif group_num in test_groups:
            split = "test"
        else:
            print(f"Warning: Group number {group_num} is not assigned to any split folder.")
            continue
        
        # Build paths
        src_img_path = os.path.join(image_groups_dir, file_name)
        dest_img_dir = os.path.join(output_base_dir, split, "images", group_folder)
        os.makedirs(dest_img_dir, exist_ok=True)
        
        dest_img_filename = original_filename
        dest_img_path = os.path.join(dest_img_dir, dest_img_filename)
        
        # Copy the image file if it exists
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dest_img_path)
        else:
            print(f"Warning: Source image {src_img_path} does not exist.")
        
        # Process annotations for this image
        anns = annotations_by_image.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            # Convert segmentation coordinates to YOLO format
            normalized_coords = convert_segmentation_to_yolo(
                ann["segmentation"],
                img["width"],
                img["height"]
            )
            
            # Format: class_id x1 y1 x2 y2 ...
            category_id = ann["category_id"]
            coords_str = " ".join([f"{coord:.6f}" for coord in normalized_coords])
            yolo_lines.append(f"{category_id} {coords_str}")
        
        # Save YOLO annotations if available
        if yolo_lines:
            dest_label_dir = os.path.join(output_base_dir, split, "labels", group_folder)
            os.makedirs(dest_label_dir, exist_ok=True)
            label_filename = f"{Path(original_filename).stem}.txt"
            label_path = os.path.join(dest_label_dir, label_filename)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
    
    # Create dataset.yaml file
    yaml_content = f"""
path: {output_base_dir}
train: train/images
val: val/images
test: test/images

# Classes
names:
{chr(10).join(f'  {i}: {cat["name"]}' for i, cat in enumerate(data['categories']))}

# Whether to use segmentation
task: segment
    """
    with open(os.path.join(output_base_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content.strip())
    
    print("Dataset processing done")

# Run the Processing
if __name__ == "__main__":
    process_dataset(coco_path, image_groups_dir, output_base_dir)
```
Check if the conversion is good in the YOLO format
```python
#confirming the masks are okay
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# path for the random selected image
image_path = f"/work3/msam/Thesis/segmentation/yolodataset_seg/train/images/group_02/00001.png" 
label_path = f"/work3/msam/Thesis/segmentation/yolodataset_seg/train/labels/group_02/00001.txt" 

# Open the image using PIL and get its size
image = Image.open(image_path)
width, height = image.size

# Create a figure and axis
fig, ax = plt.subplots(1)
ax.imshow(image)

# Read the YOLO label file
with open(label_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        category_id = int(parts[0])  # Use as needed
        coords = list(map(float, parts[1:]))

        # Convert normalized coordinates to pixel coordinates and build the polygon points
        polygon_points = []
        for i in range(0, len(coords), 2):
            x_pixel = coords[i] * width
            y_pixel = coords[i + 1] * height
            polygon_points.append((x_pixel, y_pixel))

        # Create and add the polygon patch to the axis
        polygon = patches.Polygon(polygon_points, closed=True, fill=True, edgecolor='yellow', linewidth=2)
        ax.add_patch(polygon)

# Display the image with the overlaid polygons
plt.title("YOLO Segmentation ")
plt.axis('off')
plt.show()
```
<img src="images/yolo conversion.png" alt="Alt text" width="400">
