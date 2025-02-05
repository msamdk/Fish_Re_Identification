
<h2>Training an Object detection model using latest YOLO v11 models</h2>
<img src="images/ylo.png" alt="Alt text" width="500">

Before training, we should rearrange the Autofish dataset to be compatible with the YOLO models.
As i discussed in the Data exploration section, the COCO annotation configuration is quiet different from the YOLO configuration. The image arrangement also have to be changed according to the YOLO model compatibility.
Here i deployed following steps to rearrange the dataset


```python

import json
import os
import random
import shutil
from pathlib import Path

#Pinput and output directory
coco_path = "/work3/msam/Thesis/autofish/annotations.json"
image_groups_dir = "/work3/msam/Thesis/autofish"
output_base_dir = "/work3/msam/Thesis/autofish/YOLO"
os.makedirs(output_base_dir, exist_ok=True)

#function to convert COCO annotation format to YOLO format (bounding boxes)
def convert_coco_to_yolo(x_min, y_min, width, height, img_width, img_height):
    """Convert COCO bbox format to YOLO format"""
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center, y_center, norm_width, norm_height

def process_dataset(coco_path, image_groups_dir, output_base_dir):
    #Define YOLO dataset structure
    dataset_splits = ["train", "val", "test"]
    for split in dataset_splits:
        os.makedirs(os.path.join(output_base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, "labels"), exist_ok=True)

    #Loading COCO annotations
    with open(coco_path, "r") as f:
        data = json.load(f)

    #Map image IDs to their metadata
    image_mapping = {}
    for img in data["images"]:
        group_name = img["file_name"].split('/')[0]  #Extract group folder (e.g., "group_01")
        file_name = img["file_name"].split('/')[-1]  #Extract filename (e.g., "00001.jpg")
        img_id = img["id"]  # Get unique image ID

        image_mapping[img_id] = {
            "group": group_name,
            "original_name": file_name,
            "new_name": f"{img_id}.jpg",  # Rename with unique image ID
            "width": img["width"],
            "height": img["height"]
        }

    #Shuffle and split dataset
    image_ids = list(image_mapping.keys())
    random.shuffle(image_ids)
    num_images = len(image_ids)

    train_count = int(num_images * 0.7)
    val_count = int(num_images * 0.2)

    splits = {
        "train": image_ids[:train_count],
        "val": image_ids[train_count:train_count + val_count],
        "test": image_ids[train_count + val_count:]
    }

    #Processing images and annotations
    for split, ids in splits.items():
        for img_id in ids:
            img_info = image_mapping[img_id]
            group_folder = os.path.join(image_groups_dir, img_info["group"])
            image_path = os.path.join(group_folder, img_info["original_name"])

            #Ensure the image exists
            if os.path.exists(image_path):
                new_image_path = os.path.join(output_base_dir, split, "images", img_info["new_name"])
                shutil.copy2(image_path, new_image_path)
            else:
                print(f"Warning: Image {image_path} not found")

            #Processing annotations
            yolo_annotations = []
            for ann in data["annotations"]:
                if ann["image_id"] == img_id:
                    bbox = ann["bbox"]
                    x_center, y_center, norm_width, norm_height = convert_coco_to_yolo(
                        bbox[0], bbox[1], bbox[2], bbox[3], img_info["width"], img_info["height"]
                    )
                    category_id = ann["category_id"]
                    yolo_annotations.append(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}")

            #Save label file with matching name (same as image ID)
            if yolo_annotations:
                label_filename = f"{img_id}.txt"
                label_path = os.path.join(output_base_dir, split, "labels", label_filename)
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_annotations))

    #Create dataset.yaml for YOLO training
    yaml_content = f"""
path: {output_base_dir}
train: train/images
val: val/images
test: test/images

#Classes
names:
{chr(10).join(f'  {i}: {cat["name"]}' for i, cat in enumerate(data['categories']))}
    """

    with open(os.path.join(output_base_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content.strip())

    return "Dataset processing completed!"


process_dataset(coco_path, image_groups_dir, output_base_dir)
```

