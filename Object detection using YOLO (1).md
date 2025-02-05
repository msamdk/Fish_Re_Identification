
<h2>Training an Object detection model using latest YOLO v11 models</h2>
<img src="images/group_01.png" alt="Alt text" width="1500">

Before training, we should rearrange the Autofish dataset to be compatible with the YOLO models.
As i discussed in the Data exploration section, the COCO annotation configuration is quiet different from the YOLO configuration. 
The images and the annotations were converted to YOLO compatible structure while maintaining the oroginal image groups structure. The exact train, val and test split was follwed according to the below paper. 

## Citation
> **Bengtson, S. H., Lehotský, D., Ismiroglou, V., Madsen, N., Moeslund, T. B., & Pedersen, M. (2025).**  
> *AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish.*  
> [http://arxiv.org/abs/2501.03767](http://arxiv.org/abs/2501.03767)

<details>
  <summary>View BibTeX entry</summary>

```bibtex
@article{Bengtson2025,
   author = {Stefan Hein Bengtson and Daniel Lehotský and Vasiliki Ismiroglou and Niels Madsen and Thomas B. Moeslund and Malte Pedersen},
   month = {1},
   title = {AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish},
   url = {http://arxiv.org/abs/2501.03767},
   year = {2025},
}

```
</details>

<h2>Training with bounding boxes</h2>
The code is for the compete process of importing the dataset, modifying the annotations and preparing the YOLO compatible data structure in a different directory


```python
import json
import os
import random
import shutil
from pathlib import Path

# Paths
coco_path = "/work3/msam/Thesis/autofish/annotations.json"
image_groups_dir = "/work3/msam/Thesis/autofish"
output_base_dir = "/work3/msam/Thesis/yolodataset"
os.makedirs(output_base_dir, exist_ok=True)

#defining the train, val and test split of the data
train_groups = {2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24}
val_groups   = {1, 6, 11, 17, 25}
test_groups  = {10, 14, 20, 21, 22}

#createing output directories for each split
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, "labels"), exist_ok=True)

#function to convert COCO annotaions to YOLO format
def convert_coco_to_yolo(x_min, y_min, width, height, img_width, img_height):
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center, y_center, norm_width, norm_height

#data processing
def process_dataset(coco_path, image_groups_dir, output_base_dir):
    #Load COCO annotations
    with open(coco_path, "r") as f:
        data = json.load(f)
    
    #Build a mapping from image_id to its annotations for quick lookup
    annotations_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)
    
    #Process each image defined in the oroginal annotations
    for img in data["images"]:
        img_id = img["id"]
        file_name = img["file_name"]  # example- "group_03/00001.jpg"
        parts = file_name.split('/')
        if len(parts) < 2:
            print(f"Warning: File name format is unexpected: {file_name}")
            continue
        
        #Extract the group folder and original file name
        group_folder = parts[0]      #example "group_03"
        original_filename = parts[-1]  #example "00001.jpg"
        
        #Parse the group number (assumes folder name in the format "group_XX")
        try:
            #This converts "group_03" -> "03" -> 3
            group_num = int(group_folder.split('_')[-1])
        except ValueError:
            print(f"Warning: Could not parse group number from {group_folder}")
            continue
        
        #Determine the dataset split based on the group number
        if group_num in train_groups:
            split = "train"
        elif group_num in val_groups:
            split = "val"
        elif group_num in test_groups:
            split = "test"
        else:
            print(f"Warning: Group number {group_num} is not assigned to any split folder.")
            continue
        
        #Build the source image path and the destination directory (preserving group folder)
        src_img_path = os.path.join(image_groups_dir, file_name)
        dest_img_dir = os.path.join(output_base_dir, split, "images", group_folder)
        os.makedirs(dest_img_dir, exist_ok=True)
        #Rename the image with its unique image id

        dest_img_filename = original_filename
        dest_img_path = os.path.join(dest_img_dir, dest_img_filename)
        
        #Copy the image file if it exists
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dest_img_path)
        else:
            print(f"Warning: Source image {src_img_path} does not exist.")
        
        #Process annotations for this image
        anns = annotations_by_image.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, width, height] in COCO format
            x_center, y_center, norm_width, norm_height = convert_coco_to_yolo(
                bbox[0], bbox[1], bbox[2], bbox[3],
                img["width"], img["height"]
            )
            category_id = ann["category_id"]
            yolo_lines.append(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}")
        
        #Save YOLO annotations if available
        if yolo_lines:
            dest_label_dir = os.path.join(output_base_dir, split, "labels", group_folder)
            os.makedirs(dest_label_dir, exist_ok=True)
            # Use the original file name (changing extension to .txt)
            label_filename = f"{Path(original_filename).stem}.txt"
            label_path = os.path.join(dest_label_dir, label_filename)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
    
    #create a dataset.yaml file for YOLO traiing
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
    
    print("Dataset processing done")

# ======= Run the Processing =======
if __name__ == "__main__":
    process_dataset(coco_path, image_groups_dir, output_base_dir)

```
Combined training (Separated abd touched configurations images together)
```python
from ultralytics import YOLO
model = YOLO("yolo11m.pt")
```
```python

import os
import pandas as pd


data_path = f"/work3/msam/Thesis/yolodataset/dataset.yaml"  
output_dir = f"/work3/msam/Thesis/yolodataset/results" 
os.makedirs(output_dir, exist_ok=True)

# hyperameters
model_path = "yolo11m.pt"  # YOLOv11 extra large model
epochs = 300  # Fine-tuning epochs
learning_rate = 0.001  # Fine-tuning learning rate
batch_size = 32  # Batch size
img_size = 640 
optimizer = "Adam" # Image size

#Initialize the model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = YOLO(model_path)

#output path for naming the folder with hperparameter configurations
config_name = f"xlarge_finetune_epoch{epochs}_batch{batch_size}__lr{learning_rate}"
config_output_dir = os.path.join(output_dir, config_name)
os.makedirs(config_output_dir, exist_ok=True)

#model training
print(f"Fine-tuning medium model with epochs={epochs}, lr={learning_rate}, batch={batch_size}, img_size={img_size}")
try:
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        device=0,  # Use GPU
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,# Learning rate
        project=config_output_dir,  # Save results in the fine-tuning folder
        name="finetune_results"
    )
    print(f"Fine-tuning completed for medium model. Extracting metrics...")
except Exception as e:
    print(f"Training failed: {e}")

# Validate the model and extract results
results = model.val(
    data=data_path,
    imgsz=img_size,
    save_json=True,  # Save predictions in COCO-JSON format
    save_conf=True,  # Save confidence scores
    conf=0.5,  # Confidence threshold
    save=True  # Save predictions
)

# Extract metrics
metrics = {
    "precision": results.box.mp,  # Mean precision
    "recall": results.box.mr,  # Mean recall
    "mAP50": results.box.map50,  # mAP at IoU=0.50
    "mAP50-95": results.box.map  # mAP at IoU=0.50-0.95
}

#Save metrics to a CSV file
metrics_df = pd.DataFrame([metrics])
metrics_csv_path = os.path.join(config_output_dir, 'metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved to {metrics_csv_path}")

#Extract confusion matrix
confusion_matrix = results.confusion_matrix

#Convert confusion matrix to a DataFrame and save if available
if hasattr(confusion_matrix, 'matrix'):
    cm_data = confusion_matrix.matrix
    cm_df = pd.DataFrame(cm_data)
    cm_csv_path = os.path.join(config_output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path, index=False)
    print(f"Confusion matrix saved to {cm_csv_path}")
else:
    print("Warning: Confusion matrix is not in the expected format. Skipping saving.")

# Free GPU memory
del model
import torch
torch.cuda.empty_cache()

print(f"All results and metrics saved in {config_output_dir}.")
```


