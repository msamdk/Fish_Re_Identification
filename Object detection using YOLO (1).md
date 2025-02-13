

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
The code is for the compete process of importing the dataset, modifying the annotations and preparing the YOLO compatible data structure in a different directory. The filenames were not changed because YOLO is okay with the same file name in diffrerent directories.


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
        
        #Parse the group number 
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
Here the training will is done using medium architecture of YOLO object detection model.

- To get in pace with the training with the Autofish paper, 10 random model initializations were done to calculate the statistics.
- All model architectures were tested (n,s,m,l and xl) to see how the performance vary according to the Model architecture
- 
```python
from ultralytics import YOLO
model = YOLO("yolo11m.pt")
#loaded all the architectures first in the same way to the working directory
```
```python
#training the combined configuration for the total dataset
from ultralytics import YOLO
import os
import pandas as pd
import torch

# Base paths
data_path = f"/work3/msam/Thesis/yolodataset/dataset.yaml"
output_dir = f"/work3/msam/Thesis/yolodataset/results"
os.makedirs(output_dir, exist_ok=True)

# Define all YOLOv11 architectures
yolo_architectures = {
    "nano": "yolo11n.pt",      # Nano model
    "small": "yolo11s.pt",     # Small model
    "medium": "yolo11m.pt",    # Medium model
    "large": "yolo11l.pt",     # Large model
    "extra_large": "yolo11x.pt" # Extra Large model
}

# Training hyperparameters
hyperparameters = {
    "epochs": 300,
    "learning_rate": 0.001,
    "batch_size": 32,
    "img_size": 640,
    "optimizer": "Adam"
}

# Train each architecture
for arch_name, model_path in yolo_architectures.items():
    print(f"\n{'='*50}")
    print(f"Training {arch_name.upper()} architecture")
    print(f"{'='*50}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping...")
        continue
    
    # Initialize model
    model = YOLO(model_path)
    
    # Create output directory for this architecture
    config_name = f"{arch_name}_HPC_JOB1{hyperparameters['epochs']}_batch{hyperparameters['batch_size']}_lr{hyperparameters['learning_rate']}"
    config_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Train model
    print(f"Training {arch_name} model with:")
    print(f"- Epochs: {hyperparameters['epochs']}")
    print(f"- Learning Rate: {hyperparameters['learning_rate']}")
    print(f"- Batch Size: {hyperparameters['batch_size']}")
    print(f"- Image Size: {hyperparameters['img_size']}")
    
    try:
        model.train(
            data=data_path,
            epochs=hyperparameters['epochs'],
            imgsz=hyperparameters['img_size'],
            device=0,  # Use GPU
            batch=hyperparameters['batch_size'],
            lr0=hyperparameters['learning_rate'],
            optimizer=hyperparameters['optimizer'],
            project=config_output_dir,
            name="finetune_results"
        )
        print(f"Training completed for {arch_name} model")
        
        # Validate model
        results = model.val(
            data=data_path,
            imgsz=hyperparameters['img_size'],
            save_json=True,
            save_conf=True,
            conf=0.5,
            save=True
        )
        
        # Save metrics
        metrics = {
            "architecture": arch_name,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map
        }
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_csv_path = os.path.join(config_output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Metrics saved to {metrics_csv_path}")
        
        # Save confusion matrix
        if hasattr(results.confusion_matrix, 'matrix'):
            cm_data = results.confusion_matrix.matrix
            cm_df = pd.DataFrame(cm_data)
            cm_csv_path = os.path.join(config_output_dir, 'confusion_matrix.csv')
            cm_df.to_csv(cm_csv_path, index=False)
            print(f"Confusion matrix saved to {cm_csv_path}")
        
    except Exception as e:
        print(f"Error training {arch_name} model: {e}")
    
    # Clean up GPU memory
    del model
    import torch
    torch.cuda.empty_cache()
    print(f"Completed training and evaluation for {arch_name} architecture")

# Create combined metrics file
combined_metrics = []
for arch_name in yolo_architectures.keys():
    config_name = f"{arch_name}_finetune_epoch{hyperparameters['epochs']}_batch{hyperparameters['batch_size']}_lr{hyperparameters['learning_rate']}"
    metrics_path = os.path.join(output_dir, config_name, 'metrics.csv')
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        combined_metrics.append(metrics)

if combined_metrics:
    combined_df = pd.concat(combined_metrics, ignore_index=True)
    combined_csv_path = os.path.join(output_dir, 'combined_metrics.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined metrics for all architectures saved to {combined_csv_path}")

print("\nTraining completed for all architectures!")
```

Validation/performance metrics for each class (YOLO v11 detection modep medium architecture
```text
                

```

 ## Training the YOLO models for the separate configuration
 Here the First 40 images were taken from each image group from the train/val split
 - All model architectures were tested (n,s,m,l and xl) to see how the performance vary according to the Model architecture
 - The same code was applied to the touched configuration (images 41-60)
```python
#training the separated configuration (first 40 images)
import os
import glob
import yaml

# Dataset directories
base_dataset_dir = "/work3/msam/Thesis/yolodataset2"
dataset_yaml_path = os.path.join(base_dataset_dir, "dataset_filtered.yaml")

# Function to get first 40 images per group for a given split (train/val/test)
def get_first_40_images(split):
    images_dir = os.path.join(base_dataset_dir, split, "images")
    selected_images = []

    for group in sorted(os.listdir(images_dir)):  # Iterate over group folders
        group_path = os.path.join(images_dir, group)

        if os.path.isdir(group_path):
            image_files = sorted(glob.glob(os.path.join(group_path, "*.jpg")) +
                                 glob.glob(os.path.join(group_path, "*.png")))

            # Take only the first 40 images
            selected_images.extend(image_files[:40])

    return selected_images

# Get filtered images for train, val, and test
train_images = get_first_40_images("train")
val_images = get_first_40_images("val")
test_images = get_first_40_images("test")

# Save train, val, and test image lists to text files (with the absolute paths of filtered images)
def save_image_list(image_list, file_path):
    with open(file_path, "w") as f:
        f.write("\n".join(image_list))
    print(f"Saved image list to {file_path}")

train_txt = os.path.join(base_dataset_dir, "train.txt")
val_txt = os.path.join(base_dataset_dir, "val.txt")
test_txt = os.path.join(base_dataset_dir, "test.txt")

save_image_list(train_images, train_txt)
save_image_list(val_images, val_txt)
save_image_list(test_images, test_txt)

# Create dataset.yaml for YOLO training
dataset_yaml = {
    "path": base_dataset_dir,
    "train": train_txt,  # Pointing to the text file instead of listing images
    "val": val_txt,
    "test": test_txt,
    "names": {
        0: "horse_mackerel",
        1: "whiting",
        2: "haddock",
        3: "cod",
        4: "hake",
        5: "saithe",
        6: "other"
    }
}

# Save the filtered dataset.yaml file
with open(dataset_yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f)

print(f"Filtered dataset YAML saved at {dataset_yaml_path}")

# Train YOLO model using the filtered dataset
from ultralytics import YOLO
model = YOLO("yolo11m.pt")  # Use your YOLO model file

model.train(
    data=dataset_yaml_path,
    epochs=300,
    imgsz=640,
    device=0,  # Use GPU
    batch=32,
    lr0=0.001,
    optimizer="Adam",
    project=os.path.join(base_dataset_dir, "results"),
    name="finetune_results"
)

print("Training completed.")
```
## Overall model performances in 3 configurations.
Here only a single training run has been done to identify the model performance on the different configurations
<img src="images/overall performance.png" alt="Alt text" width="1500">

