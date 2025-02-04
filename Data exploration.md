<h2>Object Re-Identification in Fisheries Science</h2>

<img src="images/Title image.png" alt="Alt text" width="500">

Object re-identification (Re-ID) is a crucial technology with real-world industrial applications, including fisheries science. The ability to count and track individual fish over time and space enhances monitoring and behavioral analysis, providing valuable insights into fisheries management.

In fisheries technology, accurately tracking the number of individuals entering a fishing gear—especially active gears like trawls—is essential. However, challenges arise when fish move with the gear, re-enter, or appear in multiple frames at different times, leading to miscounts and confusion in underwater video observations.

To address this, developing robust Re-ID models is critical. These models must maintain accurate identification of each individual while ensuring efficiency and reliability. This project will follow a structured approach to achieve this goal, incorporating advanced tracking techniques and AI-driven methodologies to improve fish identification and monitoring.

## **Steps in the project**
1. Identifying and understanding the data structure and annotations of the Autofish dataset
2. Rearranging the dataset and annotations to train a YOLO model (YOLOv11 latest models)



## **Dataset Structure for COCO Annotations in AutoFish data**

# -*- coding: utf-8 -*-
"""
AutoFish Data Processing Script

This script loads the COCO-style annotation file (annotations.json) and extracts:
- Image details (ID, file path, height, width)
- Annotation details (bounding boxes, segmentation, fish ID, side)
- Category details (class ID, fish species)

Developed for fisheries object re-identification and behavior tracking.
"""

import os
import json
from google.colab import drive

# 🔹 Step 1: Mount Google Drive (Colab users)
drive.mount('/content/drive')

# 🔹 Step 2: Define Dataset Path
DATASET_PATH = '/content/drive/MyDrive/Thesis/autofish'

# 🔹 Step 3: Check if the Dataset Exists
print("Checking dataset files...")
print(os.listdir(DATASET_PATH))

# 🔹 Step 4: Load JSON Annotations
ANNOTATIONS_FILE = os.path.join(DATASET_PATH, "annotations.json")

with open(ANNOTATIONS_FILE) as f:
    annotations = json.load(f)

print("✅ Annotation keys:", annotations.keys())

# 🔹 Step 5: Extract Categories
categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}
print("📌 Fish Categories:", categories)

# 🔹 Step 6: Extract Sample Annotation
ann_all = annotations['annotations']
print("🔍 Sample annotation:", ann_all[0])

# 🔹 Step 7: Extract Image Information
images = annotations["images"]
for img in images[:5]:  # Display first 5 images
    print(f"🖼 Image ID: {img['id']} | File: {img['file_name']} | Size: {img['width']}x{img['height']}")

print("✅ Data processing completed successfully!")
