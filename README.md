<h2>Object Re-Identification in Fisheries Science</h2>

Object re-identification (Re-ID) is a crucial technology with real-world industrial applications, including fisheries science. The ability to count and track individual fish over time and space enhances monitoring and behavioral analysis, providing valuable insights into fisheries management.

In fisheries technology, accurately tracking the number of individuals entering a fishing gear—especially active gears like trawls—is essential. However, challenges arise when fish move with the gear, re-enter, or appear in multiple frames at different times, leading to miscounts and confusion in underwater video observations.

To address this, developing robust Re-ID models is critical. These models must maintain accurate identification of each individual while ensuring efficiency and reliability. This project will follow a structured approach to achieve this goal, incorporating advanced tracking techniques and AI-driven methodologies to improve fish identification and monitoring.

#Steps in the project
1. Identifying and understanding the data structure and annotations of the Autofish dataset
2. Rearranging the dataset and annotations to train a YOLO model (YOLOv11 latest models)



   
<h2>Data structure in the AutoFish dataset</h2>

# **Dataset Structure for COCO Annotations**

This dataset consists of **1,500 images** organized into **24 folders**, named sequentially as:  
`group_01, group_02, ..., group_24`.  

Each folder contains multiple images, but **all images within a folder start with the filename `00001.png`**. Despite identical file names across folders, **each image has a unique image ID** for identification.

---

## **Dataset Organization**
The dataset follows the **COCO JSON format**, containing three main sections:

### **1. Images**
Each image entry includes:
- `id`: A unique image identifier  
- `file_name`: The relative path, e.g., `"group_01/00001.png"`  
- `height`: Image height  
- `width`: Image width  
- `group`: The folder to which the image belongs  

Example:
```json
"images": [
    {
        "id": 1,
        "file_name": "group_01/00001.png",
        "height": 1080,
        "width": 1920,
        "group": "group_01"
    },
    {
        "id": 2,
        "file_name": "group_02/00001.png",
        "height": 1080,
        "width": 1920,
        "group": "group_02"
    }
]

<img src="image 1.png" alt="Alt text" width="500">

