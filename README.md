<h2>Object Re-Identification in Fisheries Science</h2>

![Demo GIF](images/catch.gif)


Object re-identification (Re-ID) is a crucial technology with real-world industrial applications, including fisheries science. The ability to count and track individual fish over time and space enhances monitoring and behavioral analysis, providing valuable insights into fisheries management.

In fisheries technology, accurately tracking the number of individuals entering a fishing gear—especially active gears like trawls—is essential. However, challenges arise when fish move with the gear, re-enter, or appear in multiple frames at different times, leading to miscounts and confusion in underwater video observations.

To address this, developing robust Re-ID models is critical. These models must maintain accurate identification of each individual while ensuring efficiency and reliability. This project will follow a structured approach to achieve this goal, incorporating advanced tracking techniques and AI-driven methodologies to improve fish identification and monitoring.


[![arXiv](https://img.shields.io/badge/arXiv-Access_Paper-b31b1b.svg)](https://arxiv.org/abs/2512.08400)

## Citation
If you use this code or paper in your research, please cite it as follows:

```bibtex
@misc{thilakarathna2025visualreidentificationfishusing,
      title={Towards Visual Re-Identification of Fish using Fine-Grained Classification for Electronic Monitoring in Fisheries}, 
      author={Samitha Nuwan Thilakarathna and Ercan Avsar and Martin Mathias Nielsen and Malte Pedersen},
      year={2025},
      eprint={2512.08400},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08400}, 
}


## **Dataset Structure for COCO Annotations in AutoFish data**

<a href="https://huggingface.co/datasets/vapaau/autofish" target="_blank">
    <img src="https://img.shields.io/badge/View%20Dataset-AutoFish-blue?style=for-the-badge" />
</a>




This dataset consists of **1,500 images** organized into **24 folders**, named sequentially as:  
`group_01, group_02, ..., group_24`.  

Each folder contains multiple images, but **all images within a folder start with the filename `00001.png`**. Despite identical file names across folders, **each image has a unique image ID** for identification.

---

<img src="images/data_str.jpg" alt="Alt text" width="1200">
<img src="images/image 1.png" alt="Alt text" width="500">

## **Dataset Organization**
The dataset follows the **COCO JSON format**, containing three main sections:
<img src="images/cocojson.png" alt="Alt text" width="500">

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
   "height": 2056,
   "width": 2464,
   "id": 1,
   "file_name": "group_01/00001.png",
   "group": 1
  },
  {
   "height": 2056,
   "width": 2464,
   "id": 2,
   "file_name": "group_01/00002.png",
   "group": 1
  }
]
```
### **2. Annotations**
Each annotation entry includes:
- `image_id`: A unique image identifier  
- `bbox`: The bounding box coordinates  
- `segmentation`: segmentation mask coordinates  
- `category_id`: Iclass id of the objects (fish)  
- `glength`: Total length of the fish
- `fish_id`: TUnique id for each fish in the images
- `side_up`: Side of the fish whether it is left or right
- `id` : Segmentation mask ID
- `area`: Area of the object covered

Example
```json
"annotations": [
  {
   "iscrowd": 0,
   "image_id": 1,
   "bbox": [
    381.0,
    1123.0,
    822.0,
    378.0
   ],
   "segmentation": [
    [
     1192,
     1501,
     1110,
     1477,
     1038,
     1437,
     872,
     1417,
     594,
     1323,
     482,
     1263,
     393,
     1176,
     381,
     1142,
     430,
     1123,
     498,
     1123,
     800,
     1233,
     888,
     1285,
     1048,
     1415,
     1196,
     1439,
     1203,
     1448,
     1183,
     1480
    ]
   ],
   "category_id": 0,
   "length": 35.5,
   "fish_id": 316,
   "side_up": "R",
   "id": 1,
   "area": 92164
  }
]
```
### **3. Categories**
Each annotation entry includes:
- `id`: A unique class identifier  
- `name`: Name of the class  
- `supercategory`: super category of classes

```json
"categories": [
  {
   "id": 0,
   "name": "horse_mackerel",
   "supercategory": "horse_mackerel"
  },
  {
   "id": 1,
   "name": "whiting",
   "supercategory": "whiting"
  },
  {
   "id": 2,
   "name": "haddock",
   "supercategory": "haddock"
  },
  {
   "id": 3,
   "name": "cod",
   "supercategory": "cod"
  },
  {
   "id": 4,
   "name": "hake",
   "supercategory": "hake"
  },
  {
   "id": 5,
   "name": "saithe",
   "supercategory": "saithe"
  },
  {
   "id": 6,
   "name": "other",
   "supercategory": "other"
  }
 ]
}

```
 


