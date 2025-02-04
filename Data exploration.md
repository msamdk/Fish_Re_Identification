<h2>Data exploration using python</h2>

Reading the dataset from the source. This is an example if you are doing it from google collab. First you have to mount the drive and then define the dataset path and execute this. Other wise use your local disk
```python

import json

with open('/content/drive/MyDrive/path//annotations.json') as f:
    annotations = json.load(f)
```
Printing major keys in the annotation file.
COCO style annnotations typically contain these keys
1. images
2. annotations
3. categories
```python
print("Keys in annotation file:", annotations.keys())

```
Object classes in the annotation file
```python

categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}
print("Categories:", categories)
```
list an example annotation
```python

ann_ex = annotations['annotations']
print("Example annotation:", ann_ex[0])
```
<h2>pycocotools</h2>

But rather than using json to read the annotation, you can use pycocotools
```python

pip install pycocotools
```
After installation you can get to know about the functions in the library
```python
help(COCO)
```
To see a specific function
```python
help(COCO.loadCats)
```

Importing the dataset
```python
from pycocotools.coco import COCO

annotation = 'path/to//annotations.json'
coco = COCO(annotation)
```
getting category ID and category name
```python

categories = [(category['id'],category['name']) for category in categories]
print(categories)
```

A specific category iD
```python
cat_es = coco.getCatIds(catNms=['whiting'])
print(cat_es)
```

Get the sequence of image IDs
```python
img_ids = coco.getImgIds()
print(img_ids)
```

**Annotation ID**

getAnnIds() function is crucial for extracting specific annotation ids that match certain image IDs, category IDs or other criteria

**What does it do?**

it retrieves annotation ids that satisft the given conditions


> ann_ids = coco.getAnnIds(imgIds=[10, 20], catIds=[1, 2], iscrowd=None)

here filters annotations for images with ID 10 and 20, filters annotations belonging to categories 1 and 2, igores the annotations marked as crowd


Get annotation IDs for objects larger than 1000 pixels in area
```python
ann_ids_large = coco.getAnnIds(areaRng=[1000, float('inf')])
annotations_large = coco.loadAnns(ann_ids_large)
print(f"Total large annotations: {len(annotations_large)}")
```

knowing the annotation structure of the annotation.json file
```python
img_ID = 2
ann_ids = coco.getAnnIds(imgIds=[img_ID])
annotations = coco.loadAnns(ann_ids)
print(annotations[0])
```

```
{
  'iscrowd': 0,
  'image_id': 2,
  'bbox': [79.0, 479.0, 196.0, 892.0],
  'segmentation': [[140, 1371, 110, 1369, 101, 1346, 143, 1188, 129, 1068, 85, 982, 79, 912, 97, 688, 115, 616, 157, 502, 190, 479, 217, 514, 263, 652, 275, 844, 239, 1146, 197, 1210, 185, 1280],
  'category_id': 1,
  'length': 36.0,
  'fish_id': 419,
  'side_up': 'L',
  'id': 9,
  'area': 111561
  }

```
















