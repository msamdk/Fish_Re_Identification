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






















