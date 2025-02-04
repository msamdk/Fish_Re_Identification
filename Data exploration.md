<h2>Data exploration using python</h2>

Reading the dataset from the source. This is an example if you are doing it from google collab. First you have to mount the drive and then define the dataset path and execute this. Other wise use your local disk
```python
# Reading the data (Annotations)
import json

with open('/content/drive/MyDrive/Thesis/autofish/annotations.json') as f:
    annotations = json.load(f)
```
Printing major keys in the annotation file
```python
print("Keys in annotation file:", annotations.keys())
#COCO style annnotations typically contain these keys
#images
#annotations
#categories
```
Object classes in the annotation file
```python
#all object classes in the annotations
categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}
print("Categories:", categories)
```
list an example annotation
```python
#list an example annotation
ann_ex = annotations['annotations']
print("Example annotation:", ann_ex[0])
```
