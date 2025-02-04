<h2>Data exploration using python</h2>

Reading the dataset from the source. This is an example if you are doing it from google collab. First you have to mount the drive and then define the dataset path and execute this. Other wise use your local disk
```python
# Reading the data (Annotations)
import json

with open('/content/drive/MyDrive/Thesis/autofish/annotations.json') as f:
    annotations = json.load(f)
```

```python
print("Keys in annotation file:", annotations.keys())
#COCO style annnotations typically contain these keys
#images
#annotations
#categories
```
