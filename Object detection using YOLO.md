
<h2>Training an Object detection model using latest YOLO v11 models</h2>
<img src="images/ylo.png" alt="Alt text" width="500">

Before training, we should rearrange the Autofish dataset to be compatible with the YOLO models.
As i discussed in the Data exploration section, the COCO annotation configuration is quiet different from the YOLO configuration. The image arrangement also have to be changed according to the YOLO model compatibility.
Here i deployed following steps to rearrange the dataset

1. Duplicating the images of Autofish data which in the folder structure by assigning their unique image id to rename all the image files and transfer the images in to a new directory which contain train, val and test folders.
2. Insidet these folders there should be two subfolders as images and labels to be compatible with the YOLO (see the diagram)

3. Converting the COCO annotation format to YOLO format and generate text file for the corresponding image file in the respective train, val or test folder
4. Installing ultralytics and starting to train the pretrained YOLO models with this custom dataset (here the goal is to test the performance of the model to identify each of the object with the correct class with high accuracy)

   
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

