## Initial Re-Identification experiment using Ranking based on L2 normlized Vectors on Euclidian distance

<img src="images/re_id/1.jpg" alt="Alt text" width="2000">
<img src="images/re_id/2.jpg" alt="Alt text" width="2000">
<img src="images/re_id/3.jpg" alt="Alt text" width="2000">

## COCO format provides 
- segmentation masks as polygons using absolute pixel coordinates  
- uses JSON format with a list of (x,y) points that form the contour
- "segmentation": [[x1,y1, x2,y2, x3,y3, ...]]

## YOLO format
- stores segmentation masks as normalized coordinates (0-1)
- use plain txt format (.txt) with space separated values
- format per line class_id x1 y1 x2 y2 ... xn yn
- coordinates are normalized by image width and height
  


## steps for the trainins

1. Making a separate YOLO-seg model compatible dataset for the training

```python
