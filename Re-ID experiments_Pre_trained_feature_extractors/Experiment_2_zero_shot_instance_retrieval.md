## Xero shot instance retrieval experiment

In this experiment, I used a similar pipeline to the previous experiment. The Autofish dataset contains 40 instances of each fish ID across the image group to which it belongs. The image group captains separated and occluded configurations. 
IN this experiment, 
- 2 instances from the initial side separated
- 2 instances from the flipped side separated
- 5 instances from the initial side occluded
- 5 instances from the  flipped side occluded
were taken to construct the gallery database with raw feature vectors (L2 normalised).

The feature extractor was used as it is, without specifically training with similarity learning networks like triplet loss and siamese networks 

