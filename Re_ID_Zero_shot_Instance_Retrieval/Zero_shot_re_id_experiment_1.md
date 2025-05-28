## Re_Id experiment with zero-shot

This experiment uses only a few instances of a specific fish_id. This directly simulates the zero-shot learning method because the feature extractor (ResNet50) is not trained on the Autofish dataset.

14 instances from each fish_id were taken to build the gallery and 26 was taken as queries. This is because the autofish dataset has 40 instances of each fish ID. and the experiment was done only using the test split image groups (5 groups).

The image groups contain 60 images and have 3 subsets organised according to the file names.

- set_1_initial; 00001 to 00010 (separated)
- set_1_flipped: 00010 to 00020 (separated)
- set_2_intial: 00021 to 00030 (separated)
- set_2_flipped: 00031 to 00040 (separated)
- All_set_initial: 00041 to 00050 (touched)
- All_set_flipped: 00051 to 00060 (touched)

Then this 3D plot shows the raw 2048 dimensional feature vector recieved from the outer layer of the ResNet model after removing the final classification layer (FC layer)

