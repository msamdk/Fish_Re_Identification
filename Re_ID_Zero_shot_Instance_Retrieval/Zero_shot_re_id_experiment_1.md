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

Then this 3D plot shows the raw 2048-dimensional feature vector received from the outer layer of the ResNet model after removing the final classification layer (FC layer)

### 3D visualisation of the distribution of points (14 instances of each fish_ID
- Each point represents a cropped fish image and the position of 14 points in the 3D space is different due to the variation in the lighting, arrangement, occlusion levels and so on (2048 different features of the fish).
- This figure represents the complete 94 fish_id in 3-dimensional format made using t-SNE (t-distributed stochastic neighbour embedding) technique.
- This technique is commonly used in dimensionality reduction and data visualisation.
- It is a non-linear dimensionality reduction technique that focuses on preserving the local structure of the data by minimizing the Kullback-Leibler divergence between the high-dimensional and low-dimensional distributions. It is more suitable for non-linearly separable datasets and can handle outliers. t-SNE is computationally expensive and involves hyperparameters such as perplexity, learning rate, and number of steps
  
<img src="/Re_ID_Zero_shot_Instance_Retrieval/images/3.png" alt="Alt text" width="2000">
------------------------------------------------------------------------------------------------------------------------------------------------

### 3D visualisation of the distribution of 14 points of fish_id

<img src="/Re_ID_Zero_shot_Instance_Retrieval/images/2.png" alt="Alt text" width="2000">
