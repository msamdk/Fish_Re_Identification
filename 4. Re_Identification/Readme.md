# Challenges in EM video footage

<img src="images/limitations of em.png" alt="Alt text" width="1200">

# Schematic Diagram of the concept of Re-ID application in the fisheries context

<img src="images/reid.jpg" alt="Alt text" width="1200">

# t-SNE visualizations of best performing varíants of both ResNet-50 (256 batch size) and Swin-T (16 batch size)

| Swin-T (Vision Transformer) - 41.6% mAP@R & 90.3% R1 | ResNet 50 (CNN) - 13.56% mAP@R & 70.26% R1|
| :---: | :---: |
| ![Demo GIF 1](images/swin.gif) | ![Demo GIF 2](images/resnet.gif) |

# Latest Update: UMAP results. 
![Demo GIF 1](images/umap.gif)

In here, the clusters represent a better arrangement than the t-SNE results. The right side shows the species-wise clusters, and the left side shows the individual clusters inside the species cluster. The individual clusters represent the specific individuals that belong to the large species cluster.

# Fish Re-identification Pipeline and process

The Re-Identification experiments were done to represent the pre-trained (off-the-shelf) model performance using zero-shot instance retrieval and the main experiments include the finetuning the feature extractor models (Swin-T and ResNet) with Triplet learning

The Finetuning Pipeline includes
1. Ground Truth crop dataset preparation
2. Training with variable batch sizes of data (26, 32, 64 and 256). A custom PK sampler is used to make these batch sizes. PK sampler helps to make batches with number of Fish IDs (P) multiplied by the number of instances per fish ID (K)
Therefore the batch size = P*K
3. Then the model is trained using Triplet Margin loss, which is a popular metric learning  strategy which makes triplets of images:
   - A-Anchor (reference fish ID),
   - P-Positive (different image of the same fish ID)
   - N- Negative (different fish ID)
  
Triplet margin miner is used to identify the triplets. Hard triplets are used in here. Hard triplets follow the condition that the distance (A,P) > distance (A,N) which simulates the worst cases where the model cannot identify same fish ID. so the model will select these highly confusing pairs and form the triplets. so the model can learn sublte features of two highly similar fish individuals.

## Triplet loss function

```math
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \max \left( \left\| f(x_i^a) - f(x_i^p) \right\|^2 - \left\| f(x_i^a) - f(x_i^n) \right\|^2 + \alpha, 0 \right)
```

Where,

* **$L_T$**: The final **Triplet Loss**, averaged over the batch.
* **$N$**: The **number of triplets** in the batch.
* **$a_i$**: The **anchor** sample in the $i$-th triplet.
* **$p_i$**: The **positive** sample (same class as $a_i$).
* **$n_i$**: The **negative** sample (different class from $a_i$).
* **$f(x)$**: The **embedding function** that maps an input $x$ to a feature vector.
* **$\|f(a_i) - f(p_i)\|^2$**: The **squared Euclidean distance** between the anchor and positive embeddings.
* **$\|f(a_i) - f(n_i)\|^2$**: The **squared Euclidean distance** between the anchor and negative embeddings.
* **$m$**: A **margin** hyperparameter that enforces a minimum separation between pairs.
* **$[\cdot]_+$**: The **hinge loss function**, defined as `max(·, 0)`, which ensures the loss is non-negative.


___________________________________________________________________________________________________
## Model Training
This image shows the core training steps in detail. Here there are many important things to consider. when the original dataset is splitted to the training, validation and test categories the fish IDs are dispersed without a continuous format. This is causing problems during the accuracy calculation in the metric learning. To prevent unexpected errors, label encoding was performed to the images before feeding the images into the network. Simply, the fish ids are assigned a numerical ID after sorting the fish IDs in a particular subset (train, val or test) which starts from 0 to 1.
If the fish IDs are in string format, the integers are assigned in a lexicographical order. This was done for all train, val and test data.

The PK sampler is an important element in the pipeline due to several reasons
1. Getting a good representation of diverse identities (diverse triplets)
2. Prevent degenerate batches (with structured batch generation)
3. Focus on challenging exmaples-most informative triplets

PK sampler is only used in training data. The validation data are fed to the network as fixed batch sizes of 32

The feature extractor modification is also an important asset. Usually, the feature extractor models comes with a final classification layer. but in this use case, we dont need to classify each individual. We need a feature representation of each individual fish. Therefor, the final classification layer is replaces with an identity layer which passes the high-dimensional feature vector from the backbone straight into an embedding head, which projects that high-dimensioal feature vector to form a low dimensional embedding vector with 512 dimensions.


<img src="images/1.jpg" alt="Alt text" width="1200">


##  model Evaluation using the test dataset

In the evaluation, the test data was splited in to query set and a gallery set where one random instance is selected as the query and all 39 other instances are transfered to the gallery which includes all the other fish IDs as well

<img src="images/2.jpg" alt="Alt text" width="1200">

# Performance metrics for Evaluation

Here mean average precision at R (mAP@R) is used as the primary metric which displays the ranking quality of the re-id for a particular query. Rank-1 accuracy (R1) was also calculated to have an idea how many queries detect the correct match as the first match as a percentage. The calculations can be seen in the following image

<img src="images/metrics.jpg" alt="Alt text" width="1200">
