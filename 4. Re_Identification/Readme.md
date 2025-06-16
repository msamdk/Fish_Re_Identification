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

```equation
L_T (a,p,n)=1/N ∑_i^N▒max⁡(‖f(x_i^a )-├ f(x_i^p )┤‖┤_2^2-‖f(x_i^a )-├ f(x_i^p )┤‖┤_2^2+m,0) 	   (4)


```
<img src="images/1.jpg" alt="Alt text" width="1200">
<img src="images/2.jpg" alt="Alt text" width="1200">
<img src="images/figure6.jpg" alt="Alt text" width="1200">
