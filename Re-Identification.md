## Initial Re-Identification experiment using Ranking based on L2 normlized Vectors on Euclidian distance

<img src="images/re_id/1.jpg" alt="Alt text" width="2000">
<img src="images/re_id/2.jpg" alt="Alt text" width="2000">
<img src="images/re_id/3.jpg" alt="Alt text" width="2000">

## Ranking results (ResNet50)
- Rank-1  Accuracy: 35.86%
   - roughly 1 out of 3 times, the very best match the system finds (excluding the query photo itself) is the correct fish. 
- Rank-5  Accuracy: 57.05%
   - more than half the time, the correct fish appears somewhere within the top 5 most similar results
- Rank-10 Accuracy: 68.04%
   - over two thirds of tthe time, the correct fish is found within the top 10 results.




## Script

The script follows the above phases described visually in the diagrams above

```python
