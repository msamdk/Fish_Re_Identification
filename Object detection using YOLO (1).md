
<h2>Training an Object detection model using latest YOLO v11 models</h2>
<img src="images/group_01.png" alt="Alt text" width="1500">

Before training, we should rearrange the Autofish dataset to be compatible with the YOLO models.
As i discussed in the Data exploration section, the COCO annotation configuration is quiet different from the YOLO configuration. 
The images and the annotations were converted to YOLO compatible structure while maintaining the oroginal image groups structure. The exact train, val and test split was follwed according to the below paper. 

## Citation

> **Bengtson, S. H., Lehotský, D., Ismiroglou, V., Madsen, N., Moeslund, T. B., & Pedersen, M. (2025).** *AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish.* [http://arxiv.org/abs/2501.03767](http://arxiv.org/abs/2501.03767)

<details>
  <summary>View BibTeX entry</summary>

```bibtex
@article{Bengtson2025,
   author = {Stefan Hein Bengtson and Daniel Lehotský and Vasiliki Ismiroglou and Niels Madsen and Thomas B. Moeslund and Malte Pedersen},
   month = {1},
   title = {AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish},
   url = {http://arxiv.org/abs/2501.03767},
   year = {2025},
}
```

This section is for training with bounding boxes



```python


```

