<h1>Object Re-Identification in Electronic Monitoring (EM) in Fisheries</h1>

## Context & Motivation Object 

Re-identification (Re-ID) is a pivotal technology for modernizing fisheries science. In Electronic Monitoring (EM) systems, cameras record catch handling on fishing vessels to document species composition, length distributions, and bycatch (including PET species). Currently, reviewing this footage is a manual, labor-intensive, and costly process. While early automation focused on simple species recognition, effective stock assessment requires higher precision: the ability to count, track, and measure individual fish. Relying solely on classification fails to distinguish between new and previously counted individuals, leading to inaccurate biomass estimates.

## Project Objective 

This study addresses the challenge of keeping track of every commercial catch species by leveraging Object Re-ID deep learning techniques. While Re-ID is typically used for tracking objects across different camera views, we apply it here to the Autofish dataset—a collection of images mimicking conveyor belt environments with fish in randomized spatial configurations. The goal is to develop a robust methodology that can identify unique individuals regardless of their arrangement, laying the foundation for automated counting and length estimation (Here, we focus only on the Re-ID aspect, not the length measurements).

## Technical Approach 

We evaluate and benchmark two distinct deep learning architectures to determine the best approach for the marine domain:
- Convolutional Neural Networks (CNN): Represented by ResNet-50, testing standard feature extraction capabilities.
- Vision Transformers (ViT): Represented by Swin Transformer Tiny, testing the efficacy of attention mechanisms in capturing global context on sorting belts.

<img src="images/EM_FIsh.png" alt="Alt text" width="1200">
3D modelled diagram showing the Electronic Monitoring (EM) systems equipped with Computer vision algorithms that can identify species and track them to obtain a precise count for each species.

<table>
  <tr>
    <td align="center">
      <img src="images/Art_1_beam.gif" width="600" />
      <br />
      <b>EM system camera</b>
    </td>
    <td align="center">
      <img src="images/belt sort.gif" width="600" />
      <br />
      <b>Conveyor belt setup</b>
    </td>
  </tr>
</table>



## Read our Paper in ArXiv
[![arXiv](https://img.shields.io/badge/arXiv-Access_Paper-b31b1b.svg)](https://arxiv.org/abs/2512.08400)


## Citation
If you use this code or paper in your research, please cite it as follows:

```bibtex
@misc{thilakarathna2025visualreidentificationfishusing,
      title={Towards Visual Re-Identification of Fish using Fine-Grained Classification for Electronic Monitoring in Fisheries}, 
      author={Samitha Nuwan Thilakarathna and Ercan Avsar and Martin Mathias Nielsen and Malte Pedersen},
      year={2025},
      eprint={2512.08400},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08400}, 
}
```




