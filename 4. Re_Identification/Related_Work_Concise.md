# Related Work - Concise Version
## Individual Fish Re-Identification: Methods, Applications, and Species

Re-identification (Re-ID) has matured in person and vehicle domains, yet remains sparse for aquatic applications, particularly for similarly-looking commercial fish species. **Haurum et al. [1]** applied triplet metric learning for zebrafish individual identification in laboratory settings, achieving 99% mAP. **Mathisen et al. [2]** extended this to commercial aquaculture with FishNet for Atlantic salmon Re-ID through unified embeddings. Alternative approaches include **Figueroa-Mata et al. [3]** using Siamese networks for general fish identification, and **Pedersen et al. [4]** applying keypoint matching (SuperPoint/SuperGlue) for giant sunfish Re-ID. **Moskvyak et al. [5]** proposed landmark-guided embeddings for animal re-identification across diverse wildlife. Temporal adaptation has been addressed by **Olsen et al. [6]** through contrastive learning for wild corkwing wrasse population monitoring across seasons, and **Puchalla et al. [7]** through rolling window training for zebrafish identification using CNNs and Vision Transformers. **Fan et al. [8]** tackled multi-view challenges with synchronized cameras for fish re-identification in aquaculture. **Jiang et al. [9]** directly addressed commercial species using hierarchical coarse-to-fine features, though without architectural comparisons. These works reveal a critical gap: existing Re-ID research focuses on distinctive single species (zebrafish, salmon, sunfish) in controlled or wild settings, but lacks systematic evaluation of similarly-looking commercial species requiring fine-grained individual discrimination. This work addresses this gap by: (1) applying Vision Transformer (Swin-T) vs. CNN (ResNet-50) comparison for commercial fish Re-ID on the AutoFish dataset comprising six morphologically similar species; (2) validating hard triplet mining for metric learning in commercial contexts; (3) providing failure mode analysis revealing intra-species errors dominate over inter-species confusion; and (4) demonstrating practical feasibility for Electronic Monitoring systems on conveyor belts.

---

## References (IEEE Style)

[1] J. B. Haurum, A. Karpova, M. Pedersen, S. H. Bengtson, and T. B. Moeslund, "Re-identification of zebrafish using metric learning," in Proc. IEEE Winter Conf. Appl. Comput. Vis. Workshops (WACVW), 2020, pp. 1–11.

[2] B. M. Mathisen, K. Bach, E. Meidell, H. Måløy, and E. S. Sjøblom, "FishNet: A unified embedding for salmon recognition," in Proc. Eur. Conf. Artif. Intell. (ECAI), 2020, pp. 2289–2296.

[3] G. Figueroa-Mata, C. E. Gómez-Mena, E. García-Rivas, and B. Lyn, "Using a convolutional Siamese network for image-based fish identification," in Proc. 2020 IEEE Int. Conf. Syst., Man, Cybern. (SMC), 2020, pp. 1–8.

[4] M. Pedersen, M. Nyegaard, and T. B. Moeslund, "Finding Nemo's giant cousin: Keypoint matching for robust re-identification of giant sunfish," J. Mar. Sci. Eng., vol. 11, no. 5, p. 889, May 2023.

[5] O. Moskvyak, F. Maire, F. Dayoub, and M. Baktashmotlagh, "Learning landmark guided embeddings for animal re-identification," in Proc. IEEE Winter Conf. Appl. Comput. Vis. Workshops (WACVW), 2020, pp. 12–19.

[6] Ø. L. Olsen, T. Heesch, J. Eikevik, and J. G. Gundersen, "A contrastive learning approach for individual re-identification in a wild fish population," in Proc. Northern Lights Deep Learn. Conf. (NLDL), 2023, pp. 1–12.

[7] J. Puchalla, A. Serianni, and B. Deng, "Zebrafish identification with deep CNN and ViT architectures using a rolling training window," Sci. Rep., vol. 15, 2025.

[8] S. Fan, C. Song, H. Feng, and Z. Yu, "Take good care of your fish: Fish re-identification with synchronized multi-view camera system," Front. Mar. Sci., vol. 11, p. 1429459, 2024.

[9] Z. Jiang, Y. Wang, H. Jiang, and Y. Yang, "Individual fish recognition method with coarse and fine-grained feature learning," J. Aquac. Eng., vol. 102, p. 102184, 2023.

