# Robust BKSVD for Blind Color Deconvolution and Blood Detection
A Robust BKSVD Method for Blind Color Deconvolution and Blood Detection on H&E Histological Images

Fernando Pérez-Bueno, Kjersti Engan, and Rafael Molina.
[Accepted for publication in 21st International Conference of Artificial Intelligence in Medicine (AIME) Portoroz, Slovenia June 12-15, 2023 ]

## Abstract
Hematoxylin and Eosin (H\&E) color variation between histological images from different laboratories degrades the performance of Computer-Aided Diagnosis systems. Histology-specific models to solve color variation are designed taking into account the staining procedure, where most color variations are introduced. In particular, Blind Color Deconvolution (BCD) methods aim to identify the real underlying colors in the image and to separate the tissue structure from the color information. A commonly used assumption is that images are stained with and only with the pure staining colors (e.g., blue and pink for H\&E). However, this assumption does not hold true in the presence of common artifacts such as blood, where the blood cells need a third color component to be represented. Blood usually hampers the ability of color standardization algorithms to correctly identify the stains in the image, producing unexpected outputs. In this work, we propose a robust Bayesian K-Singular Value Decomposition (BKSVD) model to simultaneously detect blood and separate color from structure in histological images. Our method was tested on synthetic and real images containing different amounts of blood pixels.
Keywords: Bayesian modelling; Histological images; Blind Color Deconvolution; Blood Detection

## Citation
@inproceedings{PerezBueno2023,
  title={A Robust BKSVD Method for Blind Color Deconvolution and Blood Detection on H&E Histological Images},
  author={Pérez-Bueno, Fernando and Engan, Kjersti and Molina, Rafael},
  year={2023},
  organization={21st International Conference of Artificial Intelligence in Medicine (AIME)}
}

## Disclaimer 

- Check Example_Robust_BKSVD.m for an use case of this method.

- Check our [BKSVD repository](https://github.com/vipgugr/BKSVD) for the original paper "Bayesian K-SVD for H and E blind color deconvolution. Applications to stain normalization, data augmentation and cancer classification"

- Notice that this version detects blood and it is more robust to estimate the H&E colors, however, it is not optimized for stain normalization and might produce noisy results.

## Data

- The synthetic Blood Dataset is generated from the data in:
 N. Alsubaie et al. Stain deconvolution using statistical analysis of multi-resolution stain colour representation. PLOS ONE, 12:e0169875, 2017

- The TCGA dataset is collected from images in the [The Cancer Genome Atlas](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)
  - TCGA-A7-A5ZV-DX1
  - TCGA-AC-A2QH-DX1
   - TCGA-AQ-A54N-DX1
  - TCGA-E2-A14X-DX1
   - TCGA-GM-A3XL-DX1
   - TCGA-OL-A97C-DX1
   - TCGA-S3-AA10-DX1
