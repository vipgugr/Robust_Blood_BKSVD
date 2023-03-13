# Robust BKSVD for Blind Color Deconvolution and Blood Detection
A Robust BKSVD Method for Blind Color Deconvolution and Blood Detection on H&E Histological Images

Fernando Pérez-Bueno1[0000−0002−7404−794X], Kjersti Engan2[0000−0002−8970−0067], and Rafael Molina1[0000−0003−4694−8588]
[Accepted for publication in 21st International Conference of Artificial Intelligence in Medicine (AIME) Portoroz, Slovenia June 12-15, 2023 ]

## Abstract
Hematoxylin and Eosin (H\&E) color variation between histological images from different laboratories degrades the performance of Computer-Aided Diagnosis systems. Histology-specific models to solve color variation are designed taking into account the staining procedure, where most color variations are introduced. In particular, Blind Color Deconvolution (BCD) methods aim to identify the real underlying colors in the image and to separate the tissue structure from the color information. A commonly used assumption is that images are stained with and only with the pure staining colors (e.g., blue and pink for H\&E). However, this assumption does not hold true in the presence of common artifacts such as blood, where the blood cells need a third color component to be represented. Blood usually hampers the ability of color standardization algorithms to correctly identify the stains in the image, producing unexpected outputs. In this work, we propose a robust Bayesian K-Singular Value Decomposition (BKSVD) model to simultaneously detect blood and separate color from structure in histological images. Our method was tested on synthetic and real images containing different amounts of blood pixels.
Keywords: Bayesian modelling; Histological images; Blind Color Deconvolution; Blood Detection

