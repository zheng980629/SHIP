# Rubik's Cube: High-Order Channel Interactions with a Hierarchical Receptive Field (NeurIPS2023)

Naishan Zheng, Man Zhou, Chong Zhou, Chen Change Loy

S-Lab, Nanyang Technological University   

---
>Image restoration techniques, spanning from the convolution to the transformer paradigm, have demonstrated robust spatial representation capabilities to deliver high-quality performance. Yet, many of these methods, such as convolution and the Feed Forward Network (FFN) structure of transformers, primarily leverage the basic first-order channel interactions and have not maximized the potential benefits of higher-order modeling. To address this limitation, our research dives into understanding relationships within the channel dimension and introduces a simple yet efficient, high-order channel-wise operator tailored for image restoration. Instead of merely mimicking high-order spatial interaction, our approach offers several added benefits: Efficiency: It adheres to the zero-FLOP and zero-parameter
principle, using a spatial-shifting mechanism across channel-wise groups. Simplicity: It turns the favorable channel interaction and aggregation capabilities into element-wise multiplications and convolution units with 1 × 1 kernel. Our new formulation expands the first-order channel-wise interactions seen in previous works to arbitrary high orders, generating a hierarchical receptive field akin to a Rubik’s cube through the combined action of shifting and interactions. Furthermore, our proposed Rubik’s cube convolution is a flexible operator that can be incorporated into existing image restoration networks, serving as a drop-in replacement for the standard convolution unit with fewer parameters overhead. We conducted experiments across various low-level vision tasks, including image denoising, low-light image enhancement, guided image super-resolution, and image de-blurring. The results consistently demonstrate that our Rubik’s cube operator enhances performance across all tasks.>
---


## Applications
### 🚀: Low-Light Image Enhancement


### 🚀: Image Deblur


### 🚀: Image Denoising


### 🚀: Classification
