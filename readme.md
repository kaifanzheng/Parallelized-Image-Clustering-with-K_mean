# ClusterVision: CUDA Multi-threaded Image Analysis
**Kaifan Zheng & Noshin Chowdhury & Hongfei Liu & Weiheng Xiao**
## Introduction
ClusterVision presents an innovative approach to enhance image segmentation in computer vision. By integrating Principal Component Analysis (PCA) with K-Means clustering and CUDA parallelization, this project achieves significant improvements in both accuracy and processing speed.

- **Principal Component Analysis (PCA)**: Facilitates efficient data handling and feature extraction.
- **K-Means Clustering**: Ensures precise data segmentation.
- **CUDA Parallelization**: Leverages GPU computing for rapid execution.

This method addresses real-time processing challenges and large dataset management, refining image segmentation accuracy.

## Getting Started
Before starting, ensure your system has the correct CUDA version:
- Required CUDA Version: `cuda_11.8`

## Project Structure
![Project Structure](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/structure.jpg)

## Principal Component Analysis (PCA)
The PCA process involves several steps:
1. Standardize the Data
2. Compute the Covariance Matrix
3. Calculate Eigenvectors and Eigenvalues
4. Sort Eigenvectors by Eigenvalues
5. Choose the Top k Eigenvectors
6. Project the Data onto Lower-Dimensional Space

### Runtime Results
#### After Parallelization with CUDA
- **Data Size Comparison**: 
  ![PCA Runtime Comparison](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/PCA_time_compare.png)

  Analysis: As data size increases, both CPU and GPU runtimes rise, with GPU showing advantages for larger datasets.

- **Thread Number Comparison**:
  ![PCA Thread Runtime](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/PCA_THREAD_RUNTIME.png)

  Analysis: GPU runtime decreases significantly with more threads, indicating an optimal range for performance gains.

## K-Means Clustering
K-Means process steps:
1. Choose the Number of Clusters (K)
2. Initialize Cluster Centers
3. Assign Data Points to Nearest Cluster
4. Update Cluster Centers
5. Repeat Steps 3 and 4

### Runtime Results
#### After Parallelization with CUDA
- **Data Size Comparison**:
  ![K-Mean Runtime Comparison](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/K_MEAN_runtime_compare.png)

  Analysis: GPU significantly outperforms CPU in larger datasets, showing around 11 times faster computation.

- **Thread Number Comparison**:
  ![K-Mean Thread Runtime](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/k_mean_THREAD_runtime.png)

## Image Output Results
### Single Object
- **Original**:
  ![Original Single Object](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test4.png)

- **Result**:
  ![Result Single Object](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test4_out.png)

### Two Objects
- **Original**:
  ![Original Two Objects](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/test5.jpg)

- **Result**:
  ![Result Two Objects](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/test5_out.png)

### Multiple Objects
- **Original**:
  ![Original Multiple Objects](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test3.png)

- **Result**:
  ![Result Multiple Objects](https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/test6%20result.png)

## Challenges and Future Work
- K-means clustering sensitivity to initial conditions and outliers affects consistency.
- Exploring Gaussian Mixture Model (GMM) for improved adaptability and accuracy.
- Future efforts include parallelizing GMM for GPU computing and integrating data prefetching to optimize memory access.

For more detailed information, please refer to the [Report PDF File](https://github.com/kaifanzheng/Parallelized-Image-Clustering-with-K_mean/blob/main/proj%206%20Final%20report.pdf).
