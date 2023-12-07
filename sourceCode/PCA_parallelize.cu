#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include "lodepng.h"
#include "gputimer.h"

#define NUM_BLOCK 1
#define NUM_THREAD 1024

//cuda kernals-----------
__global__ void calculateColumnMeans(int rows, int cols, float *points, float *mean) {
    int startCol = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = startCol; j < cols; j += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = 0; i < rows; i++) {
            sum += points[i * cols + j];
        }
        mean[j] = sum / rows;
    }
}

__global__ void subtractMeans(int rows, int cols, float *points, float *mean) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = startIdx; idx < rows * cols; idx += blockDim.x * gridDim.x) {
        int i = idx / cols;
        int j = idx % cols;
        points[i * cols + j] -= mean[j];
    }
}

__global__ void calculateCovarianceMatrix(int rows, int cols, float *points, float *mean, float *covarianceMatrix) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = startIdx; index < cols * cols; index += stride) {
        int i = index / cols;
        int j = index % cols;

        if (i < cols && j < cols) {
            float cov = 0;
            for (int k = 0; k < rows; k++) {
                cov += (points[k * cols + i] - mean[i]) * (points[k * cols + j] - mean[j]);
            }
            covarianceMatrix[i * cols + j] = cov / (rows - 1);
        }
    }
}

__global__ void transposeMatrix(int rows, int cols, float *matrix, float *transposed) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = startIdx; i < rows * cols; i += stride) {
        int row = i / cols;
        int col = i % cols;
        transposed[col * rows + row] = matrix[row * cols + col];
    }
}

__global__ void matrixMultiply(int rowsA, int colsA, int rowsB, int colsB, float *a, float *b, float *result) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = startIdx; index < rowsA * colsB; index += stride) {
        int row = index / colsB;
        int col = index % colsB;

        if (row < rowsA && col < colsB) {
            float sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += a[row * colsA + k] * b[k * colsB + col];
            }
            result[row * colsB + col] = sum;
        }
    }
}

//eigen function-------
std::pair<Eigen::VectorXd, Eigen::MatrixXd> eigenDecomposition(const Eigen::MatrixXd &covariant) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariant);
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed!");
    }
    // Return a pair containing eigenvalues and eigenvectors
    return {eigensolver.eigenvalues(), eigensolver.eigenvectors()};
}

void findAndSortAndReduceDimensionEigenVector(float *covarianceMatrix, int cols, float *eigenResult){
  //the rest are not parallellble parts due to complexity of the prorgam
  Eigen::MatrixXd cov(cols, cols);
  int index = 0;
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < 3; ++j) {
      cov(i, j) = covarianceMatrix[index++];
    }
  }
  std::cout << "cov:\n" << cov << std::endl;
  try {
    // Perform eigenvalue decomposition
    auto [eigenvalues, eigenvectors] = eigenDecomposition(cov);

    // Print eigenvalues and eigenvectors
    //std::cout << "Eigenvalues:\n" << eigenvalues << std::endl;
    //std::cout << "Eigenvectors:\n" << eigenvectors << std::endl;

    //sort eigen vector by eigen value
    // Pair each eigenvalue with its corresponding eigenvector
    std::vector<std::pair<double, Eigen::VectorXd>> eigenPairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigenPairs.emplace_back(eigenvalues[i], eigenvectors.col(i));
    }

    // Sort pairs in descending order of eigenvalues
    std::sort(eigenPairs.begin(), eigenPairs.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    // Separate sorted eigenvalues and eigenvectors
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues[i] = eigenPairs[i].first;
        eigenvectors.col(i) = eigenPairs[i].second;
    }

    // Print sorted eigenvalues and eigenvectors
    //std::cout << "Sorted Eigenvalues:\n" << eigenvalues << std::endl;
    //std::cout << "Sorted Eigenvectors:\n" << eigenvectors << std::endl;

    //reduce to 2 dimension
    int counter = 0;
    for(int i=0;i<cols;i++){
      eigenResult[counter] = eigenvectors.row(i)[0];
      eigenResult[counter+1] = eigenvectors.row(i)[1];
      counter = counter + 2;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return;
  }
}
void floatArrayLogger(int rows, int cols, float *array,char *message){
    printf("%s\n",message);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        printf("%f, ",array[i*cols+j]);
      }
      printf("\n");
    }
}

void cudaErrorLogger(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void PCA(int rows,int cols,float *h_points, float *h_result){
    float h_mean[cols];
    float *d_mean;
    float *d_points;
    cudaMalloc((void**)&d_mean, cols*sizeof(float));
    cudaMalloc((void**)&d_points, cols*rows*sizeof(float));
    cudaMemcpy(d_points, h_points, cols*rows*sizeof(float), cudaMemcpyHostToDevice);
    calculateColumnMeans <<<NUM_BLOCK, NUM_THREAD>>>(rows,cols,d_points,d_mean);
    cudaDeviceSynchronize();
    cudaErrorLogger();
    printf("found mean");
    subtractMeans <<<NUM_BLOCK, NUM_THREAD>>>(rows,cols,d_points,d_mean);
    cudaDeviceSynchronize();
    cudaErrorLogger();
    //cudaMemcpy(h_points,d_points,rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
    //floatArrayLogger(rows,cols,h_points,"points after sub mean:");

    calculateColumnMeans <<<NUM_BLOCK, NUM_THREAD>>>(rows,cols,d_points,d_mean);
    cudaDeviceSynchronize();
    cudaErrorLogger();

    float h_covarianceMatrix[cols * cols];
    float *d_covarianceMatrix;
    cudaMalloc((void**)&d_covarianceMatrix, cols*cols*sizeof(float));
    calculateCovarianceMatrix <<<NUM_BLOCK, NUM_THREAD>>>(rows, cols, d_points,d_mean, d_covarianceMatrix);
    cudaDeviceSynchronize();
    cudaErrorLogger();
    cudaMemcpy(h_covarianceMatrix,d_covarianceMatrix,cols*cols*sizeof(float), cudaMemcpyDeviceToHost);
    //floatArrayLogger(cols,cols,h_covarianceMatrix,"covariant:");

    float h_eigenResult[cols*2];
    findAndSortAndReduceDimensionEigenVector(h_covarianceMatrix, cols,h_eigenResult);
    //floatArrayLogger(cols,2,h_eigenResult,"eigen vector:");

    float *d_eigenResult;
    float *d_eigenResult_T;
    float *d_points_T;
    float *d_V;
    cudaMalloc((void**)&d_eigenResult, cols*2*sizeof(float));
    cudaMalloc((void**)&d_eigenResult_T, 2*cols*sizeof(float));
    cudaMalloc((void**)&d_points_T, cols*rows*sizeof(float));
    cudaMalloc((void**)&d_V, 2*rows*sizeof(float));
    cudaMemcpy(d_eigenResult, h_eigenResult, cols*2*sizeof(float), cudaMemcpyHostToDevice);
    transposeMatrix<<<NUM_BLOCK, NUM_THREAD>>>(cols,2,d_eigenResult,d_eigenResult_T);
    cudaDeviceSynchronize();
    cudaErrorLogger();
    transposeMatrix<<<NUM_BLOCK, NUM_THREAD>>>(rows,cols,d_points,d_points_T);
    cudaDeviceSynchronize();
    cudaErrorLogger();

    float *d_result;
    cudaMalloc((void**)&d_result, rows*2*sizeof(float));
    matrixMultiply <<<NUM_BLOCK, NUM_THREAD>>>(2,cols,cols,rows,d_eigenResult_T,d_points_T,d_V);
    cudaDeviceSynchronize();
    cudaErrorLogger();
    transposeMatrix <<<NUM_BLOCK, NUM_THREAD>>>(2,rows,d_V,d_result);
    cudaDeviceSynchronize();
    cudaErrorLogger();
    cudaMemcpy(h_result,d_result,rows*2*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_mean);
    cudaFree(d_points);
    cudaFree(d_covarianceMatrix);
    cudaFree(d_eigenResult);
    cudaFree(d_eigenResult_T);
    cudaFree(d_points_T);
    cudaFree(d_V);
    cudaFree(d_result);
}

void miniTest1(){
  int rows = 5;
  int cols = 3;
  float points[15] = {1, 2, 3, 4, 5, 6,7, 8, 9,10, 11, 12,13,14,15};
  float pcaResult[rows*2];
  PCA(rows,cols,points,pcaResult);
  floatArrayLogger(rows,2,pcaResult,"result: ");
}
void miniTest2(){
  unsigned width;
  unsigned height;
  unsigned char* h_image;
  unsigned h_error = lodepng_decode24_file(&h_image, &width, &height, "Test1.png");
  if(h_error) printf("error %u: %s\n", h_error, lodepng_error_text(h_error));
  float points[width*height*3];
  for(int i=0; i<width*height*3; i++){
    points[i] = (float)h_image[i];
  }
  float pcaResult[width*height*2];
  PCA(width*height,3,points, pcaResult);
  floatArrayLogger(1000,2,pcaResult,"pcaResult:");
}

void timingAnalysis(int dataSize){
  int rows = dataSize;
  int cols = 3;
  srand(3);
  float *points = (float *)malloc(sizeof(float)*dataSize*cols);
  for(int i=0;i<dataSize*cols ; i++){
    points[i] = (float) (rand() %256);
  }
  float *pcaResult = (float *) malloc(rows*2*sizeof(float));

  struct GpuTimer timer;
  timer.Start();
  PCA(rows,cols,points,pcaResult);
  timer.Stop();
  printf("Time : %f \n",timer.Elapsed());

  free(pcaResult);
}


int main() {
  timingAnalysis(512*512);
}
