#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include "lodepng.h"

#define NUM_BLOCK 32
#define NUM_THREAD 256

//start pca cuda kernals-------------------------------------------------
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

//start k mean-----------------------------------------
struct Point_Label {
    float x;
    float y;
    int index;
    int label;
};

__global__ void array_same(int* result, float *arrayOne, float *arrayTwo, int arraySize){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arraySize; i+=stride) {
        if ((int)arrayOne[i] != (int)arrayTwo[i]) {
            *result=0;
            return;
        }
    }
    *result=1;
}


__global__ void generate_centers(float *centers, int num_centers,int max_cen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(clock64(), idx, 0, &state);  // Initialize CURAND for each thread

    for (int i = idx; i < num_centers; i += stride) {
        centers[i * 2] = curand(&state) % max_cen;
        centers[i * 2 + 1] = curand(&state) % max_cen;
        //printf("generate center update: %f\n",centers[i*2]);
    }
}

__global__ void init_point_label(float *points, Point_Label *point_label, int points_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < points_len; i += stride) {
        point_label[i].x = points[i * 2];
        point_label[i].y = points[i * 2 + 1];

        if (point_label[i].x < 0 || point_label[i].y < 0) {
            printf("fdasfd point negative!!!----\n");
        }

        point_label[i].index = i;
        point_label[i].label = 0;
    }
}

__device__ float distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

__global__ void update_labels(Point_Label *point_label, float *centers, int points_len, int centers_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < points_len; i += stride) {
        double mindistance = INFINITY;
        int minLabel = point_label[i].label;

        for (int j = 0; j < centers_len; j++) {
            float dis = distance(point_label[i].x, point_label[i].y, centers[j * 2], centers[j * 2 + 1]);
            if (dis < mindistance) {
                mindistance = dis;
                minLabel = j;
            }
        }

        point_label[i].label = (float)minLabel;
    }
}

__global__ void update_sums_and_counts(Point_Label *point_label_array, float *sum_x, float *sum_y, int *count, int point_label_array_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < point_label_array_length; i += stride) {
        int label = point_label_array[i].label;
        atomicAdd(&sum_x[label], point_label_array[i].x);
        atomicAdd(&sum_y[label], point_label_array[i].y);
        atomicAdd(&count[label], 1);
    }
}

__global__ void compute_new_centers(float *centers, float *sum_x, float *sum_y, int *count, int numLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numLabels; i += stride) {
        if (count[i] > 0) {
            centers[i * 2] = sum_x[i] / count[i];
            centers[i * 2 + 1] = sum_y[i] / count[i];
        }
    }
}

__global__ void hardCopyCenters(float *pre_centers, float *centers, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < len * 2; i += stride) {
        pre_centers[i] = centers[i];
    }
}

void cudaErrorLogger(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
void logPoint_lable(struct Point_Label *point_label, int n){
  for(int i=0;i<n;i++){
    printf("x: %f, y: %f, lable: %d\n",point_label[i].x,point_label[i].y,point_label[i].label);
  }
}

void logCenters(float *centers, int k){
  for(int i=0;i<k;i++){
    printf("x: %f, y: %f\n",centers[i*2],centers[i*2+1]);
  }
}
void k_mean_2d(float *h_points, float *h_centers, struct Point_Label *h_point_label,int points_len, int centers_len, int max_iter, int max_cen){
    float *d_pre_centers;
    float *d_centers;
    float *d_points;
    struct Point_Label *d_point_label;
    cudaMalloc((void**)&d_centers, centers_len*2*sizeof(float));
    cudaMalloc((void**)&d_pre_centers, centers_len*2*sizeof(float));
    cudaMalloc((void**)&d_points, points_len*2*sizeof(float));
    cudaMalloc((void**)&d_point_label, points_len*sizeof(struct Point_Label));
    cudaMemcpy(d_points,h_points,points_len*2*sizeof(float),cudaMemcpyHostToDevice);
    init_point_label <<<NUM_BLOCK,NUM_THREAD>>>(d_points, d_point_label,points_len);
    cudaDeviceSynchronize();
    cudaErrorLogger();

    //debug inite point label:
    //printf("after inite point_label: \n");
    //cudaMemcpy(h_point_label,d_point_label,points_len*sizeof(struct Point_Label),cudaMemcpyDeviceToHost);
    //logPoint_lable(h_point_label,points_len);

    generate_centers <<<NUM_BLOCK,NUM_THREAD>>>(d_centers,centers_len,max_cen);
    cudaDeviceSynchronize();
    cudaErrorLogger();


    //debug generate center
    //printf("debug generate center: \n");
    //cudaMemcpy(h_centers,d_centers,2*centers_len*sizeof(float),cudaMemcpyDeviceToHost);
    //logCenters(h_centers,centers_len);

    float *d_sum_x;
    float *d_sum_y;
    int *d_count;
    cudaMalloc((void**)&d_sum_x, centers_len*sizeof(float));
    cudaMalloc((void**)&d_sum_y, centers_len*sizeof(float));
    cudaMalloc((void**)&d_count, centers_len*sizeof(int));

    for(int i = 0;i<max_iter;i++){
        printf("interation :%d\n ",i);
        hardCopyCenters <<<NUM_BLOCK,NUM_THREAD>>>(d_pre_centers,d_centers,centers_len);
        cudaDeviceSynchronize();
        cudaErrorLogger();
        update_labels <<<NUM_BLOCK,NUM_THREAD>>>(d_point_label, d_centers, points_len, centers_len);

        //debug after update point label:
        //printf("after update point_label: \n");
        //cudaMemcpy(h_point_label,d_point_label,points_len*sizeof(struct Point_Label),cudaMemcpyDeviceToHost);
        //logPoint_lable(h_point_label,points_len);

        cudaMemset(d_sum_x, 0, centers_len * sizeof(float));
        cudaMemset(d_sum_y, 0, centers_len * sizeof(float));
        cudaMemset(d_count, 0, centers_len * sizeof(int));

        update_sums_and_counts <<<NUM_BLOCK,NUM_THREAD>>>(d_point_label,d_sum_x,d_sum_y,d_count,points_len);
        cudaDeviceSynchronize();
        cudaErrorLogger();

        //debug final debug bugbugbug
        //printf("sums:    ------important\n");
        //float h_sum_x[centers_len];
        //float h_sum_y[centers_len];
        //int h_count[centers_len];
        //cudaMemcpy(h_sum_x,d_sum_x,centers_len*sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_sum_y,d_sum_y,centers_len*sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_count,d_count,centers_len*sizeof(int),cudaMemcpyDeviceToHost);
        //for(int i=0;i<centers_len;i++){
        //    printf("sumx: %f, sumy: %f, count: %d\n",h_sum_x[i],h_sum_y[i],h_count[i]);
        //}

        compute_new_centers <<<NUM_BLOCK,NUM_THREAD>>>(d_centers,d_sum_x,d_sum_y,d_count,centers_len);
        cudaDeviceSynchronize();
        cudaErrorLogger();

        //debug log center after update
        //printf("ebug log center after update: \n");
        //cudaMemcpy(h_centers,d_centers,2*centers_len*sizeof(float),cudaMemcpyDeviceToHost);
        //logCenters(h_centers,centers_len);

        int *d_isSame;
        int h_isSame[1];
        cudaMalloc((void**)&d_isSame, sizeof(int));
        array_same <<<NUM_BLOCK,NUM_THREAD>>>(d_isSame,d_centers,d_pre_centers,centers_len);
        cudaMemcpy(h_isSame,d_isSame,sizeof(int),cudaMemcpyDeviceToHost);
        if(h_isSame[0] == 1){
            break;
        }
    }
    cudaMemcpy(h_point_label,d_point_label,points_len*sizeof(struct Point_Label),cudaMemcpyDeviceToHost);

    cudaFree(d_centers);
    cudaFree(d_pre_centers);
    cudaFree(d_points);
    cudaFree(d_point_label);
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_count);
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
    //printf("found mean");
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

void convertUnsignedCharArrayToFloatArray(unsigned char* ucharArray, float* floatArray, int numFloats) {
    // Assuming each float occupies 4 bytes (sizeof(float) = 4)
    for (int i = 0; i < numFloats; ++i) {
        // Accessing each set of 4 bytes from the unsigned char array
        unsigned char bytes[4];
        for (int j = 0; j < 4; ++j) {
            bytes[j] = ucharArray[i * 4 + j];
        }
        // Casting the bytes to a float
        floatArray[i] = *((float*)bytes);
    }
}

void ImageClustering(char *imagePath,int k,char *image_output_path){
    unsigned char labelColors[10*3] = {255,0,0,
                                        0,255,0,
                                        0,0,255,
                                        255,255,0,
                                        255,0,255,
                                        0,255,255,
                                        128,0,0,
                                        0,128,0,
                                        0,0,128,
                                        128,0,128};
    unsigned width;
    unsigned height;
    unsigned char* h_image;
    printf("start load img: \n");
    unsigned h_error = lodepng_decode24_file(&h_image, &width, &height, imagePath);
    printf("finish load img: \n");
    if(h_error) printf("error %u: %s\n", h_error, lodepng_error_text(h_error));
    float *points = (float *)malloc(width*height*3 * sizeof(float));
    for(int i=0; i<width*height*3; i++){
        points[i] = h_image[i];
    }

    printf("finish covert to float: \n");
    float *pcaResult=(float *)malloc(width*height*2 * sizeof(float));
    printf("start pca: \n");
    PCA(width*height,3,points, pcaResult);
    printf("finish pca: \n");
    //find max value and minimum value of pca result
    double min = INFINITY;
    double max = -INFINITY;
    for(int i=0;i<width*height*2;i++){
        if(pcaResult[i] < min){
            min = pcaResult[i];
        }
        if(pcaResult[i] > min){
            max = pcaResult[i];
        }
    }
    //bring the points to positive cordinate
    for(int i=0;i<width*height*2;i++){
        pcaResult[i] = pcaResult[i] + abs(min);
    }
    max = max + abs(min);

    float centers[k*2];
    struct Point_Label *point_label = (struct Point_Label *)malloc(width*height * sizeof(struct Point_Label));
    k_mean_2d(pcaResult,centers,point_label,width*height,k,100,max);

    unsigned char *resultImage = (unsigned char *)malloc(width*height*3*sizeof(unsigned char));
    for(int i = 0;i<width*height;i++){
        resultImage[i*3] = labelColors[point_label[i].label*3];
        resultImage[i*3+1] = labelColors[point_label[i].label*3+1];
        resultImage[i*3+2] = labelColors[point_label[i].label*3+2];
    }
    lodepng_encode24_file(image_output_path, resultImage, width, height);
    free(points);
    free(pcaResult);
    free(point_label);
    free(resultImage);
}

int main(int argc, char *argv[]){
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.png> <output.png> <k>\n", argv[0]);
        return 1;
    }

    char *inputFileName = argv[1];
    char *outputFileName = argv[2];
    ImageClustering(inputFileName,atoi(argv[3]),outputFileName);
    return 0;
}
