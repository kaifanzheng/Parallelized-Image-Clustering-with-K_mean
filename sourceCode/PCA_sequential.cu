#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>
#include <sys/time.h>

void calculateColumnMeans(int rows, int cols, float *points, float *mean) {
    for (int j = 0; j < cols; j++) {
        mean[j] = 0; // Initialize mean for each column
        for (int i = 0; i < rows; i++) {
            mean[j] += points[i * cols + j];
        }
        mean[j] /= rows;
    }
}

void subtractMeans(int rows, int cols, float *points, float *mean) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            points[i * cols + j]  = points[i * cols + j] - mean[j];
        }
    }
}

void calculateCovarianceMatrix(int rows, int cols, float *points, float *covarianceMatrix) {
    float mean[cols];
    calculateColumnMeans(rows,cols,points,mean);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            float cov = 0;
            for (int k = 0; k < rows; k++) {
                cov += (points[k * cols + i] - mean[i]) * (points[k * cols + j] - mean[j]);
            }
            covarianceMatrix[i * cols + j] = cov / (rows - 1);
        }
    }
}

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
  //std::cout << "cov:\n" << cov << std::endl;
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

void transposeMatrix(int rows, int cols, float *matrix, float *transposed) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
}

void matrixMultiply(int rowsA, int colsA, int rowsB, int colsB, float *a, float *b, float *result) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += a[i * colsA + k] * b[k * colsB + j];
            }
            result[i * colsB + j] = sum;
        }
    }
}

void PCA(int rows,int cols,float *points, float *result){
  float mean[cols];
  calculateColumnMeans(rows,cols,points,mean);
  subtractMeans(rows,cols,points,mean);

  //log after noramlize the points
  //printf("mean: \n");
  //for (int i = 0; i < rows; i++) {
  //    for (int j = 0; j < cols; j++) {
  //      printf("%f, ",points[i*cols+j]);
  //    }
  //    printf("\n");
  //}

  float covarianceMatrix[cols * cols];
  calculateCovarianceMatrix(rows, cols, points, covarianceMatrix);

  //log after find covariant martrix
  //printf("cov: \n");
  //for (int i = 0; i < cols; i++) {
  //  for (int j = 0; j < cols; j++) {
  //      printf("%f ", covarianceMatrix[i * cols + j]);
  //  }
  //  printf("\n");
  //}

  //the rest are not parallellble parts due to complexity of the prorgam
  float eigenResult[cols*2];
  findAndSortAndReduceDimensionEigenVector(covarianceMatrix, cols,eigenResult);

  //log after reduce eigen demonsion
  //printf("reduced eigen vector: \n");
  //for (int i = 0; i < cols; i++) {
  //  for (int j = 0; j < 2; j++) {
  //      printf("%f ", eigenResult[i * 2 + j]);
   // }
   // printf("\n");
  //}
  float eigenResult_T[2*cols];
  transposeMatrix(cols,2,eigenResult,eigenResult_T);
  float points_T[cols*rows];
  transposeMatrix(rows,cols,points,points_T);

  //log transposed points
  //printf("transposed points: \n");
  //for (int i = 0; i < cols; i++) {
   // for (int j = 0; j < rows; j++) {
   //     printf("%f ", points_T[i * rows + j]);
   // }
   // printf("\n");
  //}

  float V[2*rows];
  matrixMultiply(2,cols,cols,rows,eigenResult_T,points_T,V);

  //log after dot product
  //printf("V: \n");
  //for(int i = 0;i<2;i++){
   // for(int j=0;j<rows;j++){
  //    printf("%f ", V[i * rows + j]);
   // }
   // printf("\n");
  //}
  //transposeMatrix(2,rows,V,result);

}

void miniTest1(){
  int rows = 5;
  int cols = 3;
  float points[15] = {1, 2, 3, 4, 5, 6,7, 8, 9,10, 11, 12,13,14,15};
  float pcaResult[rows*2];
  PCA(rows,cols,points,pcaResult);

  printf("result of PCA: \n");
  for(int i = 0;i<rows;i++){
    for(int j=0;j<2;j++){
      printf("%f ", pcaResult[i * 2 + j]);
    }
    printf("\n");
  }
}

void timingAnalysis(int dataSize){
  int rows = dataSize;
  int cols = 3;
  srand(3);
  float *points = (float *) malloc(dataSize*cols*sizeof(float));
  for(int i=0;i<dataSize*cols ; i++){
    points[i] = (float) (rand() %256);
  }
  float *pcaResult = (float *) malloc(rows*2*sizeof(float));
  struct timeval start_time;
  struct timeval end_time;
  gettimeofday(&start_time, 0);
  PCA(rows,cols,points,pcaResult);
  gettimeofday(&end_time, 0);
  double time = (1000000.0*(end_time.tv_sec-start_time.tv_sec) + end_time.tv_usec-start_time.tv_usec)/1000.0;
  printf("CPU Time:  %f ms \n", time);

  free(pcaResult);
}
int main() {
  timingAnalysis(32*32);
}
