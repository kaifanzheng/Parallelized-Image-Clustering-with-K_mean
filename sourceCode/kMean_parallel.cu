
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand_kernel.h>
#include "gputimer.h"

#define NUM_BLOCK 1
#define NUM_THREAD 1024
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


__global__ void generate_centers(float *centers, int num_centers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(clock64(), idx, 0, &state);  // Initialize CURAND for each thread

    for (int i = idx; i < num_centers; i += stride) {
        centers[i * 2] = curand(&state) % 256;
        centers[i * 2 + 1] = curand(&state) % 256;
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
        float mindistance = INFINITY;
        int minLabel = point_label[i].label;

        for (int j = 0; j < centers_len; j++) {
            float dis = distance(point_label[i].x, point_label[i].y, centers[j * 2], centers[j * 2 + 1]);
            if (dis < mindistance) {
                mindistance = dis;
                minLabel = j;
            }
        }

        point_label[i].label = minLabel;
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
void k_mean_2d(float *h_points, float *h_centers, struct Point_Label *h_point_label,int points_len, int centers_len, int max_iter){
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

    generate_centers <<<NUM_BLOCK,NUM_THREAD>>>(d_centers,centers_len);
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
        //printf("interation :%d\n ",i);
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
        //printf("debug log center after update: \n");
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

void mini_testset_one(){
  int k =4;
  int n = 16;
  float test_data[] = {10,20,35,25,40,30,25,25,
                         10,200,20,210,25,220,30,225,
                         200,200,220,220,220,210,240,250,
                         200,10,210,20,220,25,225,30};
  float centers[k*2];
  struct Point_Label point_label[n];
  k_mean_2d(test_data,centers,point_label,n,k,10);

  //check for result:
  printf("final result---------\n");
  printf("centers after iteration: \n");
  for(int i=0;i<k;i++){
    printf("x: %f, y: %f\n",centers[i*2],centers[i*2+1]);
  }
  printf("point-label after iteration \n");
  for(int i=0;i<n;i++){
    printf("x: %f, y: %f, lable: %d\n",point_label[i].x,point_label[i].y,point_label[i].label);
  }
}

void mini_testset_three(){
  int k =5;
  int n = 20;
  float test_data[] = {10,20,35,25,40,30,25,25,
                         10,200,20,210,25,220,30,225,
                         200,200,220,220,220,210,240,250,
                         200,10,210,20,220,25,225,30,
                         100,100,101,101,101,100,100,101
                         };
  float centers[k*2];
  struct Point_Label point_label[n];
  k_mean_2d(test_data,centers,point_label,n,k,10);

  //check for result:
  printf("final result---------\n");
  printf("centers after iteration: \n");
  for(int i=0;i<k;i++){
    printf("x: %f, y: %f\n",centers[i*2],centers[i*2+1]);
  }
  printf("point-label after iteration \n");
  for(int i=0;i<n;i++){
    printf("x: %f, y: %f, lable: %d\n",point_label[i].x,point_label[i].y,point_label[i].label);
  }
}

void mini_testset_two(){
  srand(time(NULL));
  int k =10;
  int n = 1080*256;
  float test_data[n*2];
  for(int i=0;i<n*2;i++){
    test_data[i] = rand() % 256;
  }
  float centers[k*2];
  struct Point_Label point_label[n];
  clock_t start_time, end_time;
  double cpu_time_used;
  start_time = clock();
  k_mean_2d(test_data,centers,point_label,n,k,1000);
  end_time = clock();
  cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
  printf("CPU time used: %f seconds\n", cpu_time_used);

  //check for result:
  printf("final result---------\n");
  printf("centers after iteration: \n");
  for(int i=0;i<k;i++){
    printf("x: %f, y: %f\n",centers[i*2],centers[i*2+1]);
  }
  //printf("point-label after iteration \n");
  //for(int i=0;i<n;i++){
  //  printf("x: %f, y: %f, lable: %d\n",point_label[i].x,point_label[i].y,point_label[i].label);
  //}
}

void timingAnalysis(int dataSize){
  int k =10;
  int n = dataSize;
  srand(3);
  float *test_data = (float *)malloc(dataSize*2*sizeof(float));
  float centers[k*2];
  struct Point_Label *point_label = (struct Point_Label *)malloc(n*sizeof(struct Point_Label));
  for(int i=0;i<dataSize*2 ; i++){
    test_data[i] = (float) (rand() %256);
  }
  struct GpuTimer timer;
  timer.Start();
  k_mean_2d(test_data,centers,point_label,n,k,1000);
  timer.Stop();
  printf("Time: %f \n",timer.Elapsed());

  free(test_data);
  free(point_label);
}

int main() {
    timingAnalysis(256*256);
}
