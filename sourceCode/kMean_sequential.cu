#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

struct Point_Label{
    float x;
    float y;
    int index;
    int label;
};

//find distance between two points
float distance(float x, float y, float centered_x, float centered_y) {
        return (float)sqrt((centered_x - x) * (centered_x - x) + (centered_y - y) * (centered_y - y));
}
//compare if the center are matching each other
int array_compare(float *arrayOne, float *arrayTwo, int arraySize, float threshold){
    int i;
    for (i = 0; i < arraySize; ++i) {
        if ((int)arrayOne[i] != (int)arrayTwo[i]) {
            return 0; // Arrays are different
        }
    }
    return 1;
}
//for the 0 iteration auto generate centers
void generate_centers(float *centers, float num_centers){
    srand(time(NULL));
    for(int i=0;i<num_centers;i++){
      centers[i*2] = abs(rand() % 256);
      centers[i*2+1] = abs(rand() % 256);
    }
}
//convert the 2 dimensional image data into structure point_label
void inite_point_label(float *points, struct Point_Label *point_label, int points_len){
	for(int i=0;i<points_len; i++){
		point_label[i].x = points[i*2];
		point_label[i].y = points[i*2+1];
    if(points[i*2] <0 ||points[i*2+1]<0 ){
      printf("fdasfd point nefative!!!----\n");
    }
		point_label[i].index = i;
		point_label[i].label = 0;
	}
}
//update the labels of each point during the iteration
void update_labels(struct Point_Label *point_label, float *centers,int points_len, int centers_len){
	for(int i = 0;i<points_len;i++){
		float mindistance = INFINITY;
		int minLabel = point_label[i].label;
		for(int j = 0;j<centers_len;j++){
			float dis = distance(point_label[i].x, point_label[i].y, centers[j*2], centers[j*2+1]);
			if(dis<mindistance){
				mindistance = dis;
				minLabel = j;
			}
		}
		point_label[i].label = minLabel;
	}
}
//update the centers
void update_centers(struct Point_Label *point_label_array,float *centers, int point_label_array_length, int numLabels){
  float sum_x[numLabels];
  float sum_y[numLabels];
  int count[numLabels];
  memset( sum_x, 0, sizeof(sum_x));
  memset( sum_y, 0, sizeof(sum_y));
  memset( count, 0, sizeof(count));

  for (int i = 0; i < point_label_array_length; ++i) {
    int label = point_label_array[i].label;
    sum_x[label] += point_label_array[i].x;
    sum_y[label] += point_label_array[i].y;
    count[label]++;
  }

  for (int i = 0; i < numLabels; ++i) {
    if (count[i] > 0) {
        centers[i * 2] = sum_x[i] / count[i];
        centers[i * 2 + 1] = sum_y[i] / count[i];
        if(centers[i*2] <0 || centers[i * 2 + 1]<0){
          printf("negative value centers!!!!!!!%f , %f----------\n",centers[i*2],centers[i*2+1]);
        }
    }
  }
}
void hardCopyCenters(float *pre_centers, float *centers, int len){
  for(int i=0;i<len*2;i++){
    pre_centers[i] = centers[i];
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
void k_mean_2d(float *points, float *centers, struct Point_Label *point_label,int points_len, int centers_len, int max_iter){
  float pre_centers[centers_len*2];
  inite_point_label(points, point_label,points_len);
  generate_centers(centers,centers_len);
  for(int i=0;i<max_iter;i++){      //keep this for loop sequential
    //printf("interation %d, \n",i);
  	hardCopyCenters(pre_centers,centers,centers_len);
  	update_labels(point_label, centers, points_len, centers_len);
  	update_centers(point_label, centers, points_len, centers_len);
    //debug log ----------------
    //printf("points: ---------\n");
    //logPoint_lable(point_label,points_len);
    //printf("centers: ---------\n");
    //logCenters(centers,centers_len);
    //printf("prev,centers: ---------\n");
    //logCenters(pre_centers,centers_len);
    if(array_compare(pre_centers, centers, centers_len*2,0) == 1){
      break;
    }
  }
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

void timingAnalysis(int dataSize){
  int k =10;
  int n = dataSize;
  srand(3);
  float *points = (float *)malloc(dataSize*3*sizeof(float));
  float centers[k*2];
  struct Point_Label *point_label = (struct Point_Label*)malloc(n*sizeof(struct Point_Label));
  for(int i=0;i<dataSize*3 ; i++){
    points[i] = (float) (rand() %256);
  }
  struct timeval start_time;
  struct timeval end_time;
  gettimeofday(&start_time, 0);
  k_mean_2d(points,centers,point_label,n,k,10);
  gettimeofday(&end_time, 0);
  double time = (1000000.0*(end_time.tv_sec-start_time.tv_sec) + end_time.tv_usec-start_time.tv_usec)/1000.0;
  printf("CPU Time:  %f ms \n", time);
  free(point_label);
  free(points);
}

int main(){
  timingAnalysis(512*512);
  return 0;
}
