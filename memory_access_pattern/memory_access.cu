#include <iostream>
#include <math.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>
#include <assert.h>
#include <cuda_runtime.h>


#define MEM_ACC_PAT_N 1024
#define MEM_ACC_PAT_CHECK_CORRECTNESS // check for correctness

// CUDA kernel to mul elements in x and arrays
__global__ void _mul(size_t N, size_t ROUND, int *x, int *y, int* mem_acc_pat, int *output) {
  // Get the thread idx
  int thd_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("inside, thd_idx = %d\n", thd_idx);
  
  __syncthreads();

  if (thd_idx < N) {
    for (size_t rd = 0; rd < ROUND; rd++) {
      // printf("thd_idx = %d, mem_acc_pat[thd_idx] = %d\n", thd_idx, mem_acc_pat[thd_idx]);
      output[mem_acc_pat[thd_idx]] = x[mem_acc_pat[thd_idx]] * y[mem_acc_pat[thd_idx]];
    }
  }
}


void init (size_t N, int *x_host, int *y_host, 
          std::vector<int> &good_mem_acc_patt, 
          std::vector<int> &bad_mem_acc_patt) {

  // Host memory init 
  for (size_t i = 0; i < N; i++) {
    x_host[i] = 10;
    y_host[i] = i;
  }

  // Access pattern init 
  // good pattern 
  for (size_t i = 0; i < good_mem_acc_patt.size(); i++) {
    good_mem_acc_patt[i] = i;
    bad_mem_acc_patt[i] = i;  
  }

  // Shuffle bad memory access pattern
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(bad_mem_acc_patt.begin(), bad_mem_acc_patt.end(), g);
}

void check_correctness(size_t N, int *output) {
  for (size_t i = 0; i < N; i++) {
    assert (output[i] == (10*i));
  }
}

int main(int argc, char* argv[]){
  size_t N = atoi(argv[1]); 
  size_t ROUND = atoi(argv[2]);
  size_t N_size = N*sizeof(int);

  // Allocate CPU memory
  int *x_host = (int*)malloc(N_size); 
  int *y_host = (int*)malloc(N_size); 
  int *output_host = (int*)malloc(N_size); 
  memset(output_host, 0, N_size);
  // Memory access pattern 
  std::vector<int> good_mem_acc_patt; 
  std::vector<int> bad_mem_acc_patt; 
  good_mem_acc_patt.resize(N);
  bad_mem_acc_patt.resize(N);


  // Allocate GPU memory
  int *x_devc;   
  int *y_devc;   
  int *output_devc; 
  int *good_mem_acc_patt_devc; 
  int *bad_mem_acc_patt_devc; 
  cudaMalloc((void**)&x_devc, N_size);
  cudaMalloc((void**)&y_devc, N_size);
  cudaMalloc((void**)&output_devc, N_size);
  cudaMalloc((void**)&good_mem_acc_patt_devc, N_size);
  cudaMalloc((void**)&bad_mem_acc_patt_devc, N_size);

  // Memory access pattern init
  init(N, x_host, y_host, good_mem_acc_patt, bad_mem_acc_patt);

  // Memory movement 
  cudaMemcpy(x_devc, x_host, N_size, cudaMemcpyHostToDevice);
  cudaMemcpy(y_devc, y_host, N_size, cudaMemcpyHostToDevice);
  cudaMemcpy(good_mem_acc_patt_devc, good_mem_acc_patt.data(), N_size, cudaMemcpyHostToDevice);
  cudaMemcpy(bad_mem_acc_patt_devc, bad_mem_acc_patt.data(), N_size, cudaMemcpyHostToDevice);
  cudaMemcpy(output_devc, output_host, N_size, cudaMemcpyHostToDevice);

  // Launch gpu kernel 
  cudaEvent_t start;
  cudaEvent_t stop;
  float ms_good = 0., ms_bad = 0.;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // pre run
  _mul <<< 1, 1024 >>> (N, ROUND, x_devc, y_devc, good_mem_acc_patt_devc, output_devc);

  // starting test ... 
  // Good pattern 
  cudaEventRecord(start);    
    _mul <<< 1, 1024 >>> (N, ROUND, x_devc, y_devc, good_mem_acc_patt_devc, output_devc);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms_good, start, stop);

#ifdef MEM_ACC_PAT_CHECK_CORRECTNESS
  // Check correctness 
  cudaMemcpy(output_host, output_devc, N_size, cudaMemcpyDeviceToHost);
  // Host and device synchronization 
  cudaDeviceSynchronize();
  // check_correctness(N, output_host);
#endif 

  // Bad pattern 
  cudaEventRecord(start);    
    _mul <<< 1, 1024 >>> (N, ROUND, x_devc, y_devc, bad_mem_acc_patt_devc, output_devc);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms_bad, start, stop);  
  
  printf("ms_good = %.3lf, ms_bad = %.3lf\n", ms_good, ms_bad);

#ifdef MEM_ACC_PAT_CHECK
  // Check correctness 
  cudaMemcpy(output_host, output_devc, N_size, cudaMemcpyDeviceToHost);
  // Host and device synchronization 
  cudaDeviceSynchronize(N, output_host);
  check_correctness();
#endif 
 
  // Free memory
  free(x_host);
  free(y_host);
  free(output_host);

  cudaFree(x_devc);
  cudaFree(y_devc);
  cudaFree(output_devc);
  cudaFree(good_mem_acc_patt_devc);
  cudaFree(bad_mem_acc_patt_devc);

  return 0;
}