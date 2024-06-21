#include <iostream>
#include <math.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>
#include <assert.h>
#include <cuda_runtime.h>


#define MEM_ACC_PAT_N 1024
#define MEM_ACC_PAT_CHECK_CORRECTNESS // check for correctness


/* Device version */
// CUDA kernel to mul elements in x and arrays
__global__ void _mul(size_t N, int *x, int *y, int* mem_acc_pat, int *output) {
  // Get the thread idx
  int thd_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  __syncthreads();

  if (thd_idx < N) {
    output[mem_acc_pat[thd_idx]] = x[mem_acc_pat[thd_idx]] * y[mem_acc_pat[thd_idx]];
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

int get_GPU_Prop();

int main(int argc, char* argv[]){
  // int aaa = get_GPU_Prop();
  // printf("aaa = %d\n", aaa);
  
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
  
  size_t blocks = (N+1024-1)/1024;
  size_t threads = 1024; 
  // printf("blocks = %lu, threads = %lu\n", blocks, threads);

  // pre run
  _mul <<< blocks, threads >>> (N, x_devc, y_devc, good_mem_acc_patt_devc, output_devc);

  // starting test ... 
  // Good pattern 
  cudaEventRecord(start);  
  for (size_t rd = 0; rd < ROUND; rd++) {
    _mul <<< blocks, threads >>> (N, x_devc, y_devc, good_mem_acc_patt_devc, output_devc);
  }
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
  for (size_t rd = 0; rd < ROUND; rd++) {
    _mul <<< blocks, threads >>> (N, x_devc, y_devc, bad_mem_acc_patt_devc, output_devc);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms_bad, start, stop);  
  
  printf("ms_good = %.3lf (ms), ms_bad = %.3lf (ms)\n", ms_good, ms_bad);

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


int get_GPU_Prop() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("deviceProp.clockRate = %d\n", deviceProp.clockRate);
  printf("deviceProp.totalGlobalMem = %zu\n", deviceProp.totalGlobalMem);
  printf("deviceProp.warpSize = %d\n", deviceProp.warpSize);
  printf("deviceProp.totalConstMem = %zu\n", deviceProp.totalConstMem);
  printf("deviceProp.canMapHostMemory = %d\n", deviceProp.canMapHostMemory);
  printf("deviceProp.minor = %d\n", deviceProp.minor); // Minor compute capability, e.g. cuda 9.0

  // about shared memory 
  printf("\n");
  printf("deviceProp.sharedMemPerBlockOptin = %zu bytes\n", deviceProp.sharedMemPerBlockOptin);
  printf("deviceProp.sharedMemPerBlock = %zu bytes\n", deviceProp.sharedMemPerBlock);
  printf("deviceProp.sharedMemPerMultiprocessor = %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
  
  // about SM and block
  printf("\n");
  printf("deviceProp.multiProcessorCount = %d\n", deviceProp.multiProcessorCount);
  printf("deviceProp.maxBlocksPerMultiProcessor = %d\n", deviceProp.maxBlocksPerMultiProcessor);
  printf("deviceProp.maxThreadsPerBlock = %d\n", deviceProp.maxThreadsPerBlock);
  printf("deviceProp.maxThreadsPerMultiProcessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
  
  // about registers  
  printf("\n");  
  printf("deviceProp.regsPerBlock = %d\n", deviceProp.regsPerBlock);
  printf("deviceProp.regsPerMultiprocessor = %d\n", deviceProp.regsPerMultiprocessor);

  // something
  printf("\n");
  printf("deviceProp.maxGridSize[0] = %d, deviceProp.maxGridSize[1] = %d, deviceProp.maxGridSize[2] = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  return deviceProp.clockRate;
}

