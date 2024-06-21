#include <iostream>
#include <vector>       // std::vector
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);


/* Host version */
void _mul_host(size_t N, int *x, int *y, std::vector<int> &mem_acc_pat, int *output) {
  for (size_t i = 0; i < N; i++) {
    output[mem_acc_pat[i]] = x[mem_acc_pat[i]] * y[mem_acc_pat[i]];
  }
}

void init (size_t N, int *x_host, int *y_host, 
          std::vector<int> &good_mem_acc_patt) {

  // Host memory init 
  for (size_t i = 0; i < N; i++) {
    x_host[i] = 10;
    y_host[i] = i;
  }

  // Access pattern init 
  // good pattern 
  for (size_t i = 0; i < good_mem_acc_patt.size(); i++) {
    good_mem_acc_patt[i] = i;
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
  good_mem_acc_patt.resize(N);

  // Memory access pattern init
  init(N, x_host, y_host, good_mem_acc_patt);

	// time calculation
  high_resolution_clock::time_point start, end;
  duration<double, std::milli> d1; // duration

  // computation 
	start_t(start)
    for (size_t rd = 0; rd < ROUND; rd++) {
      _mul_host(N, x_host, y_host, good_mem_acc_patt, output_host); 
    }
	end_t(end)	
	duration_t(d1, start, end)

  // print time 
  printf("%.2lf (ms)\n", d1.count()); // unit: ms, d1


  // Free memory
  free(x_host);
  free(y_host);
  free(output_host);

  return 0;
}
