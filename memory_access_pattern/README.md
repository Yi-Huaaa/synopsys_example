# Example - memory_access_pattern
* This code is used to give an example showing the importance of memory access pattern for GPU.
* Comparison run time between: 
  1. good case: all memory access is aligned and coalesced 
  1. bad case: NO aligned and NO coalesced (random_shuffle the data access order)

* Pseudo code
```
for (round in ROUND) {
  // this second for loop parallelly run on GPU by launching the `_mul` kernel 
  for (i in N) {
    output_array = x_array[i] * y_array[i];
  }
}

```

# File structure
~/synopsys/synopsys_example/memory_access_pattern
|--- cpu_memory_access.cpp
|--- memory_access.cu
|--- README.md
|--- run.sh


# Compile and Results: 
* CPU version Compile and run: `clear && g++ cpu_memory_access.cpp -o cpu_memory_access && ./cpu_memory_access 16384 10000`
  * N = 16384, ROUND = 10000
* Results: 
```
449.33 (ms)
```

* GPU version Compile and run: `clear && nvcc memory_access.cu -o memory_access && ./memory_access 16384 10000`
  * N = 16384, ROUND = 10000
* Results: 
```
ms_good = 21.170 (ms), ms_bad = 41.809 (ms)
```

---

# OR run bash file

* Compile and run: `bash run.sh` 
* Results: 
```
CPU Runtime
Running with round = 10
N = 1024
0.03 (ms)
N = 2048
0.06 (ms)
N = 4096
0.11 (ms)
N = 8192
0.22 (ms)
N = 16384
0.45 (ms)
Running with round = 100
N = 1024
0.29 (ms)
N = 2048
0.56 (ms)
N = 4096
1.12 (ms)
N = 8192
2.30 (ms)
N = 16384
4.53 (ms)
Running with round = 1000
N = 1024
2.80 (ms)
N = 2048
5.68 (ms)
N = 4096
11.32 (ms)
N = 8192
22.63 (ms)
N = 16384
44.88 (ms)
Running with round = 10000
N = 1024
28.18 (ms)
N = 2048
56.04 (ms)
N = 4096
112.21 (ms)
N = 8192
225.56 (ms)
N = 16384
451.17 (ms)

GPU runtime
Running with round = 10
N = 1024
ms_good = 0.024 (ms), ms_bad = 0.031 (ms)
N = 2048
ms_good = 0.024 (ms), ms_bad = 0.036 (ms)
N = 4096
ms_good = 0.024 (ms), ms_bad = 0.038 (ms)
N = 8192
ms_good = 0.024 (ms), ms_bad = 0.042 (ms)
N = 16384
ms_good = 0.025 (ms), ms_bad = 0.045 (ms)
Running with round = 100
N = 1024
ms_good = 0.215 (ms), ms_bad = 0.281 (ms)
N = 2048
ms_good = 0.218 (ms), ms_bad = 0.338 (ms)
N = 4096
ms_good = 0.218 (ms), ms_bad = 0.362 (ms)
N = 8192
ms_good = 0.220 (ms), ms_bad = 0.401 (ms)
N = 16384
ms_good = 0.224 (ms), ms_bad = 0.427 (ms)
Running with round = 1000
N = 1024
ms_good = 2.150 (ms), ms_bad = 2.818 (ms)
N = 2048
ms_good = 2.114 (ms), ms_bad = 3.309 (ms)
N = 4096
ms_good = 2.151 (ms), ms_bad = 3.674 (ms)
N = 8192
ms_good = 2.146 (ms), ms_bad = 3.973 (ms)
N = 16384
ms_good = 2.144 (ms), ms_bad = 4.233 (ms)
Running with round = 10000
N = 1024
ms_good = 20.827 (ms), ms_bad = 27.279 (ms)
N = 2048
ms_good = 21.357 (ms), ms_bad = 33.544 (ms)
N = 4096
ms_good = 21.360 (ms), ms_bad = 35.932 (ms)
N = 8192
ms_good = 22.117 (ms), ms_bad = 39.225 (ms)
N = 16384
ms_good = 22.349 (ms), ms_bad = 42.490 (ms)
```