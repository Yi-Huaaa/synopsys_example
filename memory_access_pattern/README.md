# Example - memory_access_pattern
* This code is used to give an example showing the importance of memory access pattern for GPU. 
* pseudo code
```
for (round in ROUND) {
  // this for loop will be parallelly run on GPU
  for (i in N) {
    output_array = x_array[i] * y_array[i];
  }
}

```

# Compile and Results: 
* Compile and run: `clear && nvcc memory_access.cu -o memory_access && ./memory_access 16384 10000`
  * N = 1024, ROUND = 1000
* Results: 
```
ms_good = 21.170 (ms), ms_bad = 41.809 (ms)
```
OR\\
* Compile and run: `bash run.sh` 
* Results: 
```
Running with round=10
N=1024
ms_good = 0.026 (ms), ms_bad = 0.034 (ms)
N=2048
ms_good = 0.023 (ms), ms_bad = 0.035 (ms)
N=4096
ms_good = 0.023 (ms), ms_bad = 0.039 (ms)
N=8192
ms_good = 0.025 (ms), ms_bad = 0.042 (ms)
N=16384
ms_good = 0.024 (ms), ms_bad = 0.045 (ms)
Running with round=100
N=1024
ms_good = 0.220 (ms), ms_bad = 0.290 (ms)
N=2048
ms_good = 0.220 (ms), ms_bad = 0.344 (ms)
N=4096
ms_good = 0.217 (ms), ms_bad = 0.371 (ms)
N=8192
ms_good = 0.221 (ms), ms_bad = 0.398 (ms)
N=16384
ms_good = 0.222 (ms), ms_bad = 0.428 (ms)
Running with round=1000
N=1024
ms_good = 2.082 (ms), ms_bad = 2.886 (ms)
N=2048
ms_good = 2.129 (ms), ms_bad = 3.336 (ms)
N=4096
ms_good = 2.142 (ms), ms_bad = 3.599 (ms)
N=8192
ms_good = 2.226 (ms), ms_bad = 3.930 (ms)
N=16384
ms_good = 2.132 (ms), ms_bad = 4.200 (ms)
Running with round=10000
N=1024
ms_good = 21.039 (ms), ms_bad = 27.892 (ms)
N=2048
ms_good = 21.303 (ms), ms_bad = 33.081 (ms)
N=4096
ms_good = 21.359 (ms), ms_bad = 36.995 (ms)
N=8192
ms_good = 22.020 (ms), ms_bad = 39.291 (ms)
N=16384
ms_good = 21.170 (ms), ms_bad = 41.809 (ms)
```