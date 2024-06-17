# Example - memory_access_pattern
* This code is used to compile 

# Compile and Results: 
* `nvcc memory_access.cu -o memory_access && ./memory_access 1024 1000`
  * N = 1024, ROUND = 1000
```
ms_good = 0.190, ms_bad = 0.750
```

* `bash run.sh` 
```
Running with round=10
N=1024
ms_good = 0.006, ms_bad = 0.012
N=2048
ms_good = 0.006, ms_bad = 0.017
N=4096
ms_good = 0.006, ms_bad = 0.018
N=8192
ms_good = 0.006, ms_bad = 0.019
N=16384
ms_good = 0.006, ms_bad = 0.025
Running with round=100
N=1024
ms_good = 0.022, ms_bad = 0.081
N=2048
ms_good = 0.022, ms_bad = 0.123
N=4096
ms_good = 0.022, ms_bad = 0.143
N=8192
ms_good = 0.022, ms_bad = 0.152
N=16384
ms_good = 0.020, ms_bad = 0.203
Running with round=1000
N=1024
ms_good = 0.176, ms_bad = 0.779
N=2048
ms_good = 0.176, ms_bad = 1.207
N=4096
ms_good = 0.176, ms_bad = 1.347
N=8192
ms_good = 0.177, ms_bad = 1.441
N=16384
ms_good = 0.177, ms_bad = 2.026
Running with round=10000
N=1024
ms_good = 1.729, ms_bad = 7.745
N=2048
ms_good = 1.730, ms_bad = 11.643
N=4096
ms_good = 1.729, ms_bad = 13.716
N=8192
ms_good = 1.730, ms_bad = 14.181
N=16384
ms_good = 1.730, ms_bad = 20.035
```