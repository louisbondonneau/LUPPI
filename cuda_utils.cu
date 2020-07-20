#include <cuda.h>

#include "cuda_utils.h"

int getCUDADeviceCount() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}
