#ifndef _INCLUDED_GR_CUDA_ERROR
#define _INCLUDED_GR_CUDA_ERROR

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

void check_cuda_errors(cudaError_t rc);

#endif