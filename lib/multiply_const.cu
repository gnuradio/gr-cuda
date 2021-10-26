#include <stdio.h>

#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <iostream>

// This is the kernel that will get launched on the device
template <typename T>
__global__ void kernel_multiply_const(const T* in, T* out, T k, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = in[i] * k;
    }
}

template <>
__global__ void
kernel_multiply_const<thrust::complex<float>>(const thrust::complex<float>* in,
                                              thrust::complex<float>* out,
                                              thrust::complex<float> k,
                                              int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = in[i] * k;
    }
}

// Kernel wrapper so that the GNU Radio code doesn't have to compile with nvcc
template <typename T>
void exec_kernel_multiply_const(const T* in,
                                T* out,
                                T k,
                                int grid_size,
                                int block_size,
                                size_t n,
                                cudaStream_t stream)
{
    kernel_multiply_const<T><<<grid_size, block_size, 0, stream>>>(in, out, k, n);
    check_cuda_errors(cudaGetLastError());
}

template <>
void exec_kernel_multiply_const<std::complex<float>>(const std::complex<float>* in,
                                                     std::complex<float>* out,
                                                     std::complex<float> k,
                                                     int grid_size,
                                                     int block_size,
                                                     size_t n,
                                                     cudaStream_t stream)
{
    kernel_multiply_const<thrust::complex<float>>
        <<<grid_size, block_size, 0, stream>>>((const thrust::complex<float>*)in,
                                               (thrust::complex<float>*)out,
                                               (thrust::complex<float>)k,
                                               n);
    check_cuda_errors(cudaGetLastError());
}

template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock)
{
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_multiply_const<T>, 0, 0));
}

template <>
void get_block_and_grid<std::complex<float>>(int* minGrid, int* minBlock)
{
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_multiply_const<thrust::complex<float>>, 0, 0));
}

#define IMPLEMENT_KERNEL(T)                          \
    template void get_block_and_grid<T>(int*, int*); \
    template void exec_kernel_multiply_const<T>(     \
        const T*, T*, T, int, int, size_t, cudaStream_t);

IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)