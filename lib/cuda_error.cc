#include <gnuradio/cuda/cuda_error.h>

void check_cuda_errors(cudaError_t rc)
{
    if (rc) {
        std::cerr << "Operation returned code " << int(rc) << ": " << cudaGetErrorName(rc)
                  << " -- " << cudaGetErrorString(rc);
    }
}
