#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace cuda {
namespace load_cu {

void exec_kernel(
    const uint8_t* in, uint8_t* out, int grid_size, int block_size, int N, size_t load, cudaStream_t stream);

void get_block_and_grid(int* minGrid, int* minBlock);

} // namespace multiply_const
} // namespace blocks
} // namespace gr