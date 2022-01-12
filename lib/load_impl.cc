/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "load_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>

#include "load.cuh"

namespace gr {
namespace cuda {

load::sptr load::make(size_t iterations, size_t itemsize, bool use_cb)
{
    return gnuradio::make_block_sptr<load_impl>(iterations, itemsize, use_cb);
}


/*
 * The private constructor
 */
load_impl::load_impl(size_t iterations, size_t itemsize, bool use_cb)
    : gr::sync_block("load",
                     gr::io_signature::make(1, 1, itemsize),
                     gr::io_signature::make(1, 1, itemsize)),
      d_iterations(iterations),
      d_itemsize(itemsize),
      d_use_cb(use_cb)
{
    load_cu::get_block_and_grid(&d_min_grid_size, &d_block_size);
    d_logger->info("minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);

    if (use_cb) {
        set_input_signature(gr::io_signature::make(1, 1, itemsize, cuda_buffer::type));
        set_output_signature(gr::io_signature::make(1, 1, itemsize, cuda_buffer::type));
    } else {
        check_cuda_errors(cudaMalloc((void**)&d_dev_in, d_max_buffer_size));
        check_cuda_errors(cudaMalloc((void**)&d_dev_out, d_max_buffer_size));
    }
}

int load_impl::work(int noutput_items,
                    gr_vector_const_void_star& input_items,
                    gr_vector_void_star& output_items)
{
    auto in = static_cast<const uint8_t*>(input_items[0]);
    auto out = static_cast<uint8_t*>(output_items[0]);

    int gridSize = (noutput_items * d_itemsize + d_block_size - 1) / d_block_size;

    if (!d_use_cb) {
        check_cuda_errors(cudaMemcpyAsync(d_dev_in,
                                          in,
                                          noutput_items * d_itemsize,
                                          cudaMemcpyHostToDevice,
                                          d_stream));

        load_cu::exec_kernel(d_dev_in,
                             d_dev_out,
                             gridSize,
                             d_block_size,
                             noutput_items * d_itemsize,
                             d_iterations,
                             d_stream);
        check_cuda_errors(cudaPeekAtLastError());

        cudaMemcpyAsync(out,
                        d_dev_out,
                        noutput_items * d_itemsize,
                        cudaMemcpyDeviceToHost,
                        d_stream);

    } else {
        load_cu::exec_kernel(in,
                             out,
                             gridSize,
                             d_block_size,
                             noutput_items * d_itemsize,
                             d_iterations,
                             d_stream);
        check_cuda_errors(cudaPeekAtLastError());
    }


    cudaStreamSynchronize(d_stream);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cuda */
} // namespace gr
