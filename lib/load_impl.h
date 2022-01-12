/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_LOAD_IMPL_H
#define INCLUDED_CUDA_LOAD_IMPL_H

#include <gnuradio/cuda/load.h>
#include <gnuradio/cuda/cuda_block.h>

namespace gr {
namespace cuda {

class load_impl : public load, public cuda_block
{
private:
    size_t d_iterations;
    size_t d_itemsize;
    bool d_use_cb;
    uint8_t *d_dev_in;
    uint8_t *d_dev_out;

    size_t d_max_buffer_size = 65536*8;
public:
    load_impl(size_t iterations, size_t itemsize, bool use_cb = true);

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_LOAD_IMPL_H */
