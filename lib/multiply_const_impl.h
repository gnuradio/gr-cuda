/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_MULTIPLY_CONST_IMPL_H
#define INCLUDED_CUDA_MULTIPLY_CONST_IMPL_H

#include <gnuradio/cuda/multiply_const.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {
namespace cuda {

template <class T>
class multiply_const_impl : public multiply_const<T>
{
    T d_k;
    const size_t d_vlen;

    cudaStream_t d_stream;
    int d_min_grid_size;
    int d_block_size;
public:
    multiply_const_impl(T k, size_t vlen);

    T k() const override { return d_k; }
    void set_k(T k) override { d_k = k; }

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_MULTIPLY_CONST_IMPL_H */
