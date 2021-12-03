/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef _INCLUDED_CUDA_BLOCK_H
#define _INCLUDED_CUDA_BLOCK_H

#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {

class cuda_block
{
protected:
    cudaStream_t d_stream;
    int d_min_grid_size;
    int d_block_size;

public:
    cuda_block() { check_cuda_errors(cudaStreamCreate(&d_stream)); };
};

} // namespace gr
#endif