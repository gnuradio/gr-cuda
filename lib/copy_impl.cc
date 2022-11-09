/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "copy_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

copy::sptr copy::make(size_t itemsize)
{
    return gnuradio::make_block_sptr<copy_impl>(itemsize);
}

/*
 * The private constructor
 */
copy_impl::copy_impl(size_t itemsize)
    : gr::sync_block("copy",
                     gr::io_signature::make(1, 1, itemsize, cuda_buffer::type),
                     gr::io_signature::make(1, 1, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize)
{
}

/*
 * Our virtual destructor.
 */
copy_impl::~copy_impl() {}

int copy_impl::work(int noutput_items,
                    gr_vector_const_void_star& input_items,
                    gr_vector_void_star& output_items)
{

    auto in = static_cast<const uint8_t*>(input_items[0]);
    auto out = static_cast<uint8_t*>(output_items[0]);

    cudaMemcpy(out, in, noutput_items * d_itemsize, cudaMemcpyDeviceToDevice);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
