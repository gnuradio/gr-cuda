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

#ifndef INCLUDED_CUDA_COPY_IMPL_H
#define INCLUDED_CUDA_COPY_IMPL_H

#include <gnuradio/cuda/copy.h>

namespace gr {
namespace cuda {

class copy_impl : public copy {
private:
  size_t d_itemsize;

public:
  copy_impl(size_t itemsize);
  ~copy_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_COPY_IMPL_H */
