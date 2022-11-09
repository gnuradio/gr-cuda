/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_LOAD_H
#define INCLUDED_CUDA_LOAD_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief <+description of block+>
 * \ingroup cuda
 *
 */
class CUDA_API load : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<load> sptr;

    /*!
     * \brief Load block for testing CUDA workflows
     *
     * The load block provides a mechanism for loading down
     * the cuda processing with a for loop.  This is useful
     * for hypothetical profiling and seeing gains in data
     * transfers using the custom buffers
     */
    static sptr make(size_t iterations, size_t itemsize, bool use_cb = true);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_LOAD_H */
