/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_CUDA_H
#define INCLUDED_GR_CUDA_H

#include <gnuradio/buffer_single_mapped.h>
#include <gnuradio/buffer_type.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {

/*!
 * \brief Subclass of buffer_single_mapped for supporting blocks using NVidia's
 * CUDA runtime.
 *
 * This buffer_single_mapped subclass is designed to provide easy buffer support
 * for blocks using NVidia's CUDA runtime. The class acts as a wrapper for two
 * underlying buffers, a host buffer allocated using the cudaMallocHost()
 * function and a device buffer allocated using the cudaMalloc() function.
 * The logic contained within this class manages both buffers and the movement
 * of data between the two depending on the buffer's assigned context.
 *
 */
class GR_RUNTIME_API cuda_buffer : public buffer_single_mapped
{
public:
    mem_func_t f_cuda_memcpy;
    mem_func_t f_cuda_memmove;
    void* cuda_memcpy(void* dest, const void* src, std::size_t count);
    void* cuda_memmove(void* dest, const void* src, std::size_t count);

    static buffer_type type;

    virtual ~cuda_buffer();

    /*!
     * \brief Handle post-general_work() cleanup and data transfer
     *
     * Called directly after call to general_work() completes and
     * is used for data transfer (and perhaps other administrative
     * activities)
     *
     * \param nitems is the number of items produced by the general_work() function.
     */
    virtual void post_work(int nitems);

    /*!
     * \brief Do actual buffer allocation. Inherited from buffer_single_mapped.
     */
    bool do_allocate_buffer(size_t final_nitems, size_t sizeof_item);

    /*!
     * \brief Return a pointer to the write buffer depending on the context
     */
    virtual void* write_pointer();

    /*!
     * \brief return pointer to read buffer depending on the context
     *
     * The return value points to at least items_available() items.
     */
    virtual const void* _read_pointer(unsigned int read_index);

    /*!
     * \brief Callback function that the scheduler will call when it determines
     * that the input is blocked. Override this function if needed.
     */
    bool input_blocked_callback(int items_required, int items_avail, unsigned read_index);

    /*!
     * \brief Callback function that the scheduler will call when it determines
     * that the output is blocked
     */
    bool output_blocked_callback(int output_multiple, bool force);

    /*!
     * \brief Creates a new cuda object
     *
     * \param nitems
     * \param sizeof_item
     * \param downstream_lcm_nitems
     * \param link
     * \param buf_owner
     *
     * \return pointer to buffer base class
     */
    static buffer_sptr make_buffer(int nitems,
                                   size_t sizeof_item,
                                   uint64_t downstream_lcm_nitems,
                                   uint32_t downstream_max_out_mult,
                                   block_sptr link,
                                   block_sptr buf_owner);

private:
    cudaStream_t d_stream;
    char* d_cuda_buf; // CUDA buffer

    /*!
     * \brief constructor is private.  Use gr_make_buffer to create instances.
     *
     * Allocate a buffer that holds at least \p nitems of size \p sizeof_item.
     *
     * \param nitems is the minimum number of items the buffer will hold.
     * \param sizeof_item is the size of an item in bytes.
     * \param downstream_lcm_nitems is the least common multiple of the items to
     *                              read by downstream blocks
     * \param downstream_max_out_mult is the maximum output multiple of all 
     *                                downstream blocks
     * \param link is the block that writes to this buffer.
     * \param buf_owner if the block that owns the buffer which may or may not
     *                  be the same as the block that writes to this buffer
     *
     * The total size of the buffer will be rounded up to a system
     * dependent boundary.  This is typically the system page size, but
     * under MS windows is 64KB.
     */
    cuda_buffer(int nitems,
                size_t sizeof_item,
                uint64_t downstream_lcm_nitems,
                uint32_t downstream_max_out_mult,
                block_sptr link,
                block_sptr buf_owner);
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_H */
