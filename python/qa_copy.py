#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Josh.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
try:
    from gnuradio.cuda import copy
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.cuda import copy

class qa_copy(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = copy(gr.sizeof_gr_complex)

    def test_001_descriptive_test_name(self):
        nsamples = 10000

        input_data = list(range(nsamples))
        src = blocks.vector_source_f(input_data, False)
        op = copy(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, op, snk)

        self.tb.run()


        self.assertEqual(snk.data(), input_data)


if __name__ == '__main__':
    gr_unittest.run(qa_copy)
