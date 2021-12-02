#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Josh Morman.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
try:
    from gnuradio.cuda import multiply_const_ff
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.cuda import multiply_const_ff

class qa_multiply_const(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = multiply_const_ff(2.0, 1)

    def test_001_descriptive_test_name(self):
        # set up fg
        input_data = list(range(100000))
        expected_data = [x*2.0*3.0 for x in input_data]
        src = blocks.vector_source_f(input_data, False)
        op1 = multiply_const_ff(2.0)
        op2 = multiply_const_ff(3.0)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, op1, op2, snk)
        self.tb.run()
        self.assertEqual(snk.data(), expected_data)        

if __name__ == '__main__':
    gr_unittest.run(qa_multiply_const)
