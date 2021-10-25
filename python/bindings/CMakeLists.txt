# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT cuda_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT cuda_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND cuda_python_files
    python_bindings.cc)

GR_PYBIND_MAKE_OOT(cuda 
   ../..
   gr::cuda
   "${cuda_python_files}")

install(TARGETS cuda_python DESTINATION ${GR_PYTHON_DIR}/cuda COMPONENT pythonapi)
