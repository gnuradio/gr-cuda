find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_CUDA gnuradio-cuda)

FIND_PATH(
    GR_CUDA_INCLUDE_DIRS
    NAMES gnuradio/cuda/fapi.h
    HINTS $ENV{GR_CUDA_DIR}/include
        ${PC_GR_CUDA_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_CUDA_LIBRARIES
    NAMES gnuradio-cuda
    HINTS $ENV{GR_CUDA_DIR}/lib
        ${PC_GR_CUDA_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-cudaTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_CUDA DEFAULT_MSG GR_CUDA_LIBRARIES GR_CUDA_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_CUDA_LIBRARIES GR_CUDA_INCLUDE_DIRS)
