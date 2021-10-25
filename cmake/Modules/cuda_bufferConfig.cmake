if(NOT PKG_CONFIG_FOUND)
    INCLUDE(FindPkgConfig)
endif()
PKG_CHECK_MODULES(PC_CUDA_BUFFER cuda_buffer)

FIND_PATH(
    CUDA_BUFFER_INCLUDE_DIRS
    NAMES cuda_buffer/api.h
    HINTS $ENV{CUDA_BUFFER_DIR}/include
        ${PC_CUDA_BUFFER_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    CUDA_BUFFER_LIBRARIES
    NAMES gnuradio-cuda_buffer
    HINTS $ENV{CUDA_BUFFER_DIR}/lib
        ${PC_CUDA_BUFFER_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

#include("${CMAKE_CURRENT_LIST_DIR}/cuda_bufferTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUDA_BUFFER DEFAULT_MSG CUDA_BUFFER_LIBRARIES CUDA_BUFFER_INCLUDE_DIRS)
MARK_AS_ADVANCED(CUDA_BUFFER_LIBRARIES CUDA_BUFFER_INCLUDE_DIRS)
