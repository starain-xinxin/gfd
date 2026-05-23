cmake_minimum_required(VERSION 3.18)
project(gfd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable AVX-512 for streaming memcpy in gather threads
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -O3")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Auto-detect GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90;120")
endif()

# Library sources (device API is header-only via include/gfd/device.cuh)
set(GFD_SOURCES
    src/copy_engine.cpp
    src/cpu_polling.cpp
    src/batch_processor.cpp
    src/parallel_gather.cpp
    src/direct_submit.cpp
    src/warp_spec_session.cpp
    src/sg_warp_spec_session.cpp
)

# Static library
add_library(gfd_static STATIC ${GFD_SOURCES})
set_target_properties(gfd_static PROPERTIES
    OUTPUT_NAME gfd
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(gfd_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(gfd_static PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

# Shared library
add_library(gfd_shared SHARED ${GFD_SOURCES})
set_target_properties(gfd_shared PROPERTIES
    OUTPUT_NAME gfd
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(gfd_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(gfd_shared PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

# Example (optional, only built if examples are enabled)
option(GFD_BUILD_EXAMPLES "Build GFD examples" ON)
if(GFD_BUILD_EXAMPLES)
    add_executable(gfd_basic_transfer examples/01_basic_transfer.cu)
    target_link_libraries(gfd_basic_transfer PRIVATE gfd_static)
    set_target_properties(gfd_basic_transfer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_direct_transfer examples/02_direct_transfer.cu)
    target_link_libraries(gfd_direct_transfer PRIVATE gfd_static)
    set_target_properties(gfd_direct_transfer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_gpu_planned examples/03_gpu_planned_transfer.cu)
    target_link_libraries(gfd_gpu_planned PRIVATE gfd_static)
    set_target_properties(gfd_gpu_planned PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_benchmark examples/04_benchmark.cu)
    target_link_libraries(gfd_benchmark PRIVATE gfd_static)
    set_target_properties(gfd_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_multi_gpu_benchmark examples/05_multi_gpu_benchmark.cu)
    target_link_libraries(gfd_multi_gpu_benchmark PRIVATE gfd_static numa)
    set_target_properties(gfd_multi_gpu_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_multi_gpu_direct examples/06_multi_gpu_direct.cu)
    target_link_libraries(gfd_multi_gpu_direct PRIVATE gfd_static numa)
    set_target_properties(gfd_multi_gpu_direct PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_warp_spec examples/07_warp_spec_simple.cu)
    target_link_libraries(gfd_warp_spec PRIVATE gfd_static)
    set_target_properties(gfd_warp_spec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_multi_gpu_warp_spec examples/08_multi_gpu_warp_spec.cu)
    target_link_libraries(gfd_multi_gpu_warp_spec PRIVATE gfd_static numa)
    set_target_properties(gfd_multi_gpu_warp_spec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_sg_warp_spec examples/09_sg_warp_spec.cu)
    target_link_libraries(gfd_sg_warp_spec PRIVATE gfd_static)
    set_target_properties(gfd_sg_warp_spec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_test_sg_e2e test/test_sg_e2e.cu)
    target_link_libraries(gfd_test_sg_e2e PRIVATE gfd_static)
    set_target_properties(gfd_test_sg_e2e PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gfd_test_sg_gpu_submit test/test_sg_gpu_submit.cu)
    target_link_libraries(gfd_test_sg_gpu_submit PRIVATE gfd_static)
    set_target_properties(gfd_test_sg_gpu_submit PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

endif()

# Install targets
install(TARGETS gfd_static gfd_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/gfd DESTINATION include)
