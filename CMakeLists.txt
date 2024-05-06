cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(llama2_q4 LANGUAGES CXX CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
enable_language(CUDA)

include_directories(../re2/)
link_directories(/usr/local/lib)

include_directories(.)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main main.cpp)
target_link_libraries(main re2)
add_library(cuda_operations SHARED inference.cu)

add_executable(weight_packer weight_packer.cpp)


# Add link library for CUDA runtime and link to main.
target_link_libraries(main cuda_operations ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})

# Enable strict CUDA checks
target_compile_options(cuda_operations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall,-Wextra>)