cmake_minimum_required(VERSION 3.12)

# Project name
project(MyCudaProject LANGUAGES CXX CUDA)

# Set the CMP0104 policy to NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Specify the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find all .cu files in the src directory
file(GLOB SOURCES "src/*/*.cu" "src/*.cu")

# Add executable
add_executable(MyCudaProject ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(
    MyCudaProject
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
