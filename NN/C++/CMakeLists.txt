# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(TensorFlowLiteClone)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add the include directory so CMake can find the header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/tensor.cpp
    src/loss.cpp
    src/layer.cpp
    src/optimizer.cpp
    main.cpp
)

# Create the executable
add_executable(TensorFlowLiteClone ${SOURCES})
