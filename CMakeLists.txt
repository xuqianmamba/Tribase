cmake_minimum_required(VERSION 3.21)
project(TriBase)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Standard)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release|Standard")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)

set(MKL_ROOT "/opt/intel/oneapi/mkl/2024.1") # TODO: change back to 2024.2 after test
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)

option(ENABLE_PROFILING "Enable profiling" OFF)

if (ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# set(W_CXX_FLAGS " -Wall -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Standard")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native -DNDEBUG")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NSTATS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss)
include_directories(${CMAKE_SOURCE_DIR}/hnswlib)
add_subdirectory(third_party/faiss)
add_subdirectory(src)