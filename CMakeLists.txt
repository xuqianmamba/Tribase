cmake_minimum_required(VERSION 3.21)
project(TriBase)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)

# set(W_CXX_FLAGS " -Wall -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 ${W_CXX_FLAGS} -g ${OpenMP_CXX_FLAGS}") # -Wpedantic -Werror -fsanitize=address
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # -Wpedantic -Werror
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party)
add_subdirectory(third_party/faiss)
add_subdirectory(src)