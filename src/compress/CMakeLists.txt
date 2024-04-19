add_library(repair STATIC repair_compress.cpp)

target_link_libraries(repair PUBLIC OpenMP::OpenMP_CXX)