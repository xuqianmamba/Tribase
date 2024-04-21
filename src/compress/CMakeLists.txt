add_library(repair STATIC repair_compress.cpp check.cpp)

target_link_libraries(repair PUBLIC OpenMP::OpenMP_CXX)