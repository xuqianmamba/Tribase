include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_FILES
    Clustering.cpp
    Index.cpp
    IVF.cpp
    # utils.cpp
)

add_library(Tribase STATIC ${SRC_FILES})

target_link_libraries(Tribase PUBLIC OpenMP::OpenMP_CXX
                                    mkl_intel_lp64
                                    mkl_sequential
                                    mkl_core)

add_executable(main main.cpp)
target_link_libraries(main Tribase faiss)

add_executable(query query.cpp)
target_link_libraries(query Tribase faiss)

add_executable(uniform_dataset_generate uniform_dataset_generate.cpp)

add_subdirectory(tests)