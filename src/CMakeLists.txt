include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_FILES
    Clustering.cpp
    Index.cpp
    IVF.cpp
)

add_library(Tribase STATIC ${SRC_FILES})

target_link_libraries(Tribase PUBLIC OpenMP::OpenMP_CXX)

# target_link_libraries(TribaseLib PUBLIC other_libraries)

add_subdirectory(tests)