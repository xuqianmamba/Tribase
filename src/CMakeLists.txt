file(GLOB_RECURSE SRC_FILES "*.cpp")

list(FILTER SRC_FILES EXCLUDE REGEX "tests/.*\\.cpp$")

add_library(TribaseLib STATIC ${SRC_FILES})

target_link_libraries(TribaseLib PUBLIC OpenMP::OpenMP_CXX)

# target_link_libraries(TribaseLib PUBLIC other_libraries)