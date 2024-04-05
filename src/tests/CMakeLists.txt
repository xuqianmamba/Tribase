add_executable(test test.cpp)
target_link_libraries(test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(heap_test heap_test.cpp)
target_link_libraries(heap_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)