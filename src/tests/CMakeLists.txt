add_executable(test test.cpp)
target_link_libraries(test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(heap_test heap_test.cpp)
target_link_libraries(heap_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(scan_test scan_test.cpp)
target_link_libraries(scan_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(cluster_test cluster_test.cpp ../Clustering.cpp)
target_link_libraries(cluster_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)