add_executable(test test.cpp)
target_link_libraries(test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(heap_test heap_test.cpp)
target_link_libraries(heap_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(scan_test scan_test.cpp)
target_link_libraries(scan_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(cluster_test cluster_test.cpp ../Clustering.cpp)
target_link_libraries(cluster_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

add_executable(faiss_test faiss_test.cpp)
target_link_libraries(faiss_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread faiss)

add_executable(index_test index_test.cpp)
target_link_libraries(index_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread Tribase faiss)

add_executable(time_test time_test.cpp)
target_link_libraries(time_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread Tribase faiss)