add_subdirectory(headers)

add_executable(test_neighbour_stats neighbour_stats.cpp)
target_link_libraries(test_neighbour_stats PRIVATE percolation)
