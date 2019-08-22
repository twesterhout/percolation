
function(add_header_test MY_HEADER)
    add_executable(test_header_${MY_HEADER} ${MY_HEADER}.cpp)
    target_link_libraries(test_header_${MY_HEADER} PRIVATE percolation)
endfunction()

add_header_test(config)
add_header_test(convolution)
add_header_test(lattice)
add_header_test(magnetic_cluster)
add_header_test(neighbour_stats)
add_header_test(shuffle)
add_header_test(particle)

# add_executable(test_header_lattice lattice.cpp)
# target_link_libraries(test_header_lattice PRIVATE percolation)

# add_executable(test_header_random random.cpp ../../src/random.cpp)
# target_link_libraries(test_header_random PUBLIC Common)

# add_executable(test_header_shuffle shuffle.cpp)
# target_link_libraries(test_header_shuffle PUBLIC Common)
# 
# add_executable(test_header_particle particle.cpp)
# target_link_libraries(test_header_particle PUBLIC Common)

# add_executable(test_header_magnetic_cluster magnetic_cluster.cpp)
# target_link_libraries(test_header_magnetic_cluster PUBLIC Common)

# add_executable(test_header_geometric_cluster geometric_cluster.cpp)
# target_link_libraries(test_header_geometric_cluster PUBLIC Common)


