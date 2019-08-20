# Copyright Tom Westerhout (c) 2018
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.9)
project(Percolation)

include(CTest)
include(CheckCXXCompilerFlag)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

cmake_policy(SET CMP0074 NEW)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to RelWithDebInfo, use \
-DCMAKE_BUILD_TYPE= to override."
    )
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" TCM_BUILD_TYPE)


CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_NATIVE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_OPT_HIDDEN_SUPPORTED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# set(INTEL_ROOT /opt/intel CACHE PATH "Root directory of Intel products.")
# set(MKL_ROOT ${INTEL_ROOT}/mkl CACHE PATH "Root directory of Intel MKL 2018 installation.")
# 
# option(MKL_SDL "Use Single Dynamically Linked version of Intel MKL." OFF)
# option(MKL_MULTI_THREADED "Use multi-threaded interface of Intel MKL." ON)
# option(MKL_GNU_THREAD "Use GNU OpenMP rather than Intel OpenMP." ON)

# Warnings
################################################################################
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(TCM_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-c++14-compat -Wno-unused -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-padded
        )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(TCM_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused)
endif()

# Dependencies
################################################################################
find_package(Git REQUIRED)

# Update submodules


find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if(NOT TARGET Boost::boost)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/boost
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- libs/align
                                                             libs/core
                                                             libs/config
                                                             libs/detail
                                                             libs/pool
                                                             libs/utility
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost)
    add_library(Boost INTERFACE)
    target_include_directories(Boost SYSTEM INTERFACE
        third_party/boost/libs/align/include
        third_party/boost/libs/core/include
        third_party/boost/libs/config/include
        third_party/boost/libs/detail/include
        third_party/boost/libs/pool/include
        third_party/boost/libs/utility/include)
    add_library(Boost::boost ALIAS Boost)
endif()

if(NOT TARGET gsl::gsl-lite)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/gsl-lite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(third_party/gsl-lite)
endif()

if(TRUE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init -- third_party/pcg-cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(pcg INTERFACE)
    target_include_directories(pcg SYSTEM INTERFACE third_party/pcg/include)
endif()

function(setup_percolation_target MY_TARGET)
    target_compile_features(${MY_TARGET} PUBLIC cxx_std_17)
    target_compile_options(${MY_TARGET} PRIVATE "${TCM_WARNING_FLAGS}")

    check_cxx_compiler_flag("-march=native" TCM_COMPILER_OPT_NATIVE_SUPPORTED)
    if(TCM_COMPILER_OPT_NATIVE_SUPPORTED)
        message(STATUS "[percolation] Using -march=native for ${MY_TARGET}.")
        target_compile_options(${MY_TARGET} PUBLIC "-march=native")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(${MY_TARGET} PUBLIC -fcolor-diagnostics)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${MY_TARGET} PUBLIC -fdiagnostics-color=always)
    endif()
endfunction()

add_library(percolation SHARED
    src/convolution.cpp
    src/lattice.c
)
target_include_directories(percolation PUBLIC include)
setup_percolation_target(percolation)
target_link_libraries(percolation PUBLIC gsl::gsl-lite pcg)
if ("${TCM_BUILD_TYPE}" STREQUAL "DEBUG")
    target_compile_definitions(percolation PUBLIC
        gsl_CONFIG_CONTRACT_VIOLATION_THROWS=1)
    # TODO(twesterhout): switch to
    #   gsl_CONFIG_CONTRACT_VIOLATION_CALLS_HANDLER=1
else()
    target_compile_definitions(percolation PUBLIC
        gsl_CONFIG_CONTRACT_LEVEL_OFF=1)
    message(STATUS "[lbfgs-cpp] Disabling gsl-lite contracts")
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

if (FALSE)
    add_library(Common INTERFACE)
    target_include_directories(Common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                ${CMAKE_CURRENT_SOURCE_DIR}/external)
    target_include_directories(Common SYSTEM INTERFACE ${MKL_INCLUDES})
    target_compile_features(Common INTERFACE cxx_std_17)
    target_compile_options(Common INTERFACE "${TCM_WARNING_FLAGS}")
    target_link_libraries(Common INTERFACE OpenMP::OpenMP_CXX Boost::boost sleef ${MKL_LIBS})

    if(COMPILER_OPT_HIDDEN_SUPPORTED)
        message(STATUS "[percolation] Using -fvisibility=hidden.")
        target_compile_options(Common INTERFACE "-fvisibility=hidden")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # target_compile_options(percolation PUBLIC
        #     -m64 -mmmx -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mavx
        #     -mtune=native)
        target_compile_options(Common INTERFACE -march=native)
    else()
        message(STATUS "[percolation] Compiler ${CMAKE_CXX_COMPILER_ID} is not "
                       "(yet) supported. You are likely to run into trouble.")
    endif()

    if (FALSE)
    add_executable(test_header_shuffle test/shuffle.cpp)
    target_include_directories(test_header_shuffle PUBLIC include)
    target_compile_features(test_header_shuffle PUBLIC cxx_std_17)
    target_compile_options(test_header_shuffle PUBLIC "${TCM_WARNING_FLAGS}")

    add_executable(test_header_lattice test/lattice.cpp)
    target_include_directories(test_header_lattice PUBLIC include)
    target_compile_features(test_header_lattice PUBLIC cxx_std_17)
    target_compile_options(test_header_lattice PUBLIC "${TCM_WARNING_FLAGS}")

    add_executable(test_header_utility test/utility.cpp)
    target_include_directories(test_header_utility PUBLIC include)
    target_compile_features(test_header_utility PUBLIC cxx_std_17)
    target_compile_options(test_header_utility PUBLIC "${TCM_WARNING_FLAGS}")

    add_executable(test_header_percolation test/percolation.c)
    set_target_properties(test_header_percolation PROPERTIES LANGUAGE C)
    target_include_directories(test_header_percolation PUBLIC include)
    target_compile_options(test_header_percolation PUBLIC "${TCM_WARNING_FLAGS}")

    add_library(tcm_lattice STATIC src/lattice.c)
    target_link_libraries(tcm_lattice PUBLIC Common)
    target_compile_options(tcm_lattice PUBLIC "${TCM_WARNING_FLAGS}")
    set_target_properties(tcm_lattice PROPERTIES C_STANDARD 11)

    add_executable(test_header_particle test/headers/particle.cpp)
    target_include_directories(test_header_particle PUBLIC include)
    target_compile_options(test_header_percolation PUBLIC "${TCM_WARNING_FLAGS}")
    endif()

    add_subdirectory(test/headers)

    if (TRUE)
        #     add_library(percolation SHARED src/perc_v2.cpp
        #                                    src/random.cpp
        #                                    src/convolution.cpp
        #                                    src/lattice.c
        #                                    # src/tsallis_distribution.c
        #     )
        # 
        #     target_link_libraries(percolation PUBLIC Common)
        # 
        #     if(COMPILER_OPT_HIDDEN_SUPPORTED)
        #         message(STATUS "[percolation] Using -fvisibility=hidden.")
        #         # target_compile_options(percolation PUBLIC "-fvisibility=hidden")
        #     endif()
        # 
        #     if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        #         # target_compile_options(percolation PUBLIC
        #         #     -m64 -mmmx -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mavx
        #         #     -mtune=native)
        #         target_compile_options(Common INTERFACE -march=native)
        #     else()
        #         message(STATUS "[percolation] Compiler ${CMAKE_CXX_COMPILER_ID} is not "
        #                        "(yet) supported. You are likely to run into trouble.")
        #     endif()
        # 
        #     # add_executable(main src/main.cpp)
        #     # target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        #     # target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
        #     # target_compile_features(main PUBLIC cxx_std_17)
        # 
        #     # add_executable(simulated_annealing src/simulated_annealing.cpp)
        #     # target_include_directories(simulated_annealing SYSTEM PUBLIC ${MKL_INCLUDES})
        #     # target_link_libraries(simulated_annealing PUBLIC
        #     #     percolation
        #     # )
        # 
        #     add_executable(tsallis test/tsallis.cpp)
        #     target_link_libraries(tsallis PUBLIC percolation)
        # 
        add_executable(rastrigin src/rastrigin.cpp src/random.cpp)
        target_link_libraries(rastrigin PUBLIC Common)
        #     target_link_libraries(rastrigin PUBLIC percolation)
        # 
        #     add_executable(simple src/simple.c)
        #     target_link_libraries(simple PUBLIC percolation)

# add_executable(print_lattice src/print_lattice.cpp)
# target_link_libraries(print_lattice PRIVATE percolation)

# install(TARGETS percolation
#             LIBRARY
#             DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/Percolation
#             PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#                         GROUP_READ GROUP_EXECUTE
#                         WORLD_READ WORLD_EXECUTE)
    endif()
endif()
