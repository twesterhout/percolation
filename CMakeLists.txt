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

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to RelWithDebInfo, use \
-DCMAKE_BUILD_TYPE= to override."
    )
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()


CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_NATIVE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_OPT_HIDDEN_SUPPORTED)

# Warnings
################################################################################
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(NQS_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-unused
        -Wno-exit-time-destructors -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-global-constructors -Wno-padded
        )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(NQS_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused)
endif()

add_library(percolation SHARED src/percolation.cpp)
target_include_directories(percolation PUBLIC include)
target_compile_features(percolation PUBLIC cxx_std_17)
target_compile_options(percolation PUBLIC "${NQS_WARNING_FLAGS}")

# if(COMPILER_OPT_NATIVE_SUPPORTED)
#     message(STATUS "[percolation] Using -march=native.")
#     target_compile_options(percolation PUBLIC "-march=native")
# endif()

if(COMPILER_OPT_HIDDEN_SUPPORTED)
    message(STATUS "[percolation] Using -fvisibility=hidden.")
    target_compile_options(percolation PUBLIC "-fvisibility=hidden")
endif()

if("${CMAKE_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(percolation PUBLIC
        -m64 -mmmx -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mavx
        -mtune=native)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE percolation)

install(TARGETS percolation
        LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/percolation
        CONFIGURATIONS Release)
