# Copyright (c) 2019 Mikael Simberg
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.0)
project(MatrixMultiplication CXX)

find_package(HPX REQUIRED)


if(HPX_WITH_CUDA)
  add_hpx_executable(hpxMatrixMultiplication SOURCES hpxMatrixMultiplication.cu)
  add_hpx_executable(cpuMM SOURCES cpuMM.cpp)
endif()


