# Copyright (c) 2019 Mikael Simberg
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.0)
project(TestesHPXCUDA CXX)

find_package(HPX REQUIRED)


if(HPX_WITH_CUDA)
  add_hpx_executable(teste1 SOURCES teste1.cu)
  add_hpx_executable(teste2 SOURCES teste2.cu)
endif()


