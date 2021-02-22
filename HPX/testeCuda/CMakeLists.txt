# Copyright (c) 2019 Mikael Simberg
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.0)
project(Exercises CXX)

find_package(HPX REQUIRED)


if(HPX_WITH_CUDA)
  add_hpx_executable(solution9 SOURCES solution9.cu)
endif()

if(TARGET tutorial)
  if(HPX_WITH_CUDA)
    add_hpx_pseudo_dependencies(tutorial.exercises.solutions solution9)
  endif()
endif()
