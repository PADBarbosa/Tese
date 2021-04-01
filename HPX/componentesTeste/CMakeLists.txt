cmake_minimum_required(VERSION 3.0)
project(hpx_accumulator CXX)

find_package(HPX REQUIRED)

add_hpx_executable(program_hpx_accumulator
	SOURCES servidor/accumulator.cpp program_hpx_accumulator.cpp
	HEADERS servidor/accumulator.hpp cliente/accumulator_cliente.hpp
    COMPONENT_DEPENDENCIES iostreams)