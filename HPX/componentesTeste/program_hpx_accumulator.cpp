#include "cliente/accumulator_cliente.hpp"
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <iostream>


int main(int argc, char * argv[]) {

	hpx::id_type locality = hpx::find_here();

	accumulator_client myaccumulator(locality);

	auto f1 = myaccumulator.Add(1);

	
	auto f2 = myaccumulator.Add(23);
	
	auto f3 = hpx::dataflow([&myaccumulator](auto f1, auto f2){
		auto element = myaccumulator.Query();
		std::cout << element.get() << std::endl; // irá imprimir 24
	}, f1, f2);


	f3.then([&myaccumulator](auto f3){
		myaccumulator.Reset();
		auto element = myaccumulator.Query();
		std::cout << element.get() << std::endl; // irá imprimir 0
	}).get();

}