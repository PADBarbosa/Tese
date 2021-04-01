#include <hpx/config.hpp>
//#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/util.hpp>

#include "../servidor/accumulator.hpp"

#include <iostream>
#include <string>
#include <vector>


class accumulator_client: hpx::components::client_base<accumulator_client, accumulator>{
	public:
		//??????????????????????
		typedef hpx::components::client_base<accumulator_client, accumulator> base_type;
			accumulator_client() : base_type(hpx::local_new<accumulator>()){}
			accumulator_client(hpx::id_type locality) : base_type(hpx::new_<accumulator>(locality)){}


		hpx::future<void> Reset() {
			typedef accumulator::reset_action_accumulator action_type;
			return hpx::async<action_type>(this->get_id());
		}
		
		hpx::future<void> Add(int i) {
			typedef accumulator::add_action_accumulator action_type;
			return hpx::async<action_type>(this->get_id(), i);
		}

		hpx::future<int> Query() {
			typedef accumulator::query_action_accumulator action_type;
			return hpx::async<action_type>(this->get_id());
		}
};