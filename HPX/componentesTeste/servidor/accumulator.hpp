//#pragma once

#include <hpx/config.hpp>
//#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include <cstdint>

class accumulator: public hpx::components::locking_hook< hpx::components::component_base<accumulator> > {
	
	public:
		//typedef std::int64_t argument_type;

		//Que é isto??? Construtor?
        //accumulator() : value(0) {}

        void reset() {
            value = 0;
        }

        void add(int arg) {
            value += arg;
        }

        int query() const {
            return value;
        }

		HPX_DEFINE_COMPONENT_ACTION(accumulator, reset, reset_action_accumulator);
		HPX_DEFINE_COMPONENT_ACTION(accumulator, add, add_action_accumulator);
		HPX_DEFINE_COMPONENT_ACTION(accumulator, query, query_action_accumulator);
	
	private:
        int value;
};

HPX_REGISTER_ACTION_DECLARATION(accumulator::reset_action_accumulator);
HPX_REGISTER_ACTION_DECLARATION(accumulator::add_action_accumulator);
HPX_REGISTER_ACTION_DECLARATION(accumulator::query_action_accumulator);