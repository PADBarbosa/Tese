#include <hpx/config.hpp>
//#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>

#include "accumulator.hpp"


typedef hpx::components::component<accumulator> accumulator_type;

HPX_REGISTER_COMPONENT(accumulator_type, accumulator);

HPX_REGISTER_ACTION(accumulator::reset_action_accumulator);
HPX_REGISTER_ACTION(accumulator::add_action_accumulator);
HPX_REGISTER_ACTION(accumulator::query_action_accumulator);