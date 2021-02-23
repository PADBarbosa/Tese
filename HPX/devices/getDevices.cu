//  Copyright (c) 2019-2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>
#include <iostream>

int main()
{
    std::vector<hpx::cuda::experimental::target> targets = hpx::cuda::experimental::get_local_targets();
    //hpx::cuda::experimental::target device = targets[0];

    //std::cout << device << std::endl;

    hpx::cuda::experimental::print_local_targets();

    
     


    return 0;
}
