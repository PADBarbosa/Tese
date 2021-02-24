#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>


int fun1(hpx::cuda::experimental::default_executor exec, hpx::compute::vector<double, hpx::cuda::experimental::allocator<double>> &v)
{
    /*for(int i = 0; i < 100000000; i++){
        if(i == 99999999) std::cout << "oi" << std::endl;
    }*/
    
    hpx::ranges::for_each(hpx::execution::par.on(exec), v,
        [] HPX_HOST_DEVICE(double& x) { x *= 2.0; });

    return 0;
}


int main()
{
    using element_type = double;
    using allocator_type = hpx::cuda::experimental::allocator<element_type>;
    using executor_type = hpx::cuda::experimental::default_executor;

    constexpr std::size_t n = 1000000;


    hpx::cuda::experimental::target device;

    allocator_type alloc(device);
    executor_type exec(device);

    hpx::compute::vector<element_type, allocator_type> v(n, 2.0, alloc);
    hpx::compute::vector<element_type> vh(n, 0.0);


    hpx::shared_future<int> f = hpx::async([&exec, &v]() { return fun1(exec, v); });

    std::cout << vh[0] << std::endl;

    
    int z = f.get();
    //std::cout << vh[0] << std::endl;

    hpx::ranges::copy(hpx::execution::par, v, vh.begin());

    std::cout << vh[0] <<std::endl;

    return 0;
}