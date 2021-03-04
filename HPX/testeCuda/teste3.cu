#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>


int fun1(hpx::cuda::experimental::default_executor exec, hpx::compute::vector<double, hpx::cuda::experimental::allocator<double>> &v)
{
    /*for(int i = 0; i < 100000000; i++){
        if(i == 99999999) std::cout << "test" << std::endl;
    }*/
    
    hpx::ranges::for_each(hpx::execution::par.on(exec), v,
        [] HPX_HOST_DEVICE(double& x) { x *= 2.0; });

    return 0;
}

int fun2(hpx::compute::vector<double> &v)
{
    /*for(int i = 0; i < 100000000; i++){
        if(i == 99999999) std::cout << "test" << std::endl;
    }*/
    
    hpx::ranges::for_each(hpx::execution::par, v,
        [] (double& x) { x *= 2.0; });

    return 0;
}




int main()
{
    std::cout << "--------------TESTE3--------------" << std::endl;

    using element_type = double;
    using allocator_type = hpx::cuda::experimental::allocator<element_type>;
    using executor_type = hpx::cuda::experimental::default_executor;

    constexpr std::size_t n = 10000000;


    hpx::cuda::experimental::target device;

    allocator_type alloc(device);
    executor_type exec(device);

    hpx::compute::vector<element_type> vh(n, 2.0);

    hpx::compute::vector<element_type, allocator_type> v(n/2, alloc);
    //policy, inicioDe, fimDe, destino
    hpx::ranges::copy(hpx::execution::par, vh.begin()+(n/2), vh.end(), v.begin());



    hpx::shared_future<int> f = hpx::async([&exec, &v]() { return fun1(exec, v); });

    std::cout << vh[0] << std::endl;


    for(int i = 0; i < n/2; i++){
        vh[i] *= 2.0;
    }
    //hpx::shared_future<int> g = hpx::async( hpx::for_each(hpx::execution::par, vh.begin(), vh.begin()+(n/2), [] (double& x) { x *= 2.0; }));
    //hpx::shared_future<int> g = hpx::async([&vh]() { return fun2(vh); });


    
    int z = f.get();
    //int zz = g.get();

    //policy, inicioDe, fimDe, destino
    hpx::ranges::copy(hpx::execution::par, v.begin(), v.end(), vh.begin()+(n/2));

    std::cout << vh[0] <<std::endl;
    std::cout << vh[n-1] <<std::endl;

    for(int i=0; i < n; i++){
        if(vh[i] != 4){
            std::cout << "erro" << std::endl;
        }
    }

    return 0;
}