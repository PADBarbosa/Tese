#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>



int main()
{
    using element_type = double;

    constexpr std::size_t n = 10000000;

    hpx::compute::vector<element_type> v(n, 2.0);

    std::cout << v[0] <<std::endl;

    hpx::ranges::for_each(hpx::execution::par, v,
        [] (double& x) { x *= 2.0; });

    std::cout << v[0] <<std::endl;

    return 0;
}