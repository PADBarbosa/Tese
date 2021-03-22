#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>

#include <iostream>
#include <cstdlib>

using element_type = int;
//using allocator_type = hpx::cuda::experimental::allocator<element_type>;
using executor_type = hpx::cuda::experimental::default_executor;
constexpr std::size_t n = 4;




int aux(hpx::compute::vector<element_type> &a, hpx::compute::vector<element_type> &b, hpx::compute::vector<element_type> &c, int &i){

    //multiplicação
    //for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            c[(i*n) + j] += a[(i*n) + k] * b[(k*n) + j]; 
        }
    }
    //}

    return 0;
}

int main(int argc, char const *argv[])
{
    hpx::compute::vector<element_type> a(n*n, 0);
    hpx::compute::vector<element_type> b(n*n, 0);
    hpx::compute::vector<element_type> c(n*n, 1);

    for(int i = 0; i < n*n; i++){
        a[i] = rand() % 10 + 1;
        b[i] = rand() % 10 + 1;
    }

    std::cout << "------------Matriz A------------" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << a[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "------------Matriz B------------" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << b[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    
    for(int i = 0; i < n; i++){
        hpx::ranges::for_each(hpx::execution::par, c,
            [&a, &b, &c, &i] () { return aux(a, b, c, i); });
    }

    std::cout << "------------Matriz C------------" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << c[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}