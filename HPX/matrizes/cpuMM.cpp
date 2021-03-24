#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>

#include <iostream>
#include <cstdlib>

using element_type = int;
using executor_type = hpx::cuda::experimental::default_executor;
constexpr std::size_t n = 5;



int aux(hpx::compute::vector<element_type> &a, hpx::compute::vector<element_type> &b, hpx::compute::vector<element_type> &c, std::size_t &current){
    //std::cout << "Hello from thread " << current << std::endl;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            c[(current*n) + i] += a[(current*n) + j] * b[i + (j*n)];
        }
    }

    return 0;
}


int main(int argc, char const *argv[])
{
    hpx::compute::vector<element_type> a(n*n, 0);
    hpx::compute::vector<element_type> b(n*n, 0);
    hpx::compute::vector<element_type> c(n*n, 0);

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


    hpx::for_loop(hpx::execution::par, 0, n,
        [&a, &b, &c](std::size_t num_thread) { aux(a, b, c, num_thread); });


    std::cout << "------------Matriz C------------" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << c[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}