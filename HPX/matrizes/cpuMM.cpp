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




int aux(hpx::compute::vector<element_type> &a, hpx::compute::vector<element_type> &b, int &i){
    int res = 0;
    i++;

    for (int k = 0; k < n; k++) {
        std::cout << "i: " << i << " a: " << a[((i-1)/n) * n + k] << " b: " << b[((i-1)%n) + (k*n)] << std::endl;
        res += a[((i-1)/n) * n + k] * b[((i-1)%n) + (k*n)];
    }
    std::cout << "---------------" << std::endl;

    return res;
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

    
    int i = 0;
    hpx::ranges::for_each(hpx::execution::seq, c,
        [&a, &b, &i] (int& x) { x = aux(a, b, i); });

    std::cout << "i: " << i << std::endl;
    

    std::cout << "------------Matriz C------------" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << c[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}