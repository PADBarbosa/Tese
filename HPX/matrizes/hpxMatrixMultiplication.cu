#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/wrap_main.hpp>

#include <cstddef>
#include <cstdlib>

using element_type = int;
using allocator_type = hpx::cuda::experimental::allocator<element_type>;
using executor_type = hpx::cuda::experimental::default_executor;


constexpr std::size_t n = 4;


int aux(hpx::compute::vector<int, allocator_type> &a, hpx::compute::vector<int, allocator_type> &b, hpx::compute::vector<int, allocator_type> &c, std::size_t &current){
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


    //Alocar as três matrizes no host
    hpx::compute::vector<element_type> a_host(n*n, 0);
    hpx::compute::vector<element_type> b_host(n*n, 0);
    hpx::compute::vector<element_type> c_host(n*n, 0);



    //Preencher as matrizes com valores aleatórios
    for(int i = 0; i < n*n; i++){
        a_host[i] = rand() % 10 + 1;
        b_host[i] = rand() % 10 + 1;
    }



    //Alocar as matrizes no device (GPU)
    std::vector<hpx::cuda::experimental::target> targets = hpx::cuda::experimental::get_local_targets();
    hpx::cuda::experimental::target device = targets[0];

    allocator_type alloc(device);
    executor_type exec(device);

    hpx::compute::vector<element_type, allocator_type> a_device(n*n, alloc);
    hpx::compute::vector<element_type, allocator_type> b_device(n*n, alloc);
    hpx::compute::vector<element_type, allocator_type> c_device(n*n, 0, alloc); //vai guardar os resultados da multiplicação da matriz A por B


    //Copiar as matrizes para o GPU
    //policy, inicioDe, fimDe, destino
    hpx::ranges::copy(hpx::execution::par, a_host.begin(), a_host.end(), a_device.begin());
    hpx::ranges::copy(hpx::execution::par, b_host.begin(), b_host.end(), a_device.begin());


    hpx::for_loop(hpx::execution::par.on(exec), 0, n,
        [&a_device, &b_device, &c_device] HPX_HOST_DEVICE(std::size_t num_thread) { aux(a_device, b_device, c_device, num_thread); });


    //Copiar a matriz do GPU para CPU
    //policy, inicioDe, fimDe, destino
    hpx::ranges::copy(hpx::execution::par, c_device, c_host.begin());

    std::cout << "------------Matriz C------------" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << c_host[i*n + j] << " ";
        }
        std::cout << std::endl;
    }


    return 0;
}