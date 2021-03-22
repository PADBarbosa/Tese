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


int fun1(hpx::compute::vector<int, allocator_type> &a, hpx::compute::vector<int, allocator_type> &b, hpx::compute::vector<int, allocator_type> &c, int value, hpx::compute::vector<int, allocator_type> &vetor)
{
    int row;
    int column;


    row = value/n; //arredondar para cima

    if((column % n) == 0){
        column = n;
    }
    else{
        column %= n;
    }


    //multiplicar linha por coluna e somar o valor à celula certa
    for(int i = 0; i < n; i++){
        c[ ((row - 1) * n + column) - 1] += a[((row - 1) * n) + i] * b[(i * n) + (column - 1)];
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

    hpx::compute::vector<element_type, allocator_type> a_device(n, alloc);
    hpx::compute::vector<element_type, allocator_type> b_device(n, alloc);
    hpx::compute::vector<element_type, allocator_type> c_device(n, 0, alloc); //vai guardar os resultados da multiplicação da matriz A por B
    hpx::compute::vector<element_type, allocator_type> vetor(n, 0, alloc);



    //Copiar as matrizes para o GPU
    //policy, inicioDe, fimDe, destino
    hpx::ranges::copy(hpx::execution::par, a_host.begin(), a_host.end(), a_device.begin());
    hpx::ranges::copy(hpx::execution::par, b_host.begin(), b_host.end(), a_device.begin());

    
    hpx::compute::vector<hpx::future<int>> futuros(n*n);

    int value = 0;

    for(int i = 0; i < n; i++){
        value = i;
        futuros[i] = hpx::async([&exec, &a_device, &b_device, &c_device, &value, &vetor]() { return fun1<<<1,1>>>(a_device, b_device, c_device, value, vetor); });
    }


    /*hpx::when_all(futuros[0], futuros[1], futuros[2], futuros[3], futuros[4], futuros[5], futuros[6], futuros[7], futuros[8], futuros[9], futuros[10], futuros[11], futuros[12], futuros[13], futuros[14], futuros[15]).then([](hpx::shared_future<void>) {
        std::cout << "Terminou" << std::endl;
    });*/


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