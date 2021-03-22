#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[])
{
    int *a, *b, *c;
    int n = 4;

    a = (int*) malloc (sizeof(int) * (n*n));
    b = (int*) malloc (sizeof(int) * (n*n));
    c = (int*) malloc (sizeof(int) * (n*n));

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


    //multiplicação
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[(i*n) + j] += a[(i*n) + k] * b[(k*n) + j]; 
            }
        }
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