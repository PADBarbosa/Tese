#include <iostream>

//Usado para testar se o sistema consegue executar código CUDA sem problemas


__global__ void add( int a, int b, int* c ) {
	*c = a + b;
}

int main( void ) {
	int* c;
    cudaMallocHost((void**)&c, sizeof(int) * 1);


	int* dev_c;
	cudaMalloc( (void**)&dev_c, sizeof(int) );

	add<<<1,1>>>( 2, 11, dev_c );

	cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );

	printf( "2 + 11 = %d\n", *c );

	cudaFree( dev_c );

	return 0;
}