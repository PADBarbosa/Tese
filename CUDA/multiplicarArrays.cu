#include <iostream>


__global__ void multiply(float* input, float* output, int size) {
	for(int i = 0; i < size; i++){
		output[i] = input[i]*3;
	}
}

__global__ void multiply2(float* input, float* output, int size) {
	for(int i = 0; i < size; i++){
		output[i] = input[i]*4;
	}
}

int main( void ) {
	int size = 8;
	
	float* a;
	a = (float*) malloc(sizeof(float)*size);
	for(int i = 0; i < size; i++){
		a[i] = 1;
		std::cout << a[i] << ", ";
	}
	std::cout << std::endl;

	float* b;
	b = (float*) malloc(sizeof(float)*size);

	float* c;
	c = (float*) malloc(sizeof(float)*size);


	float* dev_a;
	cudaMalloc( (void**)&dev_a, sizeof(float)*size );
	
	float* dev_b;
	cudaMalloc( (void**)&dev_b, sizeof(float)*size );
	
	float* dev_c;
	cudaMalloc( (void**)&dev_c, sizeof(float)*size );

	//destino source, size

	cudaMemcpy( &dev_a,	a, sizeof(float)*size, cudaMemcpyHostToDevice );
	multiply<<<1,1>>>( a, dev_b, size );
	cudaMemcpy( &b,	dev_b,	sizeof(float)*size, cudaMemcpyDeviceToHost );

	for(int i = 0; i < size; i++){
		std::cout << b[i] << ", ";
	}
	std::cout << std::endl;

	cudaMemcpy( &dev_a,	a, sizeof(float)*size, cudaMemcpyHostToDevice );
	multiply2<<<1,1>>>( a, dev_c, size );
	cudaMemcpy( &c,	dev_c,	sizeof(float)*size, cudaMemcpyDeviceToHost );

	for(int i = 0; i < size; i++){
		std::cout << c[i] << ", ";
	}
	std::cout << std::endl;


	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	return 0;
}