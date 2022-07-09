#include <iostream>
#include <chrono>

void cudaCheckError() {
	cudaError_t e=cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}
}

__global__ void multiply(float* input, float* output, int size) {
	for(int i = 0; i < size/blockDim.x; i++){
		output[threadIdx.x * (size/blockDim.x) + i] = input[threadIdx.x * (size/blockDim.x) + i]*2;
		//output[i] = input[i]*2;
	}
}

__global__ void multiply2(float* input, float* output, int size) {
	for(int i = 0; i < size/blockDim.x; i++){
		output[threadIdx.x * (size/blockDim.x) + i] = input[threadIdx.x * (size/blockDim.x) + i]*4;
	}
}


int main( void ) {
	int size = 8;

	auto start = std::chrono::steady_clock::now();
	
	float* a;
	a = (float*) malloc(sizeof(float)*size);
	cudaCheckError();

	for(int i = 0; i < size; i++){
		a[i] = 1;
		std::cout << a[i] << ", ";
	}
	std::cout << std::endl;

	float* b;
	b = (float*) malloc(sizeof(float)*size);
	cudaCheckError();

	float* c;
	c = (float*) malloc(sizeof(float)*size);
	cudaCheckError();



	float* dev_a;
	cudaMalloc( (void**)&dev_a, sizeof(float)*size );
	cudaCheckError();

	float* dev_b;
	cudaMalloc( (void**)&dev_b, sizeof(float)*size );
	cudaCheckError();

	float* dev_c;
	cudaMalloc( (void**)&dev_c, sizeof(float)*size );
	cudaCheckError();



	//destino source, size

	cudaMemcpy( dev_a, a, sizeof(float)*size, cudaMemcpyHostToDevice );
	cudaCheckError();

	cudaMemcpy( dev_b, a, sizeof(float)*size, cudaMemcpyHostToDevice );
	cudaCheckError();

	multiply<<<1,1>>>( dev_a, dev_b, size );
	cudaCheckError();


	cudaMemcpy( b, dev_b, sizeof(float)*size, cudaMemcpyDeviceToHost );
	cudaCheckError();

	for(int i = 0; i < size; i++){
		std::cout << b[i] << ", ";
	}
	std::cout << std::endl;





	cudaMemcpy( dev_a, a, sizeof(float)*size, cudaMemcpyHostToDevice );
	cudaCheckError();
	
	multiply2<<<1,1>>>( dev_a, dev_c, size );
	
	cudaMemcpy( c, dev_c, sizeof(float)*size, cudaMemcpyDeviceToHost );
	cudaCheckError();

	for(int i = 0; i < size; i++){
		std::cout << c[i] << ", ";
	}
	std::cout << std::endl;

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	return 0;
}