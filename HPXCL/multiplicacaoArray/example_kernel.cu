extern "C" { 

#include<cuda_profiler_api.h>

	__global__ void multiply(float* input, float* output, int* sizebuffer) {  //é necessária distinção do M e N porque pode nao ser quadrado
		int size = sizebuffer[0];

		for(int i = 0; i < size; i++){
			output[i] = input[i]*3;
		}
	}

	__global__ void multiply2(float* input, float* output, int* sizebuffer) {  //é necessária distinção do M e N porque pode nao ser quadrado
		int size = sizebuffer[0];

		for(int i = 0; i < size; i++){
			output[i] = input[i]*4;
		}
	}
}
