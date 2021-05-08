extern "C" { 
	__global__ void multiplyMatrix(int* matrizA, int* matrizB, int* res, int* sizebuffer) {
		int size = sizebuffer[0];

		int coluna = (blockIdx.x * blockDim.x) + threadIdx.x;
		int linha = (blockIdx.y * blockDim.y) + threadIdx.y;
		
		if (linha < size && coluna < size) {
			for (int i = 0; i < size; i++) {
				res[linha * size + coluna] += matrizA[linha * size + i] * matrizB[i * size + coluna];
			}
		}
		
	}
}
