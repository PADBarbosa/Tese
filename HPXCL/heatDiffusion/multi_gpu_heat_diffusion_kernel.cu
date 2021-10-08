

extern "C" {
	__global__ void fdm3d(float* input, float* output, int* mm, int* nn, float* rr) {

		const float3 r = make_float3(rr[0], rr[0], rr[0]);
		const int m = mm[0];
		const int n = nn[0];



		extern __shared__ float internal[];

		int offset = blockDim.x + 2;


		int thx = blockIdx.x * blockDim.x + threadIdx.x +1 ;
		int thy = blockIdx.y * blockDim.y + threadIdx.y +1 ;



		int lx = threadIdx.x + 1;
		int ly = threadIdx.y + 1;

		int totel = m*n;

		float up_z;
		float down_z;


		// Poderá ser necessário fazer alterações uma vez que o valor do m foi alterado para SIZE/2
		for(int l = 0; l < n-2; ++l) {
			if(thx < m - 1 && thy < n -1) {

				internal[lx + ly * offset] = input[thx +thy*m + (l+1)*totel];
				up_z   = input[thx +thy*m + l*totel];
				down_z = input[thx +thy*m + (l+2)*totel];

				if(lx == 1) { // início do block x
					internal[ly*offset] = input[thy * m +blockIdx.x*blockDim.x + (l+1)*totel];
				}

				if(ly == 1) { // início do block y
					internal[lx] = input[thx + (thy -1)* m   + (l+1)*totel];
				}

				if(lx == blockDim.x || thx == m -2) { // fim do block x ou do cubo, caso o tamanho do ultimo bloco nao seja igual ao dos restantes
					int distance = thx - blockIdx.x * blockDim.x;
					internal[ly*offset + distance + 1] = input[thy * m + thx + 1 + (l+1)*totel];
				}

				if(ly == blockDim.y || thy == n -2) { // fim do block y ou do cubo, caso o tamanho do ultimo bloco nao seja igual ao dos restantes
					int distance = thy - blockIdx.y * blockDim.y;
					internal[lx + (distance + 1) * offset] = input[ thx + thy*m + m + (l+1)*totel];

				}
				__syncthreads(); 
				float central = internal[lx +ly * offset];

				float dx2 = r.x*(internal[lx + ly *offset -1] - 2 * central + internal[lx + ly*offset + 1] );

				float dy2 = r.y*(internal[lx + ly *offset - offset] - 2 * central + internal[lx + ly*offset + offset] );

				float dz2 = r.z*(up_z - 2 * central + down_z );

				output[thx + thy*m +(l+1)*totel] = central  + dx2 + dy2 + dz2;

				__syncthreads(); 
			}

			// Copiar a penultima coluna para a posição da coluna que deveria ser comunicada pelo segundo GPU
			for(int i = 0; i < n; i++){
				for(int j = 0; j < n; j++){
					output[((n/2)+1) + j*((n/2)+1) + i*n*((n/2)+1)] = output[(n/2) + j*((n/2)+1) + i*n*((n/2)+1)];
				}
			}

		}
	}
}
