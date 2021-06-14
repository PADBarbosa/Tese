/*
 * fdm3d.cuh
 *
 *  Created on: 10-08-2013
 *      Author: Andrzej Biborski
 *
 */
//#ifndef fdm3d_CUH_
//#define fdm3d_CUH_
/*
 *  r = d * delta_t/(delta_h * delta_h)
 */

extern "C" {
	__global__ void fdm3d(float* input, float* output, int* mm, int* nn, float* rr) {  //é necessária distinção do M e N porque pode nao ser quadrado

		const float3 r = make_float3(rr[0], rr[0], rr[0]);
		const int m = mm[0];
		const int n = nn[0];

		/*
		 * working slice matrix block internal[ m x n x 3]
		 */

		extern __shared__ float internal[];

		int offset = blockDim.x + 2; // o que é o offset?

		/*
		 * global indicies
		 */
		int thx = blockIdx.x * blockDim.x + threadIdx.x +1 ; // "1" reflects halo concept
		int thy = blockIdx.y * blockDim.y + threadIdx.y +1 ; // "1" reflects halo concept

		/*
		 * local indicies
		 */

		int lx = threadIdx.x + 1;
		int ly = threadIdx.y + 1;

		int totel = m*n; //m = n = SIZE (neste caso, podem nao ser iguais), size*size, tamanho de um quadrado (paralelepipedo com 1 de profundidade)

		float up_z; //???
		float down_z; //???

		/*
		 * global loop - iterate "deep" into the parallelepiped
		 */

		// n-2 == SIZE-2
		//for(int i = 0; i < 10; i++){
		for(int l = 0; l < n-2; ++l) {

			if(thx < m - 1 && thy < n -1) { //load slice

				internal[lx + ly * offset] = input[thx +thy*m + (l+1)*totel];
				up_z   = input[thx +thy*m + l*totel];
				down_z = input[thx +thy*m + (l+2)*totel];

				if(lx == 1) {
					internal[ly*offset] = input[thy * m +blockIdx.x*blockDim.x + (l+1)*totel];
				}

				if(ly == 1) {
					internal[lx] = input[thx + (thy -1)* m   + (l+1)*totel];
				}

				if(lx == blockDim.x || thx == m -2) {
					int distance = thx - blockIdx.x * blockDim.x;
					internal[ly*offset + distance + 1] = input[thy * m + thx + 1 + (l+1)*totel];
				}

				if(ly == blockDim.y || thy == n -2) {
					int distance = thy - blockIdx.y * blockDim.y;
					internal[lx + (distance + 1) * offset] = input[ thx + thy*m + m + (l+1)*totel];

				}
				__syncthreads(); //sincroniza as threads de um bloco

				float central = internal[lx +ly * offset];

				float dx2 = r.x*(internal[lx + ly *offset -1] - 2 * central + internal[lx + ly*offset + 1] );

				float dy2 = r.y*(internal[lx + ly *offset - offset] - 2 * central + internal[lx + ly*offset + offset] );

				float dz2 = r.z*(up_z - 2 * central + down_z );

				output[thx + thy*m +(l+1)*totel] = central  + dx2 + dy2 + dz2;

				__syncthreads(); //sincroniza as threads de um bloco

			}
		}


			/* Remove these lines and the outer for loop if using the for loop in the .cpp file,
			float* temp = 0;
			temp = input;
			input = output;
			output = temp;
			*/
		//}
	}

	__global__ void swap() {
		extern __shared__ int pointers[];

		int* temp = 0;
		temp = pointers[0];
		pointers[0] = pointers[1];
		pointers[1] = temp;
	}
}
