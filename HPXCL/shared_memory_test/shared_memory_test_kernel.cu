extern "C" { 
	__global__ void dynamicReverse(int* d, int* sizebuffer){
		int n = sizebuffer[0];

		extern __shared__ int s[64];

		int t = threadIdx.x;
		int tr = n-t-1;
		
		//d[t] = d[tr];
		
		s[t] = d[t];
		
		__syncthreads();

		d[t] = s[tr];
	}
}