#include <nvrtc.h>
#include <cuda.h>
#include <iostream>

#define SIZE 130
#define STEPS 10
#define BLOCK_SIZE 16

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

const char *fdm3d = "                                            \n\
extern \"C\" __global__                                          \n\
/*                                                               \n\
 * fdm3d.cuh \n\
 * \n\
 *  Created on: 10-08-2013 \n\
 *      Author: Andrzej Biborski \n\
 * \n\
 */ \n\
#ifndef fdm3d_CUH_ \n\
#define fdm3d_CUH_ \n\
/* \n\
 *  r = d * delta_t/(delta_h * delta_h) \n\
 */ \n\
 \n\
 \n\
__global__ void fdm3d(const float * const input, float *output, int m, int n, float r) {  //é necessária distinção do M e N porque pode nao ser quadrado \n\
 \n\
  /* \n\
   * working slice matrix block internal[ m x n x 3] \n\
   */ \n\
 \n\
 \n\
  extern __shared__ float internal[]; \n\
  int offset = blockDim.x + 2; // o que é o offset? \n\
 \n\
  /* \n\
   * global indicies \n\
   */ \n\
  int thx = blockIdx.x * blockDim.x + threadIdx.x +1 ; // 1 reflects halo concept \n\
  int thy = blockIdx.y * blockDim.y + threadIdx.y +1 ; // 1 reflects halo concept \n\
 \n\
  /* \n\
   * local indicies \n\
   */ \n\
 \n\
  int lx = threadIdx.x + 1; \n\
  int ly = threadIdx.y + 1; \n\
 \n\
  int totel = m*n; //m = n = SIZE (neste caso, podem nao ser iguais), size*size, tamanho de um quadrado (paralelepipedo com 1 de profundidade) \n\
 \n\
  float up_z; //??? \n\
  float down_z; //??? \n\
 \n\
  /* \n\
   * global loop - iterate deep into the parallelepiped \n\
   */ \n\
 \n\
  // n-2 == SIZE-2 \n\
  for(int l = 0; l < n-2; ++l) { \n\
 \n\
    if(thx < m - 1 && thy < n -1) { //load slice \n\
 \n\
      internal[lx + ly * offset] = input[thx +thy*m + (l+1)*totel]; \n\
      up_z   = input[thx +thy*m + l*totel]; \n\
      down_z = input[thx +thy*m + (l+2)*totel]; \n\
 \n\
      if(lx == 1) { \n\
        internal[ly*offset] = input[thy * m +blockIdx.x*blockDim.x + (l+1)*totel]; \n\
      } \n\
 \n\
      if(ly == 1) { \n\
        internal[lx] = input[thx + (thy -1)* m   + (l+1)*totel]; \n\
      } \n\
 \n\
      if(lx == blockDim.x || thx == m -2) { \n\
        int distance = thx - blockIdx.x * blockDim.x; \n\
        internal[ly*offset + distance + 1] = input[thy * m + thx + 1 + (l+1)*totel]; \n\
      } \n\
 \n\
      if(ly == blockDim.y || thy == n -2) { \n\
        int distance = thy - blockIdx.y * blockDim.y; \n\
        internal[lx + (distance + 1) * offset] = input[ thx + thy*m + m + (l+1)*totel]; \n\
 \n\
      } \n\
      __syncthreads(); //sincroniza as threads de um bloco \n\
 \n\
      float central = internal[lx +ly * offset]; \n\
      float dx2 = \n\
        r*(internal[lx + ly *offset -1] - 2 * central + internal[lx + ly*offset + 1] ); \n\
 \n\
      float dy2 = \n\
        r*(internal[lx + ly *offset - offset] - 2 * central + internal[lx + ly*offset + offset] ); \n\
 \n\
      float dz2 = \n\
        r*(up_z - 2 * central + down_z ); \n\
 \n\
      output[thx + thy*m +(l+1)*totel] = central  + dx2 + dy2 + dz2; \n\
 \n\
      __syncthreads(); //sincroniza as threads de um bloco \n\
 \n\
    } \n\
  } \n\
 \n\
 \n\
} \n\
#endif /*fdm3d.cuh */    \n";


void cubeCreator(uint size, float *input) {
  for(int i = 0;i < size; ++i) {
    for(int j = 0; j < size; ++j){
      for(int k = 0; k < size; ++k){
        if(i != 0)
          input[k + j*size + i*size*size] = 0.0f;
        else
          input[k + j*size + i*size*size] = 100.0f;

      }
    }
  }
}

void dump(const float * const input, int size, int I) {
  if(input) {
    for(int i = 0;i < size; ++i) {
      for(int j = 0; j < size; ++j){
        for(int k = 0; k < size; ++k){
          if( k == I ){
            printf("%d %d %d %f \n",k,j,i,input[k+ j*size + i*size*size]);
            if(j == size -1)
              printf("\n");
            }
          }
        }
      }
    }
  }


int main()
{
  // Create an instance of nvrtcProgram with the SAXPY code string.
  nvrtcProgram prog;

  NVRTC_SAFE_CALL(
	nvrtcCreateProgram(&prog,         // prog
					   fdm3d,         // buffer
					   "fdm3d.cuh",    // name
					   0,             // numHeaders
					   NULL,          // headers
					   NULL));        // includeNames
  // Compile the program with fmad disabled.
  // Note: Can specify GPU target architecture explicitly with '-arch' flag.
  const char *opts[] = {"--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
												  1,     // numOptions
												  opts); // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {
	exit(1);
  }
  // Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  // Load the generated PTX and get a handle to the SAXPY kernel.
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "fdm3d"));

  // Generate input for execution, and create output buffers.
  float r = 0.005f;
  int size = SIZE;
  CUdeviceptr d_input;
  CUdeviceptr d_output;
  float* h_input = (float*)malloc(sizeof(float) * SIZE * SIZE *SIZE); //cubo no processador
  float* h_output = (float*)malloc(sizeof(float) * SIZE * SIZE *SIZE);
  cubeCreator(SIZE, h_input);


  CUDA_SAFE_CALL(cuMemAlloc(&d_input, sizeof(float) * SIZE * SIZE * SIZE));
  CUDA_SAFE_CALL(cuMemAlloc(&d_output, sizeof(float) * SIZE * SIZE * SIZE));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_input, h_input, sizeof(float) * SIZE * SIZE * SIZE));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_output, h_input, sizeof(float) * SIZE * SIZE * SIZE));


  /*size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);
  float a = 5.1f;
  float *hX = new float[n], *hY = new float[n], *hOut = new float[n];
  for (size_t i = 0; i < n; ++i) {
	hX[i] = static_cast<float>(i);
	hY[i] = static_cast<float>(i * 2);
  }
  CUdeviceptr dX, dY, dOut;
  CUDA_SAFE_CALL(cuMemAlloc(&dX, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dY, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));*/

  // Execute SAXPY.
  //void *args[] = { &a, &dX, &dY, &dOut, &n };
  
  void *args[] = { &d_input, &d_output, &size, &size, &r };

  //CUDA_SAFE_CALL(cuLaunchKernel(kernel, dim3(8, 8, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1), 3*(sizeof(float)*(BLOCK_SIZE+2)*(BLOCK_SIZE+2), NULL, args, 0));
  for(int i = 0; i < STEPS; ++i) {
    //CUDA_SAFE_CALL(cuLaunchKernel(kernel, 8, 8, 1, BLOCK_SIZE, BLOCK_SIZE, 1, args, 3*(sizeof(float)*(BLOCK_SIZE+2)*(BLOCK_SIZE+2))));

    CUDA_SAFE_CALL(
      cuLaunchKernel(kernel,
                   8, 8, 1,    // grid dim
                   BLOCK_SIZE, BLOCK_SIZE, 1,   // block dim
                   3*(sizeof(float)*(BLOCK_SIZE+2)*(BLOCK_SIZE+2)), NULL,             // shared mem and stream
                   args, 0));           // arguments
    
    CUDA_SAFE_CALL(cuCtxSynchronize());


    CUdeviceptr d_temp = 0;
    d_temp = d_input;
    d_input = d_output;
    d_output = d_temp;
  }

  // Retrieve and print output.
  CUDA_SAFE_CALL(cuMemcpyDtoH(h_output, d_output, sizeof(float) * SIZE * SIZE * SIZE));

  dump(h_output, SIZE, SIZE/3);

  /*for (size_t i = 0; i < n; ++i) {
	std::cout << a << " * " << hX[i] << " + " << hY[i]
			  << " = " << hOut[i] << '\n';
  }*/
  // Release resources.
  /*CUDA_SAFE_CALL(cuMemFree(dX));
  CUDA_SAFE_CALL(cuMemFree(dY));
  CUDA_SAFE_CALL(cuMemFree(dOut));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] hX;
  delete[] hY;
  delete[] hOut;*/
  return 0;
}