#include <nvrtc.h>
#include <cuda.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "Constants.h"
#include "vector_types.h"
#include <chrono>



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


const char *nvidia_nbody = "                                            \n\
extern \"C\" {\n\
\n\
  __device__ void swap_args(float4* newPos, float4* oldPos) {\n\
    float4* temp = newPos;\n\
    newPos = oldPos;\n\
    oldPos = temp;\n\
  }\n\
\n\
  __device__ float my_rsqrt(float x) {\n\
    return rsqrtf(x);\n\
  }\n\
\n\
\n\
  __device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj) {\n\
      float3 r;\n\
\n\
      r.x = bj.x - bi.x;\n\
      r.y = bj.y - bi.y;\n\
      r.z = bj.z - bi.z;\n\
\n\
      float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;\n\
    distSqr += 0.00125f;\n\
\n\
\n\
      float invDist = my_rsqrt(distSqr);\n\
      float invDistCube =  invDist * invDist * invDist;\n\
\n\
      float s = bj.w * invDistCube;\n\
\n\
      ai.x += r.x * s;\n\
      ai.y += r.y * s;\n\
      ai.z += r.z * s;\n\
\n\
      return ai;\n\
  }\n\
\n\
\n\
  __device__ float3 computeBodyAccel(float4 bodyPos, float4* positions, int numTiles) {\n\
    __shared__ float4 sharedPos[256];\n\
\n\
\n\
      float3 acc = make_float3(0, 0, 0);\n\
      for (int tile = 0; tile < numTiles; tile++)\n\
      {\n\
          sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];\n\
\n\
          __syncthreads();\n\
\n\
  #pragma unroll 128\n\
\n\
          for (unsigned int counter = 0; counter < blockDim.x; counter++) {\n\
              acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);\n\
          }\n\
\n\
          __syncthreads();\n\
\n\
      }\n\
\n\
      return acc;\n\
  }\n\
\n\
\n\
  __global__ void integrateBodies(float4* newPos, float4* oldPos, float4* vel, int* deviceOffsetBuffer, int* deviceNumBodiesBuffer, float* deltaTimeBuffer, float* dampingBuffer, int* numTilesBuffer){\n\
\n\
      int deviceOffset = deviceOffsetBuffer[0];\n\
      int deviceNumBodies = deviceNumBodiesBuffer[0];\n\
      float deltaTime = deltaTimeBuffer[0];\n\
      float damping = dampingBuffer[0];\n\
      int numTiles = numTilesBuffer[0];\n\
      int index = blockIdx.x * blockDim.x + threadIdx.x;\n\
\n\
      if (index >= deviceNumBodies)\n\
      {\n\
          return;\n\
      }\n\
\n\
      float4 position = oldPos[deviceOffset + index];\n\
\n\
      \n\
      float3 accel = computeBodyAccel(position, oldPos, numTiles);\n\
\n\
      float4 velocity = vel[deviceOffset + index];\n\
\n\
      velocity.x += accel.x * deltaTime;\n\
      velocity.y += accel.y * deltaTime;\n\
      velocity.z += accel.z * deltaTime;\n\
\n\
      velocity.x *= damping;\n\
      velocity.y *= damping;\n\
      velocity.z *= damping;\n\
\n\
      position.x += velocity.x * deltaTime;\n\
      position.y += velocity.y * deltaTime;\n\
      position.z += velocity.z * deltaTime;\n\
\n\
      newPos[deviceOffset + index] = position;\n\
      vel[deviceOffset + index]    = velocity;\n\
  }\n\
\n\
\n\
}\n\
\n";



int main()
{
  std::cout << "Start" << std::endl;
  auto start = std::chrono::steady_clock::now();

  int numBodies = 4000000;
  int numTilesValue = 15625;
  int numBlocksValue = 15625;
  int iterationsValue = 10;

  std::cout << "numBodies: " << numBodies << std::endl;
  std::cout << "numTilesValue: " << numTilesValue << std::endl;
  std::cout << "numBlocksValue: " << numBlocksValue << std::endl;
  std::cout << "iterationsValue: " << iterationsValue << std::endl;



  auto start_compile = std::chrono::steady_clock::now();
  nvrtcProgram prog;

  NVRTC_SAFE_CALL(
  nvrtcCreateProgram(&prog,         // prog
             nvidia_nbody,         // buffer
             "nvidia_nbody.cuh",    // name
             0,             // numHeaders
             NULL,          // headers
             NULL));        // includeNames

  const char *opts[] = {"--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                          1,     // numOptions
                          opts); // options

  auto end_compile = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds_compile = end_compile - start_compile;
  std::cout << "elapsed time compile: " << elapsed_seconds_compile.count() << " s\n";

  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {
  exit(1);
  }

  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));


  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUfunction kernel_swap;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "integrateBodies"));
  //CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_swap, module, "swap_args"));


  CUdeviceptr d_oldPos;
  CUdeviceptr d_newPos;
  CUdeviceptr d_vel;
  CUdeviceptr d_deviceOffset;
  CUdeviceptr d_deviceNumBodies;
  CUdeviceptr d_deltaTime;
  CUdeviceptr d_damping;
  CUdeviceptr d_numTiles;
  //CUdeviceptr d_iterations;


  float4* h_oldPos = (float4*)malloc(sizeof(float) * numBodies * 4);
  float4* h_newPos = (float4*)malloc(sizeof(float) * numBodies * 4);
  float4* h_vel = (float4*)malloc(sizeof(float) * numBodies * 4);


  for(int i = 0; i < numBodies; i++){
    h_oldPos[i].x = 1*(i+1);
    h_oldPos[i].y = 2*(i+1);
    h_oldPos[i].z = 3*(i+1);
    h_oldPos[i].w = 4*(i+1);

    h_newPos[i].x = 1*(i+1);
    h_newPos[i].y = 2*(i+1);
    h_newPos[i].z = 3*(i+1);
    h_newPos[i].w = 4*(i+1);

    h_vel[i].x = 1*(i+1);
    h_vel[i].y = 2*(i+1);
    h_vel[i].z = 3*(i+1);
    h_vel[i].w = 4*(i+1);

  }

  int* deviceOffset = (int*)malloc(sizeof(int));
  deviceOffset[0] = 0;
  
  int* deviceNumBodies = (int*)malloc(sizeof(int));
  deviceNumBodies[0] = numBodies;
  
  float* deltaTime = (float*)malloc(sizeof(float));
  deltaTime[0] = 0.016;
  
  float* damping = (float*)malloc(sizeof(float));
  damping[0] = 1;
  
  int* numTiles = (int*)malloc(sizeof(int));
  numTiles[0] = numTilesValue;

  //int* iterations = (int*)malloc(sizeof(int));
  //iterations[0] = iterationsValue;



  CUDA_SAFE_CALL(cuMemAlloc(&d_oldPos, sizeof(float) * numBodies * 4));
  CUDA_SAFE_CALL(cuMemAlloc(&d_newPos, sizeof(float) * numBodies * 4));
  CUDA_SAFE_CALL(cuMemAlloc(&d_vel, sizeof(float) * numBodies * 4));
  CUDA_SAFE_CALL(cuMemAlloc(&d_deviceOffset, sizeof(int)));
  CUDA_SAFE_CALL(cuMemAlloc(&d_deviceNumBodies, sizeof(int)));
  CUDA_SAFE_CALL(cuMemAlloc(&d_deltaTime, sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&d_damping, sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&d_numTiles, sizeof(int)));
  //CUDA_SAFE_CALL(cuMemAlloc(&d_iterations, sizeof(int)));

  CUDA_SAFE_CALL(cuMemcpyHtoD(d_oldPos, h_oldPos, sizeof(float) * numBodies * 4));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_newPos, h_newPos, sizeof(float) * numBodies * 4));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_vel, h_vel, sizeof(float) * numBodies * 4));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_deviceOffset, deviceOffset, sizeof(int)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_deviceNumBodies, deviceNumBodies, sizeof(int)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_deltaTime, deltaTime, sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_damping, damping, sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_numTiles, numTiles, sizeof(int)));
  //CUDA_SAFE_CALL(cuMemcpyHtoD(d_iterations, iterations, sizeof(int)));
  
  void *args_1[] = { &d_newPos, &d_oldPos, &d_vel, &d_deviceOffset, &d_deviceNumBodies, &d_deltaTime, &d_damping, &d_numTiles };
  //void *args_2[] = { &d_newPos, &d_oldPos };

  //int currentRead = 0;

  //float4* h_tempPos = (float4*)malloc(sizeof(float) * 4 * numBodies);
  //float4* h_tempVel = (float4*)malloc(sizeof(float) * 4 * numBodies);

  for(int step = 0; step < iterationsValue; step++) {
    
    //std::cout << "---------- Iteracao " << step << " ----------" << std::endl;


    CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
                   numBlocksValue, 1, 1,    // grid dim
                   256, 1, 1,   // block dim
                   1024*sizeof(float), NULL,             // shared mem and stream
                   args_1, 0));           // arguments

    CUDA_SAFE_CALL(cuCtxSynchronize());

/*
  if(iterationsValue % 2 == 0){
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_tempPos, d_oldPos, sizeof(float) * numBodies * 4));
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_tempVel, d_vel, sizeof(float) * numBodies * 4));

    currentRead = 1;
  }
  else if(iterationsValue % 2 == 1){
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_tempPos, d_oldPos, sizeof(float) * numBodies * 4));
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_tempVel, d_vel, sizeof(float) * numBodies * 4));

    currentRead = 0;
  }


  for(int i = 0; i < 1; i++){
    std::cout << std::endl << "velX[" << i << "]: " << h_tempVel[i].x;
    std::cout << std::endl << "velY[" << i << "]: " << h_tempVel[i].y;
    std::cout << std::endl << "velZ[" << i << "]: " << h_tempVel[i].z;
    std::cout << std::endl << "velW[" << i << "]: " << h_tempVel[i].w;
    std::cout << std::endl;

    std::cout << std::endl << "posX[" << i << "]: " << h_tempPos[i].x;
    std::cout << std::endl << "posY[" << i << "]: " << h_tempPos[i].y;
    std::cout << std::endl << "posZ[" << i << "]: " << h_tempPos[i].z;
    std::cout << std::endl << "posW[" << i << "]: " << h_tempPos[i].w;
    std::cout << std::endl;
    std::cout << std::endl;
  }
*/
/*
    CUDA_SAFE_CALL(
    cuLaunchKernel(kernel_swap,
                   1, 1, 1,    // grid dim
                   1, 1, 1,   // block dim
                   0, NULL,             // shared mem and stream
                   args_1, 0));           // arguments*/
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << " s\n";
  
  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(d_oldPos));
  CUDA_SAFE_CALL(cuMemFree(d_newPos));
  CUDA_SAFE_CALL(cuMemFree(d_vel));
  CUDA_SAFE_CALL(cuMemFree(d_deviceOffset));
  CUDA_SAFE_CALL(cuMemFree(d_deviceNumBodies));
  CUDA_SAFE_CALL(cuMemFree(d_deltaTime));
  CUDA_SAFE_CALL(cuMemFree(d_damping));
  CUDA_SAFE_CALL(cuMemFree(d_numTiles));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] h_oldPos;
  delete[] h_newPos;
  delete[] h_vel;
  delete[] deviceOffset;
  delete[] deviceNumBodies;
  delete[] deltaTime;
  delete[] damping;
  delete[] numTiles;
  
  return 0;
}
