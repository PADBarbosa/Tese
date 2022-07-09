#include <nvrtc.h>
#include <cuda.h>
#include <math.h>
#include <iostream>

#include "Constants.h"


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

const char *nbody = "                                            \n\
extern \"C\" {\n\
  #include \"Constants.h\"\n\
\n\
\n\
  __global__ void interactBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass) {\n\
    int i = blockDim.x * blockIdx.x + threadIdx.x;\n\
    if(i < NUM_BODIES)\n\
    {   \n\
      float Fx=0.0f; float Fy=0.0f; float Fz=0.0f;\n\
      float xposi=xpos[i];\n\
      float yposi=ypos[i];\n\
      float zposi=zpos[i];\n\
      #pragma unroll\n\
      for(int j=0; j < NUM_BODIES; j++)\n\
      {\n\
        if(i!=j)\n\
        { \n\
          vec3 posDiff;\n\
          posDiff.x = (xposi-xpos[j])*TO_METERS;\n\
          posDiff.y = (yposi-ypos[j])*TO_METERS;\n\
          posDiff.z = (zposi-zpos[j])*TO_METERS;\n\
          float dist = sqrt(posDiff.x*posDiff.x+posDiff.y*posDiff.y+posDiff.z*posDiff.z);\n\
          float F = TIME_STEP*(G*mass[i]*mass[j]) / ((dist*dist + SOFTENING*SOFTENING) * dist);\n\
          //float Fa = F/mass[i];\n\
          Fx-=F*posDiff.x;\n\
          Fy-=F*posDiff.y;\n\
          Fz-=F*posDiff.z;\n\
        } \n\
      }\n\
      xvel[i] += Fx/mass[i];\n\
      yvel[i] += Fy/mass[i];\n\
      zvel[i] += Fz/mass[i];\n\
      xpos[i] += TIME_STEP*xvel[i]/TO_METERS;\n\
      ypos[i] += TIME_STEP*yvel[i]/TO_METERS;\n\
      zpos[i] += TIME_STEP*zvel[i]/TO_METERS;\n\
    }\n\
  }\n\
\n\
  __global__ void renderClear(char* image, float* hdImage) {\n\
    for (int i=0; i<WIDTH*HEIGHT*3; i++) {\n\
      image[i] = 0;\n\
      hdImage[i] = 0.0;\n\
    }\n\
  }\n\
\n\
  __global__ void GPUrenderBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass, float* hdImage) {\n\
    /// ORTHOGONAL PROJECTION\n\
    int i = blockIdx.x*blockDim.x+threadIdx.x;\n\
    float velocityMax = MAX_VEL_COLOR; //35000\n\
    float velocityMin = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/\n\
          (SYSTEM_SIZE*TO_METERS)); //MIN_VEL_COLOR;\n\
    if(i<NUM_BODIES)\n\
    {\n\
      float vxsqr=xvel[i]*xvel[i];\n\
      float vysqr=yvel[i]*yvel[i];\n\
      float vzsqr=zvel[i]*zvel[i];\n\
      float vMag = sqrt(vxsqr+vysqr+vzsqr);\n\
      int x = (WIDTH/2.0)*(1.0+xpos[i]/(SYSTEM_SIZE*RENDER_SCALE));\n\
      int y = (HEIGHT/2.0)*(1.0+ypos[i]/(SYSTEM_SIZE*RENDER_SCALE));\n\
\n\
      if (x>DOT_SIZE && x<WIDTH-DOT_SIZE && y>DOT_SIZE && y<HEIGHT-DOT_SIZE)\n\
      {\n\
        float vPortion = sqrt((vMag-velocityMin) / velocityMax);\n\
        float xPixel = (WIDTH/2.0)*(1.0+xpos[i]/(SYSTEM_SIZE*RENDER_SCALE));\n\
        float yPixel = (HEIGHT/2.0)*(1.0+ypos[i]/(SYSTEM_SIZE*RENDER_SCALE));\n\
        float xP = floor(xPixel);\n\
        float yP = floor(yPixel);\n\
        color c;\n\
        c.r = max(min(4*(vPortion-0.333),1.0),0.0);\n\
        c.g = max(min(min(4*vPortion,4.0*(1.0-vPortion)),1.0),0.0);\n\
        c.b = max(min(4*(0.5-vPortion),1.0),0.0);\n\
        for (int a=-DOT_SIZE/2; a<DOT_SIZE/2; a++)\n\
        {\n\
          for (int b=-DOT_SIZE/2; b<DOT_SIZE/2; b++)\n\
          {\n\
            float cFactor = PARTICLE_BRIGHTNESS /(pow(exp(pow(PARTICLE_SHARPNESS*(xP+a-xPixel),2.0)) + exp(pow(PARTICLE_SHARPNESS*(yP+b-yPixel),2.0)),/*1.25*/0.75)+1.0);\n\
            //colorAt(int(xP+a),int(yP+b),c, cFactor, hdImage);\n\
            int pix = 3*(xP+a+WIDTH*(yP+b));\n\
            hdImage[pix+0] += c.r*cFactor;\n\
            hdImage[pix+1] += c.g*cFactor;\n\
            hdImage[pix+2] += c.b*cFactor;\n\
          }\n\
        }\n\
      }\n\
    }\n\
  }\n\
}\n";

void initializeBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass){
  using std::uniform_real_distribution;
  uniform_real_distribution<float> randAngle (0.0, 200.0*PI);
  uniform_real_distribution<float> randRadius (INNER_BOUND, SYSTEM_SIZE);
  uniform_real_distribution<float> randHeight (0.0, SYSTEM_THICKNESS);
  std::default_random_engine gen (0);
  float angle;
  float radius;
  float velocity;

  //STARS
  velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
  //STAR 1
  xpos[0] = 0.0;///-BINARY_SEPARATION;
  ypos[0] = 0.0;
  zpos[0] = 0.0;
  xvel[0] = 0.0;
  yvel[0] = 0.0;//velocity;
  zvel[0] = 0.0;
  mass[0] = SOLAR_MASS;

    ///STARTS AT NUMBER OF STARS///
  float totalExtraMass = 0.0;
  for (int i=1; i<NUM_BODIES; i++)
  {
    angle = randAngle(gen);
    radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
    velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
                    / (radius*TO_METERS)), 0.5);
    xpos[i] =  radius*cos(angle);
    ypos[i] =  radius*sin(angle);
    zpos[i] =  randHeight(gen)-SYSTEM_THICKNESS/2;
    xvel[i] =  velocity*sin(angle);
    yvel[i] = -velocity*cos(angle);
    zvel[i] =  0.0;
    mass[i] = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
    totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
  }
  std::cout << "\nTotal Disk Mass: " << totalExtraMass;
  std::cout << "\nEach Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
        << "\n______________________________\n";
}

void renderClear(char* image, float* hdImage) {
  for (int i=0; i<WIDTH*HEIGHT*3; i++) {
    image[i] = 0;
    hdImage[i] = 0.0;
  }
}

float min(float x, float y){
  return !(y<x)?x:y;
}

float max(float x, float y){
  return (y<x)?x:y;
}

float clamp(float x) {
  return max(min(x,1.0), 0.0);
}

void writeRender(char* data, float* hdImage, int step) {
  
  for (int i=0; i<HEIGHT*WIDTH*3; i++)
  {
    data[i] = int(255.0*clamp(hdImage[i]));
  }

  int frame = step/RENDER_INTERVAL + 1;//RENDER_INTERVAL;
  std::string name = "images/Step"; 
  int i = 0;
  if (frame == 1000) i++; // Evil hack to avoid extra 0 at 1000
  for (i; i<4-floor(log(frame)/log(10)); i++)
  {
    name.append("0");
  }
  name.append(std::to_string(frame));
  name.append(".ppm");

  std::ofstream file (name, std::ofstream::binary);

  if (file.is_open())
  {
//    size = file.tellg();
    file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
    file.write(data, WIDTH*HEIGHT*3);
    file.close();
  }

}


int main()
{
  nvrtcProgram prog;

  NVRTC_SAFE_CALL(
	nvrtcCreateProgram(&prog,         // prog
					   nbody,         // buffer
					   "nbody.cuh",    // name
					   0,             // numHeaders
					   NULL,          // headers
					   NULL));        // includeNames

  const char *opts[] = {"--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
												  1,     // numOptions
												  opts); // options

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
  CUfunction kernel_1;
  CUfunction kernel_2;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_1, module, "interactBodies"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_2, module, "GPUrenderBodies"));

  CUdeviceptr d_xPos;
  CUdeviceptr d_yPos;
  CUdeviceptr d_zPos;
  CUdeviceptr d_xVel;
  CUdeviceptr d_yVel;
  CUdeviceptr d_zVel;
  CUdeviceptr d_mass;
  CUdeviceptr d_image;
  CUdeviceptr d_hdImage;


  float* h_xPos = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_yPos = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_zPos = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_xVel = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_yVel = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_zVel = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_mass = (float*)malloc(sizeof(float) * NUM_BODIES);
  float* h_image = (float*)malloc(sizeof(float) * WIDTH * HEIGHT * 3);
  float* h_hdImage = (float*)malloc(sizeof(float) * WIDTH * HEIGHT * 3);

  initializeBodies(h_xPos, h_yPos, h_zPos, h_xVel, h_yVel, h_zVel, h_mass);

  renderClear(h_image, h_hdImage);


  CUDA_SAFE_CALL(cuMemAlloc(&d_xPos, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_xPos, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_xPos, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_xVel, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_xVel, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_xVel, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_mass, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemAlloc(&d_image, sizeof(float) * WIDTH * HEIGHT * 3));
  CUDA_SAFE_CALL(cuMemAlloc(&d_hdImage, sizeof(float) * WIDTH * HEIGHT * 3));

  CUDA_SAFE_CALL(cuMemcpyHtoD(d_xPos, h_xPos, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_xPos, h_xPos, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_xPos, h_xPos, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_xVel, h_xVel, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_xVel, h_xVel, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_xVel, h_xVel, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_mass, h_mass, sizeof(float) * NUM_BODIES));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_image, h_image, sizeof(float) * WIDTH * HEIGHT * 3));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_hdImage, h_hdImage, sizeof(float) * WIDTH * HEIGHT * 3));
  
  void *args_1[] = { &d_xPos, &d_yPos, &d_zPos, &d_xVel, &d_yVel, &d_zVel, &d_mass };
  void *args_2[] = { &d_xPos, &d_yPos, &d_zPos, &d_xVel, &d_yVel, &d_zVel, &d_mass, &d_hdImage };

  for(int i = 0; i < STEP_COUNT; i++) {

    CUDA_SAFE_CALL(
      cuLaunchKernel(kernel_1,
                   1024, 1, 1,    // grid dim
                   (NUM_BODIES+1024-1)/1024, 1, 1,   // block dim
                   0, NULL,             // shared mem and stream
                   args_1, 0));           // arguments
    
    CUDA_SAFE_CALL(cuCtxSynchronize());


    CUDA_SAFE_CALL(
      cuLaunchKernel(kernel_2,
                   1025, 1, 1,    // grid dim
                   ((NUM_BODIES+1024-1)/1024)+1, 1, 1,   // block dim
                   0, NULL,             // shared mem and stream
                   args_2, 0));           // arguments
    
    CUDA_SAFE_CALL(cuCtxSynchronize());

  }

  // Retrieve and print output.
  CUDA_SAFE_CALL(cuMemcpyDtoH(h_xPos, d_xPos, sizeof(float) * NUM_BODIES));



  
  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(d_xPos));
  CUDA_SAFE_CALL(cuMemFree(d_yPos));
  CUDA_SAFE_CALL(cuMemFree(d_zPos));
  CUDA_SAFE_CALL(cuMemFree(d_xVel));
  CUDA_SAFE_CALL(cuMemFree(d_yVel));
  CUDA_SAFE_CALL(cuMemFree(d_zVel));
  CUDA_SAFE_CALL(cuMemFree(d_mass));
  CUDA_SAFE_CALL(cuMemFree(d_hdImage));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] h_xPos;
  delete[] h_yPos;
  delete[] h_zPos;
  delete[] h_xVel;
  delete[] h_yVel;
  delete[] h_zVel;
  delete[] h_mass;
  delete[] h_image;
  delete[] h_hdImage;
  
  return 0;
}