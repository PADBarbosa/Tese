#include <stdlib.h>

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

using namespace hpx::cuda;

//#define SIZE 130
#define SIZE 20
#define STEPS 50000
#define BLOCK_SIZE 4
//#define BLOCK_SIZE 16


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


void dump(const float* const input, int size, int I) {
	if(input) {
		for(int i = 0;i < size; ++i) {
			for(int j = 0; j < size; ++j){
				for(int k = 0; k < size; ++k){
					if( k == I ){
						std::cout << k << " " << j << " " << i << " :" << input[k + j * size + i * size * size] << std::endl;
						if(j == size -1){
							std::cout << std::endl;
						}
					}
				}
			}
		}
	}
}


int main(int argc, char* argv[]) {

	std::vector<hpx::lcos::future<void>> data_futures;

	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	

    float* inputCube;
	cudaMallocHost((void**)&inputCube, sizeof(float) * SIZE * SIZE * SIZE);
	checkCudaError("Malloc inputData");

	cubeCreator(SIZE, inputCube);

	device cudaDevice = devices[0];

	buffer inputCubeBuffer = cudaDevice.create_buffer(sizeof(float) * SIZE * SIZE * SIZE).get();

	data_futures.push_back(inputCubeBuffer.enqueue_write(0, sizeof(float) * SIZE * SIZE * SIZE, inputCube));

	program prog = cudaDevice.create_program_with_file("example_kernel.cu").get();


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Add compiler flags for compiling the kernel
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	// Compile the program
	prog.build_sync(flags, "fdm3d");
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float* outputCube;
	cudaMallocHost((void**)&outputCube, sizeof(float) * SIZE * SIZE * SIZE);
	checkCudaError("Malloc result");
	
	
	buffer outputCubeBuffer = cudaDevice.create_buffer(sizeof(float) * SIZE * SIZE).get();
	data_futures.push_back(outputCubeBuffer.enqueue_write(0, sizeof(float) * SIZE * SIZE * SIZE, outputCube));


	// Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	// Set the values for the grid dimension (número de blocks)
	grid.x = 2;
	grid.y = 2;
	grid.z = 1;

	// Set the values for the block dimension (número de threads por bloco)
	block.x = 2;
	block.y = 2;
	block.z = 1;


    //3*(sizeof(float)*(BLOCK_SIZE+2)*(BLOCK_SIZE+2))



	//Apenas 1 size porque vai ser cubo
	int* n;
	cudaMallocHost((void**)&n, sizeof(int));
	n[0] = SIZE;

	buffer sizeBuffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(sizeBuffer.enqueue_write(0, sizeof(int), n));


	float* r;
	cudaMallocHost((void**)&r, sizeof(float));
	r[0] = 0.005f; //fazer o float3 no GPU a partir deste valor

	buffer rBuffer = cudaDevice.create_buffer(sizeof(float)).get();
	data_futures.push_back(rBuffer.enqueue_write(0, sizeof(float), r));



	// Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer> args;
	args.push_back(inputCubeBuffer);
	args.push_back(outputCubeBuffer);
	args.push_back(sizeBuffer);
	args.push_back(rBuffer);

	hpx::wait_all(data_futures);

	//Run the kernel at the default stream
	auto kernel_future = prog.run(args, "fdm3d", grid, block, SIZE*SIZE*3);

	hpx::wait_all(kernel_future);

	
	//Copy the result back
	float* res = outputCubeBuffer.enqueue_read_sync<float>(0, sizeof(float) * SIZE * SIZE * SIZE);

	dump(res, SIZE, SIZE/3);


	return EXIT_SUCCESS;
}

