#include <stdlib.h>

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

using namespace hpx::cuda;

//#define SIZE 130
#define SIZE 4
#define STEPS 50000
#define BLOCK_SIZE 2
//#define BLOCK_SIZE 16


void printMatriz(int* m){
	for(int i = 0; i < SIZE; i++){
		for (int j = 0; j < SIZE; j++){
			std::cout << m[i * SIZE + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "//////////////" << std::endl;
}

void fillMatriz(int* m){
	for(int i = 0; i < SIZE; i++){
		for (int j = 0; j < SIZE; j++){
			m[i * SIZE + j] = rand() % 10 + 1;
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

	

    int* matrizA;
	cudaMallocHost((void**)&matrizA, sizeof(int) * SIZE * SIZE);
	checkCudaError("Malloc inputData");
	fillMatriz(matrizA);
	printMatriz(matrizA);

	int* matrizB;
	cudaMallocHost((void**)&matrizB, sizeof(int) * SIZE * SIZE);
	checkCudaError("Malloc inputData");
	fillMatriz(matrizB);
	printMatriz(matrizB);



	device cudaDevice = devices[0];

	buffer mAbuffer = cudaDevice.create_buffer(sizeof(int) * SIZE * SIZE).get();
	buffer mBbuffer = cudaDevice.create_buffer(sizeof(int) * SIZE * SIZE).get();

	data_futures.push_back(mAbuffer.enqueue_write(0, sizeof(int) * SIZE * SIZE, matrizA));
	data_futures.push_back(mBbuffer.enqueue_write(0, sizeof(int) * SIZE * SIZE, matrizB));

	program prog = cudaDevice.create_program_with_file("matrix_mul_kernel.cu").get();


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Add compiler flags for compiling the kernel
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	// Compile the program
	prog.build_sync(flags, "multiplyMatrix");
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int* output;
	cudaMallocHost((void**)&output, sizeof(int) * SIZE * SIZE);
	checkCudaError("Malloc result");
	
	
	buffer outbuffer = cudaDevice.create_buffer(sizeof(int) * SIZE * SIZE).get();
	data_futures.push_back(outbuffer.enqueue_write(0, sizeof(int) * SIZE * SIZE, output));


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



	int* n;
	cudaMallocHost((void**)&n, sizeof(int));
	n[0] = SIZE;

	buffer sizebuffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(sizebuffer.enqueue_write(0, sizeof(int), n));



	// Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer> args;
	args.push_back(mAbuffer);
	args.push_back(mBbuffer);
	args.push_back(outbuffer);
	args.push_back(sizebuffer);

	hpx::wait_all(data_futures);

	//Run the kernel at the default stream
	auto kernel_future = prog.run(args, "multiplyMatrix", grid, block);

	hpx::wait_all(kernel_future);

	
	//Copy the result back
	int* res = outbuffer.enqueue_read_sync<int>(0, sizeof(int) * SIZE * SIZE);

	printMatriz(res);


	return EXIT_SUCCESS;
}

