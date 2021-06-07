#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;


#define SIZE 130
#define STEPS 5e4
#define BLOCK_SIZE 16


void cubeCreator(int size, float* input) {
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



void dump(float* input, int size, int I) {
	if(input) {
		for(int i = 0;i < size; ++i) {
			for(int j = 0; j < size; ++j){
				for(int k = 0; k < size; ++k){
					if( k == I ){
						printf("%d %d %d %f \n",k,j,i,input[k+ j*size + i*size*size]);
						if(j == size -1){
							printf("\n");
						}
					}
				}
			}
		}
	}
}


int main (int argc, char* argv[]) {

	auto start = std::chrono::steady_clock::now();

	std::vector<hpx::lcos::future<void>> data_futures;

	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	

	float* input;
	cudaMallocHost((void**)&input, sizeof(float) * SIZE * SIZE * SIZE);
	checkCudaError("Malloc input");

	cubeCreator(SIZE, input);
	

	device cudaDevice = devices[0];

	buffer inbuffer = cudaDevice.create_buffer(sizeof(float) * SIZE * SIZE * SIZE).get();
	data_futures.push_back(inbuffer.enqueue_write(0, sizeof(float) * SIZE * SIZE * SIZE, input));

	program prog = cudaDevice.create_program_with_file("heat_diffusion_kernel.cu").get();

	// Add compiler flags for compiling the kernel
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	// Compile the program
	prog.build_sync(flags, "fdm3d");

	float* output;
	cudaMallocHost((void**)&output, sizeof(float) * SIZE * SIZE * SIZE);
	checkCudaError("Malloc output");

	cubeCreator(SIZE, output);

	buffer outbuffer = cudaDevice.create_buffer(sizeof(float) * SIZE * SIZE * SIZE).get();
	data_futures.push_back(outbuffer.enqueue_write(0, sizeof(float) * SIZE * SIZE * SIZE, output));


	// Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	// Set the values for the grid dimension
	grid.x = 8;
	grid.y = 8;
	grid.z = 1;

	// Set the values for the block dimension
	block.x = BLOCK_SIZE;
	block.y = BLOCK_SIZE;
	block.z = 1;



	int* n;
	cudaMallocHost((void**)&n, sizeof(int));
	checkCudaError("Malloc n");
	n[0] = SIZE;

	buffer n_buffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(n_buffer.enqueue_write(0, sizeof(int), n));



	int* m;
	cudaMallocHost((void**)&m, sizeof(int));
	checkCudaError("Malloc m");
	m[0] = SIZE;

	buffer m_buffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(m_buffer.enqueue_write(0, sizeof(int), m));



	float* r;
	cudaMallocHost((void**)&r, sizeof(float));
	checkCudaError("Malloc r");
	r[0] = 0.005;

	buffer r_buffer = cudaDevice.create_buffer(sizeof(float)).get();
	data_futures.push_back(r_buffer.enqueue_write(0, sizeof(float), r));


	// Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer> args;
	args.push_back(inbuffer);
	args.push_back(outbuffer);
	args.push_back(n_buffer);
	args.push_back(m_buffer);
	args.push_back(r_buffer);

	hpx::wait_all(data_futures);	

	float* res;
	cudaMallocHost((void**)&res, sizeof(float) * SIZE * SIZE * SIZE);

	for (int i = 0; i < 10; i++)	{
		std::vector<hpx::lcos::future<void>> data;
		auto kernel_future = prog.run(args, "fdm3d", grid, block, 3 * (sizeof(float) * (BLOCK_SIZE+2) * (BLOCK_SIZE+2)));

		wait_all(kernel_future);

		float* temp = 0;
		temp = input;
		res = outbuffer.enqueue_read_sync<float>(0, sizeof(float) * SIZE * SIZE * SIZE);
		input = res;
		output = temp;

		data.push_back(inbuffer.enqueue_write(0, sizeof(float) * SIZE * SIZE * SIZE, input));
		data.push_back(outbuffer.enqueue_write(0, sizeof(float) * SIZE * SIZE * SIZE, output));
		wait_all(data);
	}

	//hpx::wait_all(kernel_future);

	dump(input, SIZE, SIZE/3);

	return EXIT_SUCCESS;
}




