#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;

#define SIZE 24

int main(int argc, char* argv[]) {

	auto start = std::chrono::steady_clock::now();

	std::vector<hpx::lcos::future<void>> data_futures_1;

	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	

    int* input_1;
	cudaMallocHost((void**)&input_1, sizeof(int) * SIZE);
	checkCudaError("Malloc inputData");

	std::cout << "SIZE: " << SIZE << std::endl;

	for(int i = 0; i < SIZE; i++){
		input_1[i] = 1;
	}

	/*for(int i = SIZE/2; i < SIZE; i++){
		input_1[i] = 2;
	}*/

	/*std::cout << "INPUT 1:" << std::endl;
	for (int i = 0; i < SIZE-1; i++){
		std::cout << input_1[i] << ","
	}
	std::cout << input_1[SIZE-1] << std::endl;*/

	device cudaDevice_1 = devices[0];
	device cudaDevice_2 = devices[1];

	buffer inbuffer_1 = cudaDevice_1.create_buffer(sizeof(int) * SIZE/2).get();
	buffer inbuffer_2 = cudaDevice_2.create_buffer(sizeof(int) * SIZE/2).get();

	data_futures_1.push_back(inbuffer_1.enqueue_write(0, sizeof(int) * SIZE/2, input_1));
	data_futures_1.push_back(inbuffer_2.enqueue_write(sizeof(int) * SIZE/2, sizeof(int) * SIZE/2, input_1)); // Passar segunda metade do input





	program prog_1 = cudaDevice_1.create_program_with_file("array_mul_kernel.cu").get();
	program prog_2 = cudaDevice_2.create_program_with_file("array_mul_kernel.cu").get();


	// Add compiler flags for compiling the kernel
	std::vector<std::string> flags_1;
	std::string mode_1 = "--gpu-architecture=compute_";
	mode_1.append(std::to_string(cudaDevice_1.get_device_architecture_major().get()));
	mode_1.append(std::to_string(cudaDevice_1.get_device_architecture_minor().get()));
	flags_1.push_back(mode_1);

	// Add compiler flags for compiling the kernel
	std::vector<std::string> flags_2;
	std::string mode_2 = "--gpu-architecture=compute_";
	mode_2.append(std::to_string(cudaDevice_2.get_device_architecture_major().get()));
	mode_2.append(std::to_string(cudaDevice_2.get_device_architecture_minor().get()));
	flags_2.push_back(mode_2);
	
	// Compile the program
	prog_1.build_sync(flags_1, "multiply");
	prog_2.build_sync(flags_2, "multiply2");

    int* output_1;
	cudaMallocHost((void**)&output_1, sizeof(int) * SIZE/2);
	checkCudaError("Malloc result");

	int* output_2;
	cudaMallocHost((void**)&output_2, sizeof(int) * SIZE/2);
	checkCudaError("Malloc result");
	
	
	buffer outbuffer_1 = cudaDevice_1.create_buffer(sizeof(int) * SIZE/2).get();
	buffer outbuffer_2 = cudaDevice_2.create_buffer(sizeof(int) * SIZE/2).get();
	data_futures_1.push_back(outbuffer_1.enqueue_write(0, sizeof(int) * SIZE/2, output_1));
	data_futures_1.push_back(outbuffer_2.enqueue_write(0, sizeof(int) * SIZE/2, output_2));


	// Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	// Set the values for the grid dimension
	grid.x = 1;
	grid.y = 1;
	grid.z = 1;

	// Set the values for the block dimension
	block.x = 1;
	block.y = 1;
	block.z = 1;



	int* n_1;
	cudaMallocHost((void**)&n_1, sizeof(int));
	n_1[0] = SIZE/2;

	buffer sizebuffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
	data_futures_1.push_back(sizebuffer_1.enqueue_write(0, sizeof(int), n_1));


	int* n_2;
	cudaMallocHost((void**)&n_2, sizeof(int));
	n_2[0] = SIZE/2;

	buffer sizebuffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
	data_futures_1.push_back(sizebuffer_2.enqueue_write(0, sizeof(int), n_2));



	// Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer> args_1;
	args_1.push_back(inbuffer_1);
	args_1.push_back(outbuffer_1);
	args_1.push_back(sizebuffer_1);

	std::vector<hpx::cuda::buffer> args_2;
	args_2.push_back(inbuffer_2);
	args_2.push_back(outbuffer_2);
	args_2.push_back(sizebuffer_2);

	hpx::wait_all(data_futures_1);

	//Run the kernel at the default stream
	auto kernel_future_1 = prog_1.run(args_1, "multiply", grid, block);
	auto kernel_future_2 = prog_2.run(args_2, "multiply2", grid, block);


	

	/*std::cout << "RES 1:" << std::endl;
	for (int i = 0; i < SIZE-1; i++){
		std::cout << res_1[i] << ", ";
	}
	std::cout << res_1[SIZE-1] << std::endl;*/


	hpx::wait_all(kernel_future_1);
	int* res_1 = outbuffer_1.enqueue_read_sync<int>(0, sizeof(int) * SIZE/2);

	hpx::wait_all(kernel_future_2);
	int* res_2 = outbuffer_2.enqueue_read_sync<int>(0, sizeof(int) * SIZE/2);

	int* res = (int*) malloc(sizeof(int) * SIZE);

	for(int i = 0; i < SIZE/2; i++){
		res[i] = res_1[i];
		res[SIZE/2+i] = res_2[i];
	}


	std::cout << "RES 1:" << std::endl;
	for (int i = 0; i < SIZE/2-1; i++){
		std::cout << res_1[i] << ", ";
	}
	std::cout << res_1[SIZE/2-1] << std::endl;


	std::cout << "RES 2:" << std::endl;
	for (int i = 0; i < SIZE/2-1; i++){
		std::cout << res_2[i] << ", ";
	}
	std::cout << res_2[SIZE/2-1] << std::endl;


	/*std::cout << "RES:" << std::endl;
	for (int i = 0; i < SIZE-1; i++){
		std::cout << res[i] << ", ";
	}
	std::cout << res[SIZE-1] << std::endl;*/



	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}

