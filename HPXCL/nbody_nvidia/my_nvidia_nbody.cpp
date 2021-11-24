#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;


int main (int argc, char* argv[]) {
	std::cout << "Start" << std::endl;
	auto start = std::chrono::steady_clock::now();

	std::vector<hpx::lcos::future<void>> data_futures;

	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	device cudaDevice = devices[0];

	float* hPos_new;
	cudaMallocHost((void**)&hPos_new, sizeof(float) * 4 * 69632);
	checkCudaError("Malloc hPos_new");

	float* hPos_old;
	cudaMallocHost((void**)&hPos_old, sizeof(float) * 4 * 69632);
	checkCudaError("Malloc hPos_old");
	
	float* hVel;
	cudaMallocHost((void**)&hVel, sizeof(float) * 4 * 69632);
	checkCudaError("Malloc hVel");

	for(int i = 0; i < 4 * 69632; i++){
		hPos_new[i] = i;
		hPos_old[i] = i;
		hVel[i] = i;
	}

	buffer dPos_new_buffer = cudaDevice.create_buffer(sizeof(float) * 4 * 69632).get();
	data_futures.push_back(dPos_new_buffer.enqueue_write(0, sizeof(float) * 4 * 69632, hPos_new));

	buffer dPos_old_buffer = cudaDevice.create_buffer(sizeof(float) * 4 * 69632).get();
	data_futures.push_back(dPos_old_buffer.enqueue_write(0, sizeof(float) * 4 * 69632, hPos_old));

	buffer dVel_buffer = cudaDevice.create_buffer(sizeof(float) * 4 * 69632).get();
	data_futures.push_back(dVel_buffer.enqueue_write(0, sizeof(float) * 4 * 69632, hVel));

/*
	int* deviceOffset;
	cudaMallocHost((void**)&deviceOffset, sizeof(int));
	checkCudaError("Malloc deviceOffset");
	deviceOffset[0] = 0;

	buffer dOffset_buffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(dOffset_buffer.enqueue_write(0, sizeof(int), deviceOffset));


	int* deviceNumBodies;
	cudaMallocHost((void**)&deviceNumBodies, sizeof(int));
	checkCudaError("Malloc deviceNumBodies");
	deviceNumBodies[0] = 69632;

	buffer dNumBodies_buffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(dNumBodies_buffer.enqueue_write(0, sizeof(int), deviceNumBodies));


	float* deltaTime;
	cudaMallocHost((void**)&deltaTime, sizeof(float));
	checkCudaError("Malloc deltaTime");
	deltaTime[0] = 0.016;

	buffer dDeltaTime_buffer = cudaDevice.create_buffer(sizeof(float)).get();
	data_futures.push_back(dDeltaTime_buffer.enqueue_write(0, sizeof(float), deltaTime));


	float* damping;
	cudaMallocHost((void**)&damping, sizeof(float));
	checkCudaError("Malloc damping");
	damping[0] = 256;

	buffer dDamping_buffer = cudaDevice.create_buffer(sizeof(float)).get();
	data_futures.push_back(dDamping_buffer.enqueue_write(0, sizeof(float), damping));


	int* numTiles;
	cudaMallocHost((void**)&numTiles, sizeof(int));
	checkCudaError("Malloc numTiles");
	numTiles[0] = 272;

	buffer dNumTiles_buffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(dNumTiles_buffer.enqueue_write(0, sizeof(int), numTiles));
*/
	
	program prog = cudaDevice.create_program_with_file("my_nvidia_nbody_kernel.cu").get();

	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	prog.build_sync(flags, "integrateBodies");


	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	grid.x = 272;
	grid.y = 1;
	grid.z = 1;

	block.x = 256;
	block.y = 1;
	block.z = 1;


	std::vector<hpx::cuda::buffer> args;
	args.push_back(dPos_new_buffer);
	args.push_back(dPos_old_buffer);
	args.push_back(dVel_buffer);
	/*args.push_back(dOffset_buffer);
	args.push_back(dNumBodies_buffer);
	args.push_back(dDeltaTime_buffer);
	args.push_back(dDamping_buffer);
	args.push_back(dNumTiles_buffer);*/

	


	hpx::wait_all(data_futures);

	std::cout << "Before kernel launch" << std::endl;
	for (int step=0; step<2; step++) {
		auto kernel_future = prog.run(args, "integrateBodies", grid, block, 1024*sizeof(float));
		kernel_future.get();		
	}
	std::cout << "After kernel launch" << std::endl;


	float* posRes;
	cudaMallocHost((void**)&posRes, sizeof(float) * 69632 * 4);
	posRes = dPos_new_buffer.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

	float* velRes;
	cudaMallocHost((void**)&velRes, sizeof(float) * 69632 * 4);
	velRes = dVel_buffer.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

	for(int i = 0; i < 69632; i++){
		std::cout << "velDepoisX[" << i << "]: " << velRes[i*4] << std::endl;
		std::cout << "velDepoisY[" << i << "]: " << velRes[i*4+1] << std::endl;
		std::cout << "velDepoisZ[" << i << "]: " << velRes[i*4+2] << std::endl;
		std::cout << "velDepoisW[" << i << "]: " << velRes[i*4+3] << std::endl;
		std::cout << std::endl;

		std::cout << "posDepoisX[" << i << "]: " << posRes[i*4] << std::endl;
		std::cout << "posDepoisY[" << i << "]: " << posRes[i*4+1] << std::endl;
		std::cout << "posDepoisZ[" << i << "]: " << posRes[i*4+2] << std::endl;
		std::cout << "posDepoisW[" << i << "]: " << posRes[i*4+3] << std::endl;
		std::cout << std::endl;

	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}
