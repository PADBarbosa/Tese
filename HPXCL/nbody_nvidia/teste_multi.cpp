#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>
#include "vector_types.h"

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

	device cudaDevice_1 = devices[0];
	device cudaDevice_2 = devices[1];

	int totalNumBodies = 1000448;
	int numBodiesDevice1 = 500224;
	int numBodiesDevice2 = 500224; //offset do device2 é igual ao numBodies do device1
	int numTilesValue = 1954;
	int numBlocksValue = 1954;
	int iterations = 500;

	float4* hPos_new;
    cudaMallocHost((void**)&hPos_new, sizeof(float) * 4 * totalNumBodies);
	checkCudaError("Malloc hPos_new");

	float4* hPos_old;
	cudaMallocHost((void**)&hPos_old, sizeof(float) * 4 * totalNumBodies);
	checkCudaError("Malloc hPos_old");
	
	float4* hVel;
	cudaMallocHost((void**)&hVel, sizeof(float) * 4 * totalNumBodies);
	checkCudaError("Malloc hVel");

	for(int i = 0; i < totalNumBodies; i++){
		hPos_old[i].x = 1*(i+1);
	    hPos_old[i].y = 2*(i+1);
	    hPos_old[i].z = 3*(i+1);
	    hPos_old[i].w = 4*(i+1);

	    hPos_new[i].x = 1*(i+1);
	    hPos_new[i].y = 2*(i+1);
	    hPos_new[i].z = 3*(i+1);
	    hPos_new[i].w = 4*(i+1);

	    hVel[i].x = 1*(i+1);
	    hVel[i].y = 2*(i+1);
	    hVel[i].z = 3*(i+1);
	    hVel[i].w = 4*(i+1);
	}


	buffer dPos_new_buffer_1 = cudaDevice_1.create_buffer(sizeof(float) * 4 * totalNumBodies).get();
	data_futures.push_back(dPos_new_buffer_1.enqueue_write(0, sizeof(float) * 4 * totalNumBodies, hPos_new));

	buffer dPos_old_buffer_1 = cudaDevice_1.create_buffer(sizeof(float) * 4 * totalNumBodies).get();
	data_futures.push_back(dPos_old_buffer_1.enqueue_write(0, sizeof(float) * 4 * totalNumBodies, hPos_old));

	buffer dVel_buffer_1 = cudaDevice_1.create_buffer(sizeof(float) * 4 * totalNumBodies).get();
	data_futures.push_back(dVel_buffer_1.enqueue_write(0, sizeof(float) * 4 * totalNumBodies, hVel));


	buffer dPos_new_buffer_2 = cudaDevice_2.create_buffer(sizeof(float) * 4 * totalNumBodies).get();
	data_futures.push_back(dPos_new_buffer_2.enqueue_write(0, sizeof(float) * 4 * totalNumBodies, hPos_new));

	buffer dPos_old_buffer_2 = cudaDevice_2.create_buffer(sizeof(float) * 4 * totalNumBodies).get();
	data_futures.push_back(dPos_old_buffer_2.enqueue_write(0, sizeof(float) * 4 * totalNumBodies, hPos_old));

	buffer dVel_buffer_2 = cudaDevice_2.create_buffer(sizeof(float) * 4 * totalNumBodies).get();
	data_futures.push_back(dVel_buffer_2.enqueue_write(0, sizeof(float) * 4 * totalNumBodies, hVel));


	int* deviceOffset_1;
	cudaMallocHost((void**)&deviceOffset_1, sizeof(int));
	checkCudaError("Malloc deviceOffset_1");
	deviceOffset_1[0] = 0;

	buffer dOffset_buffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
	data_futures.push_back(dOffset_buffer_1.enqueue_write(0, sizeof(int), deviceOffset_1));


	int* deviceNumBodies_1;
	cudaMallocHost((void**)&deviceNumBodies_1, sizeof(int));
	checkCudaError("Malloc deviceNumBodies_1");
	deviceNumBodies_1[0] = numBodiesDevice1;

	buffer dNumBodies_buffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
	data_futures.push_back(dNumBodies_buffer_1.enqueue_write(0, sizeof(int), deviceNumBodies_1));


	float* deltaTime_1;
	cudaMallocHost((void**)&deltaTime_1, sizeof(float));
	checkCudaError("Malloc deltaTime_1");
	deltaTime_1[0] = 0.016;

	buffer dDeltaTime_buffer_1 = cudaDevice_1.create_buffer(sizeof(float)).get();
	data_futures.push_back(dDeltaTime_buffer_1.enqueue_write(0, sizeof(float), deltaTime_1));


	float* damping_1;
	cudaMallocHost((void**)&damping_1, sizeof(float));
	checkCudaError("Malloc damping_1");
	damping_1[0] = 1;

	buffer dDamping_buffer_1 = cudaDevice_1.create_buffer(sizeof(float)).get();
	data_futures.push_back(dDamping_buffer_1.enqueue_write(0, sizeof(float), damping_1));


	int* numTiles_1;
	cudaMallocHost((void**)&numTiles_1, sizeof(int));
	checkCudaError("Malloc numTiles_1");
	numTiles_1[0] = numTilesValue;

	buffer dNumTiles_buffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
	data_futures.push_back(dNumTiles_buffer_1.enqueue_write(0, sizeof(int), numTiles_1));



	int* deviceOffset_2;
	cudaMallocHost((void**)&deviceOffset_2, sizeof(int));
	checkCudaError("Malloc deviceOffset_2");
	deviceOffset_2[0] = numBodiesDevice1;

	buffer dOffset_buffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
	data_futures.push_back(dOffset_buffer_2.enqueue_write(0, sizeof(int), deviceOffset_2));


	int* deviceNumBodies_2;
	cudaMallocHost((void**)&deviceNumBodies_2, sizeof(int));
	checkCudaError("Malloc deviceNumBodies_2");
	deviceNumBodies_2[0] = numBodiesDevice2;

	buffer dNumBodies_buffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
	data_futures.push_back(dNumBodies_buffer_2.enqueue_write(0, sizeof(int), deviceNumBodies_2));


	float* deltaTime_2;
	cudaMallocHost((void**)&deltaTime_2, sizeof(float));
	checkCudaError("Malloc deltaTime_2");
	deltaTime_2[0] = 0.016;

	buffer dDeltaTime_buffer_2 = cudaDevice_2.create_buffer(sizeof(float)).get();
	data_futures.push_back(dDeltaTime_buffer_2.enqueue_write(0, sizeof(float), deltaTime_2));


	float* damping_2;
	cudaMallocHost((void**)&damping_2, sizeof(float));
	checkCudaError("Malloc damping_2");
	damping_2[0] = 1;

	buffer dDamping_buffer_2 = cudaDevice_2.create_buffer(sizeof(float)).get();
	data_futures.push_back(dDamping_buffer_2.enqueue_write(0, sizeof(float), damping_2));


	int* numTiles_2;
	cudaMallocHost((void**)&numTiles_2, sizeof(int));
	checkCudaError("Malloc numTiles_2");
	numTiles_2[0] = numTilesValue;

	buffer dNumTiles_buffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
	data_futures.push_back(dNumTiles_buffer_2.enqueue_write(0, sizeof(int), numTiles_2));



	program prog_1 = cudaDevice_1.create_program_with_file("teste_kernel.cu").get();
	program prog_2 = cudaDevice_2.create_program_with_file("teste_kernel.cu").get();

	std::vector<std::string> flags_1;
	std::string mode_1 = "--gpu-architecture=compute_";
	mode_1.append(std::to_string(cudaDevice_1.get_device_architecture_major().get()));
	mode_1.append(std::to_string(cudaDevice_1.get_device_architecture_minor().get()));
	flags_1.push_back(mode_1);

	std::vector<std::string> flags_2;
	std::string mode_2 = "--gpu-architecture=compute_";
	mode_2.append(std::to_string(cudaDevice_2.get_device_architecture_major().get()));
	mode_2.append(std::to_string(cudaDevice_2.get_device_architecture_minor().get()));
	flags_2.push_back(mode_2);

	prog_1.build_sync(flags_1, "integrateBodies");
	prog_2.build_sync(flags_2, "integrateBodies");


	hpx::cuda::server::program::Dim3 grid_1;
	hpx::cuda::server::program::Dim3 block_1;

	grid_1.x = numBlocksValue;
	grid_1.y = 1;
	grid_1.z = 1;

	block_1.x = 256;
	block_1.y = 1;
	block_1.z = 1;


	hpx::cuda::server::program::Dim3 grid_2;
	hpx::cuda::server::program::Dim3 block_2;

	grid_2.x = numBlocksValue;
	grid_2.y = 1;
	grid_2.z = 1;

	block_2.x = 256;
	block_2.y = 1;
	block_2.z = 1;


	std::vector<hpx::cuda::buffer> args_1;
	args_1.push_back(dPos_new_buffer_1);
	args_1.push_back(dPos_old_buffer_1);
	args_1.push_back(dVel_buffer_1);
	args_1.push_back(dOffset_buffer_1);
	args_1.push_back(dNumBodies_buffer_1);
	args_1.push_back(dDeltaTime_buffer_1);
	args_1.push_back(dDamping_buffer_1);
	args_1.push_back(dNumTiles_buffer_1);


	std::vector<hpx::cuda::buffer> args_2;
	args_2.push_back(dPos_new_buffer_2);
	args_2.push_back(dPos_old_buffer_2);
	args_2.push_back(dVel_buffer_2);
	args_2.push_back(dOffset_buffer_2);
	args_2.push_back(dNumBodies_buffer_2);
	args_2.push_back(dDeltaTime_buffer_2);
	args_2.push_back(dDamping_buffer_2);
	args_2.push_back(dNumTiles_buffer_2);


	hpx::wait_all(data_futures);


	//float4* h_tempPos;
	//cudaMallocHost((void**)&h_tempPos, sizeof(float) * 4 * numBodies);

	//float4* h_tempVel;
	//cudaMallocHost((void**)&h_tempVel, sizeof(float) * 4 * numBodies);

    int currentRead = 0;

	std::cout << "Before kernel launch" << std::endl;
	for (int step=0; step<iterations; step++) {
		auto kernel_future_1 = prog_1.run(args_1, "integrateBodies", grid_1, block_1, 1024*sizeof(float));
		auto kernel_future_2 = prog_2.run(args_2, "integrateBodies", grid_2, block_2, 1024*sizeof(float));

		kernel_future_1.get();
		kernel_future_2.get();

		std::iter_swap(args_1.begin(), args_1.begin()+1);
		std::iter_swap(args_2.begin(), args_2.begin()+1);


/*		// Usado para verificar os valores dos resultados a cada iteração
		if(currentRead == 0){
            h_velTemp = dVel_buffer_2.enqueue_read_parcel_sync<float>(0, sizeof(float) * 100352 * 4);
        	h_posTemp = dPos_new_buffer_2.enqueue_read_parcel_sync<float>(0, sizeof(float) * 100352 * 4);

            currentRead = 1;
        }
        else if(currentRead == 1){
            h_velTemp = dVel_buffer_2.enqueue_read_parcel_sync<float>(0, sizeof(float) * 100352 * 4);
        	h_posTemp = dPos_old_buffer_2.enqueue_read_parcel_sync<float>(0, sizeof(float) * 100352 * 4);

            currentRead = 0;
        }





        std::cout << "-------------------- Iteracao: " << step << " --------------------" << std::endl;

        std::cout << std::endl << "velDepoisX[" << 69632 << "]: " << h_velTemp[4*69632+0];
        std::cout << std::endl << "velDepoisY[" << 69632 << "]: " << h_velTemp[4*69632+1];
        std::cout << std::endl << "velDepoisZ[" << 69632 << "]: " << h_velTemp[4*69632+2];
        std::cout << std::endl << "velDepoisW[" << 69632 << "]: " << h_velTemp[4*69632+3];
        std::cout << std::endl;

        std::cout << std::endl << "posDepoisX[" << 69632 << "]: " << h_posTemp[4*69632+0];
        std::cout << std::endl << "posDepoisY[" << 69632 << "]: " << h_posTemp[4*69632+1];
        std::cout << std::endl << "posDepoisZ[" << 69632 << "]: " << h_posTemp[4*69632+2];
        std::cout << std::endl << "posDepoisW[" << 69632 << "]: " << h_posTemp[4*69632+3];
        std::cout << std::endl;
        
        std::cout << "--------------------------------------------------" <<std::endl;*/

		//hpx::wait_all(data_futures);

	}
	std::cout << "After kernel launch" << std::endl;


/*	float* posRes_1;
	cudaMallocHost((void**)&posRes_1, sizeof(float) * 69632 * 4);

	float* velRes_1;
	cudaMallocHost((void**)&velRes_1, sizeof(float) * 69632 * 4);


	float* posRes_2;
	cudaMallocHost((void**)&posRes_2, sizeof(float) * 30720 * 4);

	float* velRes_2;
	cudaMallocHost((void**)&velRes_2, sizeof(float) * 30720 * 4);


	if(currentRead == 0){
		posRes_1 = dPos_old_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);
		velRes_1 = dVel_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

		posRes_2 = dPos_old_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);
		velRes_2 = dVel_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);
	}
	else if(currentRead == 1){
		posRes_1 = dPos_new_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);
		velRes_1 = dVel_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

		posRes_2 = dPos_new_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);
		velRes_2 = dVel_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);
	}
	*/

	/*for(int i = 0; i < 100352; i++){
		std::cout << "--------------- Device 0 ---------------" << std::endl;
		std::cout << "velX[" << i << "]: " << velRes_1[i*4+0] << std::endl;
		std::cout << "velY[" << i << "]: " << velRes_1[i*4+1] << std::endl;
		std::cout << "velZ[" << i << "]: " << velRes_1[i*4+2] << std::endl;
		std::cout << "velW[" << i << "]: " << velRes_1[i*4+3] << std::endl;
		std::cout << std::endl;

		std::cout << "posX[" << i << "]: " << posRes_1[i*4+0] << std::endl;
		std::cout << "posY[" << i << "]: " << posRes_1[i*4+1] << std::endl;
		std::cout << "posZ[" << i << "]: " << posRes_1[i*4+2] << std::endl;
		std::cout << "posW[" << i << "]: " << posRes_1[i*4+3] << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "--------------- Device 1 ---------------" << std::endl;
		std::cout << "velX[" << i << "]: " << velRes_2[69632*4 + i*4+0] << std::endl;
		std::cout << "velY[" << i << "]: " << velRes_2[69632*4 + i*4+1] << std::endl;
		std::cout << "velZ[" << i << "]: " << velRes_2[69632*4 + i*4+2] << std::endl;
		std::cout << "velW[" << i << "]: " << velRes_2[69632*4 + i*4+3] << std::endl;
		std::cout << std::endl;

		std::cout << "posX[" << i << "]: " << posRes_2[69632*4 + i*4+0] << std::endl;
		std::cout << "posY[" << i << "]: " << posRes_2[69632*4 + i*4+1] << std::endl;
		std::cout << "posZ[" << i << "]: " << posRes_2[69632*4 + i*4+2] << std::endl;
		std::cout << "posW[" << i << "]: " << posRes_2[69632*4 + i*4+3] << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

	}*/


	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}
