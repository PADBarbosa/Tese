#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;

/*
---------Valores para o device: 0---------
NumBlocks: 272
BlockSize: 256
SharedMemSize: 4096
DeltaTime: 0.016
Damping: 1
NumTiles: 392
DeviceOffset: 0
Device NumBodies: 69632
Total NumBodies: 100352

---------Valores para o device: 1---------
NumBlocks: 120
BlockSize: 256
SharedMemSize: 4096
DeltaTime: 0.016
Damping: 1
NumTiles: 392
DeviceOffset: 69632
Device NumBodies: 30720
Total NumBodies: 100352

*/

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
	device cudaDevice_2 = devices[0];

	float* hPos_new;
	cudaMallocHost((void**)&hPos_new, sizeof(float) * 4 * 100352);
	checkCudaError("Malloc hPos_new");

	float* hPos_old;
	cudaMallocHost((void**)&hPos_old, sizeof(float) * 4 * 100352);
	checkCudaError("Malloc hPos_old");
	
	float* hVel;
	cudaMallocHost((void**)&hVel, sizeof(float) * 4 * 100352);
	checkCudaError("Malloc hVel");

	for(int i = 0; i < 100352; i++){
		hPos_new[i*4] = 1;
		hPos_new[i*4+1] = 2;
		hPos_new[i*4+2] = 3;
		hPos_new[i*4+3] = 4;

		hPos_old[i*4] = 1;
		hPos_old[i*4+1] = 2;
		hPos_old[i*4+2] = 3;
		hPos_old[i*4+3] = 4;


		hVel[i*4+0] = 1*(i+1);
		hVel[i*4+1] = 2*(i+1);
		hVel[i*4+2] = 3*(i+1);
		hVel[i*4+3] = 4*(i+1);
	}

	/*for(int i = 0; i < 69632; i++){
		hPos_new[i*4] = 1;
		hPos_new[i*4+1] = 2;
		hPos_new[i*4+2] = 3;
		hPos_new[i*4+3] = 4;

		hPos_old[i*4] = 1;
		hPos_old[i*4+1] = 2;
		hPos_old[i*4+2] = 3;
		hPos_old[i*4+3] = 4;

		hVel[i*4] = 1;
		hVel[i*4+1] = 2;
		hVel[i*4+2] = 3;
		hVel[i*4+3] = 4;
	}

	for(int i = 69632; i < 100352; i++){
		hPos_new[i*4] = 1;
		hPos_new[i*4+1] = 2;
		hPos_new[i*4+2] = 3;
		hPos_new[i*4+3] = 4;

		hPos_old[i*4] = 1;
		hPos_old[i*4+1] = 2;
		hPos_old[i*4+2] = 3;
		hPos_old[i*4+3] = 4;

		hVel[i*4] = 1;
		hVel[i*4+1] = 2;
		hVel[i*4+2] = 3;
		hVel[i*4+3] = 4;
	}*/

	buffer dPos_new_buffer_1 = cudaDevice_1.create_buffer(sizeof(float) * 4 * 100352).get();
	data_futures.push_back(dPos_new_buffer_1.enqueue_write(0, sizeof(float) * 4 * 100352, hPos_new));

	buffer dPos_old_buffer_1 = cudaDevice_1.create_buffer(sizeof(float) * 4 * 100352).get();
	data_futures.push_back(dPos_old_buffer_1.enqueue_write(0, sizeof(float) * 4 * 100352, hPos_old));

	buffer dVel_buffer_1 = cudaDevice_1.create_buffer(sizeof(float) * 4 * 100352).get();
	data_futures.push_back(dVel_buffer_1.enqueue_write(0, sizeof(float) * 4 * 100352, hVel));



	buffer dPos_new_buffer_2 = cudaDevice_2.create_buffer(sizeof(float) * 4 * 100352).get();
	data_futures.push_back(dPos_new_buffer_2.enqueue_write(0, sizeof(float) * 4 * 100352, hPos_new));

	buffer dPos_old_buffer_2 = cudaDevice_2.create_buffer(sizeof(float) * 4 * 100352).get();
	data_futures.push_back(dPos_old_buffer_2.enqueue_write(0, sizeof(float) * 4 * 100352, hPos_old));

	buffer dVel_buffer_2 = cudaDevice_2.create_buffer(sizeof(float) * 4 * 100352).get();
	data_futures.push_back(dVel_buffer_2.enqueue_write(0, sizeof(float) * 4 * 100352, hVel));

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
	
	program prog_1 = cudaDevice_1.create_program_with_file("multi_my_nvidia_nbody_kernel_1.cu").get();
	program prog_2 = cudaDevice_2.create_program_with_file("multi_my_nvidia_nbody_kernel_2.cu").get();

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

	grid_1.x = 272;
	grid_1.y = 1;
	grid_1.z = 1;

	block_1.x = 256;
	block_1.y = 1;
	block_1.z = 1;


	hpx::cuda::server::program::Dim3 grid_2;
	hpx::cuda::server::program::Dim3 block_2;

	grid_2.x = 120;
	grid_2.y = 1;
	grid_2.z = 1;

	block_2.x = 256;
	block_2.y = 1;
	block_2.z = 1;



	std::vector<hpx::cuda::buffer> args_1;
	args_1.push_back(dPos_new_buffer_1);
	args_1.push_back(dPos_old_buffer_1);
	args_1.push_back(dVel_buffer_1);


	std::vector<hpx::cuda::buffer> args_2;
	args_2.push_back(dPos_new_buffer_2);
	args_2.push_back(dPos_old_buffer_2);
	args_2.push_back(dVel_buffer_2);


	hpx::wait_all(data_futures);

	std::cout << "Before kernel launch" << std::endl;
	for (int step=0; step<10; step++) {
		auto kernel_future_1 = prog_1.run(args_1, "integrateBodies", grid_1, block_1, 1024*sizeof(float));
		auto kernel_future_2 = prog_2.run(args_2, "integrateBodies", grid_2, block_2, 1024*sizeof(float));

		kernel_future_1.get();
		kernel_future_2.get();

		// Vou buscar a cada GPU os dados que quero de cada array
		// Neste caso:
		// GPU0, quero os dados do 0->69632 body (69632)
		// GPU1, quero os dados do 69632->100352 body (30720)

		float* temp_posRes_1;
		cudaMallocHost((void**)&temp_posRes_1, sizeof(float) * 69632 * 4);
		temp_posRes_1 = dPos_new_buffer_1.enqueue_read_parcel_sync<float>(0, sizeof(float) * 69632 * 4);

		float* temp_velRes_1;
		cudaMallocHost((void**)&temp_velRes_1, sizeof(float) * 69632 * 4);
		temp_velRes_1 = dVel_buffer_1.enqueue_read_parcel_sync<float>(0, sizeof(float) * 69632 * 4);

		/*for(int i = 0; i < 1; i++){
			std::cout << "0tempVelX[" << i << "]: " << temp_velRes_1[i*4+0] << std::endl;
			std::cout << "0tempVelY[" << i << "]: " << temp_velRes_1[i*4+1] << std::endl;
			std::cout << "0tempVelZ[" << i << "]: " << temp_velRes_1[i*4+2] << std::endl;
			std::cout << "0tempVelW[" << i << "]: " << temp_velRes_1[i*4+3] << std::endl;
			std::cout << std::endl;

			std::cout << "0tempPosX[" << i << "]: " << temp_posRes_1[i*4+0] << std::endl;
			std::cout << "0tempPosY[" << i << "]: " << temp_posRes_1[i*4+1] << std::endl;
			std::cout << "0tempPosZ[" << i << "]: " << temp_posRes_1[i*4+2] << std::endl;
			std::cout << "0tempPosW[" << i << "]: " << temp_posRes_1[i*4+3] << std::endl;
			std::cout << std::endl;


		}*/
		


		float* temp_posRes_2;
		cudaMallocHost((void**)&temp_posRes_2, sizeof(float) * 30720 * 4);
		temp_posRes_2 = dPos_new_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);

		float* temp_velRes_2;
		cudaMallocHost((void**)&temp_velRes_2, sizeof(float) * 30720 * 4);
		temp_velRes_2 = dVel_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);

		/*for(int i = 0; i < 1; i++){
			std::cout << "1tempVelX[" << i << "]: " << temp_velRes_2[i*4+0] << std::endl;
			std::cout << "1tempVelY[" << i << "]: " << temp_velRes_2[i*4+1] << std::endl;
			std::cout << "1tempVelZ[" << i << "]: " << temp_velRes_2[i*4+2] << std::endl;
			std::cout << "1tempVelW[" << i << "]: " << temp_velRes_2[i*4+3] << std::endl;
			std::cout << std::endl;

			std::cout << "1tempPosX[" << i << "]: " << temp_posRes_2[i*4+0] << std::endl;
			std::cout << "1tempPosY[" << i << "]: " << temp_posRes_2[i*4+1] << std::endl;
			std::cout << "1tempPosZ[" << i << "]: " << temp_posRes_2[i*4+2] << std::endl;
			std::cout << "1tempPosW[" << i << "]: " << temp_posRes_2[i*4+3] << std::endl;
			std::cout << std::endl;


		}*/

		// Juntar os dados todos num array só
		for(int i = 0; i < 69632; i++){
			hPos_new[i*4+0] = temp_posRes_1[i*4+0];
			hPos_new[i*4+1] = temp_posRes_1[i*4+1];
			hPos_new[i*4+2] = temp_posRes_1[i*4+2];
			hPos_new[i*4+3] = temp_posRes_1[i*4+3];

			hVel[i*4+0] = temp_velRes_1[i*4+0];
			hVel[i*4+1] = temp_velRes_1[i*4+1];
			hVel[i*4+2] = temp_velRes_1[i*4+2];
			hVel[i*4+3] = temp_velRes_1[i*4+3];
		}

		// Juntar os dados todos num array só (ter em consideração o offset para não escrever por cima dos valores do GPU0)
		int offset = 69632;
		for(int i = 0; i < 30720; i++){
			hPos_new[offset*4 + i*4 + 0] = temp_posRes_2[i*4+0];
			hPos_new[offset*4 + i*4 + 1] = temp_posRes_2[i*4+1];
			hPos_new[offset*4 + i*4 + 2] = temp_posRes_2[i*4+2];
			hPos_new[offset*4 + i*4 + 3] = temp_posRes_2[i*4+3];

			hVel[offset*4 + i*4 + 0] = temp_velRes_2[i*4+0];
			hVel[offset*4 + i*4 + 1] = temp_velRes_2[i*4+1];
			hVel[offset*4 + i*4 + 2] = temp_velRes_2[i*4+2];
			hVel[offset*4 + i*4 + 3] = temp_velRes_2[i*4+3];
		}

		// Copiar os dados todos de volta (ineficiente mas funções atuais do HPXCL complicam muito a cópia)
		data_futures.push_back(dPos_old_buffer_1.enqueue_write(0, sizeof(float) * 4 * 100352, hPos_new));
		data_futures.push_back(dVel_buffer_1.enqueue_write(0, sizeof(float) * 4 * 100352, hVel));

		data_futures.push_back(dPos_old_buffer_2.enqueue_write(0, sizeof(float) * 4 * 100352, hPos_new));
		data_futures.push_back(dVel_buffer_2.enqueue_write(0, sizeof(float) * 4 * 100352, hVel));


		hpx::wait_all(data_futures);


/*
		std::cout << "---------- Iteracao " << step << " ----------" << std::endl;
		std::cout << "velDepoisX[" << 0 << "]: " << hVel[0] << std::endl;
		std::cout << "velDepoisY[" << 0 << "]: " << hVel[1] << std::endl;
		std::cout << "velDepoisZ[" << 0 << "]: " << hVel[2] << std::endl;
		std::cout << "velDepoisW[" << 0 << "]: " << hVel[3] << std::endl;
		std::cout << std::endl;

		std::cout << "posDepoisX[" << 0 << "]: " << hPos_new[0] << std::endl;
		std::cout << "posDepoisY[" << 0 << "]: " << hPos_new[1] << std::endl;
		std::cout << "posDepoisZ[" << 0 << "]: " << hPos_new[2] << std::endl;
		std::cout << "posDepoisW[" << 0 << "]: " << hPos_new[3] << std::endl;
		std::cout << std::endl;

*/







/*

		hPos_new = dPos_new_buffer_1.enqueue_read_parcel_sync<float>(0, sizeof(float) * 69632 * 4);
		hVel = dVel_buffer_1.enqueue_read_parcel_sync<float>(0, sizeof(float) * 69632 * 4);

		data_futures.push_back(dPos_old_buffer_2.enqueue_write(0, sizeof(float) * 4 * 69632, hPos_new));
		data_futures.push_back(dVel_buffer_2.enqueue_write(0, sizeof(float) * 4 * 69632, hVel));


		hPos_new = dPos_new_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);
		hVel = dVel_buffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 30720 * 4);

		data_futures.push_back(dPos_old_buffer_1.enqueue_write(0, sizeof(float) * 4 * 30720, hPos_new)); //sizeof(float) * 69632 * 4
		data_futures.push_back(dVel_buffer_1.enqueue_write(0, sizeof(float) * 4 * 30720, hVel)); 
		// escreve a partir do hVel(ultimo arg) com offset de 0 (primeiro arg) e escreve um totalde sizeof(float)*4*30720 (segundo arg)





		float* temp_posRes_1;
		cudaMallocHost((void**)&temp_posRes_1, sizeof(float) * 69632 * 4);
		temp_posRes_1 = dPos_new_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

		float* temp_velRes_1;
		cudaMallocHost((void**)&temp_velRes_1, sizeof(float) * 69632 * 4);
		temp_posRes_1 = dVel_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

		data_futures.push_back(dPos_old_buffer_2.enqueue_write(0, sizeof(float) * 4 * 69632, temp_posRes_1));
		data_futures.push_back(dVel_buffer_2.enqueue_write(0, sizeof(float) * 4 * 69632, temp_velRes_1));


		float* temp_posRes_2;
		cudaMallocHost((void**)&temp_posRes_2, sizeof(float) * 30720 * 4);
		temp_posRes_2 = dPos_new_buffer_2.enqueue_read_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 100352 * 4);

		float* temp_velRes_2;
		cudaMallocHost((void**)&temp_velRes_2, sizeof(float) * 30720 * 4);
		temp_posRes_2 = dVel_buffer_2.enqueue_read_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 100352 * 4);

		data_futures.push_back(dPos_old_buffer_1.enqueue_write(sizeof(float) * 69632 * 4, sizeof(float) * 4 * 100352, temp_posRes_2));
		data_futures.push_back(dVel_buffer_1.enqueue_write(sizeof(float) * 69632 * 4, sizeof(float) * 4 * 100352, temp_velRes_2));
*/
	}
	std::cout << "After kernel launch" << std::endl;

/*
	float* posRes_1;
	cudaMallocHost((void**)&posRes_1, sizeof(float) * 69632 * 4);
	posRes_1 = dPos_new_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);

	float* velRes_1;
	cudaMallocHost((void**)&velRes_1, sizeof(float) * 69632 * 4);
	velRes_1 = dVel_buffer_1.enqueue_read_sync<float>(0, sizeof(float) * 69632 * 4);





	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	//VERIFICAR SE ESTA LEITURA ESTÁ BEM FEITA
	float* posRes_2;
	cudaMallocHost((void**)&posRes_2, sizeof(float) * 30720 * 4);
	posRes_2 = dPos_new_buffer_2.enqueue_read_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 100352 * 4);

	float* velRes_2;
	cudaMallocHost((void**)&velRes_2, sizeof(float) * 30720 * 4);
	velRes_2 = dVel_buffer_2.enqueue_read_sync<float>(sizeof(float) * 69632 * 4, sizeof(float) * 100352 * 4);


	float* final_pos;
	cudaMallocHost((void**)&final_pos, sizeof(float) * 100352 * 4);

	float* final_vel;
	cudaMallocHost((void**)&final_vel, sizeof(float) * 100352 * 4);

	for(int i = 0; i < 69632; i++){
		final_pos[i+0] = posRes_1[i+0];
		final_pos[i+1] = posRes_1[i+1];
		final_pos[i+2] = posRes_1[i+2];
		final_pos[i+3] = posRes_1[i+3];

		final_vel[i+0] = velRes_1[i+0];
		final_vel[i+1] = velRes_1[i+1];
		final_vel[i+2] = velRes_1[i+2];
		final_vel[i+3] = velRes_1[i+3];
	}

	for(int i = 69632; i < 100352; i++){
		final_pos[i+0] = posRes_2[i+0];
		final_pos[i+1] = posRes_2[i+1];
		final_pos[i+2] = posRes_2[i+2];
		final_pos[i+3] = posRes_2[i+3];

		final_vel[i+0] = velRes_2[i+0];
		final_vel[i+1] = velRes_2[i+1];
		final_vel[i+2] = velRes_2[i+2];
		final_vel[i+3] = velRes_2[i+3];
	}
*/

	/*for(int i = 0; i < 100352; i++){
		std::cout << "finalPosX[" << i << "]: " << hPos_new[i*4] << std::endl;
		std::cout << "finalPosY[" << i << "]: " << hPos_new[i*4+1] << std::endl;
		std::cout << "finalPosZ[" << i << "]: " << hPos_new[i*4+2] << std::endl;
		std::cout << "finalPosW[" << i << "]: " << hPos_new[i*4+3] << std::endl;
		std::cout << std::endl;

		std::cout << "finalVelX[" << i << "]: " << hVel[i*4] << std::endl;
		std::cout << "finalVelY[" << i << "]: " << hVel[i*4+1] << std::endl;
		std::cout << "finalVelZ[" << i << "]: " << hVel[i*4+2] << std::endl;
		std::cout << "finalVelW[" << i << "]: " << hVel[i*4+3] << std::endl;
		std::cout << std::endl;
	}*/

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}
