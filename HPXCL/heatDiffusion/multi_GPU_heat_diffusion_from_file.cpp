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

void cubeCreator_2(int size, float* input) {
	for(int i = 0;i < size; ++i) {
		for(int j = 0; j < size; ++j){
			for(int k = 0; k < ((size/2)+1); ++k){
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
	

	device cudaDevice_1 = devices[0];
	device cudaDevice_2 = devices[1];


	float* input_1;
	cudaMallocHost((void**)&input_1, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));

	float* input_2;
	cudaMallocHost((void**)&input_2, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));

	// Colocar em input_1 metade+1 camadas do cubo (colocado o input_2 uma vez que os valores vão ser iguais, ver melhor o que pôr aqui de forma a fazer sentido)
	for(int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++)	{
			for (int k = 0; k < ((SIZE/2)+1); k++)	{
				input_1[k + j*SIZE + i*SIZE*SIZE] = input[k + j*SIZE + i*SIZE*SIZE];
				input_2[k + j*SIZE + i*SIZE*SIZE] = input[k + j*SIZE + i*SIZE*SIZE];

				//std::cout << input_1[k + j*SIZE + i*SIZE*SIZE] << " ";
				//std::cout << input_2[k + j*SIZE + i*SIZE*SIZE] << " ";
			}
			//std::cout << std::endl;
		}
	}


	// Copia desde o início até ao meio, mais uma linha (linha que futuramente é comunicada pelo outro GPU)
	buffer inbuffer_1 = cudaDevice_1.create_buffer(sizeof(float) * SIZE * SIZE * ((SIZE/2)+1)).get();

	// Copia desde uma linha antes do meio (linha que futuramente é comunicada pelo outro GPU), até ao final do cubo
	buffer inbuffer_2 = cudaDevice_2.create_buffer(sizeof(float) * SIZE * SIZE * ((SIZE/2)+1)).get();





	data_futures.push_back(inbuffer_1.enqueue_write(0, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1), input_1));
	data_futures.push_back(inbuffer_2.enqueue_write(0, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1), input_2));



	program prog_1 = cudaDevice_1.create_program_with_file("multi_gpu_heat_diffusion_kernel.cu").get();
	program prog_2 = cudaDevice_2.create_program_with_file("multi_gpu_heat_diffusion_kernel.cu").get();



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

	prog_1.build_sync(flags_1, "fdm3d");
	prog_2.build_sync(flags_2, "fdm3d");


	// Output está só a ser utilizado para uma das metades do cubo daí o tamanho ser SIZE/2+1
	float* output;
	cudaMallocHost((void**)&output, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));
	checkCudaError("Malloc output");

	cubeCreator_2(SIZE, output);

	buffer outbuffer_1 = cudaDevice_1.create_buffer(sizeof(float) * SIZE * SIZE * ((SIZE/2)+1)).get();
	data_futures.push_back(outbuffer_1.enqueue_write(0, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1), output));

	buffer outbuffer_2 = cudaDevice_2.create_buffer(sizeof(float) * SIZE * SIZE * ((SIZE/2)+1)).get();
	data_futures.push_back(outbuffer_2.enqueue_write(0, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1), output));


	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	grid.x = 8;
	grid.y = 8;
	grid.z = 1;

	block.x = BLOCK_SIZE;
	block.y = BLOCK_SIZE;
	block.z = 1;




	int* n;
	cudaMallocHost((void**)&n, sizeof(int));
	checkCudaError("Malloc n");
	n[0] = SIZE;

	buffer n_buffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
	data_futures.push_back(n_buffer_1.enqueue_write(0, sizeof(int), n));

	buffer n_buffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
	data_futures.push_back(n_buffer_2.enqueue_write(0, sizeof(int), n));



	int* m;
	cudaMallocHost((void**)&m, sizeof(int));
	checkCudaError("Malloc m");
	// Tamanho é alterado para SIZE/2 em relação à versão num GPU só (deveria ser SIZE/2+1??)
	m[0] = SIZE/2;

	buffer m_buffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
	data_futures.push_back(m_buffer_1.enqueue_write(0, sizeof(int), m));

	buffer m_buffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
	data_futures.push_back(m_buffer_2.enqueue_write(0, sizeof(int), m));



	float* r;
	cudaMallocHost((void**)&r, sizeof(float));
	checkCudaError("Malloc r");
	r[0] = 0.005;

	buffer r_buffer_1 = cudaDevice_1.create_buffer(sizeof(float)).get();
	data_futures.push_back(r_buffer_1.enqueue_write(0, sizeof(float), r));

	buffer r_buffer_2 = cudaDevice_2.create_buffer(sizeof(float)).get();
	data_futures.push_back(r_buffer_2.enqueue_write(0, sizeof(float), r));


	std::vector<hpx::cuda::buffer> args_1;
	args_1.push_back(inbuffer_1);
	args_1.push_back(outbuffer_1);
	args_1.push_back(n_buffer_1);
	args_1.push_back(m_buffer_1);
	args_1.push_back(r_buffer_1);

	std::vector<hpx::cuda::buffer> args_2;
	args_2.push_back(inbuffer_2);
	args_2.push_back(outbuffer_2);
	args_2.push_back(n_buffer_2);
	args_2.push_back(m_buffer_2);
	args_2.push_back(r_buffer_2);

	hpx::wait_all(data_futures);

	/*float* output_1_temp;
	cudaMallocHost((void**)&output_1_temp, sizeof(float) * SIZE * SIZE);

	float* output_2_temp;
	cudaMallocHost((void**)&output_2_temp, sizeof(float) * SIZE * SIZE);

	std::vector<hpx::lcos::future<void>> data;*/

	for (int i = 0; i < STEPS; i++)	{
		auto kernel_future_1 = prog_1.run(args_1, "fdm3d", grid, block, 3 * (sizeof(float) * (BLOCK_SIZE+2) * (BLOCK_SIZE+2)));
		auto kernel_future_2 = prog_2.run(args_2, "fdm3d", grid, block, 3 * (sizeof(float) * (BLOCK_SIZE+2) * (BLOCK_SIZE+2)));

		kernel_future_1.get();
		kernel_future_2.get();



		/*auto f_1 = outbuffer_1.p2p_copy(inbuffer_2.get_device_pointer().get(),
									  inbuffer_2.get_device_id().get(),
									  sizeof(float) * SIZE * SIZE);

		auto f_2 = outbuffer_2.p2p_copy(inbuffer_1.get_device_pointer().get(),
									  inbuffer_1.get_device_id().get(),
									  sizeof(float) * SIZE * SIZE);

		f_1.get();
		f_2.get();*/



		// Valores nas copias estão errados
		/*output_1_temp = outbuffer_1.enqueue_read_parcel_sync<float>(sizeof(float) * (z-1) * SIZE * SIZE, sizeof(float) * SIZE * SIZE );

		// Ir buscar a segunda linha do output_2 (a primeira corresponde à linha que apenas é de leitura e que é atualizada pelo outro GPU)
		output_2_temp = outbuffer_2.enqueue_read_parcel_sync<float>(sizeof(float) * SIZE * SIZE, sizeof(float) * SIZE * SIZE );
		
		// Escrita dos dados do GPU2 para o GPU1
		data.push_back(inbuffer_1.enqueue_write(sizeof(float) * (z-1) * SIZE * SIZE, sizeof(float) * SIZE * SIZE, output_2_temp)); // A linha a passar ao GPU1 é escrita na última posição
		
		// Escrita dos dados do GPU1 para o GPU2
		data.push_back(inbuffer_2.enqueue_write(0, sizeof(float) * SIZE * SIZE, output_1_temp)); // A linha a passar ao GPU2 é escrita na primeira posição

		wait_all(data);*/

		std::iter_swap(args_1.begin(), args_1.begin()+1);
		std::iter_swap(args_2.begin(), args_2.begin()+1);
	}

	// Verificar qual será a melhor maneira para fazer a junção das duas metades
	/*float* res_1;
	cudaMallocHost((void**)&res_1, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));
	res_1 = inbuffer_1.enqueue_read_sync<float>(0, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));

	float* res_2;
	cudaMallocHost((void**)&res_2, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));
	res_2 = inbuffer_2.enqueue_read_sync<float>(0, sizeof(float) * SIZE * SIZE * ((SIZE/2)+1));

	float* res;
	cudaMallocHost((void**)&res, sizeof(float) * SIZE * SIZE * SIZE);

	for(int i = 0; i < SIZE * SIZE * SIZE/2; i++){
		res[i] = res_1[i];
	}

	for(int i = SIZE * SIZE * SIZE/2; i < SIZE * SIZE * SIZE; i++){
		res[i] = res_2[i];
	}*/

	//dump(res, SIZE, SIZE/3);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}




