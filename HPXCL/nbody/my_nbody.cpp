#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include "Constants.h"

#include <chrono>

using namespace hpx::cuda;

//template <class T> const T& min (const T& a, const T& b) {
//  return !(b<a)?a:b;     // or: return !comp(b,a)?a:b; for version (2)
//}


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
	velocity = 0.67*sqrt((GR*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
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
		velocity = pow(((GR*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
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
//		size = file.tellg();
		file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
		file.write(data, WIDTH*HEIGHT*3);
		file.close();
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




	float* xpos;
	cudaMallocHost((void**)&xpos, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc xpos");

	float* ypos;
	cudaMallocHost((void**)&ypos, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc ypos");

	float* zpos;
	cudaMallocHost((void**)&zpos, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc zpos");

	float* xvel;
	cudaMallocHost((void**)&xvel, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc xvel");

	float* yvel;
	cudaMallocHost((void**)&yvel, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc yvel");

	float* zvel;
	cudaMallocHost((void**)&zvel, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc zvel");

	float* mass;
	cudaMallocHost((void**)&mass, sizeof(float) * NUM_BODIES);
	checkCudaError("Malloc mass");
	
	initializeBodies(xpos, ypos, zpos, xvel, yvel, zvel, mass);


	char* image;
	cudaMallocHost((void**)&image, sizeof(char) * WIDTH*HEIGHT*3);
	checkCudaError("Malloc image");

	float* hdImage;
	cudaMallocHost((void**)&hdImage, sizeof(float) * WIDTH*HEIGHT*3);
	checkCudaError("Malloc hdImage");

	renderClear(image, hdImage);
	




	device cudaDevice = devices[0];




	buffer xpos_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer ypos_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zpos_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer xvel_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer yvel_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zvel_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer mass_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer hdImage_buffer = cudaDevice.create_buffer(sizeof(float) * WIDTH*HEIGHT*3).get();

	
	data_futures.push_back(xpos_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, xpos));
	data_futures.push_back(ypos_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, ypos));
	data_futures.push_back(zpos_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, zpos));
	
	data_futures.push_back(xvel_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, xvel));
	data_futures.push_back(yvel_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, yvel));
	data_futures.push_back(zvel_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, zvel));
	
	data_futures.push_back(mass_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, mass));

	data_futures.push_back(hdImage_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, mass));

	wait_all(data_futures);


	program prog_1 = cudaDevice.create_program_with_file("my_nbody_kernel.cu").get();
	program prog_2 = cudaDevice.create_program_with_file("my_nbody_kernel.cu").get();



	std::vector<std::string> flags_1;
	std::string mode_1 = "--gpu-architecture=compute_";
	mode_1.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode_1.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags_1.push_back(mode_1);

	std::vector<std::string> flags_2;
	std::string mode_2 = "--gpu-architecture=compute_";
	mode_2.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode_2.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags_2.push_back(mode_2);



	prog_1.build_sync(flags_1, "interactBodies");
	prog_2.build_sync(flags_2, "GPUrenderBodies");







	/*buffer xpos_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer ypos_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zpos_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer xvel_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer yvel_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zvel_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer mass_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();*/





	hpx::cuda::server::program::Dim3 grid_1;
	hpx::cuda::server::program::Dim3 block_1;

	grid_1.x = 1024;
	grid_1.y = 1;
	grid_1.z = 1;

	block_1.x = (NUM_BODIES+1024-1)/1024;
	block_1.y = 1;
	block_1.z = 1;

	hpx::cuda::server::program::Dim3 grid_2;
	hpx::cuda::server::program::Dim3 block_2;

	grid_2.x = 1025;
	grid_2.y = 1;
	grid_2.z = 1;

	block_2.x = ((NUM_BODIES+1024-1)/1024)+1;
	block_2.y = 1;
	block_2.z = 1;

	// int nBlocks=(NUM_BODIES+1024-1)/1024;
	// interactBodies<<<nBlocks,1024>>>(d_xpos,d_ypos,d_zpos,d_xvel,d_yvel,d_zvel,d_mass);




	// void interactBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass)
	std::vector<hpx::cuda::buffer> args_1;
	args_1.push_back(xpos_buffer);
	args_1.push_back(ypos_buffer);
	args_1.push_back(zpos_buffer);
	args_1.push_back(xvel_buffer);
	args_1.push_back(yvel_buffer);
	args_1.push_back(zvel_buffer);
	args_1.push_back(mass_buffer);

	std::vector<hpx::cuda::buffer> args_2;
	args_2.push_back(xpos_buffer);
	args_2.push_back(ypos_buffer);
	args_2.push_back(zpos_buffer);
	args_2.push_back(xvel_buffer);
	args_2.push_back(yvel_buffer);
	args_2.push_back(zvel_buffer);
	args_2.push_back(mass_buffer);
	args_2.push_back(hdImage_buffer);


	hpx::wait_all(data_futures);

	std::vector<hpx::future<hpx::serialization::serialize_buffer<char>>> temp_res;

	for (int step=1; step<10; step++) {
		auto kernel_future = prog_1.run(args_1, "interactBodies", grid_1, block_1, 0);
		kernel_future.get();

		temp_res.push_back(xpos_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));
		temp_res.push_back(ypos_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));
		temp_res.push_back(zpos_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));

		temp_res.push_back(xvel_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));
		temp_res.push_back(yvel_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));
		temp_res.push_back(zvel_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));

		temp_res.push_back(mass_buffer.enqueue_read_parcel(0, sizeof(float) * NUM_BODIES));


		hpx::when_all(temp_res).then([](auto&& f){
			std::vector<hpx::future<hpx::serialization::serialize_buffer<char>>> futures = f.get();

			float* xpos_temp = reinterpret_cast<float*>(futures[0].get().data());


			for(int i = 0; i < NUM_BODIES; i++){
				std::cout << xpos_temp[0] << std::endl;
			}
		});
	}


/*	for (int step = 1; step < STEP_COUNT; step++) {
		auto kernel_future_1 = prog_1.run(args_1, "interactBodies", grid_1, block_1, 0);
		kernel_future_1.get();

		if (step%RENDER_INTERVAL == 0) {
			hpx::wait_all(data_futures);

			auto kernel_future_2 = prog_2.run(args_2, "GPUrenderBodies", grid_2, block_2, 0);
			kernel_future_2.get();
			
			hdImage = hdImage_buffer.enqueue_read_parcel_sync<float>(0, sizeof(float) * HEIGHT*WIDTH*3);
			
			writeRender(image, hdImage, step);

			renderClear(image, hdImage);

			data_futures.push_back(hdImage_buffer.enqueue_write(0, sizeof(float) * HEIGHT*WIDTH*3, hdImage));
		}
	}
*/




	float* xpos_res;
	cudaMallocHost((void**)&xpos_res, sizeof(float) * NUM_BODIES);
	float* ypos_res;
	cudaMallocHost((void**)&ypos_res, sizeof(float) * NUM_BODIES);
	float* zpos_res;
	cudaMallocHost((void**)&zpos_res, sizeof(float) * NUM_BODIES);

	float* xvel_res;
	cudaMallocHost((void**)&xvel_res, sizeof(float) * NUM_BODIES);
	float* yvel_res;
	cudaMallocHost((void**)&yvel_res, sizeof(float) * NUM_BODIES);
	float* zvel_res;
	cudaMallocHost((void**)&zvel_res, sizeof(float) * NUM_BODIES);

	// Massa vai ficar constante
	float* mass_res;
	cudaMallocHost((void**)&mass_res, sizeof(float) * NUM_BODIES);





	xpos_res = xpos_buffer.enqueue_read_sync<float>(0, sizeof(float) * NUM_BODIES);
	ypos_res = ypos_buffer.enqueue_read_sync<float>(0, sizeof(float) * NUM_BODIES);
	zpos_res = zpos_buffer.enqueue_read_sync<float>(0, sizeof(float) * NUM_BODIES);

	xvel_res = xvel_buffer.enqueue_read_sync<float>(0, sizeof(float) * NUM_BODIES);
	yvel_res = yvel_buffer.enqueue_read_sync<float>(0, sizeof(float) * NUM_BODIES);
	zvel_res = zvel_buffer.enqueue_read_sync<float>(0, sizeof(float) * NUM_BODIES);



	


	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "total elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}




