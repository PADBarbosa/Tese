#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;


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
	




	device cudaDevice = devices[0];






	buffer xpos_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer ypos_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zpos_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer xvel_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer yvel_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zvel_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer mass_buffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();






	data_futures.push_back(xpos_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, xpos));
	data_futures.push_back(ypos_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, ypos));
	data_futures.push_back(zpos_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, zpos));
	
	data_futures.push_back(xvel_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, xvel));
	data_futures.push_back(yvel_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, yvel));
	data_futures.push_back(zvel_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, zvel));
	
	data_futures.push_back(mass_buffer.enqueue_write(0, sizeof(float) * NUM_BODIES, mass));





	program prog = cudaDevice.create_program_with_file("my_nbody_kernel.cu").get();



	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);



	prog_1.build_sync(flags_1, "interactBodies");







	/*buffer xpos_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer ypos_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zpos_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer xvel_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer yvel_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();
	buffer zvel_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();

	buffer mass_outbuffer = cudaDevice.create_buffer(sizeof(float) * NUM_BODIES).get();*/





	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	grid.x = 1024;
	grid.y = 1;
	grid.z = 1;

	block.x = (NUM_BODIES+1024-1)/1024;
	block.y = 1;
	block.z = 1;

	// int nBlocks=(NUM_BODIES+1024-1)/1024;
	// interactBodies<<<nBlocks,1024>>>(d_xpos,d_ypos,d_zpos,d_xvel,d_yvel,d_zvel,d_mass);




	// void interactBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass)
	std::vector<hpx::cuda::buffer> args;
	args.push_back(xpos_buffer);
	args.push_back(ypos_buffer);
	args.push_back(zpos_buffer);
	args.push_back(xvel_buffer);
	args.push_back(yvel_buffer);
	args.push_back(zvel_buffer);
	args.push_back(mass_buffer);


	hpx::wait_all(data_futures);



	for (int step=1; step<10; step++) {
		prog.run(args, "interactBodies", grid, block, 0);
	}

	





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
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}




