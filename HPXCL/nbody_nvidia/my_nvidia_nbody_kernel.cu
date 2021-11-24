extern "C" {


	__device__ float my_rsqrt(float x) {
		return rsqrtf(x);
	}

	__device__ void bodyBodyInteraction(float* acc, float* bodyPos, float* sharedPos) {
		float r[3];
		r[0] = sharedPos[0] - bodyPos[0];
		r[1] = sharedPos[1] - bodyPos[1];
		r[2] = sharedPos[2] - bodyPos[2];

		float distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

		distSqr += 0.00125f;

		float invDist = my_rsqrt(distSqr);
		float invDistCube =  invDist * invDist * invDist;

		float s = sharedPos[3] * invDistCube;

		acc[0] += r[0] * s;
		acc[1] += r[1] * s;
		acc[2] += r[2] * s;

	}



	__device__ float* computeBodyAccel(float* accel, float* bodyPos, float* oldPos, int numTiles){
		extern __shared__ float sharedPos[1024];

		float acc[3];
		for(int i = 0; i < 3; i++){
			acc[i] = 0;
		}

		for (int tile = 0; tile < numTiles; tile++){
			sharedPos[threadIdx.x*4] = oldPos[(tile * blockDim.x + threadIdx.x)*4];
			sharedPos[threadIdx.x*4 + 1] = oldPos[(tile * blockDim.x + threadIdx.x)*4 + 1];
			sharedPos[threadIdx.x*4 + 2] = oldPos[(tile * blockDim.x + threadIdx.x)*4 + 2];
			sharedPos[threadIdx.x*4 + 3] = oldPos[(tile * blockDim.x + threadIdx.x)*4 + 3];

			__syncthreads();

			#pragma unroll 128

			for (unsigned int counter = 0; counter < blockDim.x; counter++){
				float sharedPosTemp[4];
				sharedPosTemp[0] = sharedPos[counter*4];
				sharedPosTemp[1] = sharedPos[counter*4 + 1];
				sharedPosTemp[2] = sharedPos[counter*4 + 2];
				sharedPosTemp[3] = sharedPos[counter*4 + 3];

				bodyBodyInteraction(acc, bodyPos, sharedPosTemp);
			}

			__syncthreads();
		}

		for(int i = 0; i < 3; i++){
			accel[i] = acc[i];
		}
	}



	__global__ void	integrateBodies(float* newPos, float* oldPos, float* vel){
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		int deviceOffset = 0;
		int deviceNumBodies = 69632;
		float deltaTime = 0.016;
		float damping = 1;
		int numTiles = 272;

		if (index*4 >= deviceNumBodies*4)
		{
			return;
		}

		float bodyPos[4];
		bodyPos[0] = oldPos[index*4];
		bodyPos[1] = oldPos[index*4 + 1];
		bodyPos[2] = oldPos[index*4 + 2];
		bodyPos[3] = oldPos[index*4 + 3];
		
		float accel[3];
		accel[0] = 1;
		accel[1] = 2;
		accel[2] = 3;

		computeBodyAccel(accel, bodyPos, oldPos, numTiles);
		
		float velocity[4];
		velocity[0] = vel[deviceOffset + index*4];
		velocity[1] = vel[deviceOffset + index*4 + 1];
		velocity[2] = vel[deviceOffset + index*4 + 2];
		velocity[3] = vel[deviceOffset + index*4 + 3];
		
		velocity[0] += accel[0] * deltaTime;
		velocity[1] += accel[1] * deltaTime;
		velocity[2] += accel[2] * deltaTime;

		velocity[0] *= damping;
		velocity[1] *= damping;
		velocity[2] *= damping;

		float position[3];
		position[0] = oldPos[deviceOffset + index*4];
		position[1] = oldPos[deviceOffset + index*4 + 1];
		position[2] = oldPos[deviceOffset + index*4 + 2];
		
		position[0] += velocity[0] * deltaTime;
		position[1] += velocity[1] * deltaTime;
		position[2] += velocity[2] * deltaTime;

		newPos[deviceOffset + index*4] = position[0];
		newPos[deviceOffset + index*4 + 1] = position[1];
		newPos[deviceOffset + index*4 + 2] = position[2];

		vel[deviceOffset + index*4] = velocity[0];
		vel[deviceOffset + index*4 + 1] = velocity[1];
		vel[deviceOffset + index*4 + 2] = velocity[2];

	}

}
