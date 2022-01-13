extern "C" {


	__device__ float my_rsqrt(float x) {
		return rsqrtf(x);
	}


	__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj) {
	    float3 r;

	    // r_ij  [3 FLOPS]
	    r.x = bj.x - bi.x;
	    r.y = bj.y - bi.y;
	    r.z = bj.z - bi.z;

	    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
	    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
	    //distSqr += getSofteningSquared<T>();
		distSqr += 0.00125f;


	    float invDist = my_rsqrt(distSqr);
	    float invDistCube =  invDist * invDist * invDist;

	    // s = m_j * invDistCube [1 FLOP]
	    float s = bj.w * invDistCube;

	    // a_i =  a_i + s * r_ij [6 FLOPS]
	    ai.x += r.x * s;
	    ai.y += r.y * s;
	    ai.z += r.z * s;

	    return ai;
	}


	__device__ float3 computeBodyAccel(float4 bodyPos, float4* positions, int numTiles)	{
	    //extern __shared__ float4* sharedPos;
		extern __shared__ float4 sharedPos[256];


	    float3 acc = make_float3(0, 0, 0);
	    for (int tile = 0; tile < numTiles; tile++)
	    {
	        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

	        __syncthreads();

	#pragma unroll 128

	        for (unsigned int counter = 0; counter < blockDim.x; counter++) {
	            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);
	        }

	        __syncthreads();

	    }

	    return acc;
	}


	__global__ void	integrateBodies(float4* newPos, float4* oldPos, float4* vel, int* deviceOffsetBuffer, int* deviceNumBodiesBuffer, float* deltaTimeBuffer, float* dampingBuffer, int* numTilesBuffer){
		
		int deviceOffset = deviceOffsetBuffer[0];
		int deviceNumBodies = deviceNumBodiesBuffer[0];
		float deltaTime = deltaTimeBuffer[0];
		float damping = dampingBuffer[0];
		int numTiles = numTilesBuffer[0];

	    int index = blockIdx.x * blockDim.x + threadIdx.x;

	    if (index >= deviceNumBodies)
	    {
	        return;
	    }

	    float4 position = oldPos[deviceOffset + index];

	    
	    float3 accel = computeBodyAccel(position, oldPos, numTiles);

	    float4 velocity = vel[deviceOffset + index];

	    velocity.x += accel.x * deltaTime;
	    velocity.y += accel.y * deltaTime;
	    velocity.z += accel.z * deltaTime;

	    velocity.x *= damping;
	    velocity.y *= damping;
	    velocity.z *= damping;

	    // new position = old position + velocity * deltaTime
	    position.x += velocity.x * deltaTime;
	    position.y += velocity.y * deltaTime;
	    position.z += velocity.z * deltaTime;

	    // store new position and velocity
	    newPos[deviceOffset + index] = position;
	    vel[deviceOffset + index]    = velocity;
	}

}
