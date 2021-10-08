extern "C" {
	__global__ void interactBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NUM_BODIES)
		{		
			float Fx=0.0f; float Fy=0.0f; float Fz=0.0f;
			float xposi=xpos[i];
			float yposi=ypos[i];
			float zposi=zpos[i];
			#pragma unroll
			for(int j=0; j < NUM_BODIES; j++)
			{
				if(i!=j)
				{ 
					vec3 posDiff;
					posDiff.x = (xposi-xpos[j])*TO_METERS;
					posDiff.y = (yposi-ypos[j])*TO_METERS;
					posDiff.z = (zposi-zpos[j])*TO_METERS;
					float dist = sqrt(posDiff.x*posDiff.x+posDiff.y*posDiff.y+posDiff.z*posDiff.z);
					float F = TIME_STEP*(G*mass[i]*mass[j]) / ((dist*dist + SOFTENING*SOFTENING) * dist);
					//float Fa = F/mass[i];
					Fx-=F*posDiff.x;
					Fy-=F*posDiff.y;
					Fz-=F*posDiff.z;
				}	
			}
			xvel[i] += Fx/mass[i];
			yvel[i] += Fy/mass[i];
			zvel[i] += Fz/mass[i];
			xpos[i] += TIME_STEP*xvel[i]/TO_METERS;
			ypos[i] += TIME_STEP*yvel[i]/TO_METERS;
			zpos[i] += TIME_STEP*zvel[i]/TO_METERS;
		}
	}
}