extern "C" {
	#include "Constants.h"


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

	__global__ void renderClear(char* image, float* hdImage) {
		for (int i=0; i<WIDTH*HEIGHT*3; i++) {
			image[i] = 0;
			hdImage[i] = 0.0;
		}
	}

	__global__ void GPUrenderBodies(float* xpos, float* ypos, float* zpos, float* xvel, float* yvel, float* zvel, float* mass, float* hdImage) {
		/// ORTHOGONAL PROJECTION
		int i = blockIdx.x*blockDim.x+threadIdx.x;
		float velocityMax = MAX_VEL_COLOR; //35000
		float velocityMin = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/
					(SYSTEM_SIZE*TO_METERS)); //MIN_VEL_COLOR;
		if(i<NUM_BODIES)
		{
			float vxsqr=xvel[i]*xvel[i];
			float vysqr=yvel[i]*yvel[i];
			float vzsqr=zvel[i]*zvel[i];
			float vMag = sqrt(vxsqr+vysqr+vzsqr);
			int x = (WIDTH/2.0)*(1.0+xpos[i]/(SYSTEM_SIZE*RENDER_SCALE));
			int y = (HEIGHT/2.0)*(1.0+ypos[i]/(SYSTEM_SIZE*RENDER_SCALE));

			if (x>DOT_SIZE && x<WIDTH-DOT_SIZE && y>DOT_SIZE && y<HEIGHT-DOT_SIZE)
			{
				float vPortion = sqrt((vMag-velocityMin) / velocityMax);
				float xPixel = (WIDTH/2.0)*(1.0+xpos[i]/(SYSTEM_SIZE*RENDER_SCALE));
				float yPixel = (HEIGHT/2.0)*(1.0+ypos[i]/(SYSTEM_SIZE*RENDER_SCALE));
				float xP = floor(xPixel);
				float yP = floor(yPixel);
				color c;
				c.r = max(min(4*(vPortion-0.333),1.0),0.0);
				c.g = max(min(min(4*vPortion,4.0*(1.0-vPortion)),1.0),0.0);
				c.b = max(min(4*(0.5-vPortion),1.0),0.0);
				for (int a=-DOT_SIZE/2; a<DOT_SIZE/2; a++)
				{
					for (int b=-DOT_SIZE/2; b<DOT_SIZE/2; b++)
					{
						float cFactor = PARTICLE_BRIGHTNESS /(pow(exp(pow(PARTICLE_SHARPNESS*(xP+a-xPixel),2.0)) + exp(pow(PARTICLE_SHARPNESS*(yP+b-yPixel),2.0)),/*1.25*/0.75)+1.0);
						//colorAt(int(xP+a),int(yP+b),c, cFactor, hdImage);
						int pix = 3*(xP+a+WIDTH*(yP+b));
						hdImage[pix+0] += c.r*cFactor;
						hdImage[pix+1] += c.g*cFactor;
						hdImage[pix+2] += c.b*cFactor;
					}
				}
			}
		}
	}
}