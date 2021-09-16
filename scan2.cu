
#include "scan2.h"
#include "gpuerrors.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define TILEX 1024


__global__ void kernelFunc(float* ad, float* cd, float* lasts) {
	__shared__ float as[TILEX];
	
	float h = ad[TILEX*bx+tx];
	as[tx] = h;
	__syncthreads();
	
	for (int p = 2; p <= TILEX; p *= 2){
		if((tx+1)%p == 0){
			as[tx] += as[tx - p/2];
		}
		__syncthreads();
	}
	
	if(tx+1 == TILEX){
		as[tx] = 0;
	}
	__syncthreads();
	
	for(int p = TILEX; p >1; p/= 2){
		if((tx+1)%p == 0){
			float r = as[tx];
			float l = as[tx - p/2];
			as[tx - p/2] = r;
			as[tx] = r + l;
		}
		__syncthreads();
	}

	h = as[tx] + h;
	cd[TILEX*bx+tx] = h;
	if(tx+1 == TILEX){
		lasts[bx] = h;
	}
}

__global__ void littleFunc(float* ad, float* cd, float* lasts, int tile) {
	__shared__ float as[128];
	
	float h = ad[tile*bx+tx];
	as[tx] = h;
	__syncthreads();
	
	for (int p = 2; p <= tile; p *= 2){
		if((tx+1)%p == 0){
			as[tx] += as[tx - p/2];
		}
		__syncthreads();
	}
	
	if(tx+1 == tile){
		as[tx] = 0;
	}
	__syncthreads();
	
	for(int p = tile; p >1; p/= 2){
		if((tx+1)%p == 0){
			float r = as[tx];
			float l = as[tx - p/2];
			as[tx - p/2] = r;
			as[tx] = r + l;
		}
		__syncthreads();
	}

	h = as[tx] + h;
	cd[tile*bx+tx] = h;
	if(tx+1 == tile){
		lasts[bx] = h;
	}
}

__global__ void merge(float* ad, float* cd, float num) {
	__shared__ float n;
	if(tx == 0){
		n = num;
		if(bx > 0){
			n += cd[bx-1];
		}
	}
	__syncthreads();
	ad[TILEX*bx+tx] += n;
}

void gpuKernel(float* a, float* c,int n) {
	unsigned long long int rounds = n / (1024*1024*32);
	if (rounds < 1){
		rounds = 1;
	}		
	float* ad;
	float* cd;
	float* lasts;
	float* scaned_lasts;
	float* lasts2;
	float* scaned_lasts2;
	float* nextc;
	float next[1] = {0};
	
	HANDLE_ERROR(cudaMalloc((void**)&ad, n/rounds * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&cd, n/rounds * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&lasts, n/rounds/TILEX * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&scaned_lasts, n/rounds/TILEX * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&lasts2, n/rounds/TILEX/TILEX * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&scaned_lasts2, n/rounds/TILEX/TILEX * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&nextc, sizeof(float)));
	
	for (unsigned long long int i = 0; i < rounds; i ++){
		HANDLE_ERROR(cudaMemcpy(ad, a+(i*n/rounds), n/rounds * sizeof(float), cudaMemcpyHostToDevice));
		dim3 dimBlock(TILEX);
		kernelFunc<<< n/rounds/TILEX, dimBlock >>>(ad, cd, lasts);
		kernelFunc<<< n/rounds/TILEX/TILEX, dimBlock >>>(lasts, scaned_lasts, lasts2);
		littleFunc<<< 1, n/rounds/TILEX/TILEX >>>(lasts2, scaned_lasts2, nextc, n/rounds/TILEX/TILEX);
		
		merge<<< n/rounds/TILEX/TILEX, dimBlock >>>(scaned_lasts, scaned_lasts2, next[0]);
		merge<<< n/rounds/TILEX, dimBlock >>>(cd, scaned_lasts, 0);
		float w = next[0];
		cudaMemcpy(next, nextc, sizeof(float), cudaMemcpyDeviceToHost);
		next[0] += w;
		HANDLE_ERROR(cudaMemcpy(c+i*n/rounds, cd, n/rounds * sizeof(float), cudaMemcpyDeviceToHost));
	}
}
