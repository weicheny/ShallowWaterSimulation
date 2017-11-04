#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
extern "C" {
#include "shallow2d.cuh"
#include "shallow2d_base.h"
}

typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int nx, int ny, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int nx, int ny, int field_stride);

void testShallow2d_by_reference(float* cxy, 
				float* FU, float* GU, const float* U,
                int nx, int ny, int field_stride){
	shallow2d_flux_cu(FU, GU, U, nx, ny, field_stride);
	shallow2d_speed_cu(cxy, U, nx, ny, field_stride);
}

void testShallow2d_baseline(float* cxy, 
				float* FU, float* GU, const float* U,
                int nx, int ny, int field_stride){
	shallow2d_flux(FU, GU, U, nx*ny, field_stride);
	shallow2d_speed(cxy, U, nx*ny, field_stride);
}

void print_array(float* array, int len) {
	for(int i = 0; i < len; i++) {
	    printf("%.2f ", array[i]);    
	}
	printf("\n");
}

int main(int argc, char** argv){
	cudaEvent_t start,stop;
	float ms;
	const int nx = 1, ny = 2;
	const int ncell = nx * ny;
	const int field_stride = nx * ny;
	float cxy[2] = {1.0, 2.0};
	float FU[ncell * 3], GU[ncell * 3], U[ncell * 3];
	int i;
	for (i = 0; i < ncell * 3; i++) {
    	FU[i] = 1; GU[i] = 1; U[i] = 1;
	}
	// print_array(FU, ncell*3);

	// Execute baseline code
	double t0 = omp_get_wtime();
	testShallow2d_baseline(cxy, FU, GU, U, nx, ny, field_stride);
    double t1 = omp_get_wtime();
    printf("CPU code time: %f\n", t1-t0);
    // save true values
	float tFU[ncell * 3], tGU[ncell * 3], tU[ncell * 3];
	for (i = 0; i < ncell * 3; i++) {
    	tFU[i] = FU[i]; tGU[i] = GU[i]; tU[i] = U[i];
	}

	// Reset
	for (i = 0; i < ncell * 3; i++) {
    	FU[i] = 1; GU[i] = 1; U[i] = 1;
	}

	// Execute on GPU
	// device copies of FU, GU, U
    float *dev_FU, *dev_GU, *dev_U, *dev_cxy;
    int size = ncell*3*sizeof(float);
    cudaMalloc( (void**)&dev_FU, size );
    cudaMalloc( (void**)&dev_GU, size );
    cudaMalloc( (void**)&dev_U,  size );
    cudaMalloc( (void**)&dev_cxy, 2*sizeof(float) );
    // We only need to copy to device once, this time should be amortized.
    cudaMemcpy( dev_FU, FU, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_GU, GU, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_U,  U,  size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_cxy, cxy, size, cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	testShallow2d_by_reference(dev_cxy, dev_FU, dev_GU, dev_U, nx, ny, field_stride);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU: %f ms.",ms);

	cudaMemcpy( FU, dev_FU, size, cudaMemcpyDeviceToHost );
	cudaMemcpy( GU, dev_GU, size, cudaMemcpyDeviceToHost );
	cudaMemcpy( U,  dev_U,  size, cudaMemcpyDeviceToHost );	

	printf("Check correctness ");
	for (i = 0; i < ncell * 3; i++) {
    	if (FU[i] != tFU[i] or GU[i] != tFU[i] or U[i] != tU[i]){
    		printf("Wrong! \n");
    	}
	}
	printf("\n");

	cudaFree( dev_FU );
	cudaFree( dev_GU );
	cudaFree( dev_U );
}