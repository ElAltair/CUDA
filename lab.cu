/*
 ============================================================================
 Name        : lab.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>

//#include <stdio.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

struct Params
{
	int length;
	int time;
	float dx;
	float dt;
	float startT;
	float endT;
};

void initialize(float **data, Params inParam)
{
	for (unsigned i = 0; i < inParam.time; ++i)
		data[i] = new float[inParam.length];
}

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *OldData,float *NewData, Params inParams) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
			if(idx == 0)
				NewData[idx] = inParams.startT;
			else if (idx == inParams.length - 1 )
				NewData[idx] = inParams.endT * inParams.dt + OldData[idx];
			else
			{
				NewData[idx] = ((OldData[idx+1] - 2 * OldData[idx] + OldData[idx-1])* inParams.dt)/inParams.dx * inParams.dx + OldData[idx];
			}

}

/**
 * Host function that copies the data and launches the work on GPU
 */
void gpuReciprocal(float *data, Params inParams)
{
	float *rc = new float[inParams.length];
	float *gpuOldData;
	float *gpuNewData;

	cudaEvent_t GPUstart, GPUstop;
		float GPUtime = 0.0f;


		cudaEventCreate(&GPUstart);
		cudaEventCreate(&GPUstop);

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuOldData, sizeof(float)*inParams.length));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuOldData, rc, sizeof(float)*inParams.length, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuNewData, sizeof(float)*inParams.length));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuNewData, data, sizeof(float)*inParams.length, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 26;
	const int blockCount = 1;
	for(int i = 0; i < inParams.time; ++i)
	{
		cudaEventRecord(GPUstart, 0);
		if (i % 2 == 0)
		{
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuOldData, gpuNewData, inParams);
			cudaEventRecord(GPUstop, 0);
			CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuNewData, sizeof(float)*inParams.length, cudaMemcpyDeviceToHost));

		}
			else
			{

			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuNewData, gpuOldData, inParams);
			cudaEventRecord(GPUstop, 0);
		    CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuOldData, sizeof(float)*inParams.length, cudaMemcpyDeviceToHost));
			}

		cudaEventSynchronize(GPUstop);

		std::cout << i <<": ";
		for(int i =0; i < inParams.length; ++i)
						std::cout << rc[i] << " ";
		std::cout << std::endl;
		float temp;
		cudaEventElapsedTime(&temp,GPUstart, GPUstop);
		GPUtime += temp;

	}
	printf("GPU time = %.3f ms\n",GPUtime);
}

void initializeData(float* data, unsigned size,  float defaultValue)
{
	for(int i = 0; i < size; ++i)
		data[i] = defaultValue;
}



void print(float **a, unsigned h, unsigned w)
{
	for(int i = 0; i < h; ++i)
	{
		std::cout << i << ": ";
		for(int j = 0; j < w; ++j)
			std::cout << a[i][j] << " ";
		std::cout << std::endl;
	}

}


float **cpuReciprocal(float **data,Params inParams)
{
	float CPUstart, CPUstop;
	float CPUtime = 0.0f;

	float **cpuResult = new float*[inParams.time];
	initialize(cpuResult, inParams);

	float * currentData = cpuResult[0];
    initializeData(currentData, inParams.length, 0);


     currentData[0] = inParams.startT;

	for(int i =1; i < inParams.time; ++i)
	{
		cpuResult[i][0] = inParams.startT;
		for(int j = 1; j < inParams.length - 1; ++j)
		{
			cpuResult[i][j] = ((cpuResult[i-1][j+1] - 2 * cpuResult[i-1][j] + cpuResult[i-1][j-1])* inParams.dt)/inParams.dx * inParams.dx + cpuResult[i-1][j];
		}
		cpuResult[i][inParams.length-1] = inParams.endT * inParams.dt + cpuResult[i-1][inParams.length-1];
	}

	return cpuResult;
}






int main(void)
{
	Params mainParams;
	mainParams.length = 13;
	mainParams.time = 20;
	mainParams.dx = 0.5;
	mainParams.dt = 0.1;
	mainParams.startT = 0.0;
	mainParams.endT = 5.0;

	float **data;
	data = new float*[mainParams.time];
	initialize(data, mainParams);

	std::cout << "CPU" << std::endl;
	float **recCpu = cpuReciprocal(data, mainParams);
	print(recCpu, mainParams.time, mainParams.length);

	std::cout << "GPU" << std::endl;
	float *dataTwo;
	dataTwo = new float[mainParams.length];
	gpuReciprocal(dataTwo, mainParams);
	//print(recGpu, mainParams.time, mainParams.length);
//	float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
//	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);

	/* Verify the results */
	//std::cout<<"gpuSum = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;

	/* Free memory */
	//delete[] data;
	//delete[] recCpu;
	//delete[] recGpu;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

