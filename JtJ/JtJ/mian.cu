#include<stdio.h>
#include < cuda_runtime.h>
#include <d:/book.h>
#define N   1000
#define M   64
#define threadsPerBlock   64


__global__ void jtj_each_cam( float * J,	float* blocks ,int nJEC)
{
	__shared__ float value[threadsPerBlock * 8]; 
	int index=threadIdx.x + blockIdx.x * blockDim.x;
	int colpos=  threadIdx.x & 0x7;      
	int rowpos = threadIdx.x - colpos; 
	float row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	int tid =index;
	while (tid < nJEC) 
	{
		value[threadIdx.x] = J[tid];
		for(int j = 0; j < 8; ++j)   
		if(j>=colpos)
			row[j] += (value[threadIdx.x] * value[rowpos + j]);
		tid += gridDim.x*blockDim.x;	
	}
	for(int i = 0; i < 8; ++i)   
		value[threadIdx.x * 8 + i] = row[i]; 
	int i=8*threadsPerBlock;
	while (i != 64) 
	{	
		tid =threadIdx.x;
		while(tid < i/2)
		{
			value[tid] += value[tid + i/2];
			tid += threadsPerBlock;
		}  
		i /= 2;
	}
	blocks[index]=value[threadIdx.x];
    blocks[index]=blocks[index]+blocks[index+64];
}

int main( void ) 
{
	float j[N], jtj[M];
	float *dev_j,  *dev_jtj;

	//·ÖÅäGPUÏÔ´æ
	HANDLE_ERROR( cudaMalloc( (void**)&dev_j, N * sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_jtj, N * sizeof(float) ) );



	//¶ÁÈ¡ÑÅ¿É±È¾ØÕó
	const char *rpc1Filename = "D:\\in.txt";
	FILE* fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%f ", &j[i]);
	}


	// ÑÅ¿É±È¾ØÕó¿½±´µ½GPU
	HANDLE_ERROR( cudaMemcpy( dev_jtj, jtj, M * sizeof(float),
		cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_j, j, N * sizeof(float),
		cudaMemcpyHostToDevice ) );

	//ºËº¯Êý
	dim3 grid(2), block(threadsPerBlock);
	jtj_each_cam<<<grid, block>>>(
		dev_j, dev_jtj,N);


	//´ÓCPU¿½±´»ØGPU
	HANDLE_ERROR( cudaMemcpy( jtj, dev_jtj, M * sizeof(float),cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( j, dev_j, N * sizeof(float),cudaMemcpyDeviceToHost ) );

	// display the results
	for (int i=0; i<M; i++) 
	{
		printf( "%d   ", jtj[i] );
		if((i+1)%8==0)
			printf( "\n");
	}

	// free the memory allocated on the GPU
	HANDLE_ERROR( cudaFree( dev_j ) );
	HANDLE_ERROR( cudaFree( dev_jtj ) );

	return 0;
}
