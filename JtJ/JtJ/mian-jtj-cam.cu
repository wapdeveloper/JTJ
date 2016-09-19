#include<stdio.h>
#include < cuda_runtime.h>
#include <d:/book.h>
#define N   1000
#define M   128
#define K   2
#define threadsPerBlock  64


__global__ void jtj_cam( int * J,	int* blocks ,int* cameraID ,int *cameraCount)
{
	//分配共享内存
	__shared__ int value[threadsPerBlock * 8]; 
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	//列索引
	int colpos=  threadIdx.x & 0x7;      
	//行索引
	int rowpos = threadIdx.x - colpos; 
	int row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	//取出每个相机对应的雅各比矩阵
	int tid =threadIdx.x+cameraID[blockIdx.x];
	while (tid < cameraCount[blockIdx.x]) 
	{
		//共享内存赋值
		value[threadIdx.x] = J[tid];
		for(int j = 0; j < 8; ++j)   
		if(j>colpos||j==colpos) //减少对角重复计算
			row[j] += (value[threadIdx.x] * value[rowpos + j]);
		tid += blockDim.x;
	}
	//每个线程计算完毕
	for(int i = 0; i < 8; ++i)
		value[threadIdx.x * 8 + i] = row[i]; 

	int i=8*threadsPerBlock;
	while (i != 64) 
	{	
		tid =threadIdx.x;
		while(tid < i/2)
		{
			//共享内存规约
			value[tid] += value[tid + i/2];
			tid += 64;
		}  
		i /= 2;
	}
	//得到每个相机对应的矩阵，64个值
    blocks[index]=value[threadIdx.x];
}

int main( void ) 
{
	int j[N], jtj[M],cameraID[K],cameraCount[K];
	int *dev_j,  *dev_jtj ,*dev_cameraID,*dev_cameraCount;

	//分配GPU显存
	HANDLE_ERROR( cudaMalloc( (void**)&dev_j, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_jtj, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cameraID, K * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cameraCount, K * sizeof(int) ) );

	//读取雅可比矩阵
	const char *rpc1Filename = "D:\\in.txt";
	FILE* fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &j[i]);
	}

	//读取雅可比矩阵
	rpc1Filename = "D:\\f.txt";
	fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &jtj[i]);
	}

	cameraID[0]=0;
	cameraID[1]=256;

	cameraCount[0]=256;
	cameraCount[1]=1000;


	// 雅可比矩阵拷贝到GPUx
	HANDLE_ERROR( cudaMemcpy( dev_j, j, N * sizeof(int),
		cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( dev_jtj, jtj, M * sizeof(int),
		cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraID, cameraID, K * sizeof(int),
		cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraCount, cameraCount, K * sizeof(int),
		cudaMemcpyHostToDevice ) );

	//核函数
	dim3 grid(2), block(threadsPerBlock);
	jtj_cam<<<grid, block>>>(dev_j, dev_jtj,dev_cameraID,dev_cameraCount);


	//从CPU拷贝回GPU

	HANDLE_ERROR( cudaMemcpy( j, dev_j, N * sizeof(int),cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( jtj, dev_jtj, M * sizeof(int),cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( cameraID, dev_cameraID, K * sizeof(int),cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy(cameraCount, dev_cameraCount, K * sizeof(int),cudaMemcpyDeviceToHost ) );


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
	HANDLE_ERROR( cudaFree( dev_cameraID ) );
	HANDLE_ERROR( cudaFree( dev_cameraCount ) );
	return 0;
}
