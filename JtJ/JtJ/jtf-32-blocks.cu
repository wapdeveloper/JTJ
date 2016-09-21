#include<stdio.h>
#include < cuda_runtime.h>
#include <d:/book.h>
#define N   1000
#define E    125


#define M   16
//#define M   256
#define K   2
#define threadsPerBlock  32

texture<int4, 1, cudaReadModeElementType> tex_jtf_cam;
texture<int, 1, cudaReadModeElementType> tex_f_cam;



__global__ void jtf_cam( int* blocks,int* cameraID ,int *cameraCount)
{
	////分配共享内存
	__shared__ int value[threadsPerBlock*8]; 
	int rx = 0, ry = 0, rz = 0, rw = 0; 
	int tid =threadIdx.x+cameraID[blockIdx.x]/8;
	while (tid < cameraCount[blockIdx.x]/8) 
	{
		int4 jab= tex1Dfetch(tex_jtf_cam, (tid*2)+threadIdx.y);
		int err= tex1Dfetch(tex_f_cam, tid);
		rx += jab.x * err;		ry += jab.y * err;
        rz += jab.z * err;  	rw += jab.w * err;
		tid += blockDim.x;
	}

	int index=(threadIdx.x+(threadIdx.y*threadsPerBlock))<<2;  //5->32
	value[index+0]=rx;		value[index+1]=ry;
	value[index+2]=rz;		value[index+3]=rw;

	int i=8*threadsPerBlock;
	while (i > 8) 
	{	
		tid =threadIdx.x+(threadIdx.y<<7);
		while(tid < (i/4+(threadIdx.y<<7)))
		{
			//共享内存规约
			value[tid] += value[tid + i/4];
			tid += threadsPerBlock;
		}  
		i /= 2;
	}
	if(threadIdx.x < 4)
        blocks[threadIdx.x+(threadIdx.y<<2)+(blockIdx.x<<3)]=value[threadIdx.x+(threadIdx.y<<7)];
}






int main( void ) 
{
	int j[N], f[E], jtf[M],cameraID[K],cameraCount[K];
	int *dev_j,  *dev_f, *dev_jtf ,*dev_cameraID,*dev_cameraCount;

	//分配GPU显存
	HANDLE_ERROR( cudaMalloc( (void**)&dev_j, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_f, E * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_jtf, M * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cameraID, K * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cameraCount, K * sizeof(int) ) );

	//读取雅可比矩阵
	const char *rpc1Filename = "D:\\in.txt";
	FILE* fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &j[i]);
	}

	//读取误差矩阵
	rpc1Filename = "D:\\f.txt";
	fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &f[i]);
	}

	cameraID[0]=0;
	cameraID[1]=256;

	cameraCount[0]=256;
	cameraCount[1]=1000;






	// 雅可比矩阵拷贝到GPU
	HANDLE_ERROR( cudaMemcpy( dev_j, j, N * sizeof(int),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( dev_f, f, E * sizeof(int),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( dev_jtf, jtf, M * sizeof(int),	cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraID, cameraID, K * sizeof(int),	cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraCount, cameraCount, K * sizeof(int),	cudaMemcpyHostToDevice ) );


	HANDLE_ERROR(cudaBindTexture(0, tex_jtf_cam, dev_j, sizeof(int)*N));
	HANDLE_ERROR(cudaBindTexture(0, tex_f_cam, dev_f, sizeof(int)*E));

	//核函数
	dim3 grid(2), block(threadsPerBlock,2);
	jtf_cam<<<grid, block>>>( dev_jtf,dev_cameraID,dev_cameraCount );


	HANDLE_ERROR(cudaUnbindTexture(tex_jtf_cam));
	HANDLE_ERROR(cudaUnbindTexture(tex_f_cam));

	//从CPU拷贝回GPU
	//HANDLE_ERROR( cudaMemcpy( j, dev_j, N * sizeof(int),cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy( f, dev_f, E * sizeof(int),cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( jtf, dev_jtf, M * sizeof(int),cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy( cameraID, dev_cameraID, K * sizeof(int),cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy(cameraCount, dev_cameraCount, K * sizeof(int),cudaMemcpyDeviceToHost ) );


	// display the results
	for (int i=0; i<M; i++) 
	{
		printf( "%d   ", jtf[i] );
		if((i+1)%8==0)
			printf( "\n");
	}

	FILE * pFile;
	pFile = fopen("myfile.txt", "w");


	char x[M][10] = { "", };
	for (int i = 0; i<M; i++)
	{
		itoa(jtf[i], x[i], 10);
	}

	if (pFile != NULL)
	{
		for (size_t i = 0; i < M; i++)
		{
			fputs(x[i], pFile);
			fputs("\n", pFile);
		}
		fclose(pFile);
	}



	//free the memory allocated on the CPU
	//free(j);
	//free(f);
	//free(jtf);
	//free(cameraID);
	//free(cameraCount);

	// free the memory allocated on the GPU
	//HANDLE_ERROR( cudaFree( dev_j ) );
	//HANDLE_ERROR( cudaFree( dev_f ) );
	//HANDLE_ERROR( cudaFree( dev_jtf ) );
	//HANDLE_ERROR( cudaFree( dev_cameraID ) );
	//HANDLE_ERROR( cudaFree( dev_cameraCount ) );



	return 0;
}






