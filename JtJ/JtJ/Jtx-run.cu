#include<stdio.h>
#include < cuda_runtime.h>
#include <d:/book.h>
#define N   1000
#define E    8


//#define n    7     //32-threadIdx   2^8
//#define threadsPerBlock  32


#define n    8    //64-threadIdx    2^9
#define txPerBlock  32



#define M   32
//#define M   256


#define K   2





texture<int4, 1, cudaReadModeElementType> tex_jtc_cam;
texture<int4, 1, cudaReadModeElementType> tex_x_cam;

__global__ void jtx_cam(int* blocks, int* cameraID, int *cameraCount)
{
	//分配共享内存
	//__shared__ int value[txPerBlock * 8]; 
	//取出每个相机雅各比矩阵以及对应的投影误差

	//int4 x1 = tex1Dfetch(tex_x_cam, 0);
	//int4 x2 = tex1Dfetch(tex_x_cam, 1);	
	////拾取绑定雅各比矩阵的纹理内存
	//int4 jc = tex1Dfetch(tex_jtc_cam, threadIdx.x);
	////拾取绑定投影误差的纹理内存
	//   //载入共享内存
	//int index = threadIdx.x<< 2;  
	//value[index + 0] = jc.x;	value[index + 1] = jc.y;
	//value[index + 2] = jc.z;	value[index + 3] = jc.w;
	////线程同步
	//__syncthreads();
	//index = threadIdx.x << 3;  
	//if(threadIdx.x < (txPerBlock >> 1))
	//blocks[threadIdx.x ] = value[index + 0] * x1.x + value[index + 1] * x1.y + value[index + 2] * x1.z + value[index + 3] * x1.w
	//				+ value[index + 4]* x2.x + value[index + 5] * x2.y + value[index + 6] * x2.z + value[index + 7] * x2.w;




	//__shared__ int value[txPerBlock * 8]; 
	//int4 x1 = tex1Dfetch(tex_x_cam, 0);
	//int4 x2 = tex1Dfetch(tex_x_cam, 1);	
	//////拾取绑定雅各比矩阵的纹理内存
	//int4 jc = tex1Dfetch(tex_jtc_cam, threadIdx.x);
	//////拾取绑定投影误差的纹理内存
	//   //载入共享内存
	//int index = threadIdx.x<< 2;  
	//value[index + 0] = jc.x;	value[index + 1] = jc.y;
	//value[index + 2] = jc.z;	value[index + 3] = jc.w;
	//////线程同步
	//__syncthreads();
	//index = threadIdx.x << 3;  
	//if(threadIdx.x < (txPerBlock >> 1))
	//blocks[threadIdx.x ] = value[index + 0] * x1.x + value[index + 1] * x1.y + value[index + 2] * x1.z + value[index + 3] * x1.w
	//				+ value[index + 4]* x2.x + value[index + 5] * x2.y + value[index + 6] * x2.z + value[index + 7] * x2.w;




	//jc = tex1Dfetch(tex_jtc_cam, (threadIdx.x+32));
	////拾取绑定投影误差的纹理内存
	//   //载入共享内存
	//index=(threadIdx.x)<<2;
	//value[index + 0] = jc.x;	value[index + 1] = jc.y;
	//value[index + 2] = jc.z;	value[index + 3] = jc.w;
	////////线程同步
	//__syncthreads();
	//index = ((threadIdx.x) << 3);  
	//if(threadIdx.x < (txPerBlock >> 1))
	//		blocks[threadIdx.x +16] = value[index + 0] * x1.x + value[index + 1] * x1.y + value[index + 2] * x1.z + value[index + 3] * x1.w
	//				+ value[index + 4]* x2.x + value[index + 5] * x2.y + value[index + 6] * x2.z + value[index + 7] * x2.w;



	//分配共享内存
		__shared__ int value[txPerBlock * 8]; 
	//取出每个相机雅各比矩阵以及对应的投影误差

	int4 x1 = tex1Dfetch(tex_x_cam, 0);
	int4 x2 = tex1Dfetch(tex_x_cam, 1);	
	int tid=threadIdx.x;
	int count=0;
	while(tid<64)
	{
		//拾取绑定雅各比矩阵的纹理内存
		int4 jc = tex1Dfetch(tex_jtc_cam, tid);
		//拾取绑定投影误差的纹理内存
		//载入共享内存
		int index = threadIdx.x<< 2;  
		value[index + 0] = jc.x;	value[index + 1] = jc.y;
		value[index + 2] = jc.z;	value[index + 3] = jc.w;
		//线程同步
		__syncthreads();
		index = threadIdx.x << 3;  
		if(threadIdx.x < (txPerBlock >> 1))
			blocks[threadIdx.x+count*(txPerBlock >> 1)] = value[index + 0] * x1.x + value[index + 1] * x1.y + value[index + 2] * x1.z + value[index + 3] * x1.w
			+ value[index + 4]* x2.x + value[index + 5] * x2.y + value[index + 6] * x2.z + value[index + 7] * x2.w;
		tid+=txPerBlock;
		count++;
	}
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
	const char *rpc1Filename = "in.txt";
	FILE* fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &j[i]);
	}

	//读取误差矩阵
	rpc1Filename = "f.txt";
	fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &f[i]);
	}

	cameraID[0]=0;
	cameraID[1]=256;

	cameraCount[0]=256;
	cameraCount[1]=1000;


	//int k=8*threadsPerBlock,n=0;
	//while(k>1)
	//{
	//	k/=2;
	//	n+=1;
	//}
	//printf("%d", n);
	//printf("\n");


	// 雅可比矩阵拷贝到GPU
	HANDLE_ERROR( cudaMemcpy( dev_j, j, N * sizeof(int),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( dev_f, f, E * sizeof(int),cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( dev_jtf, jtf, M * sizeof(int),	cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraID, cameraID, K * sizeof(int),	cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraCount, cameraCount, K * sizeof(int),	cudaMemcpyHostToDevice ) );


	HANDLE_ERROR(cudaBindTexture(0, tex_jtc_cam, dev_j, sizeof(int)*N));
	HANDLE_ERROR(cudaBindTexture(0, tex_x_cam, dev_f, sizeof(int)*E));

	//核函数
	dim3 grid(1), block(txPerBlock);
	jtx_cam<<<grid, block>>>( dev_jtf,dev_cameraID,dev_cameraCount );


	HANDLE_ERROR(cudaUnbindTexture(tex_jtc_cam));
	HANDLE_ERROR(cudaUnbindTexture(tex_x_cam));

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






