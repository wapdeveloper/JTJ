#include<stdio.h>
#include < cuda_runtime.h>
#include <d:/book.h>
#define N   1000
#define M   128
#define K   2
#define txPerBlock  32


__global__ void jtj_cam( int * J,	int* blocks ,int* cameraID ,int *cameraCount)
{
	//���乲���ڴ�
	__shared__ int value[txPerBlock * 8]; 
	int index=blockIdx.x*blockDim.x;
	//������
	int colpos=  threadIdx.x & 0x7;      
	//������
	int rowpos = threadIdx.x - colpos; 
	int row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	//ȡ��ÿ�������Ӧ���Ÿ��Ⱦ���
	int tid =threadIdx.x+cameraID[blockIdx.x];
	while (tid < cameraCount[blockIdx.x]) 
	{
		//���Ÿ��Ⱦ������빲���ڴ�
		value[threadIdx.x] = J[tid];
		for(int j = 0; j < 8; ++j)   
		if(j>colpos||j==colpos) //���ٶԽ��ظ�����
			row[j] += (value[threadIdx.x] * value[rowpos + j]);
		tid += txPerBlock;
	}
	//�߳�ͬ��
	__syncthreads();
	//ÿ���̼߳������
	for(int i = 0; i < 8; ++i)
		value[threadIdx.x * 8 + i] = row[i]; 
	//�����ڴ��Լ
	int i=8*txPerBlock;
	while (i > 64) 
	{	
		tid =threadIdx.x;
		while(tid < (i>>1))
		{
			value[tid] += value[tid + (i>>1)];
			tid += txPerBlock;
		}  
		i >>=1;
	}
	//�õ�ÿ�������Ӧ�ľ���64��ֵ
	tid=threadIdx.x;
	while(tid<64)
	{
  		blocks[tid+(blockIdx.x<<6)]=value[tid];	
		tid+=txPerBlock;
	}
}

int main( void ) 
{
	int j[N], jtj[M],cameraID[K],cameraCount[K];
	int *dev_j,  *dev_jtj ,*dev_cameraID,*dev_cameraCount;

	//����GPU�Դ�
	HANDLE_ERROR( cudaMalloc( (void**)&dev_j, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_jtj, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cameraID, K * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_cameraCount, K * sizeof(int) ) );

	//��ȡ�ſɱȾ���
	const char *rpc1Filename = "in.txt";
	FILE* fid21 = fopen(rpc1Filename, "rt");
	for (int i = 0; i < N; i++)
	{
		fscanf(fid21, "%d ", &j[i]);
	}

	////��ȡ�ſɱȾ���
	//rpc1Filename = "D:\\f.txt";
	//fid21 = fopen(rpc1Filename, "rt");
	//for (int i = 0; i < N; i++)
	//{
	//	fscanf(fid21, "%d ", &jtj[i]);
	//}

	cameraID[0]=0;
	cameraID[1]=256;

	cameraCount[0]=256;
	cameraCount[1]=1000;


	// �ſɱȾ��󿽱���GPUx
	HANDLE_ERROR( cudaMemcpy( dev_j, j, N * sizeof(int),
		cudaMemcpyHostToDevice ) );	
	HANDLE_ERROR( cudaMemcpy( dev_jtj, jtj, M * sizeof(int),
		cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraID, cameraID, K * sizeof(int),
		cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_cameraCount, cameraCount, K * sizeof(int),
		cudaMemcpyHostToDevice ) );

	//�˺���
	dim3 grid(2), block(txPerBlock);
	jtj_cam<<<grid, block>>>(dev_j, dev_jtj,dev_cameraID,dev_cameraCount);


	//��CPU������GPU

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
