
__global__ void jtj_cam(float* blocks, float * J, float* cameraID, float *cameraCount)
{
	//���乲���ڴ�
	__shared__ float value[txPerBlock * 8]; 
	int index = blockIdx.x * blockDim.x;
	//������
	int colpos = threadIdx.x & 0x7;      
	//������
	int rowpos = threadIdx.x - colpos; 
	int row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	//ȡ��Jc����Ԫ��
	int tid = threadIdx.x + cameraID[blockIdx.x];
	while (tid < cameraCount[blockIdx.x]) 
	{
		//��Jc�������빲���ڴ�
		value[threadIdx.x] = J[tid];
		for(int j = 0; j < 8; ++j)   
		if(j > colpos || j == colpos) //���ٶԽ��ظ�����
			row[j] += (value[threadIdx.x] * value[rowpos + j]);
		tid += txPerBlock;
	}
	//�߳�ͬ��
	__syncthreads();
	//ÿ���̼߳������
	for(int i = 0; i < 8; ++i)
		value[threadIdx.x * 8 + i] = row[i]; 
	//�����ڴ��Լ
	int i = txPerBlock * 8;
	while (i > 64) 
	{	
		tid = threadIdx.x;
		while(tid < (i >> 1))
		{
			value[tid] += value[tid + (i >> 1)];
			tid += txPerBlock;
		}  
		i >>= 1;
	}
	//�õ�ÿ�������Ӧ�ľ���64��ֵ
	tid = threadIdx.x;
	while(tid<64)
	{
  		blocks[tid + (blockIdx.x << 6)] = value[tid];	
		tid += txPerBlock;
	}
}
