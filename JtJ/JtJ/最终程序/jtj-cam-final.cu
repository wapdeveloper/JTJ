
__global__ void jtj_cam(float* blocks, float * J, float* cameraID, float *cameraCount)
{
	//分配共享内存
	__shared__ float value[txPerBlock * 8]; 
	int index = blockIdx.x * blockDim.x;
	//列索引
	int colpos = threadIdx.x & 0x7;      
	//行索引
	int rowpos = threadIdx.x - colpos; 
	int row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	//取出Jc矩阵元素
	int tid = threadIdx.x + cameraID[blockIdx.x];
	while (tid < cameraCount[blockIdx.x]) 
	{
		//将Jc矩阵载入共享内存
		value[threadIdx.x] = J[tid];
		for(int j = 0; j < 8; ++j)   
		if(j > colpos || j == colpos) //减少对角重复计算
			row[j] += (value[threadIdx.x] * value[rowpos + j]);
		tid += txPerBlock;
	}
	//线程同步
	__syncthreads();
	//每个线程计算完毕
	for(int i = 0; i < 8; ++i)
		value[threadIdx.x * 8 + i] = row[i]; 
	//共享内存规约
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
	//得到每个相机对应的矩阵，64个值
	tid = threadIdx.x;
	while(tid<64)
	{
  		blocks[tid + (blockIdx.x << 6)] = value[tid];	
		tid += txPerBlock;
	}
}
