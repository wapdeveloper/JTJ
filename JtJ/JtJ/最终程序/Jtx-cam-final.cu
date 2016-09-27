texture<float4, 1, cudaReadModeElementType> tex_jc_cam;
texture<float4, 1, cudaReadModeElementType> tex_x_cam;

__global__ void jtx_cam(int* result, int* cameraID, int *cameraCount)
{
	//分配共享内存
	__shared__ int value[txPerBlock * 8]; 
	//取出Jc矩阵元素以及对应的向量
	float4 x1 = tex1Dfetch(tex_x_cam, 0);
	float4 x2 = tex1Dfetch(tex_x_cam, 1);	
	int count = 0;
	int tid = threadIdx.x + (cameraID[blockIdx.x] >> 2);
	while (tid < cameraCount[blockIdx.x]>>2) 
	{
		//拾取绑定Jc矩阵的纹理内存
		float4 jc = tex1Dfetch(tex_jc_cam, tid);
		//载入共享内存
		int index = threadIdx.x<< 2;  
		value[index + 0] = jc.x;	value[index + 1] = jc.y;
		value[index + 2] = jc.z;	value[index + 3] = jc.w;
		//线程同步
		__syncthreads();
		index = threadIdx.x << 3;  
		if(threadIdx.x < (txPerBlock >> 1))
			result[threadIdx.x + count * (txPerBlock >> 1) + (cameraID[blockIdx.x] >> 3] 
				= value[index + 0] * x1.x + value[index + 1] * x1.y + value[index + 2] * x1.z + value[index + 3] * x1.w
			    + value[index + 4] * x2.x + value[index + 5] * x2.y + value[index + 6] * x2.z + value[index + 7] * x2.w;
		tid += txPerBlock;
		count ++ ;
	}
}