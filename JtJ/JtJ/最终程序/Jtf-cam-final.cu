texture<float4, 1, cudaReadModeElementType> tex_jtc_cam;
texture<float, 1, cudaReadModeElementType> tex_f_cam;

__global__ void jtf_cam(float* result, int* cameraID, int *cameraCount)
{
	//分配共享内存
	__shared__ float value[txPerBlock * 8]; 
	float jx = 0, jy = 0, jz = 0, jw = 0; 
	//取出每个Jtc矩阵以及对应的投影误差
	int tid = threadIdx.x + (cameraID[blockIdx.x] >> 3);
	while (tid < cameraCount[blockIdx.x]>>3) 
	{
	    //拾取绑定Jtc矩阵的纹理内存 
	    float4 jab = tex1Dfetch(tex_jtc_cam, (tid * 2) + threadIdx.y);
		//拾取绑定投影误差f的纹理内存
		float err = tex1Dfetch(tex_f_cam, tid);
		jx += jab.x * err;		jy += jab.y * err;
        jz += jab.z * err;  	jw += jab.w * err;
		tid += txPerBlock;
	}
	//线程同步
	__syncthreads();
	//将每个线程计算得到的Jtf结果载入共享内存
	int index=(threadIdx.x + (threadIdx.y * txPerBlock)) << 2;  
	value[index + 0] = jx;		value[index + 1] = jy;
	value[index + 2] = jz;		value[index + 3] = jw;
	//共享内存规约
	int i = txPerBlock * 8;
	while (i > 8) 
	{	
		tid = threadIdx.x + (threadIdx.y << n);
		while(tid < ((threadIdx.y << n) + (i >> 2)))
		{
			value[tid] += value[tid + (i >> 2)];
			tid += txPerBlock;
		}  
		i >>= 1;
	}
	//得到Jtf结果，每个相机对应8个值
	if(threadIdx.x < 4)
        result[threadIdx.x + (threadIdx.y << 2) + (blockIdx.x << 3)]=value[threadIdx.x + (threadIdx.y << n)];
}


