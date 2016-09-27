texture<float4, 1, cudaReadModeElementType> tex_jtc_cam;
texture<float, 1, cudaReadModeElementType> tex_f_cam;

__global__ void jtf_cam(float* result, int* cameraID, int *cameraCount)
{
	//���乲���ڴ�
	__shared__ float value[txPerBlock * 8]; 
	float jx = 0, jy = 0, jz = 0, jw = 0; 
	//ȡ��ÿ��Jtc�����Լ���Ӧ��ͶӰ���
	int tid = threadIdx.x + (cameraID[blockIdx.x] >> 3);
	while (tid < cameraCount[blockIdx.x]>>3) 
	{
	    //ʰȡ��Jtc����������ڴ� 
	    float4 jab = tex1Dfetch(tex_jtc_cam, (tid * 2) + threadIdx.y);
		//ʰȡ��ͶӰ���f�������ڴ�
		float err = tex1Dfetch(tex_f_cam, tid);
		jx += jab.x * err;		jy += jab.y * err;
        jz += jab.z * err;  	jw += jab.w * err;
		tid += txPerBlock;
	}
	//�߳�ͬ��
	__syncthreads();
	//��ÿ���̼߳���õ���Jtf������빲���ڴ�
	int index=(threadIdx.x + (threadIdx.y * txPerBlock)) << 2;  
	value[index + 0] = jx;		value[index + 1] = jy;
	value[index + 2] = jz;		value[index + 3] = jw;
	//�����ڴ��Լ
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
	//�õ�Jtf�����ÿ�������Ӧ8��ֵ
	if(threadIdx.x < 4)
        result[threadIdx.x + (threadIdx.y << 2) + (blockIdx.x << 3)]=value[threadIdx.x + (threadIdx.y << n)];
}


