/*****************************************************************************
 * File:        reduce.cu
 * Description: Implementing reduce computation in different ways
 *              
 * Compile:     nvcc -o reduce reduce.cu -I..
 * Run:         ./reduce num
 *          [0] : interleaved addressing with divergent branching
 *          [1] : interleaved addressing with bank conflicts
 *          [2] : sequential addressing
 *          [3] : first add during global load
 *          [4] : unroll last warp
 *          [5] ：completely unrolled
 *          [6] : multiple elements per thread
 *          [7] : Warp Shuffle
 *****************************************************************************/
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string>

#define N 32*1024*1024u
#define BLOCK_SIZE 256u
#define WARP_SIZE 32

__global__ void reduce_v0(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v1(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // if (tid % (2*s) == 0) {
        //     sdata[tid] += sdata[tid + s];
        // }
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v2(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s >>= 1) {
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v3(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s >>= 1) {
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__device__ void warpReduce4(volatile float* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce_v4(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s >>= 1) {
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce4(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
 __device__ void warpReduce(volatile float* cache,int tid){
    if(blockSize >= 64)cache[tid]+=cache[tid+32];
    if(blockSize >= 32)cache[tid]+=cache[tid+16];
    if(blockSize >= 16)cache[tid]+=cache[tid+8];
    if(blockSize >= 8)cache[tid]+=cache[tid+4];
    if(blockSize >= 4)cache[tid]+=cache[tid+2];
    if(blockSize >= 2)cache[tid]+=cache[tid+1];
}

template <unsigned int blockSize>
__global__ void reduce_v5(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce<blockSize>(sdata,tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v6(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x * NUM_PER_THREAD) + threadIdx.x;
    sdata[tid] = 0;
    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sdata[tid] += g_idata[i+iter*blockSize];
    }
    __syncthreads();

    // do reduction in shared mem
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce<blockSize>(sdata,tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSumXor(float sum) {
    if (blockSize >= 32)sum += __shfl_xor_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_xor_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_xor_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_xor_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_xor_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v7(float *g_idata,float *g_odata, unsigned int n){
    float sum = 0;

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sum += g_idata[i+iter*blockSize];
    }
    
    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0 )warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize/WARP_SIZE>(sum); 
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sum;
}


int main(int argc, char* argv[]) {
    printf("[Reduce...]\n\n");

    
    int whichKernel = std::stoi(argv[1]);

    // input
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    // output
    float *output_host; 
    float *output_device;


    printf("Implement [Kernel %d]\n\n", whichKernel);

    if (whichKernel == 0) { 
        int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        output_host= (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
        cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float)); 
        dim3 grid(N / BLOCK_SIZE, 1);
        dim3 block(BLOCK_SIZE, 1);     
        reduce_v0<<<grid, block>>>(input_device, output_device); 
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);    
    }
    else if (whichKernel == 1) {
        int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        output_host= (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
        cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float)); 
        dim3 grid(N / BLOCK_SIZE, 1);
        dim3 block(BLOCK_SIZE, 1);
        reduce_v1<<<grid, block>>>(input_device, output_device);  
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);       
    }
    else if (whichKernel == 2) {
        int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        output_host= (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
        cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float)); 
        dim3 grid(N / BLOCK_SIZE, 1);
        dim3 block(BLOCK_SIZE, 1);
        reduce_v2<<<grid, block>>>(input_device, output_device);
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);    
    }
    else if (whichKernel == 3) { 
        int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
        output_host= (float*)malloc((block_num) * sizeof(float));
        cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
        dim3 grid(block_num, 1);
        dim3 block(BLOCK_SIZE, 1);    
        reduce_v3<<<grid, block>>>(input_device, output_device); 
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);     
    }
    else if (whichKernel == 4) {
        int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
        output_host= (float*)malloc((block_num) * sizeof(float));
        cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
        dim3 grid(block_num, 1);
        dim3 block(BLOCK_SIZE, 1);  
        reduce_v4<<<grid, block>>>(input_device, output_device);  
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);      
    }
    else if (whichKernel == 5) {
        int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
        output_host= (float*)malloc((block_num) * sizeof(float));
        cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
        dim3 grid(block_num, 1);
        dim3 block(BLOCK_SIZE, 1);  
        reduce_v5<BLOCK_SIZE><<<grid, block>>>(input_device, output_device);  
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);   
    }
    else if (whichKernel == 6) {
        const int block_num = 1024;
        const int NUM_PER_BLOCK = N / block_num;
        const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
        output_host= (float*)malloc((block_num) * sizeof(float));
        cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
        dim3 grid6(block_num, 1);
        dim3 block6(BLOCK_SIZE, 1);
        reduce_v6<BLOCK_SIZE ,NUM_PER_THREAD><<<grid6, block6>>>(input_device, output_device);
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    }
    else if (whichKernel == 7) {
        const int block_num = 1024;
        const int NUM_PER_BLOCK = N / block_num;
        const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
        output_host= (float*)malloc((block_num) * sizeof(float));
        cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
        dim3 grid6(block_num, 1);
        dim3 block6(BLOCK_SIZE, 1);
        reduce_v7<BLOCK_SIZE ,NUM_PER_THREAD><<<grid6, block6>>>(input_device, output_device, N);
        cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    // const int block_num = 1024;
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int repeat_times = 20;
    for(int i = 0; i < repeat_times; i++){
            if (whichKernel == 0) { 
            int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            output_host= (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
            cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float)); 
            dim3 grid(N / BLOCK_SIZE, 1);
            dim3 block(BLOCK_SIZE, 1);     
            reduce_v0<<<grid, block>>>(input_device, output_device); 
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);    
        }
        else if (whichKernel == 1) {
            int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            output_host= (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
            cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float)); 
            dim3 grid(N / BLOCK_SIZE, 1);
            dim3 block(BLOCK_SIZE, 1);
            reduce_v1<<<grid, block>>>(input_device, output_device);  
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);       
        }
        else if (whichKernel == 2) {
            int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            output_host= (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
            cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float)); 
            dim3 grid(N / BLOCK_SIZE, 1);
            dim3 block(BLOCK_SIZE, 1);
            reduce_v2<<<grid, block>>>(input_device, output_device);
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);    
        }
        else if (whichKernel == 3) { 
            int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
            output_host= (float*)malloc((block_num) * sizeof(float));
            cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
            dim3 grid(block_num, 1);
            dim3 block(BLOCK_SIZE, 1);    
            reduce_v3<<<grid, block>>>(input_device, output_device); 
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);     
        }
        else if (whichKernel == 4) {
            int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
            output_host= (float*)malloc((block_num) * sizeof(float));
            cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
            dim3 grid(block_num, 1);
            dim3 block(BLOCK_SIZE, 1);  
            reduce_v4<<<grid, block>>>(input_device, output_device);  
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);      
        }
        else if (whichKernel == 5) {
            int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
            output_host= (float*)malloc((block_num) * sizeof(float));
            cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
            dim3 grid(block_num, 1);
            dim3 block(BLOCK_SIZE, 1);  
            reduce_v5<BLOCK_SIZE><<<grid, block>>>(input_device, output_device);  
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);   
        }
        else if (whichKernel == 6) {
            const int block_num = 1024;
            const int NUM_PER_BLOCK = N / block_num;
            const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
            output_host= (float*)malloc((block_num) * sizeof(float));
            cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
            dim3 grid6(block_num, 1);
            dim3 block6(BLOCK_SIZE, 1);
            reduce_v6<BLOCK_SIZE ,NUM_PER_THREAD><<<grid6, block6>>>(input_device, output_device);
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else if (whichKernel == 7) {
            const int block_num = 1024;
            const int NUM_PER_BLOCK = N / block_num;
            const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
            output_host= (float*)malloc((block_num) * sizeof(float));
            cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
            dim3 grid6(block_num, 1);
            dim3 block6(BLOCK_SIZE, 1);
            reduce_v7<BLOCK_SIZE ,NUM_PER_THREAD><<<grid6, block6>>>(input_device, output_device, N);
            cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    printf("Reduce: ");
    printf("Total Result :%f \n\n", output_host[1]);
    
    // Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms \n\n", milliseconds/repeat_times );

    //printf(output_host[0] == (N/block_num) * 2 ? "Test PASS\n" : "Test FAILED!\n");

    // free memory
    free(input_host);
    free(output_host);
    cudaFree(input_device);
    cudaFree(output_host);
    
    return 0;
}




















