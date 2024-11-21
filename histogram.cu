/*****************************************************************************
 * File:        histogram.cu
 * Description: Implementing text histogram computation in different ways
 *              
 * Compile:     nvcc -o histogram histogram.cu -I..
 * Debug:       nvcc -g -G histogram.cu -o  histogram
 *              gdb --args ./histogram 0
 * Run:         ./histogram num
 *          [0] : Sequential Histogram
 *          [1] : Simple Parallel Histogram
 *          [2] : Fix [1] Kernel for memory coalescing
 *          [3] : Privatized Histogram Kernel
 *          [4] : Using two phases and global memory atomic operations
 *          [5] ：Using two phases and shared memory atomic operations
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

void sequential_Histogram(int* data, int n, int* histo);
__global__ void histo_kernel(int* data, int n, int* histo, int n_bins);
__global__ void histo_kernel_2(int* data, int n, int* histo, int n_bins);
__global__ void histo_privatized_kernel(int* data, int n, int* histo, int n_bins);
__global__ void histogram_gmem_atomics(int* data, int n, int *out, int n_bins);
__global__ void histogram_smem_atomics(int* data, int n, int* histo, int n_bins);
__global__ void histogram_final_accum(int *in, int n, int *out, int n_bins);

// Privatized Aggregation Histogram Kernel
// __global__ void histo_privatized_aggregation_kernel(char* data, int n, int* histo, int n_bins);

int n_bins = 512; // the number of bins, a-d, e-h, i-l, m-p, q-t, u-x, y-z

int main(int argc, char* argv[])
{
    printf("[Histogram...]\n\n");

    int n = 1 << 24;
    int whichKernel = std::stoi(argv[1]);
    int threads = 256;
    int blocks = 256;
    

    printf("The number of elements: %d\n", n);
    printf("Threads: %d / Blocks: %d\n\n", threads, blocks);
    

    unsigned int bytes = n*sizeof(int);
    int* h_data;
    int* h_histo;
    int* h_double_kernel_temp_histo;

    // allocate host memory
    h_data = (int*)malloc(bytes);
    h_histo = (int*)malloc(n_bins*sizeof(int));
    h_double_kernel_temp_histo = (int*)malloc(n_bins*blocks*sizeof(int));

    // init
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist(0, 511);
    for (int i = 0; i < n; i++) {
        h_data[i] = dist(generator);
    }
    for (int i = 0; i < n_bins; i++)
        h_histo[i] = 0;
    for (int i = 0; i < n_bins*blocks; i++)
        h_double_kernel_temp_histo[i] = 0;

    // allocate device memory
    int* d_data;
    int *d_histo;
    int *d_double_kernel_temp_histo;
    cudaMalloc((void**)&d_data, bytes);
    cudaMalloc((void**)&d_histo, n_bins*sizeof(int));
    cudaMalloc((void**)&d_double_kernel_temp_histo, n_bins*blocks*sizeof(int));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    printf("Implement [Kernel %d]\n\n", whichKernel);
    // double start, finish;
    // GET_TIME(start);
    
    if (whichKernel == 0) {
        sequential_Histogram(h_data, n, h_histo);
    }
    else if (whichKernel == 1) {
        histo_kernel<<<blocks, threads>>>(d_data, n, d_histo, n_bins);
        cudaDeviceSynchronize();
        cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);
    }
    else if (whichKernel == 2) {
        histo_kernel_2<<<blocks, threads>>>(d_data, n, d_histo, n_bins);
        cudaDeviceSynchronize();
        cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);
    }
    else if (whichKernel == 3) {
        int smem_size = 2*n_bins*sizeof(int);
        histo_privatized_kernel<<<blocks, threads, smem_size>>>(d_data, n, d_histo, n_bins);
        cudaDeviceSynchronize();
        cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);
    }
    else if (whichKernel == 4) {
        histogram_gmem_atomics<<<blocks, threads>>>(d_data, n, d_double_kernel_temp_histo, n_bins);
        cudaDeviceSynchronize();
        histogram_final_accum<<<blocks, threads>>>(d_double_kernel_temp_histo, n_bins*blocks, d_histo, n_bins);
        cudaDeviceSynchronize();
        cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);

    }
    else if (whichKernel == 5) {
        histogram_smem_atomics<<<blocks, threads>>>(d_data, n, d_double_kernel_temp_histo, n_bins);
        cudaDeviceSynchronize();
        histogram_final_accum<<<blocks, threads>>>(d_double_kernel_temp_histo, n_bins*blocks, d_histo, n_bins);
        cudaDeviceSynchronize();
        cudaMemcpy(h_histo, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int repeat_times = 10;
    for(int i = 0; i < repeat_times; i++){
        if (whichKernel == 0) {
        sequential_Histogram(h_data, n, h_histo);
    }
    else if (whichKernel == 1) {
        histo_kernel<<<blocks, threads>>>(d_data, n, d_histo, n_bins);
    }
    else if (whichKernel == 2) {
        histo_kernel_2<<<blocks, threads>>>(d_data, n, d_histo, n_bins);
    }
    else if (whichKernel == 3) {
        int smem_size = 2*n_bins*sizeof(int);
        histo_privatized_kernel<<<blocks, threads, smem_size>>>(d_data, n, d_histo, n_bins);
    }
    else if (whichKernel == 4) {
        histogram_gmem_atomics<<<blocks, threads>>>(d_data, n, d_double_kernel_temp_histo, n_bins);
        cudaDeviceSynchronize();
        histogram_final_accum<<<blocks, threads>>>(d_double_kernel_temp_histo, n_bins*blocks, d_histo, n_bins);
    }
    else if (whichKernel == 5) {
        histogram_smem_atomics<<<blocks, threads>>>(d_data, n, d_double_kernel_temp_histo, n_bins);
        cudaDeviceSynchronize();
        histogram_final_accum<<<blocks, threads>>>(d_double_kernel_temp_histo, n_bins*blocks, d_histo, n_bins);
    }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    int total_count = 0;
    printf("histo: ");
    for (int i = 0; i < n_bins; i++) {
        printf("%d ", h_histo[i]);
        total_count += h_histo[i];
    }
    printf("\n\n");
    printf("Total Count : %d\n", total_count);
    // printf("Time: %f msec\n\n", (finish-start)*1000);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n\n", milliseconds/repeat_times );

    printf(total_count == n ? "Test PASS\n" : "Test FAILED!\n");

    // free memory
    free(h_data);
    free(h_histo);
    cudaFree(d_data);
    cudaFree(d_histo);

    return 0;
}

void sequential_Histogram(int* data, int n, int* histo)
{
    for (int i = 0; i < n; i++) {
        int alphabet_pos = data[i];
        if (alphabet_pos >= 0 && alphabet_pos < n_bins)
            histo[alphabet_pos]++;
    }
}

__global__
void histo_kernel(int* data, int n, int* histo, int n_bins)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int section_size = (n - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i*section_size;

    for (int k = 0; k < section_size; k++) {
        if (start + k < n) {
            int alphabet_pos = data[start+k];
            if (alphabet_pos >= 0 && alphabet_pos < n_bins)
                atomicAdd(&histo[alphabet_pos], 1);
        }
    }
}

__global__
void histo_kernel_2(int* data, int n, int* histo, int n_bins)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
        int alphabet_pos = data[i];
        if (alphabet_pos >= 0 && alphabet_pos < n_bins)
            atomicAdd(&histo[alphabet_pos], 1);
    }
}

__global__
void histo_privatized_kernel(int* data, int n, int* histo, int n_bins)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    // Privatized bins
    extern __shared__ int histo_s[];
    if (threadIdx.x < n_bins)
        histo_s[threadIdx.x] = 0u;
    __syncthreads();

    // histogram
    for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
        int alphabet_pos = data[i];
        if (alphabet_pos >= 0 && alphabet_pos < n_bins)
            atomicAdd(&histo_s[alphabet_pos], 1);
    }
    __syncthreads();

    // commit to global memory
    if (threadIdx.x < n_bins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}

// 
__global__ void histogram_gmem_atomics(int* data, int n, int *out, int n_bins)
{
  int tid = blockDim.x*blockIdx.x + threadIdx.x;

  // total threads in 2D block
  int nt = blockDim.x * blockDim.y; 
  
  // linear block index within 2D grid
  int g = blockIdx.x + blockIdx.y * gridDim.x;

  // initialize temporary accumulation array in global memory
  int *gmem = out + g * n_bins;

  // process pixels
  // updates our block's partial histogram in global memory
  for (int i = tid; i < n; i += blockDim.x*gridDim.x) { 
      int alphabet_pos = data[i];
      if (alphabet_pos >= 0 && alphabet_pos < n_bins)
            atomicAdd(&gmem[alphabet_pos], 1);
    }
}

__global__ void histogram_smem_atomics(int* data, int n, int* histo, int n_bins)
{

  // linear thread index within 2D block
  int tid = blockDim.x*blockIdx.x + threadIdx.x;

  // linear block index within 2D grid
  int g = blockIdx.x + blockIdx.y * gridDim.x;

  // initialize temporary accumulation array in shared memory
  __shared__ int histo_s[512];

  if (threadIdx.x < n_bins) histo_s[threadIdx.x] = 0u;

  __syncthreads();

  // process pixels
  // updates our block's partial histogram in shared memory
  for (int i = tid; i < n; i += blockDim.x*gridDim.x) { 
      
      int alphabet_pos = data[i];
      if (alphabet_pos >= 0 && alphabet_pos < n_bins)
            atomicAdd(&histo_s[alphabet_pos], 1);
    }
  __syncthreads();

  // write partial histogram into the global memory
  histo += g * n_bins;
  if (threadIdx.x < n_bins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }

}

__global__ void histogram_final_accum(int *in, int n,int *out, int n_bins)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n_bins) {
    int total = 0;
    
    for (int j = 0; i + n_bins * j < n; j++){ 
      total += in[i + n_bins * j];
      
    }
    out[i] = total;
  }
}

// __global__
// void histo_privatized_aggregation_kernel(int* data, int n, int* histo, int n_bins)
// {
//     int tid = blockDim.x*blockIdx.x + threadIdx.x;

//     // Privatized bins
//     extern __shared__ int histo_s[];
//     if (threadIdx.x < n_bins)
//         histo_s[threadIdx.x] = 0u;
//     __syncthreads();

//     int prev_index = -1;
//     int accumulator = 0;

//     // histogram
//     for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
//         int alphabet_pos = data[i];
//         if (alphabet_pos >= 0 && alphabet_pos < ) {
//             int curr_index = alphabet_pos;
//             if (curr_index != prev_index) {
//                 if (prev_index != -1 && accumulator > 0)
//                     atomicAdd(&histo_s[prev_index], accumulator);
//                 accumulator = 1;
//                 prev_index = curr_index;
//             }
//             else {
//                 accumulator++;
//             }
//         }
//     }
//     if (accumulator > 0)
//         atomicAdd(&histo_s[prev_index], accumulator);
//     __syncthreads();

//     // commit to global memory
//     if (threadIdx.x < n_bins) {
//         atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
//     }
// }