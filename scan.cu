/*****************************************************************************
 * File:        scan.cu
 * Description: Implementing scan computation in different ways
 *              
 * Compile:     nvcc -o scan scan.cu -I..
 * Run:         ./scan
 *          [0] : Scan cpu
 *          [1] : Naive Parallel Scan
 *          [2] : Parallel Block Scan
 *          [3] : Parallel Block Scan Avoiding Bank Conflicts
 *          [4] : Parallel Large Array Scan
 *          [5] : Parallel Large Array Scan Avoiding Bank Conflicts
 *****************************************************************************/


#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <chrono>
#include <tuple>
#include <random>

#define NUM_BANKS 32
#define MAX_ELEMENTS_PER_BLOCK 2048
#define MAX_THREADS_PER_BLOCK 1024
#define LOG_NUM_BANKS 5
#define ZERO_BANK_CONFLICTS
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
#define MAX_SHARE_SIZE (MAX_ELEMENTS_PER_BLOCK + CONFLICT_FREE_OFFSET(MAX_ELEMENTS_PER_BLOCK - 1))
#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            printf("CUDA Error: \n");                                                                                  \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error Code: %d\n", err);                                                                       \
            printf("    Error Text: %s\n", cudaGetErrorString(err));                                                   \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)


//--------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- utiles  ----------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------//

// 初始化一个整数数组 data，使其包含从 -10 到 100 的随机整数
void data_init(int *data, int N)
{
    //创建一个均匀分布的整数生成器，范围从 -10 到 100。这意味着生成的随机数在这个区间内是等概率的
    std::uniform_int_distribution<> int_generator(-10, 100); 
    // // 使用当前时间作为种子来初始化默认的随机数引擎 (rand_engine)。time(nullptr) 返回当前的系统时间（以秒为单位），确保每次调用程序时会有不同的初始值，从而生成不同的随机数序列。
    std::default_random_engine rand_engine(time(nullptr)); // 
    for (int i = 0; i < N; i++)
    {
        data[i] = int_generator(rand_engine);
    }
}

void results_check(int *a, int *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != b[i])
        {
            printf("results_check fail\n");
            exit(1);
        }
    }
}

// 打印一个整数数组的内容
void print_int_arr(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
}

int next_power_of_two(int x)
{
    int power = 1;
    while (power < x)
    {
        power *= 2;
    }
    return power;
}

// warm_up_kernel 和 warm_up 使用简单的kernel进行warm up
__global__ void warm_up_kernel(int *data)
{
    int tid = threadIdx.x;
    data[tid] += tid;
}

void warm_up()
{
    int N = 512;
    size_t arr_size = N * sizeof(int);
    int *data = (int *)malloc(arr_size);
    data_init(data, N);

    for (int i = 0; i < 10; i++)
    {
        int *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, arr_size));
        CUDA_CHECK(cudaMemcpy(d_data, data, arr_size, cudaMemcpyHostToDevice));

        warm_up_kernel<<<1, N>>>(d_data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(data, d_data, arr_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data));
    }

    free(data);
}

// 定义了两个计时器类：TotalTimer 和 KernelTimer
class TotalTimer
{
  private:
    //分别记录计时的开始和结束时间点
    std::chrono::high_resolution_clock::time_point m_start_point, m_end_point;

  public:
    void start()
    {
        m_start_point = std::chrono::high_resolution_clock::now();
    };
    void end()
    {
        m_end_point = std::chrono::high_resolution_clock::now();
    };
    float cost()
    {
        // 通过计算结束时间与开始时间之间的差值来得到时间间隔，并将其转换为毫秒（std::milli）
        std::chrono::duration<float, std::milli> dur = m_end_point - m_start_point;
        return dur.count();
    };
};

// 专门用于测量 CUDA 内核的执行时间
class KernelTimer
{
  private:
    cudaEvent_t m_start_event, m_end_event;

  public:
    KernelTimer()
    {
        CUDA_CHECK(cudaEventCreate(&m_start_event));
        CUDA_CHECK(cudaEventCreate(&m_end_event));
    };
    ~KernelTimer()
    {
        CUDA_CHECK(cudaEventDestroy(m_start_event));
        CUDA_CHECK(cudaEventDestroy(m_end_event));
    };
    void start()
    {
        CUDA_CHECK(cudaEventRecord(m_start_event));
    };
    void end()
    {
        CUDA_CHECK(cudaEventRecord(m_end_event));
        CUDA_CHECK(cudaEventSynchronize(m_end_event));
    };
    float cost()
    {
        float kernel_cost;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_cost, m_start_event, m_end_event));
        return kernel_cost;
    };
};


//--------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- scan  ----------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------------//

//----------------------------
//------- scan_cpu -----------
//----------------------------
float scan_cpu(int *data, int *prefix_sum, int N)
{
    TotalTimer total_timer;
    total_timer.start();

    prefix_sum[0] = 0;
    for (int i = 1; i < N; i++)
    {
        prefix_sum[i] = prefix_sum[i - 1] + data[i - 1];
    }

    total_timer.end();
    return total_timer.cost();
}


//-----------------------------------
//------- native_scan -----------
//-----------------------------------
__global__ void native_scan_kernel(int *data, int *prefix_sum, int N)
{
    prefix_sum[0] = 0;
    for (int i = 1; i < N; i++)
    {
        prefix_sum[i] = prefix_sum[i - 1] + data[i - 1];
    }
}

std::tuple<float, float> native_scan_gpu(int *data, int *prefix_sum, int N)
{
    TotalTimer total_timer;
    total_timer.start();

    int *d_data, *d_prefix_sum;
    size_t arr_size = N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_data, arr_size));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, arr_size));
    CUDA_CHECK(cudaMemcpy(d_data, data, arr_size, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    // 运行kernel
    native_scan_kernel<<<1, 1>>>(d_data, d_prefix_sum, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    float kernel_cost = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(prefix_sum, d_prefix_sum, arr_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_prefix_sum));

    total_timer.end();
    float total_cost = total_timer.cost();

    return {total_cost, kernel_cost};
}

//---------------------------------------
//------- parallel_block_scan -----------
//---------------------------------------
__global__ void parallel_block_scan_kernel(int *data, int *prefix_sum, int N)
{
    extern __shared__ int tmp[];
    int tid = threadIdx.x;
    int leaf_num = blockDim.x * 2; // equals to length of tmp

    // 单线程块加载多数据块
    tmp[tid * 2] = tid * 2 < N ? data[tid * 2] : 0;
    tmp[tid * 2 + 1] = tid * 2 + 1 < N ? data[tid * 2 + 1] : 0;
    __syncthreads();

    // 并行扫描第一阶段：reduce规约
    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // 开扫描，将最后一个元素记录为0
    if (tid == 0)
    {
        tmp[leaf_num - 1] = 0;
    }
    __syncthreads();

    // 并行扫描第二阶段：down sweep
    for (int d = 1; d < leaf_num; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }

    // 并行扫描结束将共享内存结果存回全局内存
    if (tid * 2 < N)
    {
        prefix_sum[tid * 2] = tmp[tid * 2];
    }
    if (tid * 2 + 1 < N)
    {
        prefix_sum[tid * 2 + 1] = tmp[tid * 2 + 1];
    }
}

// Bank conflict avoidance optimization (BCAO)
__global__ void parallel_block_scan_bcao_kernel(int *data, int *prefix_sum, int N)
{
    extern __shared__ int tmp[];
    int tid = threadIdx.x;
    int leaf_num = blockDim.x * 2; // not equals to length of tmp

    int ai = tid;
    int bi = tid + (leaf_num >> 1);
    int offset_ai = CONFLICT_FREE_OFFSET(ai);
    int offset_bi = CONFLICT_FREE_OFFSET(bi);

    tmp[ai + offset_ai] = ai < N ? data[ai] : 0;
    tmp[bi + offset_bi] = bi < N ? data[bi] : 0;
    __syncthreads();

    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    if (tid == 0)
    {
        tmp[leaf_num - 1 + CONFLICT_FREE_OFFSET(leaf_num - 1)] = 0;
    }
    __syncthreads();

    for (int d = 1; d < leaf_num; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }

    if (ai < N)
    {
        prefix_sum[ai] = tmp[ai + offset_ai];
    }
    if (bi < N)
    {
        prefix_sum[bi] = tmp[bi + offset_bi];
    }
}

std::tuple<float, float> parallel_block_scan_gpu(int *data, int *prefix_sum, int N, bool bcao)
{
    TotalTimer total_timer;
    total_timer.start();

    int *d_data, *d_prefix_sum;
    size_t arr_size = N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_data, arr_size));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, arr_size));
    CUDA_CHECK(cudaMemcpy(d_data, data, arr_size, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    int padding_N = next_power_of_two(N);
    if (bcao)
    {
        int share_mem_size = (padding_N + CONFLICT_FREE_OFFSET(padding_N - 1)) * sizeof(int);
        parallel_block_scan_bcao_kernel<<<1, padding_N / 2, share_mem_size>>>(d_data, d_prefix_sum, N);
    }
    else
    {
        int share_mem_size = padding_N * sizeof(int);
        parallel_block_scan_kernel<<<1, padding_N / 2, share_mem_size>>>(d_data, d_prefix_sum, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    float kernel_cost = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(prefix_sum, d_prefix_sum, arr_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_prefix_sum));

    total_timer.end();
    float total_cost = total_timer.cost();

    return {total_cost, kernel_cost};
}


//--------------------------------------------
//------- Arrays of Arbitrary Size -----------
//--------------------------------------------
__global__ void parallel_large_scan_kernel(int *data, int *prefix_sum, int N, int *sums)
{
    __shared__ int tmp[MAX_ELEMENTS_PER_BLOCK];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * MAX_ELEMENTS_PER_BLOCK;
    int leaf_num = MAX_ELEMENTS_PER_BLOCK;

    tmp[tid * 2] = tid * 2 + block_offset < N ? data[tid * 2 + block_offset] : 0;
    tmp[tid * 2 + 1] = tid * 2 + 1 + block_offset < N ? data[tid * 2 + 1 + block_offset] : 0;
    __syncthreads();

    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    if (tid == 0)
    {
        sums[bid] = tmp[leaf_num - 1];
        tmp[leaf_num - 1] = 0;
    }
    __syncthreads();

    for (int d = 1; d < leaf_num; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }

    if (tid * 2 + block_offset < N)
    {
        prefix_sum[tid * 2 + block_offset] = tmp[tid * 2];
    }
    if (tid * 2 + 1 + block_offset < N)
    {
        prefix_sum[tid * 2 + 1 + block_offset] = tmp[tid * 2 + 1];
    }
}

__global__ void parallel_large_scan_bcao_kernel(int *data, int *prefix_sum, int N, int *sums)
{
    __shared__ int tmp[MAX_SHARE_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * MAX_ELEMENTS_PER_BLOCK;
    int leaf_num = MAX_ELEMENTS_PER_BLOCK;

    int ai = tid;
    int bi = tid + (leaf_num >> 1);
    int offset_ai = CONFLICT_FREE_OFFSET(ai);
    int offset_bi = CONFLICT_FREE_OFFSET(bi);

    tmp[ai + offset_ai] = ai + block_offset < N ? data[ai + block_offset] : 0;
    tmp[bi + offset_bi] = bi + block_offset < N ? data[bi + block_offset] : 0;
    __syncthreads();

    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    if (tid == 0)
    {
        int last_idx = leaf_num - 1 + CONFLICT_FREE_OFFSET(leaf_num - 1);
        sums[bid] = tmp[last_idx];
        tmp[last_idx] = 0;
    }
    __syncthreads();

    for (int d = 1; d < leaf_num; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }

    if (ai + block_offset < N)
    {
        prefix_sum[ai + block_offset] = tmp[ai + offset_ai];
    }
    if (bi + block_offset < N)
    {
        prefix_sum[bi + block_offset] = tmp[bi + offset_bi];
    }
}

__global__ void add_kernel(int *prefix_sum, int *valus, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * MAX_ELEMENTS_PER_BLOCK;
    int ai = tid + block_offset;
    int bi = tid + (MAX_ELEMENTS_PER_BLOCK >> 1) + block_offset;

    if (ai < N)
    {
        prefix_sum[ai] += valus[bid];
    }
    if (bi < N)
    {
        prefix_sum[bi] += valus[bid];
    }
}

void recursive_scan(int *d_data, int *d_prefix_sum, int N, bool bcao)
{
    int block_num = N / MAX_ELEMENTS_PER_BLOCK;
    if (N % MAX_ELEMENTS_PER_BLOCK != 0)
    {
        block_num += 1;
    }
    int *d_sums, *d_sums_prefix_sum;
    CUDA_CHECK(cudaMalloc(&d_sums, block_num * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums_prefix_sum, block_num * sizeof(int)));

    if (bcao)
    {
        parallel_large_scan_bcao_kernel<<<block_num, MAX_THREADS_PER_BLOCK>>>(d_data, d_prefix_sum, N, d_sums);
    }
    else
    {
        parallel_large_scan_kernel<<<block_num, MAX_THREADS_PER_BLOCK>>>(d_data, d_prefix_sum, N, d_sums);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (block_num != 1)
    {
        recursive_scan(d_sums, d_sums_prefix_sum, block_num, bcao);
        add_kernel<<<block_num, MAX_THREADS_PER_BLOCK>>>(d_prefix_sum, d_sums_prefix_sum, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_sums_prefix_sum));
}

std::tuple<float, float> parallel_large_scan_gpu(int *data, int *prefix_sum, int N, bool bcao)
{
    TotalTimer total_timer;
    total_timer.start();

    int *d_data, *d_prefix_sum;
    size_t arr_size = N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_data, arr_size));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, arr_size));
    CUDA_CHECK(cudaMemcpy(d_data, data, arr_size, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    recursive_scan(d_data, d_prefix_sum, N, bcao);

    kernel_timer.end();
    float kernel_cost = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(prefix_sum, d_prefix_sum, arr_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_prefix_sum));

    total_timer.end();
    float total_cost = total_timer.cost();

    return {total_cost, kernel_cost};
}


//--------------------------------------------- main  ----------------------------------------------------------------------

int main(int argc, char **argv)
{
    warm_up();
    int nums[] = {1000, 2048, 100000, 10000000};
    int len = sizeof(nums) / sizeof(int);
    for (int i = 0; i < len; i++)
    {
        int N = nums[i];
        size_t arr_size = N * sizeof(int);
        int *data = (int *)malloc(arr_size);
        int *prefix_sum_cpu = (int *)malloc(arr_size);
        int *prefix_sum_gpu = (int *)malloc(arr_size);
        float total_cost, kernel_cost;
        data_init(data, N);
        printf("-------------------------- N = %d --------------------------\n", N);

        // ------ scan_cpu --------
        total_cost = scan_cpu(data, prefix_sum_cpu, N);
        printf("%35s - total: %10.5f ms\n", "scan_cpu", total_cost);

        // ------ native_scan --------
        std::tie(total_cost, kernel_cost) = native_scan_gpu(data, prefix_sum_gpu, N);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "native_scan_gpu", total_cost, kernel_cost);

        // 
        if (N <= MAX_ELEMENTS_PER_BLOCK)
        {
            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, false);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu", total_cost,
                   kernel_cost);

            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, true);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu with bcao", total_cost,
                   kernel_cost);
        }

        // 5. Arrays of Arbitrary Size scan
        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, false);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu", total_cost, kernel_cost);

        // 5. Arrays of Arbitrary Size scan without bank Conflicts
        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, true);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu with bcao", total_cost,
               kernel_cost);

        free(data);
        free(prefix_sum_cpu);
        free(prefix_sum_gpu);
        printf("\n");
    }
}