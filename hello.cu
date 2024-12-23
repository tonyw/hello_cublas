#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda_fp16.h>

inline cudaError_t cuda_check(cudaError_t err){
    if(err != cudaSuccess){
        printf("cuda error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
    return err;
}

void cpu_init(half *a, half *b, half *c_, size_t N){
    for(size_t c=0;c<N;c++){
        for(size_t r=0;r<N;r++){
            a[c*N+r] = __float2half((float)r);
            b[c*N+r] = __float2half((float)c);
            c_[c*N+r] = __float2half(0.0f);
            //printf("[%d,%d]: a:%d, b:%d\n", r,c,(int)__half2float(a[r*N+c]), (int)__half2float(b[r*N+c]));
        }
    }
}

// 将CPU上的矩阵A和B复制到GPU内存
void gpu_init(half *cpu_a, half *cpu_b, half *gpu_a, half *gpu_b, size_t N){
    cuda_check(cudaMemcpy(gpu_a, cpu_a, N*N*sizeof(half), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gpu_b, cpu_b, N*N*sizeof(half), cudaMemcpyHostToDevice));
}

__global__ void check_matrix_multiply_1t1e(half *a, half *b, half *c_gpu, half *epsilon,int *flag, size_t N){
    
    /*
    tidx.[y,x]: 0,2, cidx.[r,c]: [0, 2],i: 0,a_idx:0,a_data: 0, b_idx:8,b_data: 2
    tidx.[y,x]: 0,2, cidx.[r,c]: [0, 2],i: 1,a_idx:4,a_data: 0, b_idx:9,b_data: 2
    tidx.[y,x]: 0,2, cidx.[r,c]: [0, 2],i: 2,a_idx:8,a_data: 0, b_idx:10,b_data: 2
    tidx.[y,x]: 0,2, cidx.[r,c]: [0, 2],i: 3,a_idx:12,a_data: 0, b_idx:11,b_data: 2
     */
    int r_index = blockDim.y * blockIdx.y + threadIdx.y;
    int c_index = blockDim.x * blockIdx.x + threadIdx.x;
    half temp = 0;
    if(r_index < N && c_index < N){
        for(int i = 0; i < N; i++){
            int a_idx = i*(int)N+r_index;
            int b_idx = c_index*(int)N+i;
            half a_data = a[a_idx];
            half b_data = b[b_idx];
            temp = __hadd(temp, __hmul(a_data, b_data));
            if(r_index == 15 && c_index == 11){
                printf("tidx.[y,x]: %d,%d, cidx.[r,c]: [%d, %d],i: %d,a_idx:%d,a_data: %d, b_idx:%d,b_data: %d, temp: %f.05\n",threadIdx.y, threadIdx.x, r_index, c_index,i, a_idx, (int)__half2float(a[a_idx]), b_idx, (int)__half2float(b_data), __half2float(temp));
            }
            
        }
        if(r_index == 15 && c_index == 11){
            printf("temp: %f.05, gpu_c: %f.05\n", __half2float(temp), __half2float(c_gpu[r_index*N+c_index]));
        }
        if(__habs(temp - c_gpu[r_index*N+c_index]) > *epsilon){
            flag[r_index*N+c_index] = 1;
        }
    }
}


bool check_gpu_multiply(half *a, half *b, half *c_gpu, int N){
    const size_t threads_per_block = 32;
    const dim3 threads(threads_per_block,threads_per_block);
    //printf("CPU: Before kernel launch\n");
    //printf("threads: %d, %d\n", threads.x, threads.y);
    const dim3 blocks((N+threads.x-1)/threads.x,(N+threads.y-1)/threads.y);   
    //printf("blocks: %d, %d\n", blocks.x, blocks.y);
    
    half* EPSILON;
    cuda_check(cudaMallocManaged((void**)&EPSILON, sizeof(half)));
    *EPSILON = __float2half(1e-5f);
    int *flag;
    cuda_check(cudaMalloc(&flag, N*N*sizeof(int)));
    cuda_check(cudaMemset(flag, 0, N*N*sizeof(int)));
    //printf("CPU: Launching kernel...\n");
    
    check_matrix_multiply_1t1e<<<blocks,threads>>>(a, b, c_gpu, EPSILON, flag, N);
    
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    //printf("CPU: After kernel execution\n");
    
    int *flag_host = (int*)malloc(N*N*sizeof(int));
    cuda_check(cudaMemcpy(flag_host, flag, N*N*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0; i < N*N; i++){
        if(flag_host[i] == 1){
            printf("gpu error: %d\n", i);
            cuda_check(cudaFree(flag));
            free(flag_host);
            return false;
        }
    }
    cuda_check(cudaFree(flag));
    cuda_check(cudaFree(EPSILON));
    free(flag_host);
    return true;
}





// Helper function to convert cuBLAS status to a readable string
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "UNKNOWN CUBLAS STATUS";
    }
}


// Function to perform matrix multiplication using cuBLAS
void matrix_multiply_cublas(half *a, half *b, half *c_gpu, size_t N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);

    cublasStatus_t status = cublasHgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        a, N,
        b, N,
        &beta,
        c_gpu, N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS matrix multiplication failed\n");
        fprintf(stderr, "code: %d, status: %s\n", status, cublasGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    cublasDestroy(handle);
}

// 在CPU上分配页锁定内存(pinned memory)
void allocate_memory_cpu(half **a, half **b, half **c, size_t size) {
    cuda_check(cudaMallocHost(a, size));
    cuda_check(cudaMallocHost(b, size));
    cuda_check(cudaMallocHost(c, size));
    if (*a == NULL || *b == NULL || *c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

// 在GPU上分配设备内存
void allocate_memory_gpu(half **a, half **b, half **c, size_t size) {
    cuda_check(cudaMalloc(a, size));
    cuda_check(cudaMalloc(b, size));
    cuda_check(cudaMalloc(c, size));
    if (*a == NULL || *b == NULL || *c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

// 主函数：
// 1. 打印GPU设备信息
// 2. 分配内存并初始化数据
// 3. 执行GPU矩阵加法并计时
// 4. 验证结果并清理内存
int main(){
    int gpu_index=0;
    cudaGetDevice(&gpu_index);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_index);
    printf("GPU name: %s\n", prop.name);
    printf("GPU compute capability: %d.%d\n", prop.major, prop.minor);
    printf("GPU sm count: %d\n", prop.multiProcessorCount);
    printf("GPU global memory: %zu GB\n", prop.totalGlobalMem/1024/1024/1024);
    printf("GPU shared memory per block: %zu KB\n", prop.sharedMemPerBlock/1024);
    printf("GPU L2 cache size: %d KB\n", prop.l2CacheSize/1024);
    printf("GPU warp size: %d\n", prop.warpSize);
    printf("GPU maximum threads per block: %d\n", prop.maxThreadsPerBlock);

    const size_t N = 16;
    half *cpu_a,*cpu_b,*cpu_c;
    half *gpu_a,*gpu_b,*gpu_c;
    size_t size = N * N * sizeof(half);
    allocate_memory_gpu(&gpu_a, &gpu_b, &gpu_c, size);
    allocate_memory_cpu(&cpu_a, &cpu_b, &cpu_c, size);
    cpu_init(cpu_a,cpu_b,cpu_c, N);
    gpu_init(cpu_a,cpu_b, gpu_a, gpu_b, N);
    

    // CUDA TIME
    float ms;
    float avems = 0.0;
    cudaEvent_t start,end;

    //warm up
    for(int i = 0;i < 3;i++){
        matrix_multiply_cublas(gpu_a, gpu_b, gpu_c, N);
        cuda_check(cudaDeviceSynchronize());
    }
    
    for(int i = 0; i < 10; i++){

        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        matrix_multiply_cublas(gpu_a, gpu_b, gpu_c, N);
        //check_gpu_multiply(gpu_a, gpu_b, gpu_c, N);
        cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

        printf("\tIteration no. %d: %.2f ms\n", i, ms);
        avems+=ms;

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    printf("[**] Average kernel execution time: %.2fms.\n\n", avems/10.0);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    check_gpu_multiply(gpu_a, gpu_b, gpu_c, N)? printf("gpu ok\n") : printf("gpu error\n");
    cuda_check(cudaFree(gpu_a));
    cuda_check(cudaFree(gpu_b));
    cuda_check(cudaFree(gpu_c));
    cuda_check(cudaFreeHost(cpu_a));
    cuda_check(cudaFreeHost(cpu_b));
    cuda_check(cudaFreeHost(cpu_c));
    return 0;
}


