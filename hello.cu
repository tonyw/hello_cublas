#include <stdio.h>
#include <assert.h>

inline cudaError_t cuda_check(cudaError_t err){
    if(err != cudaSuccess){
        printf("cuda error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
    return err;
}

void cpu_init(int *a, int *b, int *c_, size_t N){
    for(size_t r=0;r<N;r++){
        for(size_t c=0;c<N;c++){
            a[r*N+c] = r;
            b[r*N+c] = c;
            c_[r*N+c] = 0;
        }
    }
}

void gpu_init(int *cpu_a, int *cpu_b, int *gpu_a, int *gpu_b, size_t N){
    cuda_check(cudaMemcpy(gpu_a, cpu_a, N*N*sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gpu_b, cpu_b, N*N*sizeof(int), cudaMemcpyHostToDevice));
}

bool check_gpu(int *a, int *b, int *c_cpu, int *c_gpu, int N){
    for(size_t i = 0; i < N; i++){
        for(size_t j = 0; j < N; j++){
            c_cpu[i*N+j] = a[i*N+j] + b[i*N+j];
            if(c_cpu[i*N+j] != c_gpu[i*N+j]){
                printf("index: %zu, %zu\n", i, j);
                printf("cpu: %d, gpu: %d\n", c_cpu[i*N+j], c_gpu[i*N+j]);
                return false;
            }
        }
    }
    return true;
}

__global__ void matrix_add_1t1e(int *a, int *b, int *c_gpu, size_t N){
    int r_index = blockDim.y * blockIdx.y + threadIdx.y;
    int c_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(r_index < N && c_index < N){
        c_gpu[r_index*N+c_index] = a[r_index*N+c_index] + b[r_index*N+c_index];
    }
}

void allocate_memory_cpu(int **a, int **b, int **c, size_t size) {
    cuda_check(cudaMallocHost(a, size));
    cuda_check(cudaMallocHost(b, size));
    cuda_check(cudaMallocHost(c, size));
    if (*a == NULL || *b == NULL || *c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

void allocate_memory_gpu(int **a, int **b, int **c, size_t size) {
    cuda_check(cudaMalloc(a, size));
    cuda_check(cudaMalloc(b, size));
    cuda_check(cudaMalloc(c, size));
    if (*a == NULL || *b == NULL || *c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

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
    const size_t N = 256;
    const size_t threads_per_block = 32;
    int *cpu_a,*cpu_b,*cpu_c;
    int *gpu_a,*gpu_b,*gpu_c;
    size_t size = N * N * sizeof(int);
    allocate_memory_gpu(&gpu_a, &gpu_b, &gpu_c, size);
    allocate_memory_cpu(&cpu_a, &cpu_b, &cpu_c, size);
    cpu_init(cpu_a,cpu_b,cpu_c, N);
    gpu_init(cpu_a,cpu_b, gpu_a, gpu_b, N);
    const dim3 threads(threads_per_block,threads_per_block);
    printf("threads: %d, %d\n", threads.x, threads.y);
    const dim3 blocks((N+threads.x-1)/threads.x,(N+threads.y-1)/threads.y);   
    printf("blocks: %d, %d\n", blocks.x, blocks.y);

    // CUDA TIME
    float ms;
    float avems = 0.0;
    cudaEvent_t start,end;

    //warm up
    for(int i = 0;i < 3;i++){
        matrix_add_1t1e<<<blocks,threads>>>(gpu_a, gpu_b, gpu_c, N);
    }
    
    for(int i = 0; i < 10; i++){

        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        matrix_add_1t1e<<<blocks,threads>>>(gpu_a, gpu_b, gpu_c, N);


        cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

        printf("\tIteration no. %d: %.6fsecs\n", i, ms);
        avems+=ms;

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    printf("[**] Average kernel execution time: %.2fsec.\n\n", avems/10.0);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    int *cpu_c_gpu = (int*)malloc(size);
    cuda_check(cudaMemcpy(cpu_c_gpu, gpu_c, size, cudaMemcpyDeviceToHost));
    check_gpu(cpu_a, cpu_b, cpu_c, cpu_c_gpu, N)? printf("gpu ok\n") : printf("gpu error\n");

    cuda_check(cudaFree(gpu_a));
    cuda_check(cudaFree(gpu_b));
    cuda_check(cudaFree(gpu_c));
    cuda_check(cudaFreeHost(cpu_a));
    cuda_check(cudaFreeHost(cpu_b));
    cuda_check(cudaFreeHost(cpu_c));
    free(cpu_c_gpu);
    return 0;
}


