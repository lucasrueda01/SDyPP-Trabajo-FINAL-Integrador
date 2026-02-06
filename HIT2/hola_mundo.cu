#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel(){

    int thread_in_block = threadIdx.x;
    int block_index = blockIdx.x;
    int threads_per_block = blockDim.x;

    int global_id = block_index * threads_per_block + thread_in_block;

    printf("Hola desde el hilo global %d (block %d, thread %d)\\n", global_id, block_index, thread_in_block);

}

int main(){

    const int blocks = 2;
    const int threads_per_block = 4;

    hello_kernel<<<blocks, threads_per_block>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Error en el lanzamiento del kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Error despues de sincronizar: %s\n", cudaGetErrorString(err));
        return 1;
    }

    //Todo OK
    return 0;

}