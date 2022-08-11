/*
Reference from https://zhuanlan.zhihu.com/p/416959273
*/

#include "stdio.h"


__global__ void reduce1d_kernel(float* x, float* out, int32_t elem_cnt){
    const int32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x; 
    const int32_t step = blockDim.x * gridDim.x; 

    float local_sum_val = 0.0f; 

    for(int32_t i = global_idx; i < elem_cnt; i+=step){
        local_sum_val += x[i]; 
    }
    extern __shared__ float shm[]; 
    shm[threadIdx.x] = local_sum_val; 

    __syncthreads(); 
    for(int32_t active_thread_num = blockDim.x /2; active_thread_num > 32; active_thread_num /=2){
        if(threadIdx.x < active_thread_num){
            shm[threadIdx.x] += shm[threadIdx.x + active_thread_num]; 
        }
        __syncthreads(); 
    }

    if (threadIdx.x < 32) {
        volatile float* vshm = shm;
        if (blockDim.x >= 64) {
          vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }
        float val = vshm[threadIdx.x]; 
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /=2){
            val += __shfl_down_sync(0xffffffff, val, lane_mask); 
        }
        if (threadIdx.x == 0) {
            out[blockIdx.x] = val;
        }
    }
}

int main(){
    float* host_x; 
    float* host_out; 
    host_x = (float*)malloc(sizeof(float) * 1024); 
    host_out = (float*)malloc(sizeof(float)); 

    float* device_x; 
    float* device_part_sum; 
    float* device_out;
    constexpr size_t block_num = 4; 
    cudaMalloc(&device_x, sizeof(float) * 1024);
    cudaMalloc(&device_part_sum, sizeof(float) * block_num); 
    cudaMalloc(&device_out, sizeof(float)); 

    for(int i = 0; i < 1024; i++){
        host_x[i] = i; 
    }
    cudaMemcpy(device_x, host_x, sizeof(float) * 1024, cudaMemcpyHostToDevice); 
    reduce1d_kernel<<<4, 256>>>(device_x, device_part_sum, 1024); 
    reduce1d_kernel<<<1, 256>>>(device_part_sum, device_out, block_num); 
    cudaMemcpy(host_out, device_out, sizeof(float), cudaMemcpyDeviceToHost); 

    printf("Host out is: %f \n", host_out[0]); 
    cudaFree(device_x); 
    cudaFree(device_part_sum); 
    cudaFree(device_out); 
    free(host_x); 
    free(host_out); 
    return 0; 
}
