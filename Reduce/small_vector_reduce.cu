#include "stdio.h"

__device__ float warp_reduce(float x){
    #pragma unroll 
    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
        x += __shfl_down_sync(0xffffffff, x, lane_mask);
    }
    return x; 
}


/*
grid = 1
block = (32, 1)
*/
constexpr int kWarpSize = 32; 

__global__ void reduce1d_kernel(float* x, float* out, int32_t elem_cnt){
    int32_t global_idx = threadIdx.x; 
    float sum = static_cast<float>(0.0); 
    for(int idx=global_idx; idx < elem_cnt; idx+=blockDim.x){
        sum += x[idx]; 
    }
    float row_reduce_sum = warp_reduce(sum); 
    out[0] = row_reduce_sum; 
}

__device__ float BlockReduce(float val, float* shared){
    const int32_t warp_idx = threadIdx.x / kWarpSize; 
    const int32_t lane_idx = threadIdx.x % kWarpSize; 
    float sum_val = warp_reduce(val); 
    if(lane_idx == 0){
        shared[warp_idx] = sum_val; 
    }
    __syncthreads(); 
    sum_val = (threadIdx.x < blockDim.x / kWarpSize) ? shared[lane_idx] : static_cast<float>(0.0);
    if(warp_idx == 0){
        sum_val = warp_reduce(sum_val); 
    } 
    return sum_val; 
}

__global__ void block_reduce1d_kernel(float* x, float* out, int32_t elem_cnt){
    int32_t global_idx = threadIdx.x; 
    float sum = static_cast<float>(0.0); 
    for(int idx=global_idx; idx < elem_cnt; idx+=blockDim.x){
        sum += x[idx]; 
    }
    __shared__ float shared_arr[kWarpSize]; 
    float row_reduce_sum = BlockReduce(sum, shared_arr); 
    out[0] = row_reduce_sum; 
}

int main(){
    float* host_x; 
    float* host_out; 
    host_x = (float*)malloc(sizeof(float) * 1024); 
    host_out = (float*)malloc(sizeof(float)); 

    float* device_x; 
    float* device_out;
    cudaMalloc(&device_x, sizeof(float) * 1024); 
    cudaMalloc(&device_out, sizeof(float)); 

    for(int i = 0; i < 1024; i++){
        host_x[i] = i; 
    }
    cudaMemcpy(device_x, host_x, sizeof(float) * 1024, cudaMemcpyHostToDevice); 
    // reduce1d_kernel<<<1, 32>>>(device_x, device_out, 1024); 
    block_reduce1d_kernel<<<1, 256>>>(device_x, device_out, 1024); 
    cudaMemcpy(host_out, device_out, sizeof(float), cudaMemcpyDeviceToHost); 

    printf("Host out is: %f \n", host_out[0]); 
    cudaFree(device_x); 
    cudaFree(device_out); 
    free(host_x); 
    free(host_out); 
    return 0; 
}
