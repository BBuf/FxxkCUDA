#include "stdio.h"
#include <type_traits>
#include <cstdint>
#include <algorithm>


constexpr int32_t kWarpSize = 32; 
constexpr int32_t kVecSize = 4; 


// Refer from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/elementwise.cuh
template<typename T, int pack_size>
struct GetPackType{
    using type = typename std::aligned_storage<sizeof(T)*pack_size, sizeof(T)*pack_size>::type; 
}; 

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;


template<typename T, int pack_size>
union Pack{
    __device__ Pack(T val){
        #pragma unroll
        for(int i = 0; i < pack_size; i++){
            elem[i] = val; 
        }
    }
    PackType<T, pack_size> storage; 
    T elem[pack_size];  

    __device__ void operator+=(Pack<T, pack_size> packA){
        #pragma unroll 
        for(int i = 0; i < pack_size; i++){
            elem[i] += packA.elem[i]; 
        }
    }
}; 

template<typename T, int pack_size>
__device__ T PackReduce(Pack<T, pack_size> pack){
    T res = 0.0; 
    #pragma unroll
    for(int i = 0; i < pack_size; i++){
        res += pack.elem[i]; 
    }
    return res; 
}

template<typename T>
__device__ T WarpReduce(T val){
    for(int lane_mask = 16; lane_mask > 0; lane_mask /=2){
        val += __shfl_down_sync(0xffffffff, val, lane_mask); 
    }
    return val; 
}


template<typename T>
__device__ T BlockReduce(T val, T* shm){
    const int32_t warp_idx = threadIdx.x / kWarpSize; 
    const int32_t lane_idx = threadIdx.x % kWarpSize; 

    T warp_reduce_sum = WarpReduce<T>(val); 
    if(lane_idx == 0){
        shm[warp_idx] = warp_reduce_sum; 
    }
    __syncthreads();
    T sum_val = (threadIdx.x < blockDim.x / kWarpSize) ? shm[lane_idx] : T(0.0); 
    if(warp_idx == 0){
        sum_val = WarpReduce<T>(sum_val); 
    } 
    return sum_val; 
}


__global__ void reduce1d_kernel(const float* x, float* out, int32_t elem_cnt){
    const int32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x; 
    const int32_t step = blockDim.x * gridDim.x; 

    Pack<float, 4> sum_pack(0.0); 
    Pack<float, 4> load_pack(0.0); 
    using LoadType = PackType<float, 4>; 
    for(int32_t linear_index = global_idx * kVecSize; linear_index < elem_cnt; linear_index+=step*kVecSize){
        const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
        load_pack.storage = *x_load; 
        sum_pack += load_pack; 
    }
    __shared__ float shm[kWarpSize]; 
    float pack_reduce_val = PackReduce<float, 4>(sum_pack); 
    float block_reduce_val = BlockReduce<float>(pack_reduce_val, shm); 
    if(threadIdx.x == 0){
        out[blockIdx.x] = block_reduce_val; 
    }
}


constexpr int kBlockSize = 256;
constexpr int kNumWaves = 1;

int64_t GetNumBlocks(int64_t n) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  int64_t num_blocks = std::max<int64_t>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
  return num_blocks;
}


int main(){
    float* host_x; 
    float* host_out; 
    const int32_t elem_cnt = 32*1024*1024; 
    host_x = (float*)malloc(sizeof(float) * elem_cnt); 
    host_out = (float*)malloc(sizeof(float)); 

    float* device_x; 
    float* device_part_sum; 
    float* device_out;
    
    const int64_t block_num = GetNumBlocks(elem_cnt); 
    printf("Block num is: %ld \n", block_num); 
    cudaMalloc(&device_x, sizeof(float) * elem_cnt);
    cudaMalloc(&device_part_sum, sizeof(float) * block_num); 
    cudaMalloc(&device_out, sizeof(float)); 

    for(int i = 0; i < elem_cnt; i++){
        host_x[i] = i; 
    }
    cudaMemcpy(device_x, host_x, sizeof(float) * elem_cnt, cudaMemcpyHostToDevice); 
    
    reduce1d_kernel<<<block_num, kBlockSize>>>(device_x, device_part_sum, elem_cnt); 
    reduce1d_kernel<<<1, kBlockSize>>>(device_part_sum, device_out, block_num); 

    cudaMemcpy(host_out, device_out, sizeof(float), cudaMemcpyDeviceToHost); 

    printf("Host out is: %f \n", host_out[0]); 
    cudaFree(device_x); 
    cudaFree(device_part_sum); 
    cudaFree(device_out); 
    free(host_x); 
    free(host_out); 
    return 0; 
}
