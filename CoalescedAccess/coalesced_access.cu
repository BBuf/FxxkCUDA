__global__ void AccessKernel(float* in, float* out, int64_t elem_cnt){
    for(int i = threadIdx.x; i < elem_cnt; i+=blockDim.x * gridDim.x){
        out[i] = in[i]; 
    }
}

int main(){
    int64_t elem_cnt = 64; 

    float* device_in; 
    float* device_out;
    cudaMalloc(&device_in, elem_cnt * sizeof(float)); 
    cudaMalloc(&device_out, elem_cnt * sizeof(float)); 

    AccessKernel<<<1, 32>>>(device_in, device_out, elem_cnt); 

    cudaFree(device_in); 
    cudaFree(device_out); 

    return 0; 
}