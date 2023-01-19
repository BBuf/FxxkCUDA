#include <cub/cub.cuh>
#include "cuda_fp16.h"

static inline int32_t divUp(int32_t m, int32_t n)
{
    return (m + n - 1) / n;
}

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor)
{
    int32_t maxDivisor = -1;
    for (int32_t i = 1; i <= std::sqrt(n); i++)
    {
        if (n % i == 0)
        {
            int32_t divisor1 = n / i;
            int32_t divisor2 = i;

            if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor)
            {
                maxDivisor = divisor1;
            }
            if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor)
            {
                maxDivisor = divisor2;
            }
        }
    }
    return maxDivisor;
}

size_t getWorkspaceSizeInBytes()
{
    return (sizeof(float) * 2) * 32 * 32; // sizeof(float2) * maxBatchSize * maxNumberOfGroup. float2
                                          // contians two buffers for sum and squared sum
}

static inline __device__ __host__ float sigmoid(float x)
{
    return 1.F / (1.F + expf(-x));
}

struct GroupNormNHWCParams
{
    // The output buffer. Layout NHWC.
    __half* dst;
    // The input buffer. Layout NHWC.
    __half const* src;
    // The gamma scaling factor.
    float const* gamma;
    // The beta term to add in GN.
    float const* beta;
    // The temporary buffer to do the global parallel reduction. Size:
    // BLOCKS_PER_BATCH x C x 2.
    float* redBuffer;

    // The number of instances in the batch.
    int32_t n;
    // The height and width of each activation map.
    int32_t h;
    int32_t w;
    // The number of channels.
    int32_t c;
    // The number of groups.
    int32_t groups;
    // Do we apply the Swish activation function?
    bool withSwish;

    // Precomputed values and parameters to control the execution of the kernels.

    // The number of activations per instance (h * w) and the number of
    // activations per block.
    int32_t hw;
    int32_t hwPerBlock;
    // The number of channels per group and blocks per activation in the C
    // dimension.
    int32_t cPerBlock;
    int32_t cPerGroup;

    // The precomputed stride between instances.
    int32_t hwc;
    // The inverse of hwc in floats (to compute mean/var).
    float invHWC;
    // The precomputed number of groups per block.
    int32_t groupsPerBlock;
};

struct GroupSums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
    // The sum of squares.
    float sumSq;
};

struct GroupSumsOp
{
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b)
    {
        GroupSums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams params)
{
    // The object in charge of doing the sums for the different blocks.
    typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

    // Allocate shared memory for BlockScan.
    __shared__ typename BlockScan::TempStorage tempStorage;
    // Allocate shared memory for the groups. We could reduce the amount of shared
    // memory reserved.
    __shared__ float2 smem[tTHREADS_PER_BLOCK];

    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // The sums.
    float sum = 0.F;
    float sumSq = 0.F;

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The offset.
        int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        float2 f2 = __half22float2(h2);

        // Update the sum.
        sum += f2.x + f2.y;
        // Update the sum of squares.
        sumSq += f2.x * f2.x + f2.y * f2.y;
    }

    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = threadIdx.x * 2 / params.cPerGroup;
    int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;

    // The data for the summations.
    GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

    // Do the segmented scan.
    GroupSums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

    // Store the results for the groups in shared memory (to produce coalesced
    // stores later).
    if (cj == params.cPerGroup - 2 /* 2 channels per thread */)
    {
        smem[gi] = make_float2(out.sum, out.sumSq);
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The global group index.
    int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

    // Threads that have nothing left to do, exit.
    if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups)
    {
        return;
    }

    // The first threads (those storing to global memory, load the values).
    float2 sums = smem[threadIdx.x];

    // Store to global memory.
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void groupNormNHWCSum(GroupNormNHWCParams const& params)
{
    // Make sure the values are as we expect.
    // PLUGIN_ASSERT(params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    // PLUGIN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCSumKernel<160><<<grid, 160>>>(params); break;
    case 480: groupNormNHWCSumKernel<256><<<grid, 256>>>(params); break;
    case 256: groupNormNHWCSumKernel<128><<<grid, 128>>>(params); break;
    case 128: groupNormNHWCSumKernel<64><<<grid, 64>>>(params); break;
    default: break;
    }

    // PLUGIN_CUASSERT(cudaGetLastError());
}

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams params)
{
    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = ci / params.cPerGroup;

    // Load the sum and sum of squares for the group.
    float sum = 0.F, sumSq = 0.F;
    if (gi < params.groups)
    {
        sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
        sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
    }

    // Load gamma/beta.
    float2 gammaF2, betaF2;
    if (ci < params.c)
    {
        gammaF2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
        betaF2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);
    }

    // Compute the mean.
    float mean = sum * params.invHWC;
    // Compute the variance.
    float var = sumSq * params.invHWC - (mean * mean);
    // Compute the inverse of the stddev.
    float invStdDev = var <= 0.F ? 1.F : rsqrtf(var);

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The src/dst offset.
        int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        float2 f2 = __half22float2(h2);

        // Normalize the channels.
        f2.x = (f2.x - mean) * invStdDev;
        f2.y = (f2.y - mean) * invStdDev;

        // Scale by gamma and add beta.
        f2.x = gammaF2.x * f2.x + betaF2.x;
        f2.y = gammaF2.y * f2.y + betaF2.y;

        // Apply Swish if needed.
        if (params.withSwish)
        {
            f2.x = f2.x * sigmoid(f2.x);
            f2.y = f2.y * sigmoid(f2.y);
        }

        // Store the scaled values.
        if (ci < params.c)
        {
            *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
        }
    }
}

void groupNormNHWCScale(GroupNormNHWCParams const& params)
{
    // Make sure the dimensions are aligned with what we expect.
    // PLUGIN_ASSERT(params.c % params.cPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    // PLUGIN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCScaleKernel<160><<<grid, 160>>>(params); break;
    case 480: groupNormNHWCScaleKernel<256><<<grid, 256>>>(params); break;
    case 256: groupNormNHWCScaleKernel<128><<<grid, 128>>>(params); break;
    case 128: groupNormNHWCScaleKernel<64><<<grid, 64>>>(params); break;
    // default: PLUGIN_FAIL("Not implemented");
    default: break;
    }
}

int main(){
    int n = 32; 
    int groups = 32; 
    int c = 512; 
    int h = 128; 
    int w = 128; 
    const int elem_cnt = n * c * h * w; 
    half* in; 
    half* out; 
    float* gamma; 
    float* beta; 
    float* buffer; 

    const int workspace_size = getWorkspaceSizeInBytes(); 
    cudaMalloc(&in, sizeof(half) * elem_cnt); 
    cudaMalloc(&out, sizeof(half) * elem_cnt); 
    cudaMalloc(&gamma, sizeof(float) * n * groups); 
    cudaMalloc(&beta, sizeof(float) * n * groups); 
    cudaMalloc(&buffer, workspace_size); 


    int32_t cPerBlock = 320;
    int32_t maxBlocksPerHW = 1024;
    GroupNormNHWCParams mParams{}; 
    switch (c)
    {
    case 960:
    case 1920: cPerBlock = 480; break;
    case 512:
    case 256: cPerBlock = 256; break;
    case 128: cPerBlock = 128; break;
    default: cPerBlock = 320;
    }

    mParams.withSwish = false;
    mParams.dst = out;
    mParams.src = in;
    mParams.gamma = gamma;
    mParams.beta = beta;
    mParams.redBuffer = buffer;
    mParams.n = n;
    mParams.h = h;
    mParams.w = w;
    mParams.c = c;
    mParams.groups = 32;
    mParams.hw = mParams.h * mParams.w;
    const int32_t blocksPerHW = findMaxDivisor(mParams.hw, maxBlocksPerHW);
    mParams.hwPerBlock = divUp(mParams.hw, blocksPerHW);
    mParams.cPerBlock = cPerBlock;
    mParams.cPerGroup = mParams.c / mParams.groups;
    mParams.hwc = mParams.hw * mParams.c;
    mParams.invHWC = 1.F / (float) (mParams.hw * mParams.cPerGroup);
    mParams.groupsPerBlock = cPerBlock / mParams.cPerGroup;

    // cudaMemsetAsync(mParams.redBuffer, 0, getWorkspaceSizeInBytes(), stream);
    groupNormNHWCSum(mParams);
    groupNormNHWCScale(mParams);

    return 0; 
}
