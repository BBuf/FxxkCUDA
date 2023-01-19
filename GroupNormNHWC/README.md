在Diffusion优化里，其中一个点就是NHWC格式的GroupNorm。而平时我们训练网络都是以NCHW格式为主，此时的GroupNorm其实等价于一种特殊的layernorm，只是多了一个Group组维度：
```python
def GroupNorm(x, gamma, beta): 
    N, C, H, W = x.shape
    x = np.reshape(x, (N, G, C//G, H, W)
    mean = np.mean(x, (2, 3, 4))
    var = np.var(x, (2, 3, 4))
    normalized = (x - mean) / np.sqrt(var + eps)
    result = normalized * gamma + beta
```

下面看下各家框架对于NHWC格式的GroupNorm做了哪些优化

### OneFlow
OneFlow的GroupNorm是基于layernorm模板拓展而来，在NHWC格式下，修改了Load和Store的逻辑，相当于在输入的时候做了一次转置，从NHWC->NCHW，计算逻辑依旧还是 NCHW 格式。Store的时候再转置回来。

在计算索引时候会用到大量整数除法和取余，OneFlow使用CUTLASS里的fast divmod 快速除法来对这部分索引计算进行优化：
```cpp
template<typename SRC, typename DST>
struct ChannelsLastLoad {
  using LoadType = DST;
  ChannelsLastLoad(const SRC* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
  template<int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    cuda::layer_norm::Pack<SRC, N> pack;
    const int32_t offset = (batch_idx * c0.divisor * c1.divisor * spatial_size
                            + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx)
                           / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
  const SRC* src;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};
```
使用A100 40G，测试数据为 32 * 256 * 256 * 512， group=32，OneFlow NHWC GroupNorm 耗时1.68ms，而对应NCHW GroupNorm耗时则为1.3+ms

### TensorRT
TensorRT自己写的GroupNorm插件是在 [groupNormKernel.cu](https://github.com/NVIDIA/TensorRT/blob/main/plugin/groupNormPlugin/groupNormKernel.cu)

首先介绍下输入参数有什么：
```cpp
struct GroupNormNHWCParams
{
    // 输出，输入，gamma，beta
    __half* dst;
    __half const* src;
    float const* gamma;
    float const* beta;
    // 用于存放mean和var的结果
    float* redBuffer;

    int32_t n;
    int32_t h;
    int32_t w;
    int32_t c;
    int32_t groups;
    // 是否应用Swish激活
    bool withSwish;

    // h*w数量
    int32_t hw;
    // 每个block需要处理的h*w数量
    int32_t hwPerBlock;
    
    // 每个block处理的channel数，每个Group对应的channel数
    int32_t cPerBlock;
    int32_t cPerGroup;

    // h*w*c
    int32_t hwc;
    // h*w*c / 1, 用于计算mean和var
    float invHWC;
    // 每个block处理的group数量
    int32_t groupsPerBlock;
};

```
接着我们看下整个Kernel：
```cpp
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

```

前半部分声明了一个cub的BlockScan，这个前缀和扫描在后面会用于计算mean和mean平方值。

整个block组织成3维的，blockIdx.z方向代表batch，blockIdx.y方向代表空间维度，即H*W，blockIdx.x代表channel维度。

根据params.hwPerBlock，计算出每个block负责的空间范围，得到hwBegin和hwEnd

```cpp
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
```
这里进入到一个hwBegin~hwEnd的一个循环，以half2格式读取输入值，并转成float类型进行累加，得到sum和sum_square两个变量（如果直接用half类型累加可能会导致误差，这是一个很有用的trick）

```cpp
// The group that thread works on and the channel in the group (modulus).
    int32_t gi = threadIdx.x * 2 / params.cPerGroup;
    int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;
```
接着是计算当前线程对应的group idx，和在这个group里的channel idx

下面是用前缀和做累加的操作了，我觉得是这个Kernel里的精华部分

```cpp
    // The data for the summations.
    GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

    // Do the segmented scan.
    GroupSums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());
```
它构造了一个GroupSums结构体，而GroupSum结构体长这样：
```cpp

struct GroupSums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
    // The sum of squares.
    float sumSq;
};
```
一共用3个变量：
- flag用于标记当前值是不是group里第一个元素
- sum累加值
- sum_square
并且也定义了对应的Reduce Op
```cpp
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
```
两个输入做sum，如果右边元素是group里第一个元素，那它就不会累加左边输入，反之则累加。

我们回顾下GroupNorm，在NCHW格式下，他是先变换成(N, G, C/G, H, W)，对(C/G, H, W)这后三个维度做reduce得到mean和var。那么对应到前面，我一直累加，然后直到下一个元素是另外一个Group的第1个元素，说明我这部分累加完毕了，需要重新累加。这里通过巧妙的定制化Reduce+前缀和的操作，在NHWC格式上直接完成了整个过程，而不需要做相关坐标变换。


我们还可以写一个例子来加深这里的理解，假设一块连续的元素，我希望每连续的32个元素累加到一起，那对应的example是：

```cpp
#include <cub/cub.cuh>
#include "cuda_fp16.h"

struct Sums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
};

struct SumsOp
{
    inline __device__ Sums operator()(Sums const& a, Sums const& b)
    {
        Sums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

template <int32_t tTHREADS_PER_BLOCK>
__global__ void BlockScanKernel(float* x)
{
    // The object in charge of doing the sums for the different blocks.
    typedef cub::BlockScan<Sums, tTHREADS_PER_BLOCK> BlockScan;

    // Allocate shared memory for BlockScan.
    __shared__ typename BlockScan::TempStorage tempStorage;
    // Allocate shared memory for the groups. We could reduce the amount of shared
    // memory reserved.

    int32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x; 

    float sum = x[global_idx];

    // The data for the summations.
    Sums inp{global_idx % 32 == 0 ? 1 : 0, sum};

    // Do the segmented scan.
    Sums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, SumsOp());

    printf("Global Idx is: %d, sum is: %f \n", global_idx, out.sum); 
}

int main(){
    int32_t elem_cnt = 64; 
    
    float* x; 
    x = (float*)malloc(elem_cnt * sizeof(float)); 
    for( int i = 0; i < elem_cnt; i++){
        // x[i] = i; 
        x[i] = 1; 
    }
    float* device_x; 
    cudaMalloc(&device_x, sizeof(float) * elem_cnt); 

    cudaMemcpy(device_x, x, sizeof(float)*elem_cnt, cudaMemcpyHostToDevice); 
    BlockScanKernel<64><<<1, 64>>>(device_x); 
    cudaDeviceSynchronize(); 
    cudaFree(device_x); 
    free(x); 
}

```
打印出来的结果如下：
```text
Global Idx is: 32, sum is: 1.000000 
Global Idx is: 33, sum is: 2.000000 
Global Idx is: 34, sum is: 3.000000 
Global Idx is: 35, sum is: 4.000000 
Global Idx is: 36, sum is: 5.000000 
...
Global Idx is: 63, sum is: 32.000000 
Global Idx is: 0, sum is: 1.000000 
Global Idx is: 1, sum is: 2.000000 
Global Idx is: 2, sum is: 3.000000 
Global Idx is: 3, sum is: 4.000000 
Global Idx is: 4, sum is: 5.000000 
...
Global Idx is: 31, sum is: 32.000000 
```

剩下这部分比较简单，就是将sum和sum_square存到shared memory，然后用atomicAdd的方式，将N*G个统计量存储到预先分配好的buffer里：
```cpp
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
```
另外会启一个kernel，将sum和sumsquare计算得到mean和var，并且读取gamma和beta做对应的缩放，这部分比较简单就不展开了：
```cpp
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
```

最后实测总的耗时是1.3ms

### 额外思考
今天想把两个单独的kernel合并到一起，利用shared memory缓存输入，结果一直算不对，后面发现是自己脑残了

该Kernel的特点就是用atomic来完成累加，而此时也有可能其他Block还在执行atomicAdd操作，你当前block此时取值就是错的，因此 **必须分开成两个Kernel执行**

想在此基础之上优化可能就是向量化云云，这里就不展开了。

另外一种思路之前想到，是类似Transpose，把NHWC格式的数据读取一部分到shared Memory，然后进行Transpose，转成NCHW进行处理，后续再Transpose回去，但我觉得还是TensorRT这个思路比较好

