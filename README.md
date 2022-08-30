Just a simple repo to collect high performance cuda kernel. 

### Topic1: Reduce

关于CUDA Reduce优化其实已经有很多经典的博客了，比如willzhang的这篇：

https://zhuanlan.zhihu.com/p/416959273

这里主要挑几个点讲下：

### 合并访存
CUDA访问global memory是要让一个warp访问的尽可能连续，这里摘一下英伟达的PPT：

![](https://files.mdnice.com/user/4601/5897df2e-4dfe-4802-b9ad-77bfa7adbb05.png)

而对于实际访问，如果一个线程要访问多个元素，则需要跳blockDim.x*gridDim.x来访问

![](https://files.mdnice.com/user/4601/d244f8a9-e8e5-4a02-a00a-6f01ad877a03.png)

考虑到CUDA支持load128bit，我们也可以用向量化去访问，进一步提升带宽，减少指令数量，这部分在 packed_reduce.cu 里面也有体现：

```cpp
    using LoadType = PackType<float, 4>; 
    for(int32_t linear_index = global_idx * kVecSize; linear_index < elem_cnt; linear_index+=step*kVecSize){
        const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
        load_pack.storage = *x_load;
        ...
```
他等价于：

![](https://files.mdnice.com/user/4601/46bdcbb4-0c73-4c4d-99ef-252901bc4b9e.png)

注意的是，这里实际发生访存是转成float4类型指针后才取的值，并且此时是跨 blockDim.x * gridDim.x * 4 来取值，此时访存是连续的。

而如果是float指针，一个个取，取满4个，则实际发生访存的还是float指针，也就**造成前面说的访存不连续**，注意区别！

![](https://files.mdnice.com/user/4601/8c5aac6b-f756-4b6a-8f48-d78e4879f992.png)

累加完Pack，再进入到下一个BlockReduce之前，我们需要对这个Pack先reduce，方法也很简单，就是一个展开的for循环：
```cpp
template<typename T, int pack_size>
__device__ T PackReduce(Pack<T, pack_size> pack){
    T res = 0.0; 
    #pragma unroll
    for(int i = 0; i < pack_size; i++){
        res += pack.elem[i]; 
    }
    return res; 
}
```
PS：我这里情况考虑的没那么复杂，但是涉及到累加这些操作的话，如果在fp16情况下，我更建议模板类型传入一个ComputeType，设置成fp32来做累加，避免溢出

### WarpReduce
willzhang博客其实是开辟了一片shared_memory来作为block中间存储，但其实我觉得blockreduce完全是够用了，在介绍BlockReduce之前，先介绍WarpReduce

CUDA执行单位还是warp，每32个线程构成一个warp。而CUDA自身也提供了一些线程束原语，我们可以借助他们来做warp级别的reduce

具体可参考NV官方的这篇博客：https://zhuanlan.zhihu.com/p/522714729

我们这里使用的是shfl_down_sync来做reduce，示意图如下：
![](https://files.mdnice.com/user/4601/81116856-e2ed-47b2-a7f8-53f8219d624f.png)

需要注意的是，最后只有线程0才是最终正确的reduce结果

### BlockReduce
这里参考的是Pytorch的BlockReduce实现

首先有个前提，BlockReduce借助warpreduce来做，因此blocksize必须是warp的整数倍

1. 我们先让所有线程执行warpreduce
2. 然后将每个线程束的reduce结果存储到shared memory中，注意这里是lane_id=0的线程去存储，因为前面我们提到了只有线程0上有正确的reduce结果
3. 从shared memory把数据读取出来，最后再用一个warp对其做reduce，即可获得整个block的reduce结果

### 最终的Reduce？
采取blockreduce后，每个block都有部分和，而如果要最终累加到一个标量里，那么其实有两种思路：

1. 用atomicAdd累加，但是看李少侠的说法其实会有精度问题

> global memory的atomicAdd和标准的浮点加法不等价，atomicAdd对denormalized float是round to zero的，理论上两者精度不一样

2. 启动两次Kernel，第一次Kernel reduce得到block个部分和结果。然后再启动一个只有1个block的kernel，做最终的求和

