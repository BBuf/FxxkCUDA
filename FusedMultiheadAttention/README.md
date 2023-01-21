> 在看这部分cutlass代码逻辑进行推导的时候，我的内心活动：哦对的对的 -> 不对不对，推错了 -> 对的对的对的 -> 对对对，对吗？-> 哦对的对的对的

## 前言

最近 xFormer 模型库针对 FusedMultiheadAttention 使用cutlass重写，引入了FlashAttention的思想。笔者也不是特别了解cutlass，只是想简单记录下阅读代码，如有错误纰漏也欢迎指正。

本篇博客所解读的FusedMultiheadAttention代码版本是 [41_fused_multi_head_attention](https://github.com/NVIDIA/cutlass/tree/master/examples/41_fused_multi_head_attention)，对应 commit 版本是3f2bb17。上游xFormers一直在改动，所以这部分代码在后面也可能有所变动。

## 整体结构

Attention的公式为：

$$
Softmax(\frac{Q \odot K}{\sqrt{head\_dim})}) \odot V
$$

其中QKV对应的形状为：(batch, num_head, seq_len, head_size)，而对应的矩阵乘部分操作，我们可以看作是 batch * num_head次数的矩阵乘。

Kernel对于Grid划分规则是这样：

- blockIdx.z 对应batch
- blockIdx.y 对应num_head
- blockIdx.x 对应block在seq_len要处理多少次，即 seq_len / kQueriesBlock

因此一个Block处理的部分是：

```
Query(kQueriesBlock, head_size) 
Key(head_size, seq_len)
Value(seq_len, head_size)
```

![](https://files.mdnice.com/user/4601/b38e9949-3b93-4cbd-972b-6cd814a72277.png)

下面简单介绍下AttentioonKernel里的各个结构体，主要代码是在 `kernel_forward.h`头文件里：

### struct Params

这是一个结构体，传入给AttentionKernel的所有参数都在里面，参数比较多，但是很多都是类似的，我们简单浏览一下：

```cpp
    // 输入tensor的指针
    scalar_t* query_ptr; // [num_queries, num_heads, head_dim]
    scalar_t* key_ptr;   // [num_keys, num_heads, head_dim]
    scalar_t* value_ptr; // [num_keys, num_heads, head_dim_value]
    // 这个是用于变长输入的指针，可以暂时先不管
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;

    // 输出Tensor的指针
    output_t* output_ptr; // [num_queries, num_heads, head_dim_value]
    // 输出buffer的指针，可以暂时先不管
    output_accum_t*
        output_accum_ptr; // [num_queries, num_heads, head_dim_value]
    // 存储中间结果用于求导的，可以先不管
    lse_scalar_t* logsumexp_ptr; // [num_heads, num_queries] - can be null

    // Tensor维度/stride
    int32_t head_dim; // head_size
    int32_t head_dim_value; // value_head_size
    int32_t num_queries; // query_seq_len
    int32_t num_keys; // key_seq_len
  
    // 是否对上三角做Mask
    bool causal; 
  
    /*
    矩阵计算通常被表示为
    (M, K) (K, N) -> (M, N)
    前面提到相当于做了batchsize * numhead次矩阵乘
    这里M维度的stride，其实就等于在pytorch里的
    query.stride(2) = head_size
    */
  
    int32_t q_strideM;
    int32_t k_strideM;
    int32_t v_strideM;

    // xx_strideH代表num_head维度的stride
    // 等价于query.stride(1)
    int32_t q_strideH;
    int32_t k_strideH;
    int32_t v_strideH;
    int32_t o_strideH;
  
    // xx_strideB代表batch维度的stride
    // 等价于query.stride(0)
    int64_t q_strideB;
    int64_t k_strideB;
    int64_t v_strideB;
    int64_t o_strideB;
    int32_t num_batches;
    int32_t num_heads;
```

Params内部也有一些内建方法，下面我们看下：

#### advance_to_block

该方法的作用是在计算执行之前，将数据指针移动到合适的位置，它的返回值是一个bool变量，如果返回False代表当前Block没有计算任务。由于不同Block所负责的数据块是不一样的，因此要利用blockIdx.z/y/x来提前进行移动。而计算偏移量之前，我们需要给参数传入对应的Tensor步长，即stride，这个概念和Torch的是一样的，这里就不展开了。

一段代码如下所示：

```cpp
// Advance to the current batch / head / query_start
query_ptr += batch_id * q_strideB;
query_ptr += (q_start + query_start) * q_strideM + head_id * q_strideH;
```

这里还有一段关于causal attention矩阵取下三角元素的逻辑：

```cpp
if (causal) {
    num_keys = cutlass::fast_min(
        int32_t(query_start + kQueriesPerBlock), num_keys);
  }
```

![](https://files.mdnice.com/user/4601/c8d1f2d4-f764-43ca-853f-9974b8026d59.png)

由于只要下三角元素，**我们可以根据query当前位置，来限制Key矩阵的迭代范围，进而减少不必要的计算**。

第一行query它对应只是attention matrix的一个元素，那么此时Key迭代范围只有1

第二行query对应attention matrix第二行的两个元素，那么Key迭代范围有2

第三行query对应attention matrix第三行的三个元素，那么Key迭代范围有3

### struct MM0

该结构体表示第一个矩阵乘，即 Query matmul Key 这个操作。这里它借用了cutlass example 里 [13_two_tensor_op_fusion](https://github.com/NVIDIA/cutlass/tree/master/examples/13_two_tensor_op_fusion) 的 back2back gemm操作，并且加入了一个ScaleSoftmaxUpdater，下面我们分别解释一下：

#### Back2Back Gemm

Back2Back Gemm的意思是连续的两个Gemm操作。如果单独拆成2个Gemm，则整个流程是：

- 读取第一次GEMM数据权重->计算->存回GlobalMemory
- 读取第一次GEMM计算结果->读取第二次GEMM权重->计算->存回GlobalMemory

那么换做Back2Back GEMM，由于第二次GEMM用到的数据其实是第一次GEMM的结果，**所以我们可以把第一次GEMM的结果存放在寄存器里**，然后紧接着执行第二次GEMM，以节省写入写回Global Memory的开销：

![](https://files.mdnice.com/user/4601/86dbe570-a57f-4397-b78f-ba7e598331bf.png)

#### AttentionScalingUpdaters

在MM0结构体里有这么一段：

```
using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Updater;
```

这个其实就是TiledSoftmax更新缩放值的一个Updater，在之前的一篇博客里有简单推导：[FlashAttention简记](https://zhuanlan.zhihu.com/p/582606847)，但是那篇博客里没有详细介绍保证数值稳定性情况下的TileSoftmax推导，这里就补充下。

在实际Softmax公式里，为了保证数值稳定性，我们通常需要减去一个最大值，一段numpy代码如下：

```
max_val = np.max(x)
x_shift = x - max_val
x_exp = np.exp(x_shift)
```

那么假设我们的一个向量是 `[0.1, 0.2, 0.3, 0.4]`，那么常规的softmax结果应该是（我们取0.1的softmax为例）：

$$
\begin {aligned}
\frac{e^{0.1 - 0.4}}{e^{0.1 - 0.4} + e^{0.2 - 0.4} + e^{0.3 - 0.4} + e^{0.4 - 0.4}}
\end {aligned}
$$

那么现在我们讲一下TiledSoftmax做法，首先放上 `attention_scaling_coefs_updater.h`里的注释：

```
// 先更新整个行的最新Max值
(1) Update `mi[r]` to the max value of the row `r`
// 然后进行下面的操作
(2) In a second iteration do the following:
    // accum里存放的是Q matmul K的临时结果
    (a) accum   <- exp(accum - mi) 得到softmax公式分子
    // m_prime表示前一轮的max值
    (b) m_prime <- exp(m_prime - mi)
    // s_prime表示前一轮的scale值，即softmax公式分母
    (c) s_prime <- s_prime * m_prime + sum(accum)
```

假设我们一次性只算两个数的Softmax结果，一共要算两次，那么第一次对应：

$$
\frac{e^{0.1 - 0.2}}{e^{0.1 - 0.2} + e^{0.2 - 0.2}}
$$

按照Updater流程那就是：

```
1. 更新行的Max值，mi[r] = 0.2
2. accum = [exp(0.1-0.2), exp(0.2-0.2)]
3. 由于此时是第一轮，之前没有最大值，因此是 m_prime <- exp(0.0-0.2)
4. 由于此时是第一轮，之前没有scale值，因此是 s_prime <- 0 + (exp(0.1-0.2) + exp(0.2-0.2))
```

此时进行第二次计算，行方向的Max值已经更新成0.4了，那么我们需要调整第一次tilesoftmax结果进行修正，那还是按照流程：

```
1. 更新行的Max值，mi[r] = 0.4
2. accum = [exp(0.3-0.4), exp(0.4-0.4)]
3. m_prime <- exp(0.2-0.4)
4. s_prime 计算如下：
```

$$
s\_prime * m\_prime = (e^{0.1-0.2} + e^{0.2-0.2}) * (e^{0.2 - 0.4})
$$

$$
s\_prime * m\_prime = (e^{0.1-0.4} + e^{0.2-0.4})
$$

$$
= (e^{0.3-0.4} + e^{0.4-0.4})
$$

$$
更新s\_prime = e^{0.1-0.4} + e^{0.2-0.4} + e^{0.3-0.4} + e^{0.4-0.4}
$$

是不是很神奇，而对应Attention matmul Value这部分是一种累加，所以只需要每次对上一轮的结果进行一个scaling，再加上最新的计算结果即可。

下面我们跟着代码过一遍：

```cpp
// 如果当前不是第一轮，则把上一轮的mi存储的max值存到m_prime
  if (!kIsFirst) {
      if (thread_id < kQueriesPerBlock) {
        m_prime[thread_id] = mi[thread_id];
      }
      __syncthreads();
    }


// 1. 更新每一行的MaxValue
    {
      accum_t max;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (kFullColumns || accum_n < max_col) {
              // 计算Max值
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            /*
            使用atomicMax来进行原子更新
            按照公式，qk要除一个sqrt(head_size)
            这个常数scale连带在tilesoftmax里也做了
            因此这里是 max * scaling
            */ 
            atomicMaxFloat(&mi[accum_m], max * scaling);
          });
    }
  
    // 2. 对fragment乘上scaling * log2e，这里log2e后面会消除掉，感觉可能是为了一些数值稳定性才多乘一个log2e
    frag = cutlass::multiplies<typename T::Fragment>()(scaling * kLog2e, frag);
    __syncthreads();

  
    // 3. 更新m_prime, s_prime（这里只包含了 s_prime * m_prime）
    if (thread_id < kQueriesPerBlock) {
      auto m_prime_exp = exp2f(kLog2e * (m_prime[thread_id] - mi[thread_id]));
      m_prime[thread_id] = m_prime_exp;
      s_prime[thread_id] *= m_prime_exp;
    }
    __syncthreads(); // Update output fragments
  
    // 4. 对前一轮的计算结果进行scale
    if (kKeepOutputInRF && !kIsFirst) {
      accum_t mp;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mp = m_prime[accum_m]; },
          [&](int accum_m, int accum_n, int idx) { frag_o[idx] *= mp; },
          [&](int accum_m) {});
      __syncthreads();
    }
  
    // 5. s_prime更新里的 sum(accum) 部分
    {
      accum_t mi_row, total_row;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = kLog2e * mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] = (kFullColumns || accum_n < max_col)
                ? exp2f(frag[idx] - mi_row)
                : accum_t(0.0);
          },
          [&](int accum_m) {});
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (BASE::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              atomicAdd(&s_prime[accum_m], total_row);
            }
          });
    }
```

MM0部分介绍到这里，MM1就是一个普通的矩阵乘，这里就不再赘述。

### epilogue_rescale_output.h

前面介绍Softmax操作时，虽然公式是：

$$
Softmax(x_{i}) = \frac{e^{x_{i} - max}}{\sum_{i=0}e^{x_{i} - max}}
$$

但是实际进行Attention matmul Value这部分计算时，是**只取了分子部分直接和Value进行计算**，即：

$$
e^{Attention} \odot Value
$$

到了**最后输出到Output才进行Scale**，而负责Scale的部分就是这里的 `MemoryEfficientAttentionNormalize`

我们看下对应的代码：

```cpp
CUTLASS_HOST_DEVICE
  FragmentOutput operator()(int row, FragmentAccumulator const& accumulator)
      const {
    // ...
  
    ComputeFragment intermediate;
    multiplies<ComputeFragment> mul_accumulator;
  
    // 当是最后一个块计算时候，做最终的scale
    ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;

    intermediate = mul_accumulator(
        alpha, converted_accumulator); // X =  alpha * C + uniform

    return destination_converter(intermediate);
  }
```

**上述操作均在寄存器里面完成**。在这个类还重载了一个版本的 `operator()`，用于矩阵较大需要额外的一块shared memory buffer进行处理，这部分笔者还没细看过，暂时先跳过（也欢迎了解这部分的读者帮忙补充交流）。

至此cutlass fused multihead attention的重要组件都介绍完了，当然还有很多头文件没有介绍，笔者没有细看，只是对重要的组件简单分析了下。

下面我们过一下整个Kernel的流程：

## AttentionKernel

### 相关模板参数

AttentionKernel自带的模板参数有以下几个：

- typename scalar_t_ 表示QKV的数据类型
- typename ArchTag 表示SM Arch版本
- bool isAligned_ 表示QKV指针是否Align，Align版本的速度会更快一点
- int kQueriesPerBlock，int kKeysPerBlock 表示每个Block处理的query_seq_len, key_seq_len大小
- bool kSingleValueIteration 布尔值-> `value.shape[-1] <= kKeysPerBlock`

其他的一些重要的模板参数我也列一下：

- bool kPreloadV 仅当SM Arch>=80并且是half数据类型才可以使用。**因为异步GlobalMemory拷贝到Shared Memory需要Arch>=80才支持**，如果为True，则会异步拷贝Value矩阵
- bool kKeepOutputInRF = kSingleValueIteration; SingleValueIteration我实在不知道是什么意思，但是KeepOutputInRF全称应该是**Keep Output In Register File**，就是**是否将输出矩阵保存在寄存器里，当然只有矩阵比较小的情况才可以这么做**

### 参数准备

TiledSoftmax需要维护的状态变量 `s_prime`和 `m_prime`都存放在shared memory，最开始我们需要初始化一下：

```cpp
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& si = shared_storage.after_mm0.si;
    auto& mi = shared_storage.mi;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
    }
```

并且初始化一块fragment寄存器空间，存储最终输出：

```cpp
typename MM1::Mma::FragmentC accum_o;
accum_o.clear();
```

### 矩阵乘

> 代码会省略一些不重要的部分

![](https://files.mdnice.com/user/4601/c4b01ca8-3ca1-49a7-ad2e-f0b0a2a33868.png)

对于每个Block，其计算是：

```
Q(kQueriesBlock, head_size) matmul K(head_size, seq_len)
Attention(kQueriesBlock, seq_len) matmul Value(seq_len, head_size)
```

画成图即：
![](https://files.mdnice.com/user/4601/7e4cd06c-ad90-4c0b-bbbb-c907a458617f.png)

下面for循环是迭代Key矩阵的seq_len，每次处理kKeysPerBlock大小：

```cpp
for (int32_t iter_key_start = 0; iter_key_start < p.num_keys;
         iter_key_start += kKeysPerBlock)
```

下面我们要创建QKV矩阵的Iterator(cutlass里的迭代器)，并且要移动指针到合适的位置：

```cpp
// 创建一个预取Value矩阵的lambda function
    auto prologueV = [&](int blockN) {
        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
            // 移动value_ptr，至于为什么移这么多，后续有个图一目了然
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        MM1::Mma::prologue(
            shared_storage.after_mm0.mm1.mm,
            iterator_V,
            thread_id(),
            problem_size_1_k);
      };

      // 创建Query的Iterator
      typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(
              typename MM0::MmaCore::LayoutA(p.q_strideM)),
          // 在advance_to_block里已经移动好了，因此不需要变
          p.query_ptr,
          {problem_size_0_m, problem_size_0_k},
          thread_id(),
          // tb_offset表示threadblock offset，这里其实都是0，不用管
          tb_offset_A);
        
      // 创建Key的Iterator
      typename MM0::IteratorB iterator_B(
          typename MM0::IteratorB::Params(
              typename MM0::MmaCore::LayoutB(p.k_strideM)),
          // 需要根据循环来调整Key矩阵的指针位置
          p.key_ptr + iter_key_start * p.k_strideM,
          {problem_size_0_k, problem_size_0_n},
          thread_id(),
          tb_offset_B);

```

构建好Iterator后我们要开始执行 Q matmul K的部分了：

```cpp
      // Construct thread-scoped matrix multiply
      typename MM0::Mma mma(
          shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

      typename MM0::Mma::FragmentC accum;
      // 清空输出
      accum.clear();
    
      ...
    
      // 执行矩阵乘计算
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();
    
      // 触发Value矩阵的异步拷贝
      if (kPreloadV) {
        prologueV(0);
      }
```

接下来是 causal取下三角矩阵元素，把上三角元素设置为-inf

```cpp
    if (p.causal && p.num_keys - iter_key_start <= kKeysPerBlock) {
        auto query_start = blockIdx.x * kQueriesPerBlock;
        auto lane_offset = MM0::ScalingCoefsUpdater::get_lane_offset(
            lane_id(), warp_id(), iteratorC_tile_offset);
        int32_t last_col;
        // accum_m 和 accum_n 表示寄存器里的行，列
        MM0::ScalingCoefsUpdater::iterateRows(
            lane_offset,
            [&](int accum_m) {
              // 计算当前行需要保留元素的index
              last_col = query_start + accum_m - iter_key_start;
            },
            [&](int accum_m, int accum_n, int idx) {
              // 如果当前列超过index，则设为-inf
              if (accum_n > last_col) {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }
```

然后是TileSoftmax更新Scale部分：

```cpp
DISPATCH_BOOL(iter_key_start == 0, kIsFirst, ([&] {
      DISPATCH_BOOL(
          p.num_keys - iter_key_start >= kKeysPerBlock,
          kFullColumns,
          ([&] {
            MM0::ScalingCoefsUpdater::update<
                kQueriesPerBlock,
                kFullColumns,
                kIsFirst,
                kKeepOutputInRF>(
                ...);
          }));
    }));
```

更新完Scale，来到了Attn matmul Value的矩阵乘部分，也是要**先创建Value对应的Iterator，并让前面的异步读取完成**：

```cpp
typename MM1::Mma::IteratorB iterator_V(
    typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
    p.value_ptr + iter_key_start * p.v_strideM,
    {problem_size_1_k, problem_size_1_n},
    thread_id(),
    cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
          
          
typename MM1::Mma mma_pv(...);
mma_pv.set_prologue_done(kPreloadV);
if (!kKeepOutputInRF) {
  accum_o.clear();
}

// 执行矩阵乘
mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o)
```

> 如果Value矩阵较大，则需要进入一个循环完成，这里暂时先不考虑这部分逻辑

执行完矩阵乘后，做最后的一个Normalize，对应前面我们介绍的 `MemoryEfficientAttentionNormalize`：

```cpp
auto dest_iter = createOutputIter(0);
EpilogueOutputOp rescale(s_prime, m_prime);
Epilogue epilogue(
    shared_storage.epilogue_shared_storage(),
    thread_id(),
    warp_id(),
    lane_id());
epilogue(rescale, dest_iter, accum_o);
```

Kernel整体流程介绍就到这里，最后画出分步解析图如下：

![](https://files.mdnice.com/user/4601/63fe6bb3-3e33-47bb-8fbd-d35cddbf1b68.png)

### kQueriesBlock和kKeysBlock设定逻辑

根据作者注释这两个值设定逻辑如下：

1. 如果 head_size <= 64, 那么每个block处理64个query和64个key，并且部分和结果存放在寄存器
2. 如果 head_size > 64，那么每个block处理32个query和128个key，部分和结果存放在shared_memory

## 后记

最近Fused Multihead Attention这个example又要进行一次大更新，比如格式从BHMD改成BMHD，还支持attention_bias云云，最近笔者在改支持broadcast mask也踩了一些坑，后面有机会继续分享。

## 其他：

### tile_smem_loader.h

GmemTileIterator：

```cpp
using GmemTileIterator =
      cutlass::transform::threadblock::PredicatedTileIterator<
          ThreadblockTileShape, // Shape
          scalar_t, // Element
          cutlass::layout::RowMajor, // Layout
          0, // AdvanceRank
          ThreadMap>; // ThreadMap
```

```
Templates implementing loading of tiles from pitch-linear rank=2 tensors. 

    This iterator uses masks to guard out-of-bounds accesses. The first tile this
    iterator visits maybe partial, then the remaining tiles are complete. So, we 
    only need to compute the predicates twice, once before the first tile and 
    once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
```

Predicate的意思是谓词，我理解他这里表示的就是用mask标记哪些访问是越界的，

对应的构造函数有：

```cpp
CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      /// Gather indices
      int const *indices = nullptr)
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset, indices) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}
```

smem_tile_iterator代码为：

```cpp
using SmemTileIterator = cutlass::transform::threadblock::RegularTileIterator<
      ThreadblockTileShape, // Shape
      scalar_t, // Element
      cutlass::layout::RowMajor, // Layout
      0, // AdvanceRank
      ThreadMap>; // ThreadMap
```

而RegularTileIterator的注释写的是它是用于加载pitch-linear memory的模板，而对于pitch-linear memory的解释如下：：

```
Pitched linear memory is just a linear memory 

allocation calculated from the 2D sizes you provide, 

with padding added as required to ensure 

row major access will be correctly aligned 

for coalesced memory access.
```

下面是一个示意图，pitch表示的是column + padding部分
![](https://files.mdnice.com/user/4601/397031f9-2ff6-4873-a070-97978b4f38cd.png)

简单来说他就是用来加载一个 row * column 的矩阵，并且指定leading dimension，不够的地方用padding代替

然后TileSmemLoader的一个作用就是从**global memory load一个tile到shared memory**

```cpp
/// load a tile from global memory into shared memory
  CUTLASS_DEVICE
  static void load(
      GmemTileIterator tile_load_iter,
      SmemTileIterator tile_store_iter) {
    Fragment tb_frag;
    tb_frag.clear();
    tile_load_iter.load(tb_frag);
    tile_store_iter.store(tb_frag);

    __syncthreads();
  }
```
