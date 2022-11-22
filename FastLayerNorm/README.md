基于OneFlow版本的LayerNorm优化


### BaseLine
cuda11.6 a100 80G

统一使用bias_add + residual操作, datatype为half

测试数据为 64x1024, 1024

OneFlow: 244.64us, 156个寄存器

### 1. 使用int64_t的部分改用int32_t
没啥用

### 2. 使用pack4
244.64us -> 232.19us
156寄存器 -> 110


### 3. 使用pack8
244.64us -> 228.42us
156寄存器 -> 64

Pack越多，寄存器减少的越多。在以往的Kernel里往往是不成立的，因为用了Pack通过循环展开是需要消耗更多的寄存器的

但是在LayerNorm这里，寄存器的主要消耗是在**num_pack的循环展开**，Pack越大，num_pack越少，这部分的循环展开更少，用的寄存器也就相应减少。

### 4. 用device_function替换threadIdx.x
没卵用

### 5. 改成half2操作
228.42us -> 220.70us

自定义了一个 HalfPack，在Load和Store对应的操作里，尽可能用half2操作。

经过上述操作后，寄存器受限的问题一定程度又解决了一部分，因此Dispatch逻辑又可以进一步拓展，之前是限制在1024，经过测试，能最大拓展到5120。


