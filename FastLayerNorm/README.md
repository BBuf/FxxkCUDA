基于OneFlow版本的LayerNorm优化


### BaseLine
cuda11.6 a100 80G

统一使用bias_add + residual操作

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

### 4. 用device_function替换threadIdx.x
没卵用

### 5. 改成half2操作
228.42us -> 220.70us



