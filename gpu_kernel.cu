#pragma once
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "common.h"


// utility function to load from memory (try different cache hints)
#define USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS 0 // TODO 不要为数据加载使用缓存 为防止编译错误 这里将0转换为1
#define USE_LDCS_FOR_WEIGHT_LOADS 0

// __device__ 只能在GPU中使用的函数、不能在CPU中使用
// __global__ 标记内核函数(kernel function)。内核函数是 GPU 并行执行的入口点,由主机(CPU)端代码调用
// __global__ 函数必须从主机端代码调用,才能在设备上执行、它们不能在设备函数中被调用,也不能在主机端执行

__forceinline__ __device__ uint4 loadFromMem(const uint4* ptr) {
    // 在内存中加载一个uint4类型的数据 从GPU内存中读取数据
    uint4 ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    // 则使用ld.global.L1::no_allocate.v4.u32内联PTX汇编指令从内存加载数据。
    // 这种方式告诉GPU不要为加载的数据分配L1缓存,可能会提高一些内存带宽密集型应用的性能
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    // __ldcs Data Cache Streaming Load L1 流式缓存
    ret = __ldcs(ptr);
#else
    ret = *ptr; // 默认读取方式、直接从GPU内存拷贝到GPU中
#endif
    return ret;
}

__forceinline__ __device__ uint32_t loadFromMem(const uint32_t* ptr) {
    // 从GPU内存中读取 int32数据
    uint32_t ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__forceinline__ __device__ half loadFromMem(const half* ptr) {
    // half是 float16 数据结果、从GPU内存中读取Half数据到GPU
    half ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    uint16_t temp;
    asm volatile("ld.global.L1::no_allocate.u16 %0, [%1];" : "=h"(temp) : "l"(ptr));
    ret = __ushort_as_half(temp);
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}




// ----------------------------------------------------------------------------
// GPU kernels
// __global__ 函数 CPU入口 GPU内部被执行 不能被核函数调用
__global__ void convert_fp16_to_fp32(float* out, half* in, int elements) {
    // elements 线程索引号
    // blockIdx.x 当前线程块在网格X维度的索引 blockDim.x 线程块在X维度上的大小
    // threadIdx.x 当前线程在所在线程块的x维度的索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}

// __restric__ 编译器优化
__global__ void copy_embedding_kernel(half* x, const half* __restrict__ table, int size, int* tokens, int* pPos)
{
    /*
     *
     * size 每个token的大小
     * 将Embedding值移动到kernel中
     * 本质上还是从GPU内存中送入到GPU Kernel中
     * */
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程索引
    if (index >= size) return;
    int pos = *pPos;                                    // 真实pos的值
    int token = tokens[pos];                            // 对应的token值
    int table_index = index + token * size;             // 二维table的索引
    x[index] = table[table_index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread) {
    /*
     * o  输出归一化向量
     * x  输入向量
     * weight   权重系数
     * size     输入向量x和weight的长度
     * elementPerThread     每个线程需要处理的元素数量
     * 作用对输入向量x 进行均方归一化、并且乘以权重向量 weight
     * */
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024; // 当前线程在向量中的索引 一共拆成1024个线程块
        if (index < size) {
            float val = (float)x[index];
            ss += val * val;    // 计算处理元素的平方和
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);             // 规约计算ss中的和

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;             // 仅在第一个线程中完成该部分操作
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;         // 分母计算 1 / (ss + alpha) ** 0.5
    }
    __syncthreads();
    ss = shared_ss;             // 所有线程从共享内存中读取数据

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];       // ResNorm计算
            o[index] = (half)val;                   // 存储在 o 中
        }
    }
}


// Only used for the final linear layer to get logits (for most other layers we use the INT4 version below)
__global__ void mat_vec_kernel(half* op, const half* ip, const half* wt, int n, int d, int numSerialLoads,
                               int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    /*
     * 矩阵向量乘法运算
     * op: 输出向量指针
     * ip: 输入向量指针
     * wt: 权重矩阵指针
     * n: 输入向量长度
     * d: 输出向量长度
     * numSerialLoads:  循环中迭代步数
     * ip_stride：       输入向量的步长
     * w_stride:         权重矩阵的步长
     * op_stride：       输出矩阵的步长
     * w_row_stride:     权重矩阵行的步长
     * alpha:            乘法的比例因子
     * */
    int index = blockIdx.x * blockDim.y + threadIdx.y;  // 线程索引
    if (index >= d)     // 当前索引大于输出向量的长度、直接输出
        return;
    const half* __restrict__ input = ip + blockIdx.y * ip_stride;   // 当前线程块需要处理输入的起始位置
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;   // 当前线程块需要处理权重矩阵的起始位置
    half* output = op + blockIdx.y * op_stride;                     // 当前线程需要处理的输出的起始位置

    float sum = 0;                                                  // 保存乘法结果之和

    for (int i = 0; i < numSerialLoads; i++) {
        // 循环 numSerialLoads 次，该循环用于逐个处理输入向量和权重矩阵的元素
        int j = (i * 32 + threadIdx.x) * 8; // 当前线程在输入向量和权重矩阵中的起始位置
        if (j < n) {
            half w[8];      //  定义长度为8的weight矩阵
            half ip[8];     //  定义长度为8的输入矩阵
            *((uint4*)(&w)) = loadFromMem((uint4*)(&weight[index * w_row_stride + j])); // 将weight的值拷贝到w中
            *((uint4*)(&ip)) = *((uint4*)(&input[j]));                                  // 将iput中所有值拷贝到ip中
            for (int el = 0; el < 8; el++)
                sum += float(w[el]) * float(ip[el]);                                    // 8个元素求和
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);                                                   // 将线程束内的sum求和 并且同步等待
    sum *= alpha;                                                                      // 乘以规约因子

    if (threadIdx.x == 0)
        output[index] = (half)sum;                                                     // 输出位置
}

// Simpler version of the above - handles non multiple of 8 dimensions too (used only by MHA block)
__global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int numSerialElements,
                                      int ip_stride, int w_stride, int w_row_stride, float alpha, int* pPos, int kv_mul) {
    /*
     * 多头注意力内部 矩阵乘法
     * op: 输出向量
     * ip: 输入向量
     * wt: 权重向量
     * n: 输入向量长度
     * numSerialLoads:  循环中迭代步数
     * ip_stride：       输入向量的步长
     * w_stride:         权重矩阵的步长
     * w_row_stride:     权重矩阵行的步长
     * alpha:            缩放因子
     * n_Pos:
     * kv_mul:           支持GQA 多头映射的缩放系数
     * */

    int op_stride = *pPos + 1;                              // 输出结果的步长
    int index = blockIdx.x * blockDim.y + threadIdx.y;      // 当前线程的索引
    if (index >= op_stride)
        return;

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;      // 线程对应输入的起始位置
    const half* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;  // 线程对应的weight起始位置
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;       // 线程束的索引
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);   // 矩阵的乘法
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);    // 线程束求和
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;          // 输出
}

// hardcoded for group-count = 128
// __forceinline__ 编译器自动内联
__forceinline__ __device__ float get_mat_vec_int4(int index, const half* __restrict__ input,
                                                  const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
                                                  int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height) {
    /*
     * 量化版本的矩阵想成 input 为 float16 weight 为 int32
     * index: 当前线程的索引
     * input: 输入数据的指针
     * q_weight: 量化权重矩阵的指针
     * q_zero:   ：量化后的零点值的指针，以`uint32_t`类型存储，用于解压缩权重矩阵。
     * - `scales`：缩放因子的指针，以`half`类型存储，用于对权重进行缩放。
       - `inputElements`：输入数据的元素个数。
       - `opElements`：输出数据的元素个数。
       - `packed_zeros_height`：压缩后的零点值矩阵的高度（行数）。
       - `scales_height`：缩放因子矩阵的高度（行数）。
     - `packed_weights_height`：压缩后的权重矩阵的高度（行数）。
     * */

    float sum = 0;
    for (int ygq = 0; ygq * 128 + threadIdx.x * 4 < packed_weights_height; ygq++) {   // each iteration of this loop covers 8 x 128 elements in y dimension of weight matrix (weight matrix is column major)
        // 外部循环，迭代矩阵的y维度。每次循环处理128个元素的一维切片。

        // 从内存中加载矩阵的量化零点值
        uint32_t packed_q_z = loadFromMem(&q_zeros[index * packed_zeros_height + ygq]);    // int32的值

        // load weights in one go (32 elements from weight matrix loaded by each thread in one read)
        // 从内存中加载矩阵的量化权重值。每个线程加载4个权重值
        uint32_t loaded_packed_wts[4];      // 读取4个int32 权重值
        *((uint4*)(&loaded_packed_wts[0])) = loadFromMem((uint4*)(&q_weight[index * packed_weights_height + ygq * 128 + threadIdx.x * 4]));

        // 计算矩阵的y维度分组值
        int group_y = ygq * 8 + (threadIdx.x / 4);


        /*
         * 这一行代码的作用是从变量packed_q_z中提取出当前线程的偏移量，用于确定要加载的32位无符号整数中的特定位。
         * 具体而言，表达式(4 * (threadIdx.x / 4))计算出当前线程在所属的4个线程组中的索引，然后乘以4，得到每个线程组的偏移量。
         * 通过右移操作符(>>)将packed_q_z向右移动以达到偏移量的目的。然后使用按位与(&)操作符与0xF进行按位与运算，提取出偏移量后4位中的值。
         * 最终，结果转换为浮点数，并赋值给变量q_z。这个值在后续计算中使用，用于计算权重
         * */
        float q_z = (float)(packed_q_z >> (4 * (threadIdx.x / 4)) & 0xF);            // 当前量化-0点值的标准值
        float scale = (float)loadFromMem(&scales[index * scales_height + group_y]);  // 取出当前量化的Scale相值
        int y_base = ygq * 1024 + threadIdx.x * 32;  // 当前线程输入的索引

        for (int qi = 0; qi < 4; qi++) {                 // each iteration of this loop covers 256 elements in y dimension of weight matrix

            // 内部循环，迭代矩阵的y维度。每次循环处理256个元素的一维切片
            int ys = y_base + qi * 8;   // 输入的Index

            if (ys < inputElements) {
                // 检查是否超过输入向量的维

                // 从加载的量化权重值中选取当前迭代所需的值
                uint32_t packed_q_w = loaded_packed_wts[qi];

                // 从内存中加载8个int4的值
                half ip[8];
                *((uint4*)(&ip)) = *((uint4*)(&input[ys]));

                for (int i = 0; i < 8; i++) { // 内层循环，迭代矩阵的y维度的每个元素

                    // 因为量化值为32位 迭代8次每次右移4位

                    float q_wt = (float)(packed_q_w & 0xF); // 计算量化权重值

                    float w = (q_wt - q_z) * scale; // 计算归一化权重
                    sum += w * float(ip[i]);            // 计算乘积并累加到结果中
                    packed_q_w = (packed_q_w >> 4);     // 将量化权重值右移4
                }
            }
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    return sum;
}


__device__ void mat_vec_int4(half* __restrict__ output, const half* __restrict__ input,
                             const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
                             int inputElements, int opElements, int packed_zeros_height,
                             int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    /*
     * // float16 和 int32 其实上是8个int4的矩阵相乘的矩阵计算
     * - `half* __restrict__ output` 表示输出数组，其中 `half` 是 16 位浮点数的类型。 `__restrict__` 关键字表示输出数组不会与其他数组重叠。
      - `const half* __restrict__ input` 表示输入矩阵，也是一个 `half` 类型的数组。
      - `const uint32_t* __restrict__ q_weight`
     `const uint32_t* __restrict__ q_zeros` 是用来量化权重和计算补零的数组。
      - `const half* __restrict__ scales` 表示一个用于缩放的数组，也是 `half` 类型的。
      - `int inputElements` 表示输入矩阵的元素数。
      - `int opElements` 表示输出数组的元素数。
      - `int packed_zeros_height` 表示补零操作的高度。
      - `int scales_height` 表示缩放数组的高度。
      - `int packed_weights_height` 表示量化权重数组的高度。
      - `bool accum` 表示是否对输出进行累加。
      - `int loff` 是一个偏移量，用于确定输出数组的位置。
      - `int* pPos` 是一个指针，指向一个当前位置的变量。
     *
     * */
    int index = blockIdx.x * blockDim.y + threadIdx.y;  // 当前线程的索引
    if (index >= opElements)
        return;


    float sum = get_mat_vec_int4(index, input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);
    // 计算当前量化矩阵相乘的输出值
    if (threadIdx.x == 0) {
        if (loff != -1) {
            output += loff + (*pPos * opElements);  // 计算 output的偏移量
        }

        if (accum)
            sum += (float)output[index];            // 求和
        output[index] = (half)sum;                  // 输出值
    }
}

__global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input,
                                    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
                                    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    // float16 和 int32 其实上是8个int4的矩阵相乘的矩阵计算
    mat_vec_int4(output, input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, accum, loff, pPos);
}

__global__ void qkv_matvec_kernel(half* __restrict__ q, half* __restrict__ key_cache, half* __restrict__ value_cache,
                                  const half* __restrict__ input,
                                  const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ q_scales,
                                  const uint32_t* __restrict__ k_weight, const uint32_t* __restrict__ k_zeros, const half* __restrict__ k_scales,
                                  const uint32_t* __restrict__ v_weight, const uint32_t* __restrict__ v_zeros, const half* __restrict__ v_scales,
                                  int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, int loff, int* pPos)
{
    /*  q: 输出矩阵
     *  key_cache: key_cache 矩阵
     *  value_cache
     *  input
     * */
    // 其实 在真实计算中 blockIdx的取值只能是 [0, 1, 2]
    if (blockIdx.y == 0)
        // 计算 Input和 Wq的矩阵
        mat_vec_int4(q, input, q_weight, q_zeros, q_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, -1, nullptr);
    else if (blockIdx.y == 1)
        // 计算 Input和 key_cache 的计算结果
        mat_vec_int4(key_cache, input, k_weight, k_zeros, k_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, loff, pPos);
    else // if (blockIdx.y == 2)
        // 计算 Inpu 和 value_cached 的计算结果
        mat_vec_int4(value_cache, input, v_weight, v_zeros, v_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, loff, pPos);
}

__global__ void  ffn_matvec_silu_kernel(half* __restrict__ output, const half* __restrict__ input,
                                        const uint32_t* __restrict__ g_weight, const uint32_t* __restrict__ g_zeros, const half* __restrict__ g_scales,
                                        const uint32_t* __restrict__ u_weight, const uint32_t* __restrict__ u_zeros, const half* __restrict__ u_scales,
                                        int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height) {
    // ，用于计算全连接层前向传播中的矩阵向量乘法，并应用SILU（Sigmoid Linear Unit）激活函数。
    int index = blockIdx.x * blockDim.y + threadIdx.y;      // 线程索引
    if (index >= opElements)
        return;

    float g_val = get_mat_vec_int4(index, input, g_weight, g_zeros, g_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);
    float u_val = get_mat_vec_int4(index, input, u_weight, u_zeros, u_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    // apply silu and write the result
    if (threadIdx.x == 0) {
        float val = g_val;      // 经过一层激活函数
        val *= 1.0f / (1.0f + expf(-val));  // val 的sigmoid 函数
        val *= u_val;                       // 在乘以 u_val
        output[index] = (half)val;
    }
}

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
__global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt,
                               int N, int* pPos, int w_stride, int op_stride, int w_row_stride, int kv_mul) {
    /*
    - `op` ：输出矩阵的指针。该矩阵包含了转置后的结果。
    - `ip` ：输入矩阵的指针。该矩阵包含了要转置的数据。
    - `wt` ：权重矩阵的指针。该矩阵包含了计算过程中需要的权重值。
    - `N` ：矩阵的大小（N x N）。
    - `pPos` ：包含一个整数指针，指向一个变量，该变量表示每个线程块处理的矩阵行数。
    - `w_stride` ：权重矩阵的列数。
    - `op_stride` ：输出矩阵的列数。
    - `w_row_stride` ：权重矩阵每一行的跨度。
    - `kv_mul` ：线程块处理的KV头数。
     * */
    int K = *pPos + 1;              // // 从pPos指向的位置读取K的值，并加1 其实位置
    const half* __restrict__ input = ip + blockIdx.y * K;           // input 位置
    const half* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;        // 权重位置
    half* output = op + blockIdx.y * op_stride;                                     // 输出位置

    int start_n = blockIdx.x * 32;                                                   // 起始列

    int i = start_n + threadIdx.y;                                                   // 当前线程的行索引

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts
    __shared__ half loaded_fragment[2][32][32 + 2];                                    // 声明共享内存

    // OOB check
    if (i >= N)                                                                        // 检查输入行
        return;

    // load the first 32x32 fragment
    int n = start_n + threadIdx.x;                                                     // 当前线程的列索引
    int k = threadIdx.y;                                                               // 当前线程束的行索引
    int offset = k * w_row_stride + n;                                                 // 当前线程的偏移量
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;
    // 将weight的值送到 loaded_fragment 中

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; ;) {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        if (start_k >= K) break;    // 超过要处理的列数
        k = start_k + threadIdx.x;  // 正在处理的列
        int buf_i = e & 1;
        // 因为当前多个线程可能要共用 threadIdx.x 和 threadIdx.y
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float)input[k] : 0.0f);
        // 相反是因为矩阵的转至操作
        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        int offset = k * w_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[i] = (half)sum;
}

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sq, half* sk_base, int num_kv_heads, int head_size,
                                    int* pPos, int loff, float rope_theta) {
    /*
    - `half* sq`：输入的数据，在这个函数中用于存储每个头部的数据
        这个指针指向一个包含多个头部的数组，每个头部的大小为`head_size`。
    - `half* sk_base`：输入的键值对数据，在这个函数中用于存储每个键值对的数据。指针指向一个包含多个键值对的数组，每个键值对的大小为`head_size`。这个数组中存储着`num_kv_heads`个头部的键值对数据。
    - `int num_kv_heads`：键值对头部的数量。
    - `int head_size`：每个头部的大小。
    - `int* pPos`：指向一个表示当前处理位置的整数指针。
    - `int loff`：在`sk_base`数组中的偏移量。
    - `float rope_theta`：Rope的参数
     * */
    int pos = *pPos;        // 获取当前位置

    int h = blockIdx.x;     // 获取当前block的索引
    half* q = sq + h * head_size;   // 获取当前线程block 输入数据处理位置
    int i = threadIdx.x;                // 线程索引
    int head_dim = (i * 2) % head_size; // 当前线程对应的多头注意力头数
    float freq = 1.0f / powf(rope_theta, head_dim / (float)head_size);
    // freq 对应计算的是
    // inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

    float val = pos * freq;
    // 对应 freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

    float fcr = cosf(val);
    float fci = sinf(val);
    // 上述操作对应
    // cos = emb.cos()
    // sin = emb.sin()

    float q0 = q[i];                            // 对应操作 x1 = x[..., : x.shape[-1] // 2]
    float q1 = q[i + head_size / 2];            // 对应操作 x2 = x[..., x.shape[-1] // 2 :]
    q[i] = q0 * fcr - q1 * fci;                     // 对应操作 x[..., : x.shape[-1] // 2] = -x2 * cos + x1 * sin
    q[i + head_size / 2] = q0 * fci + q1 * fcr;     // 对应操作 x[..., x.shape[-1] // 2 :] = x1 * cos + x2 * sin
    if (h < num_kv_heads) { // 如果当前head索引小于总共的kv头部数目
        half* sk = sk_base + loff + pos * num_kv_heads * head_size;  // 计算sk数组的起始位置
        half* k = sk + h * head_size;                                // 获取当前head在sk数组中的起始位置
        float k0 = k[i];
        float k1 = k[i + head_size / 2];
        k[i] = k0 * fcr - k1 * fci;
        k[i + head_size / 2] = k0 * fci + k1 * fcr;
    }
}

__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int* pPos) {
    // Shared-Memory最大长度 8192 对应共享变量数组
    __shared__ float att[MAX_SEQ_LEN_SMEM_KERNEL];
    int h = blockIdx.x;                 // 线程块索引
    int tid = threadIdx.x;              // 线程索引
    int step = blockDim.x;              // 线程块维度
    int size = *pPos + 1;               // size大小 整体维度

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float)arr[h * size + t];      // 将arr数据拷贝到 att 中
    __syncthreads();                            // 同步操作

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;                // 共享变量

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];                  // 计算该线程中最大值

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max()); // 规约计算最大值
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;                                   // 将最大值传播到各个线程

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);                    // 分子做归一化
        sum += att[i];                                      // 线程内分母计算
    }

    sum = BlockReduce(temp).Sum(sum);                       // sum总体求和
    if (threadIdx.x == 0)
        shared_val = sum;                                   // 将分母值共享到 shared_val 并且传播到各个线程
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half)(att[t] / sum);          // softmax 操作
}

__global__ void softmax_kernel_no_smem(half* arr, int num_heads, int* pPos) {
    int h = blockIdx.x;                     // 线程块索引
    int tid = threadIdx.x;                  // 线程索引
    int step = blockDim.x;                  // 线程维度
    int size = *pPos + 1;                   // 维度

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;      // 规约操作
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? (float)arr[h * size + tid] : 0;
    for (int i = tid + step; i < size; i += step)
    {
        float val = (float)arr[h * size + i];
        if (val > max_val)
            max_val = val;                                  // 计算当前线程上最大值
    }

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());    // 规约最大值
    if (threadIdx.x == 0)
        shared_val = max_val;                                   // 将最大值共享到shared_val 并且复制到各个线程
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float val = (float)arr[h * size + i];
        val = expf(val - max_val);
        arr[h * size + i] = (half)val;                         // 直接在原数组上做分子归一化操作
        sum += val;                                            // 计算整体值
    }

    sum = BlockReduce(temp).Sum(sum);                          // 规约计算分母
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half)(float(arr[h * size + t]) / sum);     // 除以分母
}

__global__ void argmax_kernel(half* __restrict__ x, int size, int* result, volatile int* pPos, int* pPosGpu, bool write_token) {
    // 仅返回全局最大值的索引
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp; // 规约操作
    __shared__ float shared_val;                       // 线程内贡献值

    int tid = threadIdx.x;                              // 线程索引
    int step = blockDim.x;                              // 线程块维度

    // find local max value and its position
    float max_val = tid < size ? (float)x[tid] : -INFINITY;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if ((float)x[i] > max_val) {
            max_val = x[i];                             // 保存最大值
            max_pos = i;                                // 保存最大值索引
        }
    }

    // find the global max value
    float global_max_val;
    global_max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;                       // 规约最大值 并且将最大值通过shared_val 传播到各个线程

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.
    __shared__ int global_max_pos;
    if (max_val == global_max_val) {
        global_max_pos = max_pos;                       // 规约最大值对应索引 并且传递到各个线程
    }
    __syncthreads();

    // write next token to the current token location
    if (threadIdx.x == 0) {
        int token_pos = *pPos;
        token_pos++;

        if (write_token)
            result[token_pos] = global_max_pos;   // 将全局最大位置写入到result中

        // update the token indices (unblocks the CPU)
        *pPos = token_pos;                  // 将 token_pos 位置保存在 pPos 和 pPosGpu 中
        *pPosGpu = token_pos;
    }
}

// This is used for Top-P sampling. We do the following:
// 1. Divide the logits by temperature
// 2. Compute softmax
// 3. Write the indices in an array
__global__ void softmax_logits_kernel(half* __restrict__ logits, int size, float temperature, int *indices) {
    /*
     * logits 概率输出指针
     * size 词表大小
     * temperature: 温度
     * indices 索引指针
     * 本质上是将温度的softmax操作
     * */
    int tid = threadIdx.x;              // 线程索引
    int step = blockDim.x;              // 线程块维度


    for (int t = tid; t < size; t += step)
    {
        // first just write the indices array
        indices[t] = t;  // 在每个线程中，将t的值赋给indices数组中的相应位置，用于记录元素索引

        // divide by temperature
        float val = (float)logits[t];               // 概率值
        val /= temperature;                         // 除以温度
        logits[t] = (half)val;
    }
    __syncthreads();                                // 同步

    // Compute the softmax
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? ((float)logits[tid]) : -FLT_MAX;
    for (int i = tid + step; i < size; i += step)
        if ((float)logits[i] > max_val)
            max_val = logits[i];                   // 计算线程内最大值

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;                         // 规约最大值、并且同步到每一个线程

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float v = expf(float(logits[i]) - max_val);         // 分子归一化
        logits[i] = (half)v;
        sum += v;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;                           // 规约求和值、并且同步到每一个线程

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        logits[t] = (half)(float(logits[t]) / sum);         // 分子softmax操作
}

// ----------------------------------------------------------------------------

// find the index in the array that crosses top-p threshold
__global__ void sample_top_p_kernel(half* sorted_logits_prefix_sum, int* indices, int n, float top_p_threshold, int* result, volatile int* pPos, int* pPosGpu)
{
    /*
     * - `sorted_logits_prefix_sum`: 一个包含排序后的logits的数组的指针
        - `indices`: 一个包含logits的索引的数组的指针
        - `n`: 数组的大小
        - `top_p_threshold`: 一个浮点数，表示top-p阈值
        - `result`: 一个存储结果的数组的指针
        - `pPos`: 一个指向一个整数变量的指针，用于跟踪结果数组的插入位置
        - `pPosGpu`: 一个指向一个整数变量的指针，在GPU上跟踪结果数组的插入位置
     *
     * */
    int tid = threadIdx.x;  // 线程索引
    int step = blockDim.x;  // 线程块维度

    int min_index = n - 1;  // 最小的索引、因为已经排好序了

    for (int t = tid; t < n; t += step) {
        if ((float)(sorted_logits_prefix_sum[t]) >= top_p_threshold) {
            if (t < min_index) {
                min_index = t;     // 确定概率小于 top_p_threshold 的 最小概率索引
            }
        }
    }

    // find the min across the block
    using BlockReduce = cub::BlockReduce<int, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    int min_index_global = BlockReduce(temp).Reduce(min_index, cub::Min());         // 规约最小的索引
    if (threadIdx.x == 0)
    {
        int token_pos = *pPos;
        token_pos++;
        result[token_pos] = indices[min_index_global];                             // 将 top_p_threshold 中的索引记录在 result 中

        // update the token indices
        *pPos = token_pos;                                                         // 更新索引
        *pPosGpu = token_pos;
    }
}

