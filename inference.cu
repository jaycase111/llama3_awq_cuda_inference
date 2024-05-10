#include "common.h"
#include "cuda_runtime.h"
#include "cuda_inference.h"
#include "gpu_kernel.cu"

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>

constexpr int group_size = 128; // hardcoded for this implementation 强定义 group_size 128

cudaStream_t stream;

size_t getPackedWeightHeight(size_t height)
{

    // Each uint32 element in the packed weight matrix contain 8 elements from the original matrix.
    // Also we load 4 uint's (32 elements) in a single instruction for getting better memory efficiency
    // This requires us to align the "height" dimension to a multiple of 4 uint (or 32 elements)
    return divUp(height, 32) * 4;
}

void malloc_run_state(RunState* s, Config* p) {
    // 中间变量都是 int4类型 但是logits还是float16 说明在采样之前有一个精度转换操作
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));

    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, sizeof(half) * p->n_layers * p->seq_len * kv_dim);    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, sizeof(half) * p->n_layers * p->seq_len * kv_dim);
    cudaMalloc((void**)&s->pos, sizeof(int));

    cudaMallocHost((void**)&s->shared_data, sizeof(SharedData));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->pos || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->logits || !s->key_cache
        || !s->value_cache || !s->shared_data) {
        printf("malloc failed for allocaing run state!\n");
        exit(EXIT_FAILURE);
    }

}

void allocQWeight(QWeight* pWeight, size_t height, size_t width) {
    size_t packed_wt_height = getPackedWeightHeight(height);    // 计算量化权重打包维度
    size_t scales_height = divUp(height, group_size);           // 计算量化-scale维度
    size_t packed_zeros_height = divUp(scales_height, 8);    // 计算量化-zero维度

    cudaMalloc((void**)&pWeight->weight, packed_wt_height * width * sizeof(uint32_t));  // 量化权重内存分配
    cudaMalloc((void**)&pWeight->zeros, packed_zeros_height * width * sizeof(uint32_t));    // 量化scale内存分配
    cudaMalloc((void**)&pWeight->scales, scales_height * width * sizeof(half));     // 量化zero内存分配
}


void readWeight(void* op, FILE* fp, size_t bytes, void* scratch) {
    if (fread(scratch, 1, bytes, fp) != bytes) { printf("error reading weights");  exit(EXIT_FAILURE); }a
    cudaMemcpyAsync(op, scratch, bytes, cudaMemcpyHostToDevice);
}


void uploadQWeight(QWeight& weight, FILE* fp, size_t height, size_t width, void* scratch) {
    int meta_height = divUp(height, group_size);
    int packed_wt_height = getPackedWeightHeight(height);
    int packed_zeros_height = divUp(meta_height, 8);

    readWeight(weight.weight, fp, packed_wt_height * width * sizeof(uint32_t), scratch);
    readWeight(weight.zeros,  fp, packed_zeros_height * width * sizeof(uint32_t), scratch);
    readWeight(weight.scales, fp, meta_height * width * sizeof(half), scratch);
}

void malloc_weights(TransformerWeights* w, Config* p) {

    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half));
    w->layers = (PerLayerWeight*)malloc(p->n_layers * sizeof(PerLayerWeight));
    w->num_layers = p->n_layers;
    for (int l = 0; l < p->n_layers; l++)
    {

        PerLayerWeight* layer = &(w->layers[l]);
        cudaMalloc((void**)&layer->rms_att_weight,  p->dim * sizeof(half));
        cudaMalloc((void**)&layer->rms_ffn_weight,  p->dim * sizeof(half));
        allocQWeight(&layer->wq_q, p->dim, p->dim);
        allocQWeight(&layer->wq_k, p->dim, kv_dim);
        allocQWeight(&layer->wq_v, p->dim, kv_dim);
        allocQWeight(&layer->wq_o, p->dim, p->dim);
        allocQWeight(&layer->wq_gate, p->dim, p->hidden_dim);
        allocQWeight(&layer->wq_up, p->dim, p->hidden_dim);
        allocQWeight(&layer->wq_down, p->hidden_dim, p->dim);
    }

    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(half));
    cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->layers ||
        !w->rms_final_weight || !w->wcls) {
        // 确认分配操作全部成功
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}


int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f) {
    size_t scratch_size = std::max(p->vocab_size, p->hidden_dim) * p->dim; // 计算 词表 * dim 和 hidden_dim * dim 最大值
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;                    // 计算kv_dim 的维度
    scratch_size *= sizeof(half);
    void* scratchCpu = malloc(scratch_size);                               // cpu上分配一个较大的内存空间

    printf("\nLoading Weights... ");

    readWeight(w->token_embedding_table, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    readWeight(w->wcls, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    readWeight(w->rms_final_weight, f, p->dim * sizeof(half), scratchCpu);

    // upload decoder block weight for each layer
    for (int i = 0; i < p->n_layers; i++) {
        uploadQWeight(w->layers[i].wq_q, f, p->dim, p->dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_k, f, p->dim, kv_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_v, f, p->dim, kv_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_o, f, p->dim, p->dim, scratchCpu);

        uploadQWeight(w->layers[i].wq_up  , f, p->dim, p->hidden_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_gate, f, p->dim, p->hidden_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_down, f, p->hidden_dim, p->dim, scratchCpu);

        readWeight(w->layers[i].rms_att_weight, f, p->dim * sizeof(half), scratchCpu);
        readWeight(w->layers[i].rms_ffn_weight, f, p->dim * sizeof(half), scratchCpu);
    }

    printf("done!\n");
    free(scratchCpu);
    return 0;
}




void build_transformer(Transformer* t, char* checkpoint_path) {
    FILE* file = nullptr;
    file = fopen(checkpoint_path, "rb");            // 读取模型文件
    if (!file) { printf("Couldn't open file %s\n", checkpoint_path); exit(1); }

    if (fread(&t->config, sizeof(Config), 1, file) != 1) { printf("Invalid header size\n");  exit(1); }
    printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\nrope_theta: %g\n",
           t->config.dim, t->config.hidden_dim, t->config.n_heads, t->config.n_kv_heads, t->config.n_layers, t->config.seq_len, t->config.vocab_size, t->config.rope_theta);

    malloc_weights(&t->weights, &t->config);
    if (checkpoint_init_weights(&t->weights, &t->config, file)) { exit(1); }

    malloc_run_state(&t->state, &t->config);
    fclose(file);
}


// TODO 将Transformer数据结构 和 Sampler做解耦

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;

    cudaMalloc((void**) & sampler->indices, vocab_size * sizeof(int));
}

void destroy_sampler(Sampler* sampler) {
    cudaFree(sampler->indices);
    cudaFree(sampler->tempStorage_sort);
    cudaFree(sampler->tempStorage_scan);
}

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}


// sample the token given the logits and some hyperparameters
void sample(Sampler* sampler, RunState* s, bool gen_token, cudaStream_t stream) {
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);

    if (sampler->temperature == 0.0f || !gen_token) {
        // greedy argmax sampling: take the token with the highest probability
        argmax_kernel << <1, 1024, 0, stream >> > (s->logits, sampler->vocab_size, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos, gen_token);
    }
    else {
        // apply the temperature to the logits, and then perform softmax
        softmax_logits_kernel <<<1, 1024, 0, stream >>> (s->logits, sampler->vocab_size, sampler->temperature, sampler->indices);

        float threshold = 0.0f;
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            threshold = coin;
        }
        else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            if (sampler->temp_storage_bytes_sort == 0) {
                cub::DeviceRadixSort::SortPairsDescending(sampler->tempStorage_sort, sampler->temp_storage_bytes_sort, s->logits, s->logits, sampler->indices, sampler->indices,
                                                          sampler->vocab_size, 0, sizeof(half) * 8, stream);
                cudaMalloc(&sampler->tempStorage_sort, sampler->temp_storage_bytes_sort);
            }

            cub::DeviceRadixSort::SortPairsDescending(sampler->tempStorage_sort, sampler->temp_storage_bytes_sort, s->logits, s->logits, sampler->indices, sampler->indices,
                                                      sampler->vocab_size, 0, sizeof(half) * 8, stream);
            threshold = coin * sampler->topp;
        }

        // Sample from the predicted probability distribution
        if (sampler->temp_storage_bytes_scan == 0) {
            cub::DeviceScan::InclusiveSum(sampler->tempStorage_scan, sampler->temp_storage_bytes_scan, s->logits, s->logits, sampler->vocab_size, stream);
            cudaMalloc(&sampler->tempStorage_scan, sampler->temp_storage_bytes_scan);
        }
        cub::DeviceScan::InclusiveSum(sampler->tempStorage_scan, sampler->temp_storage_bytes_scan, s->logits, s->logits, sampler->vocab_size, stream);

        sample_top_p_kernel << <1, 1024, 0, stream >> > (s->logits, sampler->indices, sampler->vocab_size, threshold, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos);
    }
}




long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void res_add(half* o, half* x, int size){
    int elementsPerThread = divUp(size, 1024);
    resadd_kernel <<<1, 1024, 0, stream>>> (o, x, size, elementsPerThread);
}


void copydata(half* o, half* x, int size){
    int elementsPerThread = divUp(size, 1024);
    copy_kernel <<<1, 1024, 0, stream>>> (o, x, size, elementsPerThread);
}

void rmsnorm(half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<< 1, 1024, 0, stream>>> (o, x, weight, size, elementsPerThread);
}

void matmul(half* xout, half* x, half* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    if ((n & 7) || (d & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;
    mat_vec_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha);
}

void matmul(half* xout, half* x, QWeight &w, int inpSize, int opSize, bool accum = false, int loff = -1, int *pPos = nullptr) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    mat_vec_kernel_int4 <<<grid_dim, block_dim, 0, stream >>> (xout, x, w.weight, w.zeros, w.scales, inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, accum, loff, pPos);
}

void RoPERotation(half *q, half *k, int num_heads, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
    RoPERotation_kernel <<<num_heads, head_size / 2, 0, stream >>> (q, k, num_kv_heads, head_size, pPos, loff, rope_theta);
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half * value_cache, half *att, int num_heads, int head_size, int kv_mul, int max_seq_len, int *pPos) {
    int dim = head_size * num_heads;
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(max_seq_len, 32), num_heads);      // using max_seq_len instead of real seq_len here has measurable impact on perf (2%) :-/
    mat_vec_kernel_simple <<< grid_dim1, block_dim, 0, stream >>> (att, q, key_cache, head_size, serialElements, head_size, head_size, dim / kv_mul, 1.0 / sqrt(head_size), pPos, kv_mul);

    // 2. Run softmax kernel
    if (max_seq_len <= MAX_SEQ_LEN_SMEM_KERNEL)
        softmax_kernel <<< num_heads, 1024, 0, stream >>> (att, num_heads, pPos);
    else
        softmax_kernel_no_smem <<< num_heads, 1024, 0, stream >>> (att, num_heads, pPos);

    // 3. weighted sum of the values to get the final result
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim, 0, stream >>> (output, att, value_cache, head_size, pPos, head_size, head_size, dim / kv_mul, kv_mul);
}


void qkv_matvec(half* q, half *key_cache, half *value_cache, half* x, QWeight& qw, QWeight& kw, QWeight& vw, int inpSize, int opSize, int loff, int* pPos) {


    // inpSize 或者 opSize 如果不能被7整除
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 3);
    // 一次计算 Query * x --> q | Key * x --> Key_Cache | Value * x -> Value_Cache
    qkv_matvec_kernel <<<grid_dim, block_dim, 0, stream >>> (q, key_cache, value_cache, x,
                                                             qw.weight, qw.zeros, qw.scales,
                                                             kw.weight, kw.zeros, kw.scales,
                                                             vw.weight, vw.zeros, vw.scales,
                                                             inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, loff, pPos);
}

void ffn_matvec_silu(half* xout, half* x, QWeight& gate_w, QWeight& up_w, int inpSize, int opSize) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    ffn_matvec_silu_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, gate_w.weight, gate_w.zeros, gate_w.scales,
                                                                  up_w.weight, up_w.zeros, up_w.scales,
                                                                  inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height);
}

void run_llama_network(int *pPos, Config* p, RunState* s, TransformerWeights* w, int seq_len_bin) {

    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;


    copy_embedding_kernel <<<divUp(dim, 256), 256, 0, stream >>> (x, w->token_embedding_table, dim, s->shared_data->tokens, pPos);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        copydata(s->xb2, s->xb, dim);
        printf("after s->xb3\n");
        rmsnorm(s->xb, x, w->layers[l].rms_att_weight, dim);

        int loff = l * p->seq_len * kv_dim;

        if (dim == kv_dim) {
            qkv_matvec(s->q, s->key_cache, s->value_cache, s->xb, w->layers[l].wq_q, w->layers[l].wq_k, w->layers[l].wq_v, dim, dim, loff, pPos);
        }
        else {
            matmul(s->q, s->xb, w->layers[l].wq_q, dim, dim);
            matmul(s->key_cache, s->xb, w->layers[l].wq_k, dim, kv_dim, false, loff, pPos);
            matmul(s->value_cache, s->xb, w->layers[l].wq_v, dim, kv_dim, false, loff, pPos);
        }

        RoPERotation(s->q, s->key_cache, p->n_heads, p->n_kv_heads, head_size, pPos, loff, p->rope_theta);

        MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, kv_mul, seq_len_bin, pPos);
        matmul(s->x, s->xb, w->layers[l].wq_o, dim, dim, true);

        copydata(s->xb2, x, dim);
        rmsnorm(s->xb, x, w->layers[l].rms_ffn_weight, dim);
        ffn_matvec_silu(s->hb, s->xb, w->layers[l].wq_gate, w->layers[l].wq_up, dim, hidden_dim);

        matmul(s->x, s->hb, w->layers[l].wq_down, hidden_dim, dim, true);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

void run_transformer(bool gen_token, Config* p, RunState* s, TransformerWeights* w,
                     bool copyLogits, Sampler *pSampler) {

#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
#endif

    int seq_len = s->shared_data->pos + 1;
#if USE_CUDA_GRAPHS
    int graphIndex;
    int seq_len_bin = 128;
    for (graphIndex = 0; graphIndex < MAX_GRAPHS - 1; seq_len_bin *= 2, graphIndex++)
        // 遍历找到合适的 graphIndex | 分配一倍的空间
        if (seq_len <= seq_len_bin) break;
    if ((seq_len > seq_len_bin) || (graphIndex == MAX_GRAPHS - 1)) seq_len_bin = p->seq_len;    // last bin holds max seq len

    if (!graphCaptured[graphIndex])
    {
        cudaGraph_t graph = {};
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        run_llama_network(s->pos, p, s, w, seq_len_bin);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&cudaGraphInstance[graphIndex], graph, 0);
        cudaGraphDestroy(graph);
        graphCaptured[graphIndex] = true;
    }
    cudaGraphLaunch(cudaGraphInstance[graphIndex], stream);
#else
    run_llama_network(s->pos, p, s, w, seq_len);
#endif

    if (copyLogits) {
        // copy to the right slot in logits_array (and convert to FP32)
        // we compute perplexity on the CPU later.
        float* pOutput = s->logits_array + p->vocab_size * s->shared_data->pos;

        convert_fp16_to_fp32 << < divUp(p->vocab_size, 128), 128, 0, stream >> > (pOutput, s->logits, p->vocab_size);
    }

    sample(pSampler, s, gen_token, stream);

#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf(" t: %g ", time);        // 打印推理耗费时间
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}


void generate_cuda(
        Transformer*  transformer,
        int* prompt_tokens,
        int num_prompt_tokens,
        int steps,
        Sampler* sampler,
        std::vector<uint64_t>& generate_token_vector

){
    long start = time_in_ms();    // used to time our code
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence

    cudaMemset(transformer->state.pos, 0, sizeof(int));
    transformer->state.shared_data->pos = 0;
    memcpy(&transformer->state.shared_data->tokens, prompt_tokens, sizeof(int) * num_prompt_tokens);

    while (pos < steps) {
        // wait for GPU work for previous iteration to complete
        // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
        cudaStreamSynchronize(stream);

        run_transformer(pos >= num_prompt_tokens - 1, &transformer->config, &transformer->state, &transformer->weights, false, sampler); // forward the transformer to get next token

        if (pos > 0) {
            next = transformer->state.shared_data->tokens[pos];  // Note: this is output token from previous iteration
            if (next >= transformer->config.vocab_size) next = 0;   // skip garbage tokens (can happen with NANs)
            generate_token_vector.push_back(next);
            std::cout << token  << std::endl;
            if (next == eos_token) break;   // break if EOS token is reached
            token = next;
        }
        pos++;
    }
    printf("\n");

    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    int timed_tokens = pos - 1;
    printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);

    free(prompt_tokens);

}

