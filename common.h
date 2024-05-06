#pragma once

#include <stdint.h>
#include <cuda_fp16.h>

// 定义最大长度 以及 Shared-Memory最大长度
constexpr int MAX_SEQ_LEN_SMEM_KERNEL = 8192; // 8k is the max sequence length supported by the kernel that uses shared memory
constexpr int MAX_SEQ_LEN = 128 * 1024;       // Can be arbitirarily large, but we need to allocate memory for the whole sequence

constexpr int bos_token = 128000;
constexpr int eos_token = 128001;

typedef struct {
    int dim; // transformer dimension   Transformer输出的维度
    int hidden_dim; // for ffn layers       前向层输出的维度
    int n_layers; // number of layers       Transformer块 总层数
    int n_heads; // number of query heads   多头数
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery) 支持GQA 同一组的Q共用一对KV
    int vocab_size; // vocabulary size, usually 32000 for llama2 models.    词表数
    int seq_len; // max sequence length                                     最大序列长度
    float rope_theta; // theta for the rope rotational embedding            Rope相关设置参数
} Config;       // 模型整体配置

struct QWeight {
    uint32_t* weight;
    uint32_t* zeros;
    half* scales;
};  // Attention层 中需要用到的矩阵

struct PerLayerWeight {
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)   rmsnorm 前向Weight
    QWeight wq_q;               // Wq
    QWeight wq_k;               // Wk
    QWeight wq_v;               // Wv
    QWeight wq_o;               // Wo
    QWeight wq_gate;            // Wq-Gate
    QWeight wq_up;              // Wq-up
    QWeight wq_down;            // Wq-down
};          // Attention 相关参数

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)            // Word-Embedding 层
    // classifier weights for the logits, on the last layer
    half* wcls;                                                     // 分类层
    // final rmsnorm
    half* rms_final_weight; // (dim,)                               // RMS-Final层
    // Per layer weights
    PerLayerWeight* layers;                                         // 多层Transformer层
    int num_layers;
} TransformerWeights;       //  Llama2的模型权重

// data shared between CPU and GPU (allocated in host memory)
struct SharedData {
    volatile int pos;         // current token index
    int tokens[MAX_SEQ_LEN];  // seq_len (tokens processed/generated so far) allocated in host memory so that CPU can read this
};      // 分词器存放 数据结构

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)             激活值输出结果
    half* xb; // same, but inside a residual branch (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)   前向输出的缓存结果
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; //     query (dim,)                                        当前Token计算的 query值
    half* att; // buffer for scores/attention values (n_heads, seq_len)
    half* logits; // output logits      Causal-LM 输出的概率
    // kv cache
    half* key_cache;   // (layer, seq_len, kv_dim)
    half* value_cache; // (layer, seq_len, kv_dim)  KV-Cached 缓存结果

    int* pos;  // GPU copy of the current position (just 1 element)
    SharedData* shared_data;

    float* logits_array;  // array of output logits used to compute perplexity (seq_len, vocab_size)
} RunState;     // 数据推理 输出及中间结果 存放地址

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;      // Llama 整体数据结构

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}
