//
// Created by jaycase on 2024/5/3.
//

#ifndef LLAMA2_Q4_CUDA_INFERENCE_H
#define LLAMA2_Q4_CUDA_INFERENCE_H

#include <cstdio>
#include <vector>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size = 0;
    int* indices = nullptr;
    void* tempStorage_scan = nullptr;
    void* tempStorage_sort = nullptr;
    size_t temp_storage_bytes_scan = 0;
    size_t temp_storage_bytes_sort = 0;
    float temperature = 0;
    float topp = 0;
    unsigned long long rng_state = 0;
} Sampler;


void build_transformer(Transformer* t, char* checkpoint_path);


void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);


void generate_cuda(
        Transformer*  transformer,
        int* prompt_tokens,
        int num_prompt_tokens,
        int steps,
        Sampler* sampler,
        std::vector<uint64_t>& generate_token_vector
);
#ifdef __cplusplus
}
#endif

#endif // CUDA_INTERFACE_H
