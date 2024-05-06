//
// Created by jaycase on 2024/5/3.
//

#ifndef LLAMA2_Q4_SAMPLER_H
#define LLAMA2_Q4_SAMPLER_H
#include "common.h"

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

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);

void sample(Sampler* sampler, RunState* s, bool gen_token, cudaStream_t stream);


#endif //LLAMA2_Q4_SAMPLER_H
