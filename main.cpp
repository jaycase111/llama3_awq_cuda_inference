//
// Created by jaycase on 2024/5/3.
// 当前已经完成环境适配
//

#include <re2/re2.h>
#include "common.h"
#include "tokenize.h"
#include "cuda_inference.h"
#include "tokenizer-main/src/sw/tokenizer/tiktoken.h"

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
constexpr int group_size = 128; // hardcoded for this implementation 强定义 group_size 128

/*
    1、完成Llama3 分词器C++实现   2024/04/29
    2、补充原始模型-AWQ-Int4量化代码、使用默认数据校准、后续可根据自定义数据完成校准 2024/04/30
    3、实现weight-packer 2024/04/30
    4、实现整体运行环境调通、完成CMakeLists.txt 2024/05/03
    5、实现模型结构定义
    6、实现采样参数定义
    7、实现分词器接入 推理完整完成
    8、错误原因排查: 输出结果异常
                    a.  排除Prompt构造过程
                    b.  分词器使用问题
                    c.  重点排查模型参数
 *
 * */


void error_usage(char *argv[]) {
    fprintf(stderr, "Usage:   %s <checkpoint> [options]\n", argv[0]);
    fprintf(stderr, "Example: %s model.bin -n 256 -i \"Write a poem on GPUs\"\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <int>    max number of steps to run for, default = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -f <string> path to file containing input prompt. Can be used with for multi-line prompts.\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.5\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|perplexity, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -q <string> dataset file for computing perplexity\n");
    exit(EXIT_FAILURE);
}

void encode(Tiktoken * t, char* text, int* tokens, int* n_tokens){
    std::string prompt_str(text);
    std::vector<uint64_t> token_vector = t->encode(prompt_str);
    token_vector.insert(token_vector.begin(), bos_token);

//    std::vector<u_int64_t> test_token_vectors = {27,    91,   318,  5011,    91,    29,  9125,   198,  2675,   527,
//                                                    264, 64694, 18328,   430,  8779,  4320,  4860, 16134,    91,   318,
//                                                    6345,    91,   397,    27,    91,   318,  5011,    91,    29,   882,
//                                                    198, 14965,   374,   597, 15784, 76514,    91,   318,  6345,    91,
//                                                    397};
    *n_tokens = token_vector.size();


//    token_vector.push_back(eos_token);
    //*n_tokens = token_vector.size();
    for(int i=0;i<*n_tokens;i++){
        tokens[i] = token_vector[i];
    }

}

void generate(Transformer* transformer, Tiktoken * tokenizer, Sampler* sampler, char* prompt, int steps) {
    // prompt 未经过System-Prompt拼接
    char empty_prompt[] = "";
    if (prompt == NULL) { prompt = empty_prompt; } // 设置空Prompt

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;

    // 增加提示符号
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    printf("\nEncoding Prompt... ");   // Encoding can take a long time, print a message to show progress

    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);  // 分词
    printf("Done!\n");

    std::cout <<  num_prompt_tokens << std::endl;
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    std::vector<uint64_t> generate_token_vector;
    generate_cuda(transformer,prompt_tokens, num_prompt_tokens, steps, sampler,
                  generate_token_vector);

    std::string generate_text = tokenizer->decode(generate_token_vector);
    std::cout << generate_text << std::endl;

}



int main(int argc, char *argv[])
{
    char* checkpoint_path = NULL;                      // int4  模型保存地址
    char default_tokenizer_path[] = "tokenizer.model";        // 分词器地址
    char* tokenizer_path = default_tokenizer_path;          // 分词器地址
    int steps = 0;              // number of steps to run for   // 生成步数 后续被-n更新
    char* prompt = nullptr;     // prompt string            正向Prompt

    float temperature = 0.5f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher 温度
    float topp = 0.6f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower 最大温度和
    unsigned long long rng_seed = 0; // seed rng with time by default   随机种子

    std::string model_ = "tokenizer.model";
    sw::tokenizer::TiktokenFactory tiktoken_factory(model_, 0);

    std::unordered_map<std::string, uint64_t> speical_tokens = tiktoken_factory.get_special_tokens();
    for (const auto& pair :  speical_tokens) {
        const std::string& key = pair.first;
        uint64_t value = pair.second;
        // 在此处使用 key 和 value 进行操作
        std::cout << "key: " + key + " value: " + std::to_string(value) << std::endl;
    }

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(argv); }

    for (int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(argv); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(argv); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(argv); } // must be -x (one dash, one letter)   参数行必须为-*
        // read in the args
        switch (argv[i][1]) {
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;                  // 设置prompt
            case 'z': tokenizer_path = argv[i + 1]; break;          // 分词器地址
            case 't': temperature = atof(argv[i + 1]); break;       // 温度
            case 'p': topp = atof(argv[i + 1]); break;              // topp
            case 's': rng_seed = atoi(argv[i + 1]); break;          // seed
            default: error_usage(argv);
        }
    } // 传入参数

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.6;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;                           // 设置正确参数

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    if (steps <= 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // 分词器构建
    std::string model(tokenizer_path);
    Tiktoken tokenizer = load_tokenizer(model);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);    // 采样配置对象 相应属性分配

    generate(&transformer, &tokenizer, &sampler, prompt, steps);
}