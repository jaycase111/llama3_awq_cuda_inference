//
// Created by jaycase on 2024/5/3.
//

#ifndef LLAMA2_Q4_TOKENIZE_H
#define LLAMA2_Q4_TOKENIZE_H

#include "tokenizer-main/src/sw/tokenizer/tiktoken.h"

using TiktokenFactory = sw::tokenizer::TiktokenFactory;
using Tiktoken = sw::tokenizer::Tiktoken;

Tiktoken load_tokenizer(std::string model){
    TiktokenFactory tiktoken_factory(model, 0);
    Tiktoken tiktoken = tiktoken_factory.create_using_model(model);
    return tiktoken;
}


#endif //LLAMA2_Q4_TOKENIZE_H
