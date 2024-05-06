运行步骤:
    1、awq_quant.py 自定义量化模型 ｜ merge_safetensors 将已量化的safetensors 转换为 AWQ-bin文件
    2、convert_awq_to_bin 将 bin文件按照key落盘到一个个小文件到output
    3、调用上层 weight_packer 将output文件文件转换CUDA推理模型文件