from safetensors.torch import save_file
from safetensors.torch import load_file
import torch

"""
    将AWQ量化后的多个SafeTensors模型-融合为一个bin文件
"""

save_file = "llama3_awq_download_cache/pytorch_model.bin"
file_list = ["llama3_awq_download_cache/model-00001-of-00002.safetensors", "llama3_awq_download_cache/model-00002-of-00002.safetensors"]

save_dict = {}
for file in file_list:
    pt_state_dict = load_file(file, device="cpu")
    save_dict.update(pt_state_dict)

torch.save(save_dict, save_file)
