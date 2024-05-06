from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModel


class AwqQuant:

    def __init__(self,
                 model_path: str):
        self.model_path = model_path

    def quant(self, save_path: str):
        model = AutoAWQForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(save_path, safetensors=False, shard_size="10GB")
        tokenizer.save_pretrained(save_path)



if __name__ == '__main__':
    """
        使用默认数据集对Model进行 AWQ-Int4量化
        量化对齐数据使用默认数据生成效果较差
        建议使用: https://huggingface.co/bartowski/OpenBioLLM-Llama3-8B-AWQ 已经完成对齐量化模型
        
    """
    model_path = "/home/ubuntu/llama_3_download"
    save_path = "./awq_model"

    quantizer = AwqQuant(model_path)
    quantizer.quant(save_path)
