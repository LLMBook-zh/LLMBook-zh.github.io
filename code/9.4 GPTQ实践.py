# GPTQ 实战
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
name = "yulan-team/YuLan-Chat-2-13b-fp16"

# 4bit模型量化
tokenizer = AutoTokenizer.from_pretrained(name)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", quantization_config=quantization_config)
print(f"memory usage: {torch.cuda.memory_allocated()/1000/1000/1000} GB") 