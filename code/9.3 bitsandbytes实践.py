# bitsandbytes 实战
from transformers import AutoModelForCausalLM
name = "yulan-team/YuLan-Chat-2-13b-fp16"

# 8bit模型量化
model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
print(f"memory usage: {torch.cuda.memory_allocated()/1000/1000/1000} GB") 


# 4bit模型量化
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_4bit=True)
print(f"memory usage: {torch.cuda.memory_allocated()/1000/1000/1000} GB") 