import vllm

# 符合LLaMA-2 Chat格式的三个提示
prompts = [
    '[INST] How are you? [/INST]',
    '[INST] 1 + 1 = ? [/INST]',
    '[INST] Can you tell me a joke? [/INST]',
]

# 初始化vLLM的模型
model = vllm.LLM(model='meta-llama/Llama-2-7b-chat-hf')

# 设置vLLM的解码参数
sampling_params = vllm.SamplingParams(
    temperature=0, # 温度设置为0表示贪心搜索
    max_tokens=2048, # 新生成token数上限
    presence_penalty=0, # 存在惩罚系数
    frequency_penalty=0, # 频率惩罚系数
)

# 调用vLLM的模型进行生成
out = model.generate(prompts, sampling_params=sampling_params)
for prompt, it in zip(prompts, out):
    print(f'input = {prompt!r}\noutput = {it.outputs[0].text!r}')

# 样例1
# input = '[INST] How are you? [/INST]'
# output = ' I\'m just an AI, I don\'t have feelings or emotions like humans do, so I don\'t have a physical state of being such as "good" or "bad." I\'m here to help answer your questions and provide information to the best of my ability, so please feel free to ask me anything!'

# 样例2
# input = '[INST] 1 + 1 = ? [/INST]'
# output = ' The answer to 1 + 1 is 2.'

# 样例3
# input = '[INST] Can you tell me a joke? [/INST]'
# output = " Of course! Here's a classic one:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!\n\nI hope that made you smile! Do you want to hear another one?"