import json


class SFTDataset:
    IGNORE_INDEX = -100
    # 定义指令模板格式
    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
    format_template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            "{input}" + response_template
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            response_template
        ),
    }

    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        self.input_ids, self.labels = self.process(self.tokenizer)

    # 数据集长度
    def __len__(self):
        return len(self.input_ids)

    # 获取第 i 条数据
    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    # 对输入和输出进行分词并标记输出位置
    def encode_src_tgt(self, s, t, tokenizer):
        source_id = tokenizer.encode(s, max_length=tokenizer.model_max_length, truncation=True)
        tokenizer.add_eos_token = True
        input_id = tokenizer.encode(s + t, max_length=tokenizer.model_max_length, truncation=True,
                                    return_tensors='pt')[0]
        tokenizer.add_eos_token = False
        label = input_id.clone()
        label[:len(source_id)] = self.IGNORE_INDEX
        return input_id, label

    # 调用数据集加载、分词、批次化
    def process(self, tokenizer):
        input_ids = []
        labels = []
        list_data_dict = json.load(open(self.args.dataset))

        for example in list_data_dict:
            example['response'] = example.pop('output')
            s = self.format_template["prompt_input"].format_map(example) if 'input' in example.keys(
            ) else self.format_template["prompt_no_input"].format_map(example)
            t = example['response'].strip()
            input_id, label = self.encode_src_tgt(s, t, tokenizer)
            input_ids.append(input_id)
            labels.append(label)
        return input_ids, labels