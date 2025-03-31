from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.hf_argparser import HfArg
from trl import DPOTrainer


@dataclass
class Arguments(TrainingArguments):
    # 模型结构
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `yulan-team/YuLan-Chat-12B-v3`",
    )
    # DPO 训练数据集
    data_path: str = HfArg(
        default=None,
        help="The path of preference dataset, e.g., `Anthropic/hh-rlhf`",
    )
    # 上下文窗口大小
    model_max_length: int = HfArg(default=512, help="Maximum sequence length.")
    # 使用 BF16 混合精度训练
    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit.",
    )
    # DPO 中使用的超参数 beta
    beta: float = HfArg(
        default=0.1,
        help="The beta factor in DPO loss."
        "Higher beta means less divergence from the initial policy.",
    )


# 加载训练数据集，并处理成相应的格式
def get_data(split, data_path):
    dataset = load_dataset(split=split, path=data_path)

    def split_prompt_and_responses_hh(sample):
        search_term = "\n\nAssistant:"
        search_term_idx = sample["chosen"].rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        prompt = sample["chosen"][:search_term_idx + len(search_term)]
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }

    return dataset.map(split_prompt_and_responses_hh)


def train():
    # 解析命令行参数
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    # 加载策略模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # 加载参考模型
    model_ref = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # 加载模型
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=True,
    )
    # 准备训练数据
    train_dataset = get_data("train", args.data_path)
    # 初始化训练器并开始训练
    kwargs = dict(
        model=model,
        ref_model=model_ref,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    dpo_trainer = DPOTrainer(**kwargs)
    dpo_trainer.train()
    dpo_trainer.save_state()


if __name__ == "__main__":
    train()