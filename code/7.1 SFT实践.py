import torch
from dataclasses import dataclass
from dataset.sft_dataset import SFTDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArg

IGNORE_INDEX = -100


# 用户输入超参数
@dataclass
class Arguments(TrainingArguments):
    # 模型结构
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf`",
    )
    # 训练数据集
    dataset: str = HfArg(
        default="",
        help="Setting the names of data file.",
    )
    # 上下文窗口大小
    model_max_length: int = HfArg(
        default=2048,
        help="The maximum sequence length",
    )
    # 只保存模型参数（不保存优化器状态等中间结果）
    save_only_model: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.",
    )
    # 使用BF16混合精度训练
    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit.",
    )


# 批次化数据，并构建序列到序列损失
@dataclass
class DataCollatorForSupervisedDataset():
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def train():
    # 解析命令行参数
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=False,
    )
    # 加载模型，并使用FlashAttention
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2")
    # 初始化训练器、准备训练数据并开始训练
    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=SFTDataset(args, tokenizer),
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )

    trainer = Trainer(**kwargs)
    trainer.train()
    trainer.save_model(args.output_dir + "/checkpoint-final")
    trainer.save_state()


if __name__ == "__main__":
    train()