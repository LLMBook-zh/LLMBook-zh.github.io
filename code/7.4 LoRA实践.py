...
# 加载PEFT模块相关接口
from peft import (
    LoraConfig,
    TaskType,
    AutoPeftModelForCausalLM,
    get_peft_model,
)
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
...


@dataclass
class Arguments(TrainingArguments):
    ...
    # LoRA相关超参数
    lora: Optional[bool] = HfArg(default=False, help="whether to train with LoRA.")

    lora_r: Optional[int] = HfArg(default=16, help='Lora attention dimension (the "rank")')

    lora_alpha: Optional[int] = HfArg(default=16, help="The alpha parameter for Lora scaling.")

    lora_dropout: Optional[float] = HfArg(default=0.05, help="The dropout probability for Lora layers.")


...


def train():
    ...
    # 加载LoRA配置并初始化LoRA模型
    if args.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
    ...
    # 将LoRA参数合并到原始模型中
    if args.lora:
        if is_deepspeed_zero3_enabled():
            unset_hf_deepspeed_config()
        subdir_list = os.listdir(args.output_dir)
        for subdir in subdir_list:
            if subdir.startswith("checkpoint"):
                print("Merging model in ", args.output_dir + "/" + subdir)
                peft_model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir + "/" + subdir)
                merged_model = peft_model.merge_and_unload()
                save_path = args.output_dir + "/" + subdir + "-merged"
                merged_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    train()