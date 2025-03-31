import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaForCausalLM,

class LlamaRewardModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # 初始化线性变换层，将隐含状态映射为标量，用于输出最终奖励
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)

    def _forward_rmloss(self, input_ids, attention_mask, **kargs):
        # input_ids：输入词元的标号序列。
        # attention_mask：与输入相对应的注意力掩码

        # 将输入词元通过大语言模型进行编码，转化为隐含状态
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            return_dict=True,
            use_cache=False
        )
        # 使用线性变换层，将隐含状态映射为标量
        logits = self.reward_head(output.last_hidden_state).squeeze(-1)
        return logits
    
    def _forward_lmloss(self, prompt_ids, lm_attn_mask, response_ids):
        # prompt_ids：输入词元和输出词元拼接后的标号序列
        # lm_attn_mask：对应的注意力掩码
        # response_ids：计算交叉熵损失时目标的标号序列
        
        # 将输入词元通过大语言模型进行编码，转化为隐含状态
        outputs = self.model.forward(
            input_ids=prompt_ids,
            attention_mask=lm_attn_mask,
            return_dict=True,
            use_cache=False,
        )
        # 使用交叉熵计算模仿学习的损失，作为最终损失函数中的正则项
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, self.config.vocab_size)
        response_ids = response_ids.view(-1)
        loss = loss_fct(logits, response_ids)
        return loss
        
    def forward(self, sent1_idx, attention_mask_1, sent2_idx, attention_mask_2, labels, prompt_ids, lm_attn_mask, response_ids, **kargs):
        # sent1_idx：输入词元和正例输出词元拼接后的标号序列。
        # attention_mask_1：sent1_idx对应的注意力掩码。
        # sent2_idx：输入词元和负例输出词元拼接后的标号序列。
        # attention_mask_2：sent2_idx对应的注意力掩码。
        # labels：正例输出所在的序列（均为0，表示正例在sent1_idx中）。
        # prompt_ids：输入词元和正例输出词元拼接后的标号序列。
        # lm_attn_mask：prompt_ids对应的注意力掩码。
        # response_ids：计算交叉熵损失时目标的标号序列。

        # 计算正例输出的奖励值
        reward0 = self._forward_rmloss(
            input_ids = sent1_idx,
            attention_mask = attention_mask_1
        )
        # 计算负例输出的奖励值
        reward1 = self._forward_rmloss(
            input_ids = sent2_idx,
            attention_mask = attention_mask_2
        )
        # 计算对比式训练方法的损失函数
        logits = reward0 - reward1
        rm_loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="mean")

        # 计算模仿学习的正则项的损失函数
        lm_loss = self._forward_lmloss(prompt_ids, lm_attn_mask, response_ids)

        # 计算最终损失
        loss = rm_loss + lm_loss
        return loss