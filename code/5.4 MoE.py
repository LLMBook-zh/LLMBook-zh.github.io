class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, 
        num_experts_per_tok: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts) # 所有专家的列表
        self.gate = gate # 路由网络
        self.num_experts_per_tok = num_experts_per_tok # 每个词元选择的专家数目

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, 
                                        self.num_experts_per_tok) 
        # 使用路由网络选择出top-k个专家
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype) 
        #计算出选择的专家的权重
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        # 将每个专家的输出加权相加作为最终的输出
        return results