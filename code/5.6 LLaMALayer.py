class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx) # 注意力层
        self.mlp = LlamaMLP(config) #前馈网络层
        
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力层和前馈网络层前的RMSNorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # 注意力层前使用RMSNorm进行归一化
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        ) 
        # 进行注意力模块的计算
        hidden_states = residual + hidden_states
        # 残差连接

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 前馈网络层前使用RMSNorm进行归一化
        hidden_states = self.mlp(hidden_states)
        # 进行前馈网络层的计算
        hidden_states = residual + hidden_states
        # 残差连接
        outputs = (hidden_states,)
        return outputs