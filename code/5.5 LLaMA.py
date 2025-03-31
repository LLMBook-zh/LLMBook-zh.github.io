class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        # LLaMA的词表大小
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # LLaMA的词嵌入矩阵，将输入的id序列转化为词向量序列
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 所有的Transformer解码器层
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )

@add_start_docstrings_to_model_forward(Llama_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,

    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # 将输入的input id序列转化为词向量序列
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        # 创建单向注意力的注意力掩盖矩阵

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )[0]
            # 用每个LLaMA解码器层对词元的隐含状态进行映射
        hidden_states = self.norm(hidden_states)
        # 对每个词元的隐含状态使用RMSNorm归一化
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )