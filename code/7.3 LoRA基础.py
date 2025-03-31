# 继承 PyTorch 的线性变换类
class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, config, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # 从配置中获取LoRA的秩，这决定了低秩矩阵A和B的大小
        self.r = config.lora_r
        
        # 初始化A，将输入映射到低秩空间r
        self.A = nn.Linear(in_features, self.r, bias=False)
        # 初始化B，将低秩空间映射回原始输出空间
        self.B = nn.Linear(self.r, out_features, bias=False)
        
        # 初始化一个Dropout层，用于在输入传递给A之前进行正则化
        self.dropout = nn.Dropout(p=config.lora_dropout)
        
        # 使用标准差为0.02的正态分布初始化A的权重
        self.A.weight.data.normal_(std=0.02)
        # B的权重初始化为零
        self.B.weight.data.zero_()

    def forward(self, input):
        # 原始权重对应输出
        linear_output = F.linear(input, self.weight, self.bias)
        
        # LoRA模块对应输出
        lora_output = self.B(self.A(self.dropout(input)))
        
        # 将标准线性输出与缩放后的LoRA输出相加，得到最终输出
        return linear_output + lora_output