import torch
import numpy as np

def quantize_func(x, scales, zero_point, n_bits=8):
    x_q = (x.div(scales) + zero_point).round()
    x_q_clipped = torch.clamp(x_q, min=alpha_q, max=beta_q)
    return x_q_clipped

def dequantize_func(x_q, scales, zero_point):
    x_q = x_q.to(torch.int32)
    x = scales * (x_q - zero_point)
    x = x.to(torch.float32)
    return x
if __name__ == "__main__":
    # 输入配置
    random_seed = 0
    np.random.seed(random_seed)
    m = 2
    p = 3
    alpha = -100.0 # 输入最小值为-100
    beta = 80.0 # 输入的最大值为80
    X = np.random.uniform(low=alpha, high=beta,
                              size=(m, p)).astype(np.float32)
    float_x = torch.from_numpy(X)
    # 量化参数配置
    num_bits = 8
    alpha_q = -2**(num_bits - 1)
    beta_q = 2**(num_bits - 1) - 1
    # 计算scales和zero_point
    S = (beta - alpha) / (beta_q - alpha_q)
    Z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))
    # 量化过程
    x_q_clip = quantize_func(float_x, S, Z)
    print(f"输入：\n{float_x}\n")
    # tensor([[ -1.2136,  28.7341,   8.4974],
    #        [ -1.9210, -23.7421,  16.2609]])
    print(f"{num_bits}比特量化后：\n{x_q_clip}")
    # tensor([[ 11.,  54.,  25.],
    #        [ 10., -21.,  36.]])
    x_re = dequantize_func(x_q_clip,S,Z)
    print(f"反量化后：\n{x_re}")
    # tensor([[ -1.4118,  28.9412,   8.4706],
    #         [ -2.1176, -24.0000,  16.2353]])