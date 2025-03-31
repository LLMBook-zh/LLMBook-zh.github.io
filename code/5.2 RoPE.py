def rotate_half(x):
    # 将向量每两个元素视为一个子空间
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # 获得各个子空间旋转的正余弦值
    cos = cos[position_ids].unsqueeze(1)  
    sin = sin[position_ids].unsqueeze(1)
    # 将每个子空间按照特定角度进行旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed