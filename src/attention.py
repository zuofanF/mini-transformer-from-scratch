import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, seq_len_q, d_k)
            K: (batch_size, seq_len_k, d_k)
            V: (batch_size, seq_len_v, d_v)  # 通常 d_v = d_k
            mask: (batch_size, seq_len_q, seq_len_k) 或 broadcastable shape
        
        Returns:
            output: (batch_size, seq_len_q, d_v)
            attn: (batch_size, seq_len_q, seq_len_k)  # attention weights
        """
        
        # TODO 1: 计算 d_k (key 的维度)
        d_k = Q.shape[-1]
        
        # TODO 2: 计算 scores = Q @ K^T / sqrt(d_k)
        # 提示: 使用 torch.matmul 和 .transpose()
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        
        # TODO 3: 如果有 mask，将 mask 位置的 scores 设为 -1e9
        if mask is not None:
            # 确保 mask 的维度正确
            if mask.dim() == 2:
                # (seq, seq) → (1, 1, seq, seq)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch, seq, seq) → (batch, 1, seq, seq)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # TODO 4: 对 scores 应用 softmax (在最后一个维度)
        attn = torch.softmax(scores, dim=-1)
        
        # TODO 5: 计算 output = attn @ V
        output = torch.matmul(attn, V)
        
        return output, attn





# ===== Day2 新增 =====
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W_o
    where headᵢ = Attention(Q W_Qⁱ, K W_Kⁱ, V W_Vⁱ)
    """
    
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: 模型维度（比如 512）
            num_heads: 注意力头数（比如 8）
        """
        super().__init__()
        
        # TODO 1: 检查 d_model 能否被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        # TODO 2: 计算每个头的维度 d_k
        self.d_k = d_model // num_heads
        
        # TODO 3: 定义 Q, K, V 的线性变换层
        # 提示: 输入 d_model, 输出 d_model (因为要分成 num_heads 个 d_k)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # TODO 4: 定义输出线性层 W_o
        self.W_O = nn.Linear(d_model, d_model)
        
        # 复用 Day1 的 attention
        self.attention = ScaledDotProductAttention()
        
    def split_heads(self, x):
        """
        将输入拆分成多个头
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        
        # TODO 5: reshape 成 (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # TODO 6: 转置成 (batch, num_heads, seq_len, d_k)
        # 提示: 使用 .transpose(1, 2)
        return x.transpose(1, 2)
        
    def combine_heads(self, x):
        """
        合并多个头
        
        Args:
            x: (batch, num_heads, seq_len, d_k)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        
        # TODO 7: 转置回 (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        
        # TODO 8: reshape 成 (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) 或 broadcastable
        Returns:
            output: (batch, seq_len, d_model)
            attn: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)
        
        # TODO 9: 通过线性层变换 Q, K, V
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # TODO 10: 拆分成多个头
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # TODO 11: 调整 mask 的维度以适配多头
        # 提示: mask 需要从 (batch, seq, seq) 变成 (batch, 1, seq, seq)
        #       这样可以 broadcast 到所有 heads
        # if mask is not None:
        #     mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
        
        # TODO 12: 应用 attention
        output, attn = self.attention(Q, K, V, mask)
        # output: (batch, num_heads, seq_len, d_k)
        # attn: (batch, num_heads, seq_len, seq_len)
        
        # TODO 13: 合并多个头
        output = self.combine_heads(output)  # (batch, seq_len, d_model)
        
        # TODO 14: 通过输出线性层
        output = self.W_O(output)
        
        return output, attn


# ===== Mask 生成函数 =====
def generate_padding_mask(seq, pad_idx=0):
    """
    生成 Padding Mask
    
    Args:
        seq: (batch, seq_len) - token IDs
        pad_idx: padding token 的 ID (默认 0)
    
    Returns:
        mask: (batch, 1, seq_len) - 1 表示有效位置，0 表示 padding
    
    Example:
        seq = [[1, 2, 3, 0, 0],
               [1, 2, 0, 0, 0]]
        
        mask = [[1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0]]
    """
    # TODO 15: 生成 mask (seq != pad_idx)
    # 提示: (seq != pad_idx) 会返回 True/False，需要转成 int
    mask = (seq != pad_idx).int()
    return mask.unsqueeze(1)  # (batch, 1, seq_len)


def generate_subsequent_mask(size):
    """
    生成 Subsequent (Causal) Mask - 防止看到未来信息
    
    Args:
        size: sequence length
    
    Returns:
        mask: (size, size) - 下三角矩阵
    
    Example:
        size = 4
        mask = [[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]]
    """
    # TODO 16: 生成下三角矩阵
    # 提示: 使用 torch.tril(torch.ones(size, size))
    mask = torch.tril(torch.ones(size, size))
    return mask


# ===== 测试代码 =====
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Day2 测试")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    
    # 创建随机输入
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化 Multi-Head Attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attn = mha(Q, K, V)
    
    print(f"\n✅ MultiHeadAttention 测试")
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"d_k per head: {mha.d_k}")
    
    # 测试 Padding Mask
    print(f"\n✅ Padding Mask 测试")
    seq = torch.tensor([[1, 2, 3, 0, 0],
                        [1, 2, 0, 0, 0]])
    pad_mask = generate_padding_mask(seq, pad_idx=0)
    print(f"Sequence:\n{seq}")
    print(f"Padding mask shape: {pad_mask.shape}")
    print(f"Padding mask:\n{pad_mask.squeeze(1)}")
    
    # 测试 Subsequent Mask
    print(f"\n✅ Subsequent Mask 测试")
    sub_mask = generate_subsequent_mask(4)
    print(f"Subsequent mask shape: {sub_mask.shape}")
    print(f"Subsequent mask:\n{sub_mask}")
    
    print("=" * 60)





# # ===== Day 1 测试代码 =====
# if __name__ == "__main__":
#     # 设置随机种子
#     torch.manual_seed(42)
    
#     # 创建测试数据
#     batch_size = 2
#     seq_len = 4
#     d_k = 8
    
#     Q = torch.randn(batch_size, seq_len, d_k)
#     K = torch.randn(batch_size, seq_len, d_k)
#     V = torch.randn(batch_size, seq_len, d_k)
    
#     # 初始化 attention
#     attention = ScaledDotProductAttention()
    
#     # Forward pass
#     output, attn_weights = attention(Q, K, V)
    
#     # 打印结果
#     print("=" * 50)
#     print("✅ Day1 测试结果")
#     print("=" * 50)
#     print(f"Q shape: {Q.shape}")
#     print(f"K shape: {K.shape}")
#     print(f"V shape: {V.shape}")
#     print(f"\nOutput shape: {output.shape}")
#     print(f"Attention weights shape: {attn_weights.shape}")
#     print(f"\nAttention weights sum (应该≈1.0): {attn_weights[0, 0].sum().item():.4f}")
#     print("=" * 50)