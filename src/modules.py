import torch
import torch.nn as nn
from attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度 (512)
            d_ff: FFN 中间层维度 (2048)
            dropout: dropout 比例
        """
        super().__init__()
        
        # TODO 1: 定义第一个线性层 (d_model → d_ff)
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # TODO 2: 定义第二个线性层 (d_ff → d_model)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # TODO 3: 定义 dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # TODO 4: x → linear1 → ReLU → dropout → linear2 → dropout
        # 提示: 使用 torch.relu() 或 F.relu()
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """
    一层 Encoder
    
    包含:
    1. Multi-Head Self-Attention + Add & Norm
    2. Feed-Forward Network + Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN 中间层维度
            dropout: dropout 比例
        """
        super().__init__()
        
        # TODO 5: 定义 Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # TODO 6: 定义 Feed-Forward Network
        self.ffn = PositionwiseFeedForward(d_model, num_heads)
        
        # TODO 7: 定义两个 LayerNorm (一个给 attention，一个给 ffn)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # TODO 8: 定义两个 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)
        Returns:
            (batch, seq_len, d_model)
        """
        # TODO 9: Multi-Head Attention + Add & Norm
        # 残差连接: x = x + dropout(attention(x))
        # LayerNorm: x = norm(x)
        
        # Step 1: Self-Attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        
        # Step 2: Add (residual) & Dropout
        x = x + self.dropout1(attn_output)
        
        # Step 3: Norm
        x = self.norm1(x)
        
        # TODO 10: Feed-Forward + Add & Norm
        # Step 1: FFN
        ffn_output = self.ffn(x)
        
        # Step 2: Add (residual) & Dropout
        x = x + self.dropout2(ffn_output)
        
        # Step 3: Norm
        x = self.norm2(x)
        
        return x


class Encoder(nn.Module):
    """
    完整的 Encoder (堆叠多层 EncoderLayer)
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers: Encoder 层数 (比如 6)
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN 中间层维度
            dropout: dropout 比例
        """
        super().__init__()
        
        # TODO 11: 创建 num_layers 个 EncoderLayer
        # 提示: 使用 nn.ModuleList
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)
        Returns:
            (batch, seq_len, d_model)
        """
        # TODO 12: 依次通过每一层
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ===== 测试代码 =====
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Day3 测试")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    # 创建随机输入
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试 FFN
    print("\n✅ PositionwiseFeedForward 测试")
    ffn = PositionwiseFeedForward(d_model, d_ff)
    ffn_output = ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"FFN output shape: {ffn_output.shape}")
    
    # 测试 EncoderLayer
    print("\n✅ EncoderLayer 测试")
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    layer_output = encoder_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"EncoderLayer output shape: {layer_output.shape}")
    
    # 测试完整 Encoder
    print("\n✅ Encoder 测试")
    encoder = Encoder(num_layers, d_model, num_heads, d_ff)
    encoder_output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Number of layers: {num_layers}")
    
    print("=" * 60)