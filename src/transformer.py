import torch
import torch.nn as nn
from attention import MultiHeadAttention
from modules import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    一层 Decoder
    
    包含:
    1. Masked Self-Attention + Add & Norm
    2. Cross-Attention + Add & Norm
    3. Feed-Forward + Add & Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # TODO 1: 定义 Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # TODO 2: 定义 Cross-Attention (Encoder-Decoder Attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # TODO 3: 定义 Feed-Forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # TODO 4: 定义 3 个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # TODO 5: 定义 3 个 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder 输入 (batch, tgt_seq_len, d_model)
            encoder_output: Encoder 输出 (batch, src_seq_len, d_model)
            src_mask: Encoder 的 padding mask (batch, 1, src_seq_len)
            tgt_mask: Decoder 的 subsequent mask (tgt_seq_len, tgt_seq_len)
        Returns:
            (batch, tgt_seq_len, d_model)
        """
        # TODO 6: Masked Self-Attention + Add & Norm
        # Q, K, V 都来自 x，使用 tgt_mask
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # TODO 7: Cross-Attention + Add & Norm
        # Q 来自 x (decoder)，K 和 V 来自 encoder_output，使用 src_mask
        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(cross_output)
        x = self.norm2(x)
        
        # TODO 8: Feed-Forward + Add & Norm
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)
        
        return x


class Decoder(nn.Module):
    """
    完整的 Decoder (堆叠多层 DecoderLayer)
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # TODO 9: 创建 num_layers 个 DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch, tgt_seq_len, d_model)
            encoder_output: (batch, src_seq_len, d_model)
            src_mask: (batch, 1, src_seq_len)
            tgt_mask: (tgt_seq_len, tgt_seq_len)
        Returns:
            (batch, tgt_seq_len, d_model)
        """
        # TODO 10: 依次通过每一层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """
    完整的 Transformer = Encoder + Decoder
    """
    
    def __init__(
        self,
        src_vocab_size,      # 源语言词汇表大小
        tgt_vocab_size,      # 目标语言词汇表大小
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=5000
    ):
        super().__init__()
        
        from modules import Encoder
        
        # TODO 11: 定义 Encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # TODO 12: 定义 Decoder
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # TODO 13: 定义源语言和目标语言的 Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # TODO 14: 定义位置编码 (Positional Encoding)
        # 先用一个简单的可学习的位置编码
        self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # TODO 15: 定义最后的线性层 (d_model → tgt_vocab_size)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: 源序列 (batch, src_seq_len)
            tgt: 目标序列 (batch, tgt_seq_len)
            src_mask: 源序列 mask
            tgt_mask: 目标序列 mask
        Returns:
            (batch, tgt_seq_len, tgt_vocab_size)
        """
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        # TODO 16: 源序列的 embedding + 位置编码
        src_pos = torch.arange(0, src_seq_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        src_embedded = self.dropout(
            self.src_embedding(src) + self.src_pos_embedding(src_pos)
        )
        
        # TODO 17: 目标序列的 embedding + 位置编码
        tgt_pos = torch.arange(0, tgt_seq_len).unsqueeze(0).repeat(batch_size, 1).to(tgt.device)
        tgt_embedded = self.dropout(
            self.tgt_embedding(tgt) + self.tgt_pos_embedding(tgt_pos)
        )
        
        # TODO 18: 通过 Encoder
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # TODO 19: 通过 Decoder
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # TODO 20: 通过最后的线性层
        output = self.fc_out(decoder_output)
        
        return output


# ===== 测试代码 =====
if __name__ == "__main__":
    from attention import generate_subsequent_mask
    
    print("=" * 60)
    print("🧪 Day4 测试")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # 创建随机输入
    torch.manual_seed(42)
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 创建 mask
    tgt_mask = generate_subsequent_mask(tgt_seq_len)
    
    # 测试 DecoderLayer
    print("\n✅ DecoderLayer 测试")
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    layer_output = decoder_layer(x, encoder_output, None, tgt_mask)
    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"DecoderLayer output shape: {layer_output.shape}")
    
    # 测试完整 Decoder
    print("\n✅ Decoder 测试")
    decoder = Decoder(num_layers, d_model, num_heads, d_ff)
    decoder_output = decoder(x, encoder_output, None, tgt_mask)
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # 测试完整 Transformer
    print("\n✅ Transformer 测试")
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
    output = model(src, tgt, None, tgt_mask)
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Transformer output shape: {output.shape}")
    print(f"Expected shape: (batch={batch_size}, tgt_seq_len={tgt_seq_len}, tgt_vocab={tgt_vocab_size})")
    
    print("=" * 60)