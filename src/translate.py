import torch
from transformer import Transformer
from attention import generate_subsequent_mask

def translate(sentence, model, src_vocab, tgt_vocab, device, max_len=50):
    """翻译一个句子"""
    model.eval()
    
    # 编码输入
    src_tokens = src_vocab.encode(sentence.lower(), max_len)
    src = torch.tensor([src_tokens]).to(device)
    
    # 生成位置编码
    src_pos = torch.arange(0, src.size(1)).unsqueeze(0).to(device)
    
    # Encoder
    with torch.no_grad():
        src_embedded = model.dropout(
            model.src_embedding(src) + model.src_pos_embedding(src_pos)
        )
        encoder_output = model.encoder(src_embedded, None)
    
    # Decoder (逐词生成)
    tgt_tokens = [1]  # <sos>
    
    for _ in range(max_len):
        tgt = torch.tensor([tgt_tokens]).to(device)
        tgt_pos = torch.arange(0, tgt.size(1)).unsqueeze(0).to(device)
        tgt_mask = generate_subsequent_mask(tgt.size(1)).to(device)
        
        with torch.no_grad():
            tgt_embedded = model.dropout(
                model.tgt_embedding(tgt) + model.tgt_pos_embedding(tgt_pos)
            )
            decoder_output = model.decoder(tgt_embedded, encoder_output, None, tgt_mask)
            output = model.fc_out(decoder_output[:, -1, :])
            next_token = output.argmax(dim=-1).item()
        
        tgt_tokens.append(next_token)
        
        if next_token == 2:  # <eos>
            break
    
    return tgt_vocab.decode(tgt_tokens)

# 测试
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("📦 加载模型...")
    checkpoint = torch.load('best_model.pt', map_location=device, weights_only=False)

    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        dropout=0.1,
        max_seq_len=100
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ 模型加载完成！\n")
    
    # 测试
    test_sentences = [
        "hello",
        "i love you",
        "thank you",
        "good morning",
        "how are you"
    ]
    
    print("🌐 翻译测试：")
    print("=" * 60)
    for sent in test_sentences:
        translation = translate(sent, model, src_vocab, tgt_vocab, device)
        print(f"English:  {sent}")
        print(f"Japanese: {translation}")
        print("-" * 60)