import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import time
import matplotlib.pyplot as plt  # 新增

from transformer import Transformer
from data_tatoeba import get_dataloaders
from attention import generate_subsequent_mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 准备 decoder 输入和目标
        tgt_input = tgt[:, :-1]  # 去掉最后一个词
        tgt_output = tgt[:, 1:]  # 去掉第一个词 (<sos>)
        
        # 生成 mask
        tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask)
        
        # 计算 loss
        output = output.reshape(-1, output.size(-1))  # (batch*seq, vocab)
        tgt_output = tgt_output.reshape(-1)           # (batch*seq)
        
        loss = criterion(output, tgt_output)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask)
            
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(num_epochs=30, batch_size=32, num_samples=10000):
    """完整的训练流程"""

    # 记录 loss 历史
    train_losses = []  # 新增
    val_losses = []    # 新增
        
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载数据
    print("\n📦 加载数据...")
    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=batch_size, 
        num_samples=num_samples
    )
    
    # 创建模型
    print("\n🏗️  创建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,        # 减小模型以加快训练
        num_heads=8,
        num_layers=3,       # 减少层数
        d_ff=512,           # 减小 FFN
        dropout=0.1,
        max_seq_len=100
    ).to(device)
    
    print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 <pad>
    
    # 训练
    print(f"\n🚀 开始训练 {num_epochs} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 评估
        val_loss = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # 记录 loss
        train_losses.append(train_loss)  # 新增
        val_losses.append(val_loss)      # 新增
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time:       {epoch_time:.2f}s")
        print("-" * 60)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, 'best_model.pt')
            print("💾 保存最佳模型")
    
    print("\n✅ 训练完成！")
    
    # 绘制 loss 曲线
    print("\n📊 绘制 loss 曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    print("✅ loss 曲线已保存到 loss_curve.png")
    plt.show()
    
    return model, src_vocab, tgt_vocab


if __name__ == "__main__":
    model, src_vocab, tgt_vocab = train(
        num_epochs=30,
        batch_size=32,
        num_samples=10000
    )