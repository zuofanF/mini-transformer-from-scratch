import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import urllib.request
import zipfile
import os

class Vocabulary:
    """构建词汇表"""
    
    def __init__(self, is_japanese=False):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.word_count = Counter()
        self.is_japanese = is_japanese  # 新增：标记是否是日语
        
    def add_sentence(self, sentence):
        """添加句子到词汇表"""
        if self.is_japanese:
            # 日语：字符级别
            for char in sentence:
                if char.strip():  # 跳过空格
                    self.word_count[char] += 1
        else:
            # 英语：词级别
            for word in sentence.split():
                self.word_count[word] += 1
            
    def build_vocab(self, min_count=1):  # 改成 min_count=1
        """构建词汇表"""
        idx = 4
        for word, count in self.word_count.items():
            if count >= min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence, max_len=None):
        """将句子转换为 ID 序列"""
        if self.is_japanese:
            # 日语：字符级别
            tokens = [self.word2idx.get(char, 3) for char in sentence if char.strip()]
        else:
            # 英语：词级别
            tokens = [self.word2idx.get(word, 3) for word in sentence.split()]
        
        # 添加 <sos> 和 <eos>
        tokens = [1] + tokens + [2]
        
        # Padding
        if max_len:
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
                
        return tokens
    
    def decode(self, indices):
        """将 ID 序列转换回句子"""
        words = []
        for idx in indices:
            if idx == 2:  # <eos>
                break
            if idx not in [0, 1]:  # 跳过 <pad> 和 <sos>
                words.append(self.idx2word.get(idx, "<unk>"))
        
        # 日语不需要空格
        if self.is_japanese:
            return "".join(words)
        else:
            return " ".join(words)


class TatoebaDataset(Dataset):
    """Tatoeba 数据集"""
    
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=50):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        
        src_ids = self.src_vocab.encode(src_text, self.max_len)
        tgt_ids = self.tgt_vocab.encode(tgt_text, self.max_len)
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def download_tatoeba():
    """下载 Tatoeba EN-JP 数据集"""
    url = "https://www.manythings.org/anki/jpn-eng.zip"
    zip_path = "jpn-eng.zip"
    
    if not os.path.exists("jpn.txt"):
        print("📥 下载 Tatoeba EN-JP 数据集...")
        
        # 添加 User-Agent 头避免 406 错误
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                with open(zip_path, 'wb') as out_file:
                    out_file.write(response.read())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            os.remove(zip_path)
            print("✅ 下载完成！")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("\n💡 备用方案：请手动下载数据集")
            print("1. 访问：https://www.manythings.org/anki/jpn-eng.zip")
            print("2. 解压得到 jpn.txt")
            print("3. 把 jpn.txt 放到当前目录")
            raise
    else:
        print("✅ 数据集已存在")


def load_data(num_samples=10000):
    """加载并处理数据"""
    download_tatoeba()
    
    print(f"📖 加载前 {num_samples} 个样本...")
    
    pairs = []
    with open("jpn.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng = parts[0].lower()
                jpn = parts[1]
                pairs.append((eng, jpn))
    
    print(f"✅ 加载了 {len(pairs)} 个句子对")
    
    # 构建词汇表
    print("🔨 构建词汇表...")
    src_vocab = Vocabulary(is_japanese=False)  # 英语
    tgt_vocab = Vocabulary(is_japanese=True)   # 日语
    
    for eng, jpn in pairs:
        src_vocab.add_sentence(eng)
        tgt_vocab.add_sentence(jpn)
    
    src_vocab.build_vocab(min_count=2)
    tgt_vocab.build_vocab(min_count=1)  # 日语用 min_count=1
    
    print(f"✅ 英语词汇量: {len(src_vocab)}")
    print(f"✅ 日语词汇量: {len(tgt_vocab)}")
    
    return pairs, src_vocab, tgt_vocab


def get_dataloaders(batch_size=32, num_samples=10000):
    """创建 DataLoader"""
    pairs, src_vocab, tgt_vocab = load_data(num_samples)
    
    # 划分训练集和验证集
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    
    train_dataset = TatoebaDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TatoebaDataset(val_pairs, src_vocab, tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, src_vocab, tgt_vocab


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 测试数据加载")
    print("=" * 60)
    
    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=4, num_samples=100)
    
    # 打印一个 batch
    for src, tgt in train_loader:
        print(f"\nSource shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        
        print(f"\n示例句子:")
        print(f"English: {src_vocab.decode(src[0].tolist())}")
        print(f"Japanese: {tgt_vocab.decode(tgt[0].tolist())}")
        break
    
    print("=" * 60)
