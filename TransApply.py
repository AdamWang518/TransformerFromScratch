import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from datasets import load_dataset
import random
from torch.utils.data import DataLoader, Dataset
import spacy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def create_masks(src, trg, src_pad_idx, trg_pad_idx):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    
    trg_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_mask = trg_mask & subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    
    return src_mask, trg_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return subsequent_mask == 0

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, dtype=torch.float32)  # 设置数据类型为浮点型
        self.W_k = nn.Linear(d_model, d_model, dtype=torch.float32)  # 设置数据类型为浮点型
        self.W_v = nn.Linear(d_model, d_model, dtype=torch.float32)  # 设置数据类型为浮点型
        self.W_o = nn.Linear(d_model, d_model, dtype=torch.float32)  # 设置数据类型为浮点型
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.head_dim)
        
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(energy, dim=-1)
        x = torch.matmul(attention_weights, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.W_o(x)
        
        return x, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        attention_output, _ = self.multihead_attention(x, x, x, mask)
        x = x + self.dropout(attention_output)
        x = self.layer_norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.multihead_self_attention = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask, trg_mask):
        self_attention_output, _ = self.multihead_self_attention(x, x, x, trg_mask)
        x = x + self.dropout(self_attention_output)
        x = self.layer_norm1(x)
        
        attention_output, _ = self.multihead_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(attention_output)
        x = self.layer_norm2(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm3(x)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return x

# 加載TED Talks數據集
ted_dataset = load_dataset("ted_multi")
# for i, example in enumerate(ted_dataset["train"]):
#     print(example)
#     if i == 5:  # 只打印前5个样本以检查结构
#         break

# 提取源語言和目標語言數據
src_lang = "en"  # 英語作為源語言
trg_lang = "zh"  # 中文作為目標語言

# 使用spacy進行分詞
spacy_en = spacy.load("en_core_web_sm")
spacy_zh = spacy.load("zh_core_web_sm")

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]

class TEDTalksTranslationDataset(Dataset):
    def __init__(self, dataset, src_lang, trg_lang, max_length):
        self.examples = []
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_length = max_length
        
        for example in dataset:
            translations_dict = example["translations"]
            languages = translations_dict["language"]
            translations = translations_dict["translation"]

            if src_lang in languages and trg_lang in languages:
                src_index = languages.index(src_lang)
                trg_index = languages.index(trg_lang)

                src_text = translations[src_index]
                trg_text = translations[trg_index]

                if len(tokenize_en(src_text)) <= self.max_length and len(tokenize_zh(trg_text)) <= self.max_length:
                    self.examples.append((src_text, trg_text))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        src_text, trg_text = self.examples[index]
        # print(f"src_text (type {type(src_text)}): {src_text}")
        # print(f"trg_text (type {type(trg_text)}): {trg_text}")
        return src_text, trg_text


# 創建自定義數據集實例
max_seq_length = 50  # 設定最大序列長度
# 在创建自定义数据集实例时，确保将原始文本传递给数据集
train_dataset = TEDTalksTranslationDataset(ted_dataset["train"], src_lang, trg_lang, max_seq_length)

# 确保 train_dataset 包含原始文本而不是索引
train_dataset = [(src_text, trg_text) for src_text, trg_text in train_dataset]


# 創建源語言和目標語言詞彙表
src_vocab = set()
trg_vocab = set()

for src_text, trg_text in train_dataset:
    src_tokens = tokenize_en(src_text)
    trg_tokens = tokenize_zh(trg_text)
    src_vocab.update(src_tokens)
    trg_vocab.update(trg_tokens)

src_vocab = list(src_vocab)
trg_vocab = list(trg_vocab)
src_vocab.append("<pad>")
trg_vocab.append("<pad>")
src_vocab.append("<sos>")
trg_vocab.append("<sos>")
src_vocab.append("<eos>")
trg_vocab.append("<eos>")
src_vocab.append("<unk>")
trg_vocab.append("<unk>")

src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)

src_vocab_stoi = {word: index for index, word in enumerate(src_vocab)}
src_vocab_itos = {index: word for index, word in enumerate(src_vocab)}
trg_vocab_stoi = {word: index for index, word in enumerate(trg_vocab)}
trg_vocab_itos = {index: word for index, word in enumerate(trg_vocab)}

# 轉換文本數據為索引數據
def text_to_indices(text, vocab_stoi):
    tokens = tokenize_en(" ".join(text))  # 合并分词后的列表并传递给 tokenize_en 函数
    indices = [vocab_stoi.get(token, vocab_stoi["<unk>"]) for token in tokens]
    return indices


def indices_to_text(indices, vocab_itos):
    tokens = [vocab_itos.get(index, "<unk>") for index in indices]
    text = " ".join(tokens)
    return text


# 定義模型和超參數（使用前面定義的Transformer模型）
d_model = 256
num_heads = 4
d_ff = 1024
dropout = 0.1
num_layers = 6

encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout).to(device)
decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout).to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab_stoi["<pad>"])
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# 自定義數據迭代器
batch_size = 64
# 转换文本数据为索引数据时，将文本传递给 text_to_indices 函数
train_dataset = [(text_to_indices(src_text, src_vocab_stoi), text_to_indices(trg_text, trg_vocab_stoi)) for src_text, trg_text in train_dataset]
def collate_fn(batch):
    src_texts, trg_texts = zip(*batch)
    
    # 找到批处理中最长的源语言和目标语言序列
    max_src_len = max(len(src_text) for src_text in src_texts)
    max_trg_len = max(len(trg_text) for trg_text in trg_texts)
    
    # 填充序列
    padded_src_texts = [src_text + [src_vocab_stoi["<pad>"]] * (max_src_len - len(src_text)) for src_text in src_texts]
    padded_trg_texts = [trg_text + [trg_vocab_stoi["<pad>"]] * (max_trg_len - len(trg_text)) for trg_text in trg_texts]
    
    # 转换为张量
    src_tensors = torch.LongTensor(padded_src_texts)
    trg_tensors = torch.LongTensor(padded_trg_texts)
    
    return src_tensors, trg_tensors

# 创建自定义数据加载器时，指定 collate_fn 参数
train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_iterator:
        src, trg = batch
        src = torch.LongTensor(src).to(device).float()
        trg = torch.LongTensor(trg).to(device).float()

        optimizer.zero_grad()
        
        src_pad_idx = src_vocab_stoi["<pad>"]
        trg_pad_idx = trg_vocab_stoi["<pad>"]
        src_mask, trg_mask = create_masks(src, trg, src_pad_idx, trg_pad_idx)
        
        encoder_output = encoder(src, src_mask)
        decoder_output = decoder(trg, encoder_output, src_mask, trg_mask)
        
        output_dim = decoder_output.shape[-1]
        decoder_output = decoder_output.view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(decoder_output, trg)
        loss.backward()
        optimizer.step()
        