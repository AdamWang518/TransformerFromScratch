import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
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

# 創建虛擬數據（隨機生成）
input_dim = 32
output_dim = 32
batch_size = 64
sequence_length = 10

# 創建虛擬的訓練數據和目標數據（使用隨機張量）
train_input = torch.randn(batch_size, sequence_length, input_dim)
train_target = torch.randn(batch_size, sequence_length, output_dim)

# 定義模型和超參數
d_model = 32
num_heads = 4
d_ff = 128
dropout = 0.1
num_layers = 2

encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    encoder_output = encoder(train_input, mask=None)
    decoder_output = decoder(train_target, encoder_output, src_mask=None, trg_mask=None)
    loss = criterion(decoder_output, train_target)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")