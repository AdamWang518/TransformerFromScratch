import torch

# 假設數據
# src 是輸入序列, tgt 是目標序列
# 這裡的數據是隨機生成的，只是為了示例
src = torch.rand((10, 32, 512))  # (序列長度, 批次大小, 特徵數)
tgt = torch.rand((20, 32, 512))  # (序列長度, 批次大小, 特徵數)

import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model, num_heads, num_encoder_layers, num_decoder_layers):
        super(SimpleTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers)
        self.input_linear = nn.Linear(input_dim, dim_model)
        self.output_linear = nn.Linear(dim_model, output_dim)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output

# 初始化模型
model = SimpleTransformer(input_dim=512, output_dim=512, dim_model=512, num_heads=8, num_encoder_layers=3, num_decoder_layers=3)
# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 簡單訓練循環
for epoch in range(100):
    optimizer.zero_grad()
    output = model(src, tgt)
    loss = criterion(output, tgt)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")
