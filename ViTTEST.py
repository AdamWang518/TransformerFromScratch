import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import matplotlib.pyplot as plt
import numpy as np
best_loss = float('inf')
# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # 将灰度图像转换为三通道图像
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 调整为三通道的正规化
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# 定义 ViT 模型
class ViTForMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTForMNIST, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

model = ViTForMNIST().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
# 选择一些测试图像进行展示
# 設置模型為評估模式
model.eval()

# 加載測試數據
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 獲取預測
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 將圖像轉換回 CPU
images = images.cpu()
predicted = predicted.cpu()
labels = labels.cpu()

# 繪製圖像
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, ax in enumerate(axes.flat):
    if i < 16:
        # 將圖像轉換回28x28灰度圖像
        img = transforms.Resize((28, 28))(images[i].unsqueeze(0))[0][0].numpy()  # 只取第一個通道
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}", fontsize=8)

plt.show()