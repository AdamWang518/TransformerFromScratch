import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import matplotlib.pyplot as plt
import numpy as np
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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 定义 ViT 模型
class ViTForMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTForMNIST, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

model = ViTForMNIST().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5  # 举例，实际迭代次数可能需要调整
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        running_loss += loss.item()
    # 每个epoch结束后打印平均损失
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# # 绘制训练损失
# plt.figure(figsize=(10, 5))
# plt.plot(losses)
# plt.title("Training Loss")
# plt.xlabel("Batch")
# plt.ylabel("Loss")
# plt.show()

# # 选择一些测试图像进行展示
# model.eval()  # 设置模型为评估模式
# dataiter = iter(test_loader)
# images, labels = dataiter.next()
# images, labels = images.to(device), labels.to(device)

# # 获取预测
# outputs = model(images)
# _, predicted = torch.max(outputs, 1)

# # 将图像转换回 CPU 并绘制
# images = images.cpu()
# predicted = predicted.cpu()
# labels = labels.cpu()

# fig, axes = plt.subplots(6, 6, figsize=(12, 12))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)

# for i, ax in enumerate(axes.flat):
#     if i < 36:
#         img = images[i].numpy().squeeze()
#         ax.imshow(img, cmap="gray")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}")