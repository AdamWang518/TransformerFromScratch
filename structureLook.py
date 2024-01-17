import timm

# 创建ViT模型实例
model = timm.create_model('vit_small_patch16_224', pretrained=True)

# 打印模型结构
print(model)
