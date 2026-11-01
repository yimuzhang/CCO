import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import timm
def extract_attention_maps(model, image_tensor, layer_index=-1):
    """
    提取指定层的attention maps
    
    Args:
        model: ViT模型
        image_tensor: 输入图像张量 [1, 3, H, W]
        layer_index: 层索引，-1表示最后一层
    
    Returns:
        attention_maps: attention权重 [num_heads, num_patches, num_patches]
    """
    # 保存原始的前向传播方法
    original_forward = model.blocks[layer_index].attn.forward
    
    # 用于存储attention权重的钩子
    attention_weights = []
    
    def attention_hook(module, input, output):
        # output[1] 包含attention权重
        attention_weights.append(output.detach().cpu())
    
    # 注册钩子
    hook = model.blocks[layer_index].attn.register_forward_hook(attention_hook)
    
    # 前向传播
    with torch.no_grad():
        model(image_tensor)
    
    # 移除钩子
    hook.remove()
    model.blocks[layer_index].attn.forward = original_forward
    
    if attention_weights:
        return attention_weights[0]  # [num_heads, num_patches+1, num_patches+1]
    return None

def visualize_all_heads(attention_weights, num_cols=4):
    """
    可视化所有attention heads
    """
    num_heads = attention_weights.shape[0]
    num_rows = (num_heads + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 3 * num_rows))
    
    for i in range(num_heads):
        plt.subplot(num_rows, num_cols, i + 1)
        attn = attention_weights[i, 0, 1:].reshape(14, 14)  # 对于224×224输入
        plt.imshow(attn, cmap='viridis')
        plt.title(f'Head {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def attention_to_image(attention_map, original_image, patch_size=16):
    """
    将attention map映射到原图上
    """
    # 调整attention map大小以匹配原图
    h, w = original_image.shape[1], original_image.shape[2]
    attn_resized = torch.nn.functional.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # 转换为numpy
    attn_np = attn_resized.numpy()
    
    # 创建热力图覆盖
    plt.figure(figsize=(12, 6))
    
    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention叠加图
    plt.subplot(1, 2, 2)
    plt.imshow(original_image.permute(1, 2, 0))
    plt.imshow(attn_np, cmap='jet', alpha=0.5)
    plt.title('Attention Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return attn_resized
def visualize_attention_single_head(attention_weights, image, head_idx=0, patch_size=16):
    """
    可视化单个attention head
    """
    # attention_weights: [num_heads, num_patches+1, num_patches+1]
    num_patches = attention_weights.shape[1] - 1  # 减去cls token
    
    # 获取指定head的attention权重（忽略cls token）
    print(attention_weights.shape)
    attn = attention_weights[head_idx, 1: ,0].reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)),-1)
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Attention Map - Head {head_idx}')
    plt.show()
    
    return attn

def analyze_attention_for_image(model, image_path, layer_index=-1, head_idx=0):
    """
    完整分析单张图像的attention
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    
    # 提取attention
    attention_weights = extract_attention_maps(model, image_tensor, layer_index)
    
    if attention_weights is None:
        print("无法提取attention weights")
        return
    
    print(f"Attention weights形状: {attention_weights.shape}")
    print(f"Number of heads: {attention_weights.shape[0]}")
    print(f"Number of patches: {attention_weights.shape[1] - 1}")
    
    # 可视化
    #visualize_all_heads(attention_weights)
    
    # 查看特定head
    specific_attn = visualize_attention_single_head(attention_weights, image_tensor.squeeze(), head_idx)
    
    # 映射到原图
    attention_to_image(specific_attn, image_tensor.squeeze())
    
    return attention_weights,image_tensor

def load_state_dict(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载模型的状态字典"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # 重新创建模型
    model = timm.create_model(
    'vit_base_patch16_224',  # 使用灵活的ViT变体
    pretrained=True,
    patch_size=16,       # patch大小（16×16）
    num_classes=2
)

    
    # 加载参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型状态字典已从 {filepath} 加载")
    return model
def main():
    

    model = load_state_dict('/data1/ymzhang/ckpt/vit/vit_model_250.pth',device='cpu')
    model.eval()
    
    # 分析图像
    image_path = "/home/ymzhang/casulity/CUB/images/001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg"  # 替换为您的图像路径
    attention_weights,image_tensor = analyze_attention_for_image(model, image_path)
    
    # 分析不同层的attention
    print("\n分析不同层的attention...")
    for layer_idx in range(len(model.blocks)):
        print(f"Layer {layer_idx}:")
        attention_weights = extract_attention_maps(model, image_tensor, layer_idx)
        if attention_weights is not None:
            print(f"  Shape: {attention_weights.shape}")

if __name__ == "__main__":
    main()

def analyze_specific_patch_attention(attention_weights, patch_idx, num_heads=12):
    """
    分析特定patch的attention模式
    """
    plt.figure(figsize=(15, 10))
    
    for head_idx in range(num_heads):
        # 获取该patch在所有其他patches上的attention
        patch_attention = attention_weights[head_idx, patch_idx + 1, 1:].reshape(14, 14)
        
        plt.subplot(3, 4, head_idx + 1)
        plt.imshow(patch_attention, cmap='viridis')
        plt.title(f'Head {head_idx} - Patch {patch_idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_cls_token_attention(attention_weights):
    """
    分析CLS token对其他patches的attention
    """
    cls_attention = attention_weights[:, 0, 1:]  # [num_heads, num_patches]
    
    plt.figure(figsize=(15, 5))
    for i in range(cls_attention.shape[0]):
        plt.subplot(2, 6, i + 1)
        attn_map = cls_attention[i].reshape(14, 14)
        plt.imshow(attn_map, cmap='hot')
        plt.title(f'Head {i} - CLS Attention')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

