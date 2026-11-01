import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import timm
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
import os
import random
from dataset_utils import crop_and_resize, combine_and_mask
def generate_train(r_water, r_land, n_sample_train, cub_dir='./CUB', places_dir='./data_large'):
    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']              # Water backgrounds
    ]
    
    train_wb = []
    train_lb = []
    with open('trainlb.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            train_lb.append(line)

    with open('trainwb.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            train_wb.append(line)
    
    im_size = 224
    # 修改为存储两张拼接后的图像
    images = torch.zeros(2, 3, im_size, im_size )  # 两张拼接图像
    y = np.zeros(n_sample_train)  # 标签（基于左侧图像）
    z = np.zeros(n_sample_train)  # 背景类型
    
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor()
    ])
    

    iswater1 = np.random.binomial(1, 0.5, 1)
    if iswater1[0] == 1:
        bird_name1 = random.choice(train_wb)
    else:
        bird_name1 = random.choice(train_lb)
    
    img_path1 = os.path.join(cub_dir, 'images', bird_name1)
    seg_path1 = os.path.join(cub_dir, 'segmentations', bird_name1.replace('.jpg', '.png'))
    img_np1 = np.asarray(Image.open(img_path1).convert('RGB'))
    seg_np1 = np.asarray(Image.open(seg_path1).convert('RGB')) / 255

    if iswater1[0] == 1:
        background1 = np.random.binomial(1, r_water, 1)
    else:
        background1 = np.random.binomial(1, 1 - r_land, 1)
    
    background_path1 = random.choice(target_places[background1[0]])
    num_background1 = str(random.randint(0, 4999) + 1).rjust(8, '0') + ".jpg"
    place_path1 = os.path.join(places_dir, background_path1[0], background_path1, num_background1)
    place1 = Image.open(place_path1).convert('RGB')
    img_array = np.array(place1)
    img_array[:] = 0  # 将所有像素值设为0
    place1 = Image.fromarray(img_array)

    img_black1 = Image.fromarray(np.around(img_np1 * seg_np1).astype(np.uint8))
    combined_img1 = combine_and_mask(place1, seg_np1, img_black1)
    combined_img1 = transform(combined_img1)
    
    # 生成第二张图像（右侧）
    if iswater1[0] == 1:
        iswater2 = np.random.binomial(1, r_water, 1)
    else:
        iswater2 = np.random.binomial(1, 1-r_land, 1)
    if iswater2[0] == 1:
        bird_name2 = random.choice(train_wb)
    else:
        bird_name2 = random.choice(train_lb)
    
    img_path2 = os.path.join(cub_dir, 'images', bird_name2)
    seg_path2 = os.path.join(cub_dir, 'segmentations', bird_name2.replace('.jpg', '.png'))
    img_np2 = np.asarray(Image.open(img_path2).convert('RGB'))
    seg_np2 = np.asarray(Image.open(seg_path2).convert('RGB')) / 255

    if iswater2[0] == 1:
        background2 = np.random.binomial(1, r_water, 1)
    else:
        background2 = np.random.binomial(1, 1 - r_land, 1)
    
    background_path2 = random.choice(target_places[background2[0]])
    num_background2 = str(random.randint(0, 4999) + 1).rjust(8, '0') + ".jpg"
    place_path2 = os.path.join(places_dir, background_path2[0], background_path2, num_background2)
    place2 = Image.open(place_path2).convert('RGB')
    img_array = np.array(place2)
    img_array[:] = 0  # 将所有像素值设为0
    place2 = Image.fromarray(img_array)

    img_black2 = Image.fromarray(np.around(img_np2 * seg_np2).astype(np.uint8))
    combined_img2 = combine_and_mask(place2, seg_np2, img_black2)
    combined_img2 = transform(combined_img2)
    
    # 将两张图像左右拼接
    concatenated_img = torch.cat([combined_img1, combined_img2], dim=2)  # 在宽度维度拼接
   
    
    images = Image.fromarray((255*concatenated_img).to(torch.uint8).permute(1, 2, 0).numpy())
    # plt.imshow((255*images).to(torch.uint8).permute(1, 2, 0), cmap='viridis')
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    y = iswater1[0]  # 使用左侧图像的标签
    z = iswater2[0]  # 使用左侧图像的背景类型
    
    return images
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default="/home/ymzhang/casulity/CUB/images/196.House_Wren/House_Wren_0098_187368.jpg",
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='min',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    # model = torch.hub.load('/data1/ymzhang/ckpt/vit/vit_model_250.pth',
    #     'deit_tiny_patch16_224',  source='local',pretrained=True)
    # model = torch.hub.load('facebookresearch/deit:main', 
    #     'deit_tiny_patch16_224', pretrained=True)
    
    checkpoint = torch.load('/data1/ymzhang/ckpt/vit/vit_model_z_100.pth')
    model = timm.create_model(
    'deit_small_patch16_224',  # 使用灵活的ViT变体
    pretrained=True,
   # patch_size=16,       # patch大小（16×16）
    num_classes=2
)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img= Image.open('Yellow_Headed_Blackbird_0003_8337.jpg')#generate_train(0.5,0.5,1)# Image.open('input.png')
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
            args.discard_ratio, args.head_fusion)


    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    cv2.waitKey(-1)