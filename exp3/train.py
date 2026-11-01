import os
import numpy as np
import random
import pandas as pd
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataset_utils import crop_and_resize, combine_and_mask
import torch
import torch.nn as nn
from torchvision import models,transforms
import random
import timm
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ResNetForImageClassification

def split(l,rate):
    n_total=len(l)
    offset=int(n_total*rate)
    random.shuffle(l)
    train=l[:offset]
    test=l[offset:]
    return train,test
def generate_train(r_water,r_land,n_sample_train,cub_dir = './CUB',places_dir = './data_large'):

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds\
    
    train_wb=[]
    train_lb=[]
    with open('trainlb.txt','r') as f:
        for line in f:
            line=line.strip('\n')
            train_lb.append(line)

    with open('trainwb.txt','r') as f:
          for line in f:
            line=line.strip('\n')
            train_wb.append(line)
    

    im_size=224
    images=torch.zeros(3,im_size,im_size)
    y=np.zeros(n_sample_train)
    z=np.zeros(n_sample_train)
    transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
        # Load bird image and segmentation

    iswater = np.random.binomial(1,0.5,1)
    if iswater[0]==1:
        bird_name=random.choice(train_wb)
    else:
        bird_name=random.choice(train_lb)
    img_path = os.path.join(cub_dir, 'images', bird_name)
    seg_path = os.path.join(cub_dir, 'segmentations', bird_name.replace('.jpg','.png'))
    img_np = np.asarray(Image.open(img_path).convert('RGB'))
    seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

    if iswater[0]==1:
        background= np.random.binomial(1,r_water,1)
    else:
        background= np.random.binomial(1,1-r_land,1)
    background_path=random.choice(target_places[background[0]])
    num_background=str(random.randint(0,4999)+1).rjust(8,'0')+".jpg"

    place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
    place = Image.open(place_path).convert('RGB')

    img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
    combined_img = combine_and_mask(place, seg_np, img_black)
    combined_img = transform(combined_img.resize((im_size,im_size)))
    images=combined_img

    y=iswater[0]
    z=background[0]
    return images,y,z
def generate_bird(n_sample_train,cub_dir = './CUB',places_dir = './data_large'):

    #model_name: resnet50, resnet34, wideresnet50
    
    

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds
    ######################################################################################
    
    train_wb=[]
    train_lb=[]
    transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    with open('trainlb.txt','r') as f:
        for line in f:
            line=line.strip('\n')
            train_lb.append(line)

    with open('trainwb.txt','r') as f:
          for line in f:
            line=line.strip('\n')
            train_wb.append(line)
    

    im_size=224
    images=torch.zeros(3,im_size,im_size)
    y=np.zeros(n_sample_train)
    z=np.zeros(n_sample_train)
    iswater = np.random.binomial(1,0.5,1)
    if iswater[0]==1:
        bird_name=random.choice(train_wb)
    else:
        bird_name=random.choice(train_lb)
    img_path = os.path.join(cub_dir, 'images', bird_name)
    img_np = Image.fromarray(np.asarray(Image.open(img_path).convert('RGB')).astype(np.uint8))


    
    images=transform(img_np.resize((im_size,im_size)))
    y=iswater[0]
    return images,y
def generate_bg(n_sample_train,cub_dir = './CUB',places_dir = './data_large'):

    #model_name: resnet50, resnet34, wideresnet50
    
    

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds
    transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    ######################################################################################
    
   
    im_size=224
    background= np.random.binomial(1,0.5,1)
    background_path=random.choice(target_places[background[0]])
    num_background=str(random.randint(0,4999)+1).rjust(8,'0')+".jpg"

    place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
    place = Image.open(place_path).convert('RGB')
    images=transform(place.resize((im_size,im_size)))
    z=background[0]
    return images,z



class CausalDataset(Dataset):
    def __init__(self, n_samples,batchsize, wr=0.5,lr=0.5,device='cpu'):
        self.n_samples = n_samples
        self.batchsize=batchsize
        self.wr=wr
        self.lr=lr
        self.device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X,Y,Z=generate_train(self.wr,self.lr,self.batchsize)
        # if idx<10:
        #     print(Y,Z)
        return X,Y
    
class CausalDataset_bird(Dataset):
    def __init__(self, n_samples,batchsize,device='cpu'):
        self.n_samples = n_samples
        self.batchsize=batchsize
        self.device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X,Z=generate_bird(self.batchsize)
        return X,Z
class CausalDataset_bg(Dataset):
    def __init__(self, n_samples,batchsize,device='cpu'):
        self.n_samples = n_samples
        self.batchsize=batchsize
        self.device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X,Z=generate_bg(self.batchsize)
        return X,Z

# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for r_train in [0.5]:#0.5,0.6,0.7,0.8,0.9,0.95,0.98,
    print('!!!',r_train)
    n_samples=40000
    batchsize=16
    dataset = CausalDataset(n_samples,batchsize,r_train,r_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataset1 = CausalDataset(4096,256,0.02,0.02)
    dataloader1 = DataLoader(test_dataset1, batch_size=256, shuffle=True, num_workers=4)


    num_classes = 2  # 根据您的任务修改
    model = timm.create_model(
        'deit_small_patch16_224',  # 使用灵活的ViT变体
        pretrained=True,
       # patch_size=16,       # patch大小（16×16）
        num_classes=num_classes
    )
    # model = models.resnet34(weights='IMAGENET1K_V1')
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    
    # model = models.efficientnet_b4(pretrained=True)
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_features, num_classes)
    #model = CompactCNN(num_classes=num_classes)  # 以10分类为例

    total_params = count_parameters(model)
    print(f"总参数数量: {total_params}")  # 约为11.5M参数
    print(f"模型大小: {total_params * 4 / (1024**2):.2f} MB") 


    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 训练循环
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    it=0
    for epoch in range(1):
        model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # 验证输入尺寸
            if batch_idx == 0:
                print(f"输入张量形状: {images.shape}") 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if it%500==499:
                model.eval()
                with torch.no_grad():
                    for r in [0.95,0.99]:#0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,
                        correct = 0
                        total = 0
                        test_loss = 0
                        test_dataset1 = CausalDataset(4096,512,r,r)
                        dataloader1 = DataLoader(test_dataset1, batch_size=256, shuffle=True, num_workers=4)
                        
                        for batch_idx, (images, labels) in enumerate(dataloader1):
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            
                            # 计算测试损失
                            loss = criterion(outputs, labels)
                            test_loss += loss.item()
                            
                            # 计算准确率
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        
                        # 计算并打印准确率
                        accuracy = 100 * correct / total
                        avg_test_loss = test_loss / len(dataloader1)
                        print(r,accuracy,avg_test_loss)


                model.train()


            if it%100==0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, f'/data1/ymzhang/ckpt/vit/vit_model_z_{it}.pth')

                print(f"模型已保存为 /data1/ymzhang/ckpt/vit/vit_model_z_{it}.pth")
            it=it+1
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')