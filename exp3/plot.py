import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# 读取log文件并解析数据
bias=[0.3,0.2,0.1,0.05,0.01]
f6=[89,88,92,81,83]
f0=[61,64,57,53,52.3]
f1=[88.45,91.87,96.02,96.55,96.65]
f2=[81.07,84.69,88.84,91.75,95.97]
plt.figure(figsize=(12,12))
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 26
plt.plot(bias,f0,label=r'$p_{train}=0.7$,background',marker='o',color='r')
plt.plot(bias,f1,label=r'$p_{train}=0.7$,oracle',linestyle='--',marker='v',color='r')
plt.plot(bias,f6,label=r'$p_{train}=0.9$,background',marker='o',color='b')
plt.plot(bias,f2,label=r'$p_{train}=0.9$,oracle',linestyle='--',marker='v',color='b')
plt.xlabel(r'$p_{flip}$')
plt.ylabel('Test Accuracy')
plt.legend(loc='right')
plt.show()

def read_log_file_basic(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # 去除空白字符并分割数据
            values = line.strip().split()
            if len(values) == 3:
                # 转换为相应的数据类型
                try:
                    x = float(values[0])
                    y = float(values[1])
                    z = float(values[2])
                    data.append((x, y, z))
                except ValueError as e:
                    print(f"错误:{e}")
                    continue
    return data

# 使用示例
data = read_log_file_basic('merge0.log')
cnt=0
vit=[]
bias=[0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98,0.99]
for point in data:
    if cnt%22==11:
        print(point)
        vit.append(point[1])
    cnt=cnt+1

data1 = read_log_file_basic('merge_resnet.log')
cnt=0
resnet=[]
for point in data1:
    if cnt%22==11:
        resnet.append(point[1])
    cnt=cnt+1

data2 = read_log_file_basic('merge_b4.log')
cnt=0
b4=[]
for point in data2:
    if cnt%22==11:
        b4.append(point[1])
    cnt=cnt+1
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 22 
plt.rcParams["figure.figsize"] = [15, 15]
plt.figure(figsize=(12,12))
plt.plot(bias,vit,label='ViT',marker='o')
plt.plot(bias,resnet,label='ResNet',linestyle='--',marker='v')
plt.plot(bias,b4,label='Efficientnet_b4',linestyle=':',marker='^')
plt.axhline(y=99.2, color='green', linestyle='--', alpha=0.7, linewidth=2,label='Oracle')
plt.xlabel('Bias Strength')
plt.ylabel('Test Loss')
plt.ylim(70,100)
plt.legend()
plt.show()

# data3 = read_log_file_basic('merge0.log')
# bias=[0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
# cnt=0
# f0=[]
# f1=[]
# f2=[]
# f3=[]
# f4=[]
# f5=[]
# plt.figure(figsize=(12,12))
# for point in data3:
#     if cnt//22==0 and cnt%22>10:
#         f0.append(point[1])
#     if cnt//22==2 and cnt%22>10:
#         f1.append(point[1])
#     if cnt//22==3 and cnt%22>10:
#         f2.append(point[1])
#     if cnt//22==4 and cnt%22>10:
#         f3.append(point[1])
#     if cnt//22==5 and cnt%22>10:
#         f4.append(point[1])
#     if cnt//22==6 and cnt%22>10:
#         f5.append(point[1])
#     cnt=cnt+1
# plt.plot(bias,f0,label=r'$p_{train}=0.5$',marker='o')
# plt.plot(bias,f1,label=r'$p_{train}=0.7$',marker='o')
# plt.plot(bias,f2,label=r'$p_{train}=0.8$',marker='o')
# plt.plot(bias,f3,label=r'$p_{train}=0.85$',marker='o')
# plt.plot(bias,f4,label=r'$p_{train}=0.9$',linestyle='--',marker='v')
# plt.plot(bias,f5,label=r'$p_{train}=0.95$',linestyle='--',marker='v')
# plt.xlabel('Test Bias Strength')
# plt.ylabel('Test Accuracy')
# plt.legend()
# plt.show()

data3 = read_log_file_basic('flip.log')
bias=[0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
cnt=0
f0=[]
f1=[]
f2=[]
f3=[]
f4=[]
f5=[]
f6=[]
plt.figure(figsize=(12,12))
for point in data3:
    if cnt//24==28 and cnt%24>11:
        f0.append(point[1])
    if cnt//24==29 and cnt%24>11:
        f1.append(point[1])
    if cnt//24==30 and cnt%24>11:
        f2.append(point[1])
    if cnt//24==31 and cnt%24>11:
        f3.append(point[1])
    if cnt//24==32 and cnt%24>11:
        f4.append(point[1])
    if cnt//24==33 and cnt%24>11:
        f5.append(point[1])
    if cnt//24==34 and cnt%24>11:
        f6.append(point[1])
    cnt=cnt+1
plt.plot(bias,f0,label=r'$p_{flip}=0.5$',linestyle='--',marker='v')
plt.plot(bias,f1,label=r'$p_{flip}=0.3$',linestyle='--',marker='v')
plt.plot(bias,f2,label=r'$p_{flip}=0.2$',linestyle='--',marker='v')
plt.plot(bias,f3,label=r'$p_{flip}=0.1$',linestyle='--',marker='v')
plt.plot(bias,f4,label=r'$p_{flip}=0.05$',linestyle='--',marker='v')
plt.plot(bias,f5,label=r'$p_{flip}=0.02$',linestyle='--',marker='v')
plt.plot(bias,f6,label=r'$p_{flip}=0.0$',marker='o')
plt.xlabel('Test Bias Strength')
plt.ylabel('Test Accuracy')
plt.legend(loc='lower right')
plt.show()

bias=[0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
cnt=0
f0=[]
f1=[]
f2=[]
f3=[]
f4=[]
f5=[]
plt.figure(figsize=(12,12))
for point in data3:
    if cnt//24==11 and cnt%24>11:
        f0.append(point[1])
    if cnt//24==12 and cnt%24>11:
        f1.append(point[1])
    if cnt//24==13 and cnt%24>11:
        f2.append(point[1])
    if cnt//24==14 and cnt%24>11:
        f3.append(point[1])
    if cnt//24==15 and cnt%24>11:
        f4.append(point[1])
    if cnt//24==16 and cnt%24>11:
        f5.append(point[1])
    cnt=cnt+1
plt.plot(bias,f0,label=r'$p_{flip}=0.5$',linestyle='--',marker='v')
plt.plot(bias,f1,label=r'$p_{flip}=0.4$',linestyle='--',marker='v')
plt.plot(bias,f2,label=r'$p_{flip}=0.3$',linestyle='--',marker='v')
plt.plot(bias,f3,label=r'$p_{flip}=0.2$',linestyle='--',marker='v')
plt.plot(bias,f4,label=r'$p_{flip}=0.1$',linestyle='--',marker='v')
plt.plot(bias,f5,label=r'$p_{flip}=0.0$',marker='o')
plt.xlabel('Test Bias Strength')
plt.ylabel('Test Accuracy')
plt.legend(loc='lower right')
plt.show()
# adj_noun=[1.3728,0.8676,0.8230,0.8102,0.6922]#,[0.5892,0.6595,0.6792,0.6800,0.7197]
# adj_verb=[1.3722,0.8652,0.8237,0.8309,0.8197]#,[0.5944,0.6625,0.6805,0.6710,0.6816]
# verb_noun=[1.3769,0.8354,0.7956,0.7875,0.675]#,[0.5917,0.6654,0.6852,0.6861,0.7234]
# noun=[1.3716,0.7727,0.7275,0.7073,0.6812]#,[0.5927,0.6943,0.7049,0.7173,0.7235]
# verb=[1.3688,0.7779,0.7281,0.7036,0.6814]#,[0.5908,0.6869,0.7088,0.7176,0.7226]
# adj=[1.3712,0.8143,0.7467,0.7303,0.6991]#,[0.5937,0.6769,0.7024,0.7077,0.7172]
# base=[1.3645,0.7366,0.6745,0.6533,0.6077]#,[0.5955,0.7002,0.7281,0.7319,0.7542]
# x=[10,1000,5000,10000,20000]
adj_noun=[1.3728,0.9452,0.8676,0.8356,0.8230,0.8102,0.6922]#,[0.5892,0.6595,0.6792,0.6800,0.7197]
adj_verb=[1.3722,0.9522,0.8652,0.8344,0.8237,0.8309,0.8197]#,[0.5944,0.6625,0.6805,0.6710,0.6816]
verb_noun=[1.3769,0.9321,0.8354,0.8112,0.7956,0.7875,0.675]#,[0.5917,0.6654,0.6852,0.6861,0.7234]
noun=[1.3716,0.9012,0.7727,0.7410,0.7275,0.7073,0.6812]#,[0.5927,0.6943,0.7049,0.7173,0.7235]
verb=[1.3688,0.9111,0.7779,0.7355,0.7281,0.7036,0.6814]#,[0.5908,0.6869,0.7088,0.7176,0.7226]
adj=[1.3712,0.9321,0.8143,0.7588,0.7467,0.7303,0.6991]#,[0.5937,0.6769,0.7024,0.7077,0.7172]
base=[1.3645,0.8876,0.7366,0.6978,0.6745,0.6533,0.6077]#,[0.5955,0.7002,0.7281,0.7319,0.7542]
x=[10,500,1000,3000,5000,10000,20000]
plt.figure(figsize=(14,14))
plt.plot(x,base,label='Origin text')
# plt.plot(x,adj,label='ADJ',linestyle=':')
# plt.plot(x,noun,label='NUON',linestyle=':')
# plt.plot(x,verb,label='VERB',linestyle=':')
plt.plot(x,adj_noun,label='ADJ+NUON',linestyle='--')
plt.plot(x,adj_verb,label='ADJ+VERB',linestyle='--')
plt.plot(x,verb_noun,label='NOUN+VERB',linestyle='--')
plt.ylim(0.58,1)
x_line = 1010
plt.axvline(x=x_line, color='green', linestyle='--', alpha=0.7, linewidth=2)

# 左侧标注'occupation'
plt.text(850, 0.69, 'occupation', 
            ha='right', va='top', fontsize=24, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 右侧标注'crowding out'
plt.text(1300, 0.69, 'crowding out', 
            ha='left', va='top', fontsize=24, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
plt.xlabel('Iteration')
plt.ylabel('Test Loss')
plt.xlim(0,10000)
plt.legend()
plt.show()
from scipy.interpolate import CubicSpline
x_smooth = np.linspace(0,10000, 500)

spline = CubicSpline(x, base)
base = spline(x_smooth)
spline = CubicSpline(x, adj)
adj = spline(x_smooth)
spline = CubicSpline(x, noun)
noun = spline(x_smooth)
spline = CubicSpline(x, verb)
verb = spline(x_smooth)
spline = CubicSpline(x,verb_noun)
verb_noun = spline(x_smooth)
spline = CubicSpline(x, adj_verb)
adj_verb = spline(x_smooth)
spline = CubicSpline(x, adj_noun)
adj_noun = spline(x_smooth)
x=x_smooth

plt.plot(x,base,label='Origin text')
plt.plot(x,adj,label='ADJ',linestyle=':')
plt.plot(x,noun,label='NUON',linestyle=':')
plt.plot(x,verb,label='VERB',linestyle=':')
plt.plot(x,adj_noun,label='ADJ+NUON',linestyle='--')
plt.plot(x,adj_verb,label='ADJ+VERB',linestyle='--')
plt.plot(x,verb_noun,label='NOUN+VERB',linestyle='--')
plt.ylim(0.58,1)
x_line = 1100
plt.axvline(x=x_line, color='green', linestyle='--', alpha=0.7, linewidth=2)

# 左侧标注'occupation'
plt.text(900, 0.675, 'occupation', 
            ha='right', va='top', fontsize=24, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 右侧标注'crowding out'
plt.text(1300, 0.675, 'crowding out', 
            ha='left', va='top', fontsize=24, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
plt.xlabel('Iteration')
plt.ylabel('Test Loss')
plt.xlim(0,10000)
plt.legend()
plt.show()