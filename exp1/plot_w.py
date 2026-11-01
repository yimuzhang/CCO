import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 24  
# 读取数据
path="/home/ymzhang/casulity/exp1/causal_run_20250924_215233/"
w_data = pd.read_csv(path+'w.csv')
print("数据形状:", w_data.shape)
print("\n前5行数据:")
print(w_data.head())
print("\n数据类型:")
print(w_data.dtypes)

# 创建绘图
plt.figure(figsize=(12, 8))

# 绘制热力图
# 如果数据包含非数值列，请先选择数值列
# 例如：numeric_data = w_data.select_dtypes(include=[np.number])
heatmap = sns.heatmap( w_data.iloc[:1000, 1:] ,  # 直接使用DataFrame
                      annot=False,        # 大数据集建议关闭数值显示
                      cmap='coolwarm',    # 颜色映射方案
                      center=0,           # 颜色中心点（对于有正负值的数据很有用）
                      cbar_kws={'label': 'Value', 'shrink': 0.8},
                      square=False)       # 单元格是否为正方形


plt.xlabel('W', fontsize=12)
plt.ylabel('Iteration', fontsize=12)

# 如果行/列名太长，可以旋转标签
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 调整布局并显示
plt.tight_layout()
plt.show()




print(w_data.iloc[1:,1:2])
plt.plot(w_data.iloc[1:,1]*5/3,label=r'$w_1$')
plt.plot(w_data.iloc[1:,2]*5/3,label=r'$w_2$')
plt.plot(w_data.iloc[1:,3]*5/3,label=r'$w_3$')
plt.plot(w_data.iloc[1:,4]*5/3,label=r'$w_4$')
plt.plot(w_data.iloc[1:,5]*5/3,label=r'$w_5$')
plt.plot(w_data.iloc[1:,6]*5/3,label=r'$w_6$')
plt.plot(w_data.iloc[1:,7]*5/3,label=r'$w_7$')
plt.plot(w_data.iloc[1:,8]*5/3,label=r'$w_8$')
plt.plot(w_data.iloc[1:,9]*5/3,label=r'$w_9$')
plt.legend()
plt.xlim(0,1000)
plt.xlabel('Iteration')
plt.ylabel('Weight')
plt.show()