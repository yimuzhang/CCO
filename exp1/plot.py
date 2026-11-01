import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 24  
# 读取数据
path="/home/ymzhang/casulity/exp1/causal_run_20250924_172952/"
path10="/home/ymzhang/casulity/exp1/causal_run_20250924_062054/"
path20="/home/ymzhang/casulity/exp1/causal_run_20250924_062922/"
w_data = pd.read_csv(path+'w.csv')
p_data = pd.read_csv(path+'pm.csv')
w10_data = pd.read_csv(path10+'w.csv')
p10_data = pd.read_csv(path10+'pm.csv')
w20_data = pd.read_csv(path20+'w.csv')
p20_data = pd.read_csv(path20+'pm.csv')
# 创建图形和双y轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制第一个数据集（左侧y轴）
color1 = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_xlim(0,1000)
ax1.set_ylim(-0.1,2.1)
x=np.arange(len(p_data.iloc[1:]))*2
ax1.set_ylabel(r'$||w-w^*||_2$', color=color1)
line1 = ax1.plot(x,w_data.iloc[1:], 
                color=color1, label=r'$||w-w^*||_2$,d=5')
line3 = ax1.plot(w10_data.iloc[1:], 
                color=color1, label=r'$||w-w^*||_2$,d=10',linestyle='--')
# line4 = ax1.plot(w20_data.iloc[1:]*2-0.7, 
#                 color=color1, label=r'$||w-w^*||_2$,d=20',linestyle='-.')
ax1.tick_params(axis='y', labelcolor=color1)

# 创建第二个y轴（右侧）
ax2 = ax1.twinx()
ax2.set_ylim(-0.02,1.02)
color2 = 'tab:red'
ax2.set_ylabel('Average of p', color=color2)

line2 = ax2.plot(p_data.iloc[1:],
                color=color2, label='p,d=5',)
line5 = ax2.plot(p10_data.iloc[1:],
                color=color2, label='p,d=10',linestyle='--')
# line6 = ax2.plot(p20_data.iloc[1:],
#                 color=color2, label='p,d=20',linestyle='-.')
ax2.tick_params(axis='y', labelcolor=color2)

x_line = 120
ax1.axvline(x=x_line, color='green', linestyle='--', alpha=0.7, linewidth=2)

# 获取y轴范围用于文本定位
y1_min, y1_max = ax1.get_ylim()
y2_min, y2_max = ax2.get_ylim()

# 左侧标注'occupation'
ax1.text(100, y1_max * 0.15, 'occupation', 
            ha='right', va='top', fontsize=24, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 右侧标注'crowding out'
ax1.text(180, y1_max * 0.15, 'crowding out', 
            ha='left', va='top', fontsize=24, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
# 添加图例
# lines = line1 +line3+line4+ line2+line5+line6
lines = line1 +line3+ line2+line5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

fig.tight_layout()

# 显示图形
plt.show()
