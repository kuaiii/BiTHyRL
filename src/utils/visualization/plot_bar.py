import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 全局设置（全元素加粗+超大字体） --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'  # 全局字体加粗
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.major.size'] = 8    # x轴刻度长度
plt.rcParams['ytick.major.size'] = 8    # y轴刻度长度
plt.rcParams['xtick.labelsize'] = 18    # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18    # y轴刻度字体大小
plt.rcParams['axes.linewidth'] = 1.5    # 坐标轴边框宽度（加粗）
plt.rcParams['text.usetex'] = True      # 启用Latex支持斜体

# -------------------------- 数据定义与计算 --------------------------
total_nodes = 18  # 总节点数

# 1. Bimodal Network数据
bimodal_degree = [1, 7]
bimodal_prob = [0.8333, 0.1667]
bimodal_count = [round(p * total_nodes) for p in bimodal_prob]
bimodal_max_prob = max(bimodal_prob)  # 最大概率值

# 2. BA Network数据
ba_degree = [1, 2, 4, 6, 8]
ba_prob = [0.6667, 0.1667, 0.0556, 0.0556, 0.0556]
ba_count = [round(p * total_nodes) for p in ba_prob]
ba_max_prob = max(ba_prob)  # 最大概率值

# 3. Random Network数据
random_degree = [1, 2, 3]
random_prob = [0.2222, 0.5556, 0.2222]
random_count = [round(p * total_nodes) for p in random_prob]
random_max_prob = max(random_prob)  # 最大概率值

# 定义柱状图颜色（RGB 169,233,228 归一化到0-1）
bar_color = (169/255, 233/255, 228/255)
bar_edge_width = 2  # 边框加粗

# -------------------------- 绘制Bimodal Network图表 --------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))
# 绘制柱状图（边框黑色加粗）
bars1 = ax1.bar(bimodal_degree, bimodal_prob, color=bar_color, width=0.8,
                edgecolor='black', linewidth=bar_edge_width)

# 横坐标仅显示1、7两个刻度，且限定x轴范围在1-7（缩小间距）
ax1.set_xticks(bimodal_degree)
ax1.set_xticklabels(bimodal_degree, weight='bold', fontsize=18)
ax1.set_xlim(0, 8)  # 微调范围，避免柱子贴边，同时缩小1和7的间距

# 设置y轴（贴合数据最大值，仅显示到90%）
y_max1 = np.ceil(bimodal_max_prob * 10) / 10  # 向上取整到0.1的倍数（0.9）
ax1.set_ylim(0, y_max1)
y_ticks1 = np.arange(0, y_max1 + 0.1, 0.1)  # 刻度步长0.1
ax1.set_yticks(y_ticks1)
ax1.set_yticklabels([f'{x:.0%}' for x in y_ticks1], weight='bold')

# 坐标轴标签（K和P(k)斜体，显式加粗）
ax1.set_xlabel(r'Degree $K$', fontsize=20, weight='bold')
ax1.set_ylabel(r'Probability $P(k)$', fontsize=20, weight='bold')

# 在柱子顶部标注节点数量（加粗）
for bar, count in zip(bars1, bimodal_count):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'n={count}', ha='center', va='bottom', fontsize=18, weight='bold')

# 去除顶部和右侧边框，保留的边框加粗
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)  # 左轴加粗
ax1.spines['bottom'].set_linewidth(1.5)  # 底轴加粗

plt.tight_layout()
plt.savefig('bimodal_network.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 绘制BA Network图表 --------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))
bars2 = ax2.bar(ba_degree, ba_prob, color=bar_color, width=0.8,
                edgecolor='black', linewidth=bar_edge_width)

# 横坐标仅显示有数据的刻度（1,2,4,6,8）
ax2.set_xticks(ba_degree)
ax2.set_xticklabels(ba_degree, weight='bold', fontsize=18)
ax2.set_xlim(0, 9)

# 设置y轴（贴合数据最大值，仅显示到70%）
y_max2 = np.ceil(ba_max_prob * 10) / 10  # 向上取整到0.1的倍数（0.7）
ax2.set_ylim(0, y_max2)
y_ticks2 = np.arange(0, y_max2 + 0.1, 0.1)
ax2.set_yticks(y_ticks2)
ax2.set_yticklabels([f'{x:.0%}' for x in y_ticks2], weight='bold')

# 坐标轴标签（显式加粗）
ax2.set_xlabel(r'Degree $K$', fontsize=20, weight='bold')
ax2.set_ylabel(r'Probability $P(k)$', fontsize=20, weight='bold')

# 标注节点数量（加粗）
for bar, count in zip(bars2, ba_count):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'n={count}', ha='center', va='bottom', fontsize=18, weight='bold')

# 边框加粗+隐藏冗余边框
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('ba_network.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 绘制Random Network图表 --------------------------
fig3, ax3 = plt.subplots(figsize=(8, 6))
bars3 = ax3.bar(random_degree, random_prob, color=bar_color, width=0.8,
                edgecolor='black', linewidth=bar_edge_width)

# 横坐标仅显示有数据的刻度（1,2,3）
ax3.set_xticks(random_degree)
ax3.set_xticklabels(random_degree, weight='bold', fontsize=18)
ax3.set_xlim(0, 4)

# 设置y轴（贴合数据最大值，仅显示到60%）
y_max3 = np.ceil(random_max_prob * 10) / 10  # 向上取整到0.1的倍数（0.6）
ax3.set_ylim(0, y_max3)
y_ticks3 = np.arange(0, y_max3 + 0.1, 0.1)
ax3.set_yticks(y_ticks3)
ax3.set_yticklabels([f'{x:.0%}' for x in y_ticks3], weight='bold')

# 坐标轴标签（显式加粗）
ax3.set_xlabel(r'Degree $K$', fontsize=20, weight='bold')
ax3.set_ylabel(r'Probability $P(k)$', fontsize=20, weight='bold')

# 标注节点数量（加粗）
for bar, count in zip(bars3, random_count):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'n={count}', ha='center', va='bottom', fontsize=18, weight='bold')

# 边框加粗+隐藏冗余边框
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(1.5)
ax3.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('random_network.png', dpi=300, bbox_inches='tight')
plt.show()