import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 数据填写 =====================
# 👉 这里填你的成功率数据（按顺序：基础、中等、优秀）
data = np.array([
    [0.80, 0.65],  # 基础学生：顺序=80%，路径=65%
    [0.75, 0.78],  # 中等学生：顺序=75%，路径=78%
    [0.60, 0.92]   # 优秀学生：顺序=60%，路径=92%
])

# 👉 填写 P-value（显著性）
p_values = np.array([
    [0.0005, 0.04],   # 基础：顺序极显著，路径显著
    [0.06, 0.02],     # 中等：不显著，显著
    [0.01, 0.0001]    # 优秀：显著，极显著
])

# 标签
students = ['基础学生', '中等学生', '优秀学生']
paths = ['顺序学习路径', '路径学习路径']

# ===================== 绘图 =====================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 1. 画热力图
im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)

# 2. 设置轴标签
ax.set_xticks(np.arange(len(paths)))
ax.set_yticks(np.arange(len(students)))
ax.set_xticklabels(paths, fontsize=12)
ax.set_yticklabels(students, fontsize=12)

# 3. 添加数值标注
for i in range(len(students)):
    for j in range(len(paths)):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=11, fontweight='bold')

# 4. 画网络连接线（模拟你给的参考图样式）
# 这里用折线模拟关联，颜色区分显著性
from matplotlib.patches import ConnectionPatch

# 定义颜色：黄色=极显著，蓝色=显著，灰色=不显著
def get_color(p):
    if p < 0.01:
        return 'gold'
    elif p < 0.05:
        return 'darkblue'
    else:
        return 'gray'

# 定义线宽：成功率越高，线越粗
def get_width(val):
    return val * 4 + 1

# 绘制连接：左节点（路径） -> 右节点（学生）
for i, student in enumerate(students):
    for j, path in enumerate(paths):
        # 路径节点坐标 (j, -0.5)
        # 学生节点坐标 (i, 0.5)
        xyA = (j, -0.3)   
        xyB = (i, 0.3)    
        
        color = get_color(p_values[i, j])
        width = get_width(data[i, j])
        
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="axes", coordsB="axes",
                              color=color, linewidth=width, alpha=0.8)
        ax.add_artist(con)

# 5. 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Success Rate', fontsize=12)

# 6. 标题与美化
plt.title('Learning Path Success Rate by Student Level', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('success_rate_network.png', dpi=300, bbox_inches='tight')
plt.show()