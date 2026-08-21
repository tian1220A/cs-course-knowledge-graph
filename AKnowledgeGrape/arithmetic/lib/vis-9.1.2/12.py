import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ====================== 你的数据 ======================
labels = ["知识点1", "知识点2", "知识点3", "知识点4", "知识点5", "知识点6"]
x = np.arange(len(labels))

# 两种算法权重
wA = np.array([0.2, 0.5, 0.9, 0.7, 0.4, 0.2])
wB = np.array([0.1, 0.4, 0.8, 0.7, 0.5, 0.1])
# ======================================================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')

# 颜色映射（权重越高颜色越深）
norm = plt.Normalize(0, max(wA.max(), wB.max()))

# ====================== 绘制算法 A (y=0) ======================
for xi, weight in zip(x, wA):
    # 绘制一个垂直面：高度=weight，颜色=热力色
    X = [xi, xi+0.001]
    Y = [0, 0]
    Z = [0, weight]
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[0, weight], [0, weight]])
    
    ax.plot_surface(X, Y, Z, color=cm.Reds(norm(weight)), 
                    alpha=0.8, shade=False)

# 算法A 节点线 + 星号
ax.plot(x, [0]*len(x), wA, 
        color='darkred', lw=3, 
        marker='*', ms=10, label='算法A')

# ====================== 绘制算法 B (y=1) ======================
for xi, weight in zip(x, wB):
    X = [xi, xi+0.001]
    Y = [1, 1]
    Z = [0, weight]
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[0, weight], [0, weight]])
    
    ax.plot_surface(X, Y, Z, color=cm.Blues(norm(weight)), 
                    alpha=0.8, shade=False)

# 算法B 节点线 + 星号
ax.plot(x, [1]*len(x), wB, 
        color='darkblue', lw=3, 
        marker='*', ms=10, label='算法B')

# ====================== 坐标轴 ======================
ax.set_xlabel('知识点')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20)

ax.set_ylabel('算法')
ax.set_yticks([0,1])
ax.set_yticklabels(['算法A','算法B'])

ax.set_zlabel('权重')
ax.set_zlim(0, 1)

# 颜色条
mappable = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
fig.colorbar(mappable, ax=ax, shrink=0.5, label='权重（越高颜色越深）')

ax.legend()
plt.title('两种算法权重对比 3D热力面积图', fontsize=14)
ax.view_init(elev=25, azim=-55)
plt.tight_layout()
plt.show()