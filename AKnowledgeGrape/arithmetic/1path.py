import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import logging
import matplotlib.patches as mpatches

# --------------------------
# 全局样式（紧凑美观版）
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HadoopStarPath")

# 样式配置
PATH_COLOR_MAIN = "#0099FF"
PATH_COLOR_BRANCH = "#33CC33"
NODE_COLOR = "#45B7D1"
STAR_COLOR = "#FF6B6B"
TRIANGLE_COLOR = "#45B7D1"
SAVE_DIR = "path_visualization_results"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --------------------------
# 数据构建（全部英文节点+标题）
# --------------------------
def build_graph_from_your_image():
    G = nx.DiGraph()

    nodes = [
        ("HDFS_ReadWrite", {"title": "HDFS Read/Write Strategy", "type": "normal"}),
        ("Heterogeneous_Storage", {"title": "Heterogeneous Storage Comparison", "type": "triangle"}),
        ("HDFS", {"title": "Hadoop Distributed File System", "type": "triangle"}),
        ("HDFS_Federation", {"title": "HDFS Federation Architecture", "type": "normal"}),
        ("Data_Redundancy", {"title": "Data Redundancy & Fault Tolerance", "type": "normal"}),
        ("NameNode_HA", {"title": "NameNode High Availability", "type": "normal"}),
        ("Cloud_Storage", {"title": "HDFS & Cloud Storage Integration", "type": "normal"}),
        ("HBase_Model", {"title": "HBase Data Model & Structure", "type": "normal"}),
        ("Secondary_Index", {"title": "Secondary Index Solution", "type": "normal"}),
        ("Region_Split", {"title": "Region Split & Load Balance", "type": "normal"}),
        ("TimeSeries_Opt", {"title": "Time Series Data Optimization", "type": "normal"}),
        ("HBase_vs_RDBMS", {"title": "HBase & RDBMS Performance Contrast", "type": "normal"}),
        ("YARN", {"title": "YARN Resource Scheduling Framework", "type": "triangle"}),
        ("Hive_MetaData", {"title": "Hive Metadata Management", "type": "normal"}),
        ("Meta_Version", {"title": "Metadata Version Control Mechanism", "type": "triangle"}),
    ]
    G.add_nodes_from(nodes)

    edges = [
        ("HDFS_ReadWrite", "Heterogeneous_Storage", {"color": PATH_COLOR_MAIN, "style": "solid"}),
        ("Heterogeneous_Storage", "HDFS", {"color": PATH_COLOR_MAIN, "style": "solid"}),
        ("HDFS", "YARN", {"color": PATH_COLOR_MAIN, "style": "solid"}),
        ("YARN", "Hive_MetaData", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("YARN", "Meta_Version", {"color": PATH_COLOR_MAIN, "style": "solid"}),

        ("HDFS", "HDFS_Federation", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("HDFS", "Data_Redundancy", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("HDFS", "NameNode_HA", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("HDFS", "Cloud_Storage", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("YARN", "HBase_Model", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("YARN", "Secondary_Index", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("YARN", "Region_Split", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("YARN", "TimeSeries_Opt", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
        ("YARN", "HBase_vs_RDBMS", {"color": PATH_COLOR_BRANCH, "style": "dashed"}),
    ]
    G.add_edges_from(edges)

    start_node = "HDFS_ReadWrite"
    goal_node = "Meta_Version"
    return G, {n: d for n, d in nodes}, start_node, goal_node

# --------------------------
# 紧凑布局
# --------------------------
def get_star_layout(G):
    pos = {
        "HDFS": np.array([-0.18, -0.08]),
        "YARN": np.array([0.0, 0.0]),
        "HDFS_ReadWrite": np.array([-0.42, -0.04]),
        "Heterogeneous_Storage": np.array([-0.30, -0.02]),
        "Hive_MetaData": np.array([0.18, 0.05]),
        "Meta_Version": np.array([0.30, 0.03]),

        "HDFS_Federation": np.array([-0.36, -0.22]),
        "Data_Redundancy": np.array([-0.18, -0.30]),
        "NameNode_HA": np.array([0.0, -0.30]),
        "Cloud_Storage": np.array([0.18, -0.22]),
        "HBase_Model": np.array([-0.18, 0.14]),
        "Secondary_Index": np.array([-0.04, 0.20]),
        "Region_Split": np.array([0.10, 0.20]),
        "TimeSeries_Opt": np.array([0.12, -0.12]),
        "HBase_vs_RDBMS": np.array([0.22, 0.10]),
    }
    return pos

# --------------------------
# 紧凑好看的可视化
# --------------------------
def visualize_star_path(G, node_info, case_start, case_goal):
    pos = get_star_layout(G)

    plt.figure(figsize=(16, 10), dpi=300)
    ax = plt.gca()
    ax.set_facecolor('white')

    # 加大缩进，箭头完全不贴节点
    shrink_val = 12

    # 分支虚线边
    for u, v, data in G.edges(data=True):
        if data["color"] == PATH_COLOR_BRANCH:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=PATH_COLOR_BRANCH, 
                                linestyle='--', lw=1.2, alpha=0.7,
                                shrinkA=shrink_val, shrinkB=shrink_val),
                zorder=50
            )

    # 主路径实线边
    for u, v, data in G.edges(data=True):
        if data["color"] == PATH_COLOR_MAIN:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=PATH_COLOR_MAIN, 
                                lw=2, shrinkA=shrink_val, shrinkB=shrink_val),
                zorder=60
            )

    # 绘制所有节点
    for node in G.nodes():
        info = node_info[node]
        x, y = pos[node]

        if info["type"] == "star":
            ax.scatter(x, y, s=450, c=STAR_COLOR, marker='*', edgecolors='blue', linewidths=1.5, zorder=15)
        elif info["type"] == "triangle":
            ax.scatter(x, y, s=450, c=TRIANGLE_COLOR, marker='*', edgecolors='blue', linewidths=1.5, zorder=15)
        else:
            ax.scatter(x, y, s=220, c=NODE_COLOR, marker='o', alpha=0.85, zorder=10)

        ax.text(x, y + 0.03, info["title"], fontsize=10, ha='center', va='bottom', zorder=21)

    # 起点 Start（英文）
    if case_start in pos:
        x, y = pos[case_start]
        ax.scatter(x, y, s=2200, marker='*', c='gold', edgecolors='darkorange', linewidths=5, alpha=0.95, zorder=120)
        ax.text(x, y + 0.08, 'Start', fontsize=16, fontweight='bold', ha='center', color='darkred',
                bbox=dict(facecolor='white', alpha=0.9, pad=3, boxstyle='round,pad=0.3'), zorder=121)
        ax.text(x, y - 0.08, f'ID: {case_start}', fontsize=10, ha='center', color='black',
                bbox=dict(facecolor='white', alpha=0.8, pad=2), zorder=120)

    # 终点 End（英文）
    if case_goal in pos:
        x, y = pos[case_goal]
        ax.scatter(x, y, s=2200, marker='*', c='purple', edgecolors='darkviolet', linewidths=5, alpha=0.95, zorder=120)
        ax.text(x, y + 0.08, 'End', fontsize=16, fontweight='bold', ha='center', color='darkblue',
                bbox=dict(facecolor='white', alpha=0.9, pad=3, boxstyle='round,pad=0.3'), zorder=121)
        ax.text(x, y - 0.08, f'ID: {case_goal}', fontsize=10, ha='center', color='black',
                bbox=dict(facecolor='white', alpha=0.8, pad=2), zorder=120)

    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "hadoop_compact_path.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ English version graph saved")

# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    G, node_info, start_node, goal_node = build_graph_from_your_image()
    visualize_star_path(G, node_info, start_node, goal_node)