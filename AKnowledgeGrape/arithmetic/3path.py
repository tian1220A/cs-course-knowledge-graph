import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import networkx as nx
import os
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
import logging
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PathComparison")

# 重复测试次数
REPEAT_TIMES = 1

# Neo4j连接信息
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "123456789"
}

# 路径颜色配置（更鲜明的颜色）
PATH_COLORS = {
    "Dijkstra": "#FF4444",    # 亮红色
    "Enhanced1": "#0099FF",   # 亮蓝色（最突出）
    "RippleNet": "#33CC33"    # 亮绿色
}

# 路径形状配置
PATH_MARKERS = {
    "Dijkstra": 'o',  # 圆形
    "Enhanced1": '*',  # 正方形
    "RippleNet": 's'   # 三角形
}

# 路径视觉权重
PATH_WEIGHTS = {
    "Dijkstra": 1.8,
    "Enhanced1": 2.0,
    "RippleNet": 1.5
}

# 边错位偏移量
EDGE_OFFSET = 0.03

# 路径节点间距调整参数
PATH_NODE_SPACING = 1.0
PATH_LAYOUT_FORCE = 1.5

# 保存文件夹
SAVE_DIR = "path_visualization_results"

# 自定义测试用例配置
CUSTOM_TEST_CASES = [
    {"start": 374, "goal": 641, "desc": "自定义测试用例1"},
    {"start": 100, "goal": 200, "desc": "自定义测试用例2"},
]

# 所有路径节点统一大小
UNIFORM_NODE_SIZE = 800

# --------------------------
# 初始化算法类
# --------------------------
def init_algorithms():
    from dijkstra_analyzer import DijkstraPathAnalyzer
    from new import EnhancedCollaborativeLearning12
    from collaborative_learning import EnhancedCollaborativeLearning

    dijkstra = DijkstraPathAnalyzer(
        neo4j_uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    dijkstra.load_graph()

    collab1 = EnhancedCollaborativeLearning12(
        neo4j_uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    collab1.load_graph()

    collab2 = EnhancedCollaborativeLearning(
        neo4j_uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    collab2.load_graph()

    return {
        "Dijkstra": dijkstra,
        "Enhanced1": collab1,
        "RippleNet": collab2
    }

# --------------------------
# 获取自定义测试用例
# --------------------------
def get_custom_test_cases(algorithms):
    base_algorithm = algorithms["Dijkstra"]
    G = base_algorithm.G

    valid_test_cases = []

    for case in CUSTOM_TEST_CASES:
        start = case["start"]
        goal = case["goal"]

        start_exists = start in G.nodes
        goal_exists = goal in G.nodes

        if not start_exists or not goal_exists:
            logger.warning(f"测试用例'{case['desc']}'中的节点不存在（start={start_exists}, goal={goal_exists}）")
            continue

        try:
            if G.is_directed():
                path = nx.shortest_path(G, source=start, target=goal)
            else:
                path = nx.shortest_path(G, source=start, target=goal)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning(f"测试用例'{case['desc']}'中没有路径从{start}到{goal}")
            continue

        start_info = base_algorithm.node_info.get(start, {})
        goal_info = base_algorithm.node_info.get(goal, {})

        start_title = start_info.get('title', f"节点{start}")[:20]
        goal_title = goal_info.get('title', f"节点{goal}")[:20]

        valid_test_cases.append({
            "start": start,
            "goal": goal,
            "desc": f"{case['desc']}（{start_title} → {goal_title}）"
        })

    if not valid_test_cases:
        logger.warning("没有有效的自定义测试用例，使用默认节点")
        valid_test_cases = [{"start": 374, "goal": 641, "desc": "默认测试用例"}]

    logger.info(f"使用{len(valid_test_cases)}个自定义测试用例")
    return valid_test_cases

# --------------------------
# 交互式获取测试用例
# --------------------------
def get_interactive_test_cases(algorithms):
    base_algorithm = algorithms["Dijkstra"]
    G = base_algorithm.G

    print("\n=== 交互式测试用例配置 ===")
    print(f"图谱中共有{G.number_of_nodes()}个节点")

    test_cases = []

    while True:
        try:
            start_input = input("\n请输入起始节点ID（或按Enter结束）：")
            if not start_input:
                break

            start = int(start_input)
            if start not in G.nodes:
                print(f"节点{start}不存在！")
                continue

            goal_input = input("请输入结束节点ID：")
            if not goal_input:
                continue

            goal = int(goal_input)
            if goal not in G.nodes:
                print(f"节点{goal}不存在！")
                continue

            desc = input("请输入测试用例描述（可选）：") or f"自定义测试（{start}→{goal}）"

            try:
                if G.is_directed():
                    path = nx.shortest_path(G, source=start, target=goal)
                else:
                    path = nx.shortest_path(G, source=start, target=goal)
                print(f"✓ 找到从{start}到{goal}的路径，长度为{len(path)-1}")
            except nx.NetworkXNoPath:
                print(f"✗ 从{start}到{goal}没有路径")
                if input("是否仍然使用此测试用例？(y/N) ").lower() != 'y':
                    continue

            start_info = base_algorithm.node_info.get(start, {})
            goal_info = base_algorithm.node_info.get(goal, {})

            start_title = start_info.get('title', f"节点{start}")[:20]
            goal_title = goal_info.get('title', f"节点{goal}")[:20]

            test_cases.append({
                "start": start,
                "goal": goal,
                "desc": f"{desc}（{start_title} → {goal_title}）"
            })

        except ValueError:
            print("请输入有效的节点ID（数字）")
        except Exception as e:
            print(f"错误：{e}")

    if not test_cases:
        print("使用默认测试用例")
        test_cases = [{"start": 374, "goal": 641, "desc": "默认测试用例"}]

    return test_cases

# --------------------------
# 执行路径生成
# --------------------------
def generate_paths(algorithms, test_cases):
    paths = {}

    for case in test_cases:
        print(f"\n=== 测试用例：{case['desc']}（{case['start']}→{case['goal']}）===")

        for alg_name, algorithm in algorithms.items():
            start_exists = case["start"] in algorithm.G.nodes
            goal_exists = case["goal"] in algorithm.G.nodes
            if not start_exists or not goal_exists:
                paths[alg_name] = {"path": None, "cost": float('inf'), "algorithm": algorithm}
                print(f"  {alg_name} 算法：节点不存在（start={start_exists}, goal={goal_exists}）")
                continue

            try:
                if alg_name == "Dijkstra":
                    path, cost = algorithm.dijkstra_search(case["start"], case["goal"])
                else:
                    path, cost = algorithm.collaborative_iteration(
                        start=case["start"],
                        goal=case["goal"],
                        max_iters=20
                    )
                paths[alg_name] = {
                    "path": path,
                    "cost": cost,
                    "algorithm": algorithm,
                    "case": case
                }
                print(f"  {alg_name} 算法：路径生成成功，成本{cost:.2f}，节点数{len(path) if path else 0}")
            except Exception as e:
                print(f"  {alg_name} 算法：生成失败 - {str(e)}")
                paths[alg_name] = {"path": None, "cost": float('inf'), "algorithm": algorithm, "case": case}

    print("\n=== 路径生成结果汇总 ===")
    for alg_name, data in paths.items():
        path = data["path"]
        if path and len(path) >= 2:
            print(f"{alg_name}：有效路径，节点数{len(path)}，成本{data['cost']:.2f}")
        else:
            print(f"{alg_name}：无效路径（{path}）")
    return paths

# --------------------------
# ✅ 关键函数：过滤掉 BLOOM=0 的节点
# --------------------------
def clean_path_remove_bloom_zero(original_path, node_info_dict):
    """
    清理路径：自动跳过 Bloom=0 的章节节点
    """
    cleaned = []
    for node_id in original_path:
        bloom = node_info_dict.get(node_id, {}).get('bloom_level', 1)
        if bloom != 0:  # 只保留非0节点
            cleaned.append(node_id)
    return cleaned

# --------------------------
# 计算边的偏移位置
# --------------------------
def get_offset_pos(pos, u, v, offset_idx, path_idx):
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        return (x1, y1), (x2, y2)

    if path_idx == 0:
        nx = -dy / length
        ny = dx / length
        offset = (offset_idx + 1) * EDGE_OFFSET
    elif path_idx == 1:
        nx = dy / length
        ny = -dx / length
        offset = offset_idx * EDGE_OFFSET
    else:
        nx = dx / length
        ny = dy / length
        offset = (offset_idx - 0.5) * EDGE_OFFSET

    x1_offset = x1 + nx * offset
    y1_offset = y1 + ny * offset
    x2_offset = x2 + nx * offset
    y2_offset = y2 + ny * offset

    return (x1_offset, y1_offset), (x2_offset, y2_offset)

# --------------------------
# 优化路径节点布局
# --------------------------
def optimize_path_layout(G, paths, pos):
    path_nodes = set()
    for alg_name, data in paths.items():
        path = data.get("path", [])
        if path and len(path) >= 2:
            path_nodes.update(path)

    if not path_nodes:
        return pos

    path_edges = []
    for alg_name, data in paths.items():
        path = data.get("path", [])
        if path and len(path) >= 2:
            for i in range(len(path)-1):
                path_edges.append((path[i], path[i+1]))

    pos_optimized = pos.copy()

    iterations = 50
    for _ in range(iterations):
        for node1 in path_nodes:
            for node2 in path_nodes:
                if node1 != node2:
                    x1, y1 = pos_optimized[node1]
                    x2, y2 = pos_optimized[node2]

                    dx = x2 - x1
                    dy = y2 - y1
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < PATH_NODE_SPACING:
                        force = (PATH_NODE_SPACING - distance) * PATH_LAYOUT_FORCE / distance
                        pos_optimized[node1] = (x1 - dx * force, y1 - dy * force)
                        pos_optimized[node2] = (x2 + dx * force, y2 + dy * force)

        for u, v in path_edges:
            if u in pos_optimized and v in pos_optimized:
                x1, y1 = pos_optimized[u]
                x2, y2 = pos_optimized[v]

                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx**2 + dy**2)

                if distance > PATH_NODE_SPACING * 2:
                    force = (distance - PATH_NODE_SPACING * 2) * 0.1 / distance
                    pos_optimized[u] = (x1 + dx * force, y1 + dy * force)
                    pos_optimized[v] = (x2 - dx * force, y2 - dy * force)

    return pos_optimized

# --------------------------
# 多路径知识图谱可视化（已优化：跳过Bloom=0节点）
# --------------------------
def visualize_multi_path_knowledge_graph(algorithms, paths, case_desc, case_start, case_goal):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    base_algorithm = algorithms["Dijkstra"]
    G = base_algorithm.G
    if G.number_of_nodes() == 0:
        logger.warning("图谱为空，无法可视化")
        return

    # 准备路径数据
    path_list = []
    path_labels = []
    path_colors = []
    path_markers = []
    path_weights = []
    node_path_mapping = defaultdict(list)

    for alg_name in ["Dijkstra", "Enhanced1", "RippleNet"]:
        data = paths.get(alg_name, {"path": None})
        path = data["path"]
        algo = data["algorithm"]
        
        # ✅ 核心：清理路径，跳过 Bloom=0
        if path:
            path = clean_path_remove_bloom_zero(path, algo.node_info)

        path_list.append(path)
        path_labels.append(alg_name if path else f"{alg_name}（无路径）")
        path_colors.append(PATH_COLORS[alg_name] if path else "#cccccc")
        path_markers.append(PATH_MARKERS[alg_name])
        path_weights.append(PATH_WEIGHTS.get(alg_name, 1.5))

        if path:
            for node in path:
                node_path_mapping[node].append(alg_name)

    # 准备节点样式数据
    node_types = {n['label'] for n in base_algorithm.node_info.values()}
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color_map = ListedColormap(base_colors[:len(node_types)])

    bloom_cmap = LinearSegmentedColormap.from_list(
        'bloom_cmap', ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c']
    )

    node_details = {}
    node_size = []
    node_list = list(G.nodes())

    for nid in node_list:
        info = base_algorithm.node_info.get(nid, {})
        type_idx = list(node_types).index(info['label']) if info.get('label') in node_types else 0
        base_color = color_map(type_idx)

        edu_pr = base_algorithm.node_attrs.get(nid, {}).get('edu_pr', 0.0)
        bloom_level = info.get('bloom_level', 0)
        size = 300 + 800 * edu_pr + 150 * bloom_level
        node_size.append(max(200, size))

        node_details[nid] = {
            'title': info.get('title', '未知节点')[:12] if len(info.get('title', '')) > 12 else info.get('title', '未知节点'),
            'chapter': info.get('chapter_id', '未知'),
            'bloom': bloom_level,
            'base_color': base_color,
            'size': size
        }

    # 准备边样式数据
    edge_styles = {}
    edge_colors = {}
    for u, v, data in G.edges(data=True):
        rel_type = data.get('rel_type', '未知')
        if rel_type == 'REQUIRES_PREREQUISITE':
            edge_styles[(u, v)] = 'solid'
            edge_colors[(u, v)] = '#d62728'
        elif rel_type == 'NEXT_CHAPTER':
            edge_styles[(u, v)] = 'dashed'
            edge_colors[(u, v)] = '#2ca02c'
        elif rel_type == 'CONTAINS':
            edge_styles[(u, v)] = 'dotted'
            edge_colors[(u, v)] = '#9467bd'
        else:
            edge_styles[(u, v)] = 'dashdot'
            edge_colors[(u, v)] = '#7f7f7f'

    # 初始布局计算
    pos = nx.spring_layout(G, k=0.5, iterations=200, seed=42, scale=2)

    # 优化路径节点布局
    pos = optimize_path_layout(G, paths, pos)

    # 创建绘图
    plt.figure(figsize=(24, 20), dpi=300)
    ax = plt.gca()

    # 设置背景色
    ax.set_facecolor('#F8F9FA')
    plt.gcf().patch.set_facecolor('white')

    # 收集所有路径节点和边
    all_path_nodes = set()
    all_path_edges = defaultdict(list)
    path_node_sizes = {}

    for idx, (path, weight) in enumerate(zip(path_list, path_weights)):
        if path and len(path) >= 2:
            all_path_nodes.update(path)
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            for edge in path_edges:
                all_path_edges[edge].append(idx)

            for nid in path:
                path_node_sizes[nid] = UNIFORM_NODE_SIZE

    non_path_nodes = [n for n in G.nodes() if n not in all_path_nodes]
    non_path_edges = [(u, v) for u, v in G.edges() if (u, v) not in all_path_edges and (v, u) not in all_path_edges]

    # 1. 绘制非路径边
    nx.draw_networkx_edges(
        G, pos, edgelist=non_path_edges,
        edge_color=[edge_colors.get(e, '#7f7f7f') for e in non_path_edges],
        style=[edge_styles.get(e, 'dashdot') for e in non_path_edges],
        width=0.8, alpha=0.2, ax=ax
    )

    # 2. 绘制非路径节点
    non_path_nodes_sorted = sorted(non_path_nodes, key=lambda x: node_details[x]['size'], reverse=True)
    non_path_sizes = [node_details[n]['size'] * 0.5 for n in non_path_nodes_sorted]
    non_path_colors = [node_details[n]['base_color'] for n in non_path_nodes_sorted]

    nx.draw_networkx_nodes(
        G, pos, nodelist=non_path_nodes_sorted,
        node_color=non_path_colors,
        node_size=non_path_sizes,
        alpha=0.15, ax=ax
    )

    # 3. 绘制路径边
    path_drawing_order = [2, 1, 0]

    for path_idx in path_drawing_order:
        path = path_list[path_idx]
        if not path or len(path) < 2:
            continue

        color = path_colors[path_idx]
        weight = path_weights[path_idx]

        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_idx = 0

            start_pos, end_pos = get_offset_pos(pos, u, v, edge_idx, path_idx)

            edge_width = 1.0

            ax.annotate("",
            xy=end_pos, xycoords='data',
            xytext=start_pos, textcoords='data',
            arrowprops=dict(
                arrowstyle="->,head_width=0.5,head_length=0.8",
                color=color,
                lw=1.8,
                alpha=1.0,
                shrinkA=0,
                shrinkB=1,
                connectionstyle="arc3,rad=0"
            ),
            zorder=10 + path_idx
)

    # 4. 绘制路径节点 + 序号（重复节点自动错开偏移）
    node_drawing_order = [0, 1, 2]

    for path_idx in node_drawing_order:
        path = path_list[path_idx]
        color = path_colors[path_idx]
        marker = path_markers[path_idx]
        weight = path_weights[path_idx]

        if not path or len(path) < 2:
            continue

        # 三个算法固定偏移方向，避免重叠
        if path_idx == 0:
            # Dijkstra 红色 → 左上偏移
            off_x, off_y = -0.06,  0.06
        elif path_idx == 1:
            # Enhanced1 蓝色 → 正下偏移
            off_x, off_y =  0.00, -0.08
        else:
            # RippleNet 绿色 → 右上偏移
            off_x, off_y =  0.06,  0.06

        # 只绘制这条路径上的节点
        for step_idx, nid in enumerate(path):
            if nid not in pos:
                continue

            x, y = pos[nid]
            px = x + off_x
            py = y + off_y

            # 画节点
            ax.scatter(
                px, py,
                s=UNIFORM_NODE_SIZE,
                c=color,
                marker=marker,
                alpha=0.95,
                edgecolors='black',
                linewidths=2.5,
                zorder=20 + path_idx,
            )

            # 光晕
            ax.scatter(
                px, py,
                s=UNIFORM_NODE_SIZE * 1.3,
                c=color,
                marker=marker,
                alpha=0.25,
                edgecolors='none',
                zorder=19 + path_idx
            )

            # 路径序号
            ax.text(
                px, py + 0.08,
                str(step_idx + 1),
                fontsize=12,
                fontweight='bold',
                color=color,
                ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color, alpha=0.9),
                zorder=25 + path_idx
            )
    # 5. 绘制标签
    displayed_nodes = set()

    for nid in all_path_nodes:
        if nid in displayed_nodes:
            continue

        details = node_details.get(nid, {})
        x, y = pos[nid]

        ax.text(
            x, y + 0.25,
            details.get('title', '未知'),
            fontsize=15, color='black', fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.95, pad=3, boxstyle='round,pad=0.5', edgecolor='gray'),
            zorder=100
        )

        info_text = f"章节: {details.get('chapter', '未知')} | Bloom: {details.get('bloom', 0)}"
        ax.text(
            x, y - 0.25,
            info_text,
            fontsize=11, ha='center', va='top', color='darkblue',
            bbox=dict(facecolor='white', alpha=0.9, pad=2, boxstyle='round,pad=0.3'),
            zorder=99
        )

        path_names = node_path_mapping.get(nid, [])
        if len(path_names) > 1:
            path_text = " | ".join(path_names)
            ax.text(
                x, y - 0.35,
                f"路径: {path_text}",
                fontsize=10, ha='center', va='top', color='gray',
                bbox=dict(facecolor='white', alpha=0.8, pad=2),
                zorder=98
            )

        displayed_nodes.add(nid)

    # 起点终点标记
    if case_start in pos:
        x, y = pos[case_start]
        ax.scatter(x, y, s=3000, marker='*', c='gold', alpha=0.95,
                  edgecolors='darkorange', linewidths=6, zorder=120)
        ax.text(x, y + 0.5, '起点', fontsize=18, fontweight='bold',
               ha='center', color='darkred',
               bbox=dict(facecolor='white', alpha=0.9, pad=5, boxstyle='round,pad=0.5'),
               zorder=121)
        ax.text(x, y - 0.5, f'ID: {case_start}', fontsize=12, fontweight='bold',
               ha='center', color='black',
               bbox=dict(facecolor='white', alpha=0.8, pad=3),
               zorder=120)

    if case_goal in pos:
        x, y = pos[case_goal]
        ax.scatter(x, y, s=3000, marker='*', c='purple', alpha=0.95,
                  edgecolors='darkviolet', linewidths=6, zorder=120)
        ax.text(x, y + 0.5, '终点', fontsize=18, fontweight='bold',
               ha='center', color='darkblue',
               bbox=dict(facecolor='white', alpha=0.9, pad=5, boxstyle='round,pad=0.5'),
               zorder=121)
        ax.text(x, y - 0.5, f'ID: {case_goal}', fontsize=12, fontweight='bold',
               ha='center', color='black',
               bbox=dict(facecolor='white', alpha=0.8, pad=3),
               zorder=120)

    plt.title(f'多路径知识图谱对比分析\n{case_desc}', fontsize=24, pad=40, fontweight='bold', color='#2C3E50')

    # 图例
    legend_elements = []
    for i, (alg_name, color, marker) in enumerate(zip(["Dijkstra（红色）", "Enhanced1（蓝色）", "RippleNet（绿色）"],
                                       [PATH_COLORS["Dijkstra"], PATH_COLORS["Enhanced1"], PATH_COLORS["RippleNet"]],
                                       [PATH_MARKERS["Dijkstra"], PATH_MARKERS["Enhanced1"], PATH_MARKERS["RippleNet"]])):
        legend_elements.append(
            mpatches.Patch(
                color=color,
                label=alg_name,
                alpha=0.8,
                ec='black',
                lw=2
            )
        )

    legend_elements.extend([
        mpatches.Patch(color='gold', label='起点', alpha=0.8, ec='darkorange', lw=2),
        mpatches.Patch(color='purple', label='终点', alpha=0.8, ec='darkviolet', lw=2),
        mpatches.Patch(color='#1f77b4', alpha=0.3, label='非路径节点'),
        mpatches.Patch(color='#7f7f7f', alpha=0.3, label='非路径边'),
    ])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=14, framealpha=0.95, fancybox=True, shadow=True)

    plt.axis('off')
    plt.tight_layout()

    save_filename = f"multi_path_comparison_{case_desc.replace(' ', '_').replace('(', '').replace(')', '').replace('→', '_')}.png"
    save_path = os.path.join(SAVE_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    logger.info(f"多路径知识图谱已保存至：{save_path}")

# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    algorithms = init_algorithms()

    print("请选择测试用例模式：")
    print("1. 使用预定义的自定义测试用例")
    print("2. 交互式输入测试用例")
    print("3. 使用随机测试用例")

    choice = input("\n请输入选择（1/2/3，默认1）：").strip() or "1"

    if choice == "1":
        TEST_CASES = get_custom_test_cases(algorithms)
    elif choice == "2":
        TEST_CASES = get_interactive_test_cases(algorithms)
    else:
        TEST_CASES = [{"start": 374, "goal": 641, "desc": "默认测试用例"}]

    if not TEST_CASES:
        logger.error("未能生成测试用例，程序退出")
        exit(1)

    for case in TEST_CASES:
        paths = generate_paths(algorithms, [case])
        visualize_multi_path_knowledge_graph(algorithms, paths, case["desc"], case["start"], case["goal"])

        print(f"\n=== 详细路径信息：{case['desc']} ===")
        for alg_name, data in paths.items():
            path = data["path"]
            if path and len(path) >= 2:
                print(f"\n{alg_name}：有效路径，节点数{len(path)}，成本{data['cost']:.2f}")
                print("路径详情：")
                base_algorithm = algorithms[alg_name]
                for i, node_id in enumerate(path):
                    node_info = base_algorithm.node_info.get(node_id, {})
                    print(f"  {i+1}. {node_info.get('title', '未知节点')} (ID: {node_id})")
                    print(f"     章节: {node_info.get('chapter_id', '未知')}")
                    print(f"     Bloom层级: {node_info.get('bloom_level', 0)}")
            else:
                print(f"\n{alg_name}：无效路径（{path}）")