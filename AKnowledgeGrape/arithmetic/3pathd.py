import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import networkx as nx
import os
import logging
import random
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.gridspec import GridSpec
import copy

# 设置字体（优先英文，避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0  # 关闭最大打开图形警告

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

# 保存文件夹
SAVE_DIR = "path_analysis_results"
VISUAL_DIR = os.path.join(SAVE_DIR, "visualizations")

# 随机测试用例配置
RANDOM_TEST_CONFIG = {
    "num_cases": 1000,  
    "min_path_length": 2,  
    "max_attempts": 50,    
    "connected_only": True 
}

# 颜色配置
COLORS = {
    "Dijkstra": "#2E86AB",
    "Enhanced1": "#A23B72", 
    "RippleNet": "#F18F01",
    "background": "#F8F9FA",
    "text": "#2C3E50"
}

# 中英文标签映射（统一管理英文文本）
LABEL_MAPPING = {
    "覆盖率": "Coverage",
    "章节连贯性": "Chapter Coherence",
    "Bloom连贯性": "Bloom Coherence",
    "关系连贯性": "Relation Coherence",
    "综合平滑度": "Overall Smoothness",
    "知识点覆盖率": "Knowledge Coverage",
    "算法类型": "Algorithm Type",
    "得分": "Score",
    "平滑度维度": "Smoothness Dimensions",
    "算法多维度性能对比雷达图（含关系连贯性）": "Algorithm Multi-Dimensional Performance Radar Chart (Including Relation Coherence)",
    "路径平滑度各模块性能波形对比（含关系连贯性）": "Path Smoothness Module Performance Waveform Comparison (Including Relation Coherence)",
    "算法多维度性能对比平行坐标图（含关系连贯性）": "Algorithm Multi-Dimensional Performance Parallel Coordinates (Including Relation Coherence)",
    "算法覆盖率、平滑度与关系连贯性对比": "Algorithm Coverage, Smoothness & Relation Coherence Comparison",
    "平滑度各维度分析热力图（含关系连贯性）": "Smoothness Dimension Heatmap Analysis (Including Relation Coherence)",
    "各算法关系连贯性分布": "Relation Coherence Distribution by Algorithm",
    "各算法平均关系连贯性对比": "Average Relation Coherence Comparison by Algorithm",
    "关系连贯性详细分析": "Detailed Relation Coherence Analysis",
    "各维度平滑度分布对比": "Smoothness Distribution Comparison by Dimension",
    "算法性能综合分析报告": "Comprehensive Algorithm Performance Analysis Report",
    "核心知识点覆盖率分析": "Core Knowledge Point Coverage Analysis",
    "路径平滑度分析（整合章节/Bloom/关系连贯性）": "Path Smoothness Analysis (Integrating Chapter/Bloom/Relation Coherence)",
    "算法对比总结": "Algorithm Comparison Summary",
    "覆盖率排名": "Coverage Ranking",
    "综合平滑度排名": "Overall Smoothness Ranking",
    "关系连贯性排名": "Relation Coherence Ranking"
}

# --------------------------
# 辅助函数：计算章节连贯性
# --------------------------
def calculate_chapter_coherence(path, algorithm):
    if len(path) < 2:
        return 0
    chapter_changes = 0
    for i in range(1, len(path)):
        prev_chapter = algorithm.node_info.get(path[i-1], {}).get('chapter_id', '')
        curr_chapter = algorithm.node_info.get(path[i], {}).get('chapter_id', '')
        if prev_chapter != curr_chapter:
            chapter_changes += 1
    total_transitions = len(path) - 1
    return 1 - (chapter_changes / total_transitions) if total_transitions > 0 else 0

# --------------------------
# 辅助函数：计算Bloom层级连贯性
# --------------------------
def calculate_bloom_coherence(path, algorithm):
    if len(path) < 2:
        return 0
    bloom_scores = []
    bloom_levels = []
    for node_id in path:
        bloom_levels.append(algorithm.node_info.get(node_id, {}).get('bloom_level', 0))
    
    # 计算单步层级得分
    for i in range(1, len(bloom_levels)):
        diff = abs(bloom_levels[i] - bloom_levels[i-1])
        if diff <= 1:
            bloom_scores.append(1)
        elif diff == 2:
            bloom_scores.append(0.7)
        else:
            bloom_scores.append(0.3)
    
    # 趋势加分（逐级递增）
    trend_bonus = 0
    if len(bloom_levels) >= 3:
        increasing = all(bloom_levels[i] >= bloom_levels[i-1] for i in range(1, len(bloom_levels)))
        if increasing:
            trend_bonus = 0.1
    
    avg_bloom_score = np.mean(bloom_scores) if bloom_scores else 0
    return min(avg_bloom_score + trend_bonus, 1.0)  # 上限1.0

# --------------------------
# 辅助函数：计算边关系连贯性
# --------------------------
def calculate_relation_coherence(path, algorithm):
    """Calculate relation coherence: measure consistency and rationality of relation types in the path"""
    if len(path) < 3:  # Need at least 3 nodes to calculate relation coherence
        return 0
    
    # Get all relation types in the path
    relation_types = []
    for i in range(len(path)-1):
        edge_data = algorithm.G.get_edge_data(path[i], path[i+1], {}) or {}
        rel_type = None
        for key in ['relation_type', 'type', 'rel_type']:
            if key in edge_data:
                rel_type = edge_data[key]
                break
        relation_types.append(rel_type)
    
    if not relation_types:
        return 0
    
    # 1. Relation type consistency score
    rel_counter = Counter(relation_types)
    most_common_rel = rel_counter.most_common(1)[0][1]
    consistency_score = most_common_rel / len(relation_types)
    
    # 2. Relation transition rationality score
    transition_scores = []
    for i in range(len(relation_types)-1):
        curr_rel = relation_types[i]
        next_rel = relation_types[i+1]
        
        # Same relation type gets highest score
        if curr_rel == next_rel:
            transition_scores.append(1.0)
        # Rational relation combinations (adjust based on actual needs)
        elif (curr_rel in ['is_a', 'part_of'] and next_rel in ['has_property', 'related_to']) or \
             (curr_rel in ['prerequisite', 'depends_on'] and next_rel in ['prerequisite', 'leads_to']):
            transition_scores.append(0.8)
        # Other combinations
        else:
            transition_scores.append(0.5)
    
    transition_score = np.mean(transition_scores) if transition_scores else 0
    
    # Comprehensive score
    relation_coherence = (consistency_score * 0.6) + (transition_score * 0.4)
    
    return relation_coherence

# --------------------------
# 辅助函数：预处理图数据（统一关系类型属性）
# --------------------------
def preprocess_graph_relations(algorithm):
    """Preprocess graph data to unify relation type attribute names"""
    if hasattr(algorithm, 'G') and algorithm.G is not None:
        # Supported relation type attribute names
        RELATION_KEYS = ['relation_type', 'type', 'rel_type', 'RELATION_TYPE', 'TYPE']
        
        # Iterate all edges to unify attribute names
        for u, v, data in algorithm.G.edges(data=True):
            if data:
                # Find relation type
                rel_type = None
                for key in RELATION_KEYS:
                    if key in data:
                        rel_type = data[key]
                        break
                
                # Unified attribute name: 'relation_type'
                if rel_type:
                    data['relation_type'] = rel_type
    
    return algorithm

# --------------------------
# 初始化算法类（增强数据隔离和预处理）
# --------------------------
def init_algorithms():
    """Initialize algorithm classes (ensure data isolation and preprocessing)"""
    from dijkstra_analyzer import DijkstraPathAnalyzer
    from new import EnhancedCollaborativeLearning12
    from collaborative_learning1 import KnowledgePathPlanner

    # Create independent instances for each algorithm to ensure data isolation
    print("Initializing Dijkstra algorithm...")
    dijkstra = DijkstraPathAnalyzer(
        neo4j_uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    dijkstra.load_graph()
    dijkstra = preprocess_graph_relations(dijkstra)  # Preprocess relation types
    print(f"Dijkstra algorithm loaded nodes: {len(dijkstra.G.nodes)}")
    
    # Check edge attributes of Dijkstra
    sample_edges = list(dijkstra.G.edges(data=True))[:5]
    print(f"Dijkstra graph edge attribute examples: {sample_edges}")

    print("Initializing Enhanced1 algorithm...")
    collab1 = EnhancedCollaborativeLearning12(
        neo4j_uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    collab1.load_graph()
    collab1 = preprocess_graph_relations(collab1)  # Preprocess relation types
    print(f"Enhanced1 algorithm loaded nodes: {len(collab1.G.nodes) if hasattr(collab1, 'G') else 'N/A'}")

    print("Initializing RippleNet algorithm...")
    collab2 = KnowledgePathPlanner(
        neo4j_uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    collab2.load_graph()
    collab2 = preprocess_graph_relations(collab2)  # Preprocess relation types
    print(f"RippleNet algorithm loaded nodes: {len(collab2.G.nodes) if hasattr(collab2, 'G') else 'N/A'}")
    
    # Force deep copy to ensure complete isolation
    if hasattr(collab2, 'G'):
        collab2.G = copy.deepcopy(collab2.G)
    if hasattr(collab2, 'node_info'):
        collab2.node_info = copy.deepcopy(collab2.node_info)
    if hasattr(collab2, 'node_attrs'):
        collab2.node_attrs = copy.deepcopy(collab2.node_attrs)
    
    # Verify memory addresses (different means isolation successful)
    print(f"Dijkstra.G id: {id(dijkstra.G)}")
    print(f"RippleNet.G id: {id(collab2.G) if hasattr(collab2, 'G') else 'N/A'}")

    return {
        "Dijkstra": dijkstra,
        "Enhanced1": collab1,
        "RippleNet": collab2
    }

# --------------------------
# 从图谱中随机抽取测试节点对（适配旧版本NetworkX）
# --------------------------
def get_random_test_cases(algorithms):
    """Randomly select valid test node pairs from the graph"""
    base_algorithm = algorithms["Dijkstra"]
    G = base_algorithm.G
    
    if G.number_of_nodes() == 0:
        logger.error("Graph is empty, cannot generate random test cases")
        return []
    
    all_nodes = list(G.nodes())
    test_cases = []
    
    # Connectivity handling for directed/undirected graphs
    if RANDOM_TEST_CONFIG["connected_only"]:
        try:
            if G.is_directed():
                connected_components = list(nx.strongly_connected_components(G))
            else:
                connected_components = list(nx.connected_components(G))
            
            large_components = [comp for comp in connected_components if len(comp) >= RANDOM_TEST_CONFIG["min_path_length"]]
            
            if not large_components:
                logger.warning("No sufficiently large connected components, will use all nodes (may include disconnected nodes)")
                RANDOM_TEST_CONFIG["connected_only"] = False
        except:
            logger.warning("Connected component analysis failed, will use all nodes (may include disconnected nodes)")
            RANDOM_TEST_CONFIG["connected_only"] = False
    
    attempts = 0
    while len(test_cases) < RANDOM_TEST_CONFIG["num_cases"] and attempts < RANDOM_TEST_CONFIG["max_attempts"]:
        attempts += 1
        
        if RANDOM_TEST_CONFIG["connected_only"]:
            component = random.choice(large_components)
            component_nodes = list(component)
            
            start = random.choice(component_nodes)
            remaining_nodes = [n for n in component_nodes if n != start]
            if not remaining_nodes:
                continue
            goal = random.choice(remaining_nodes)
            
            try:
                if G.is_directed():
                    path = nx.shortest_path(G, source=start, target=goal)
                else:
                    path = nx.shortest_path(G, source=start, target=goal)
                path_length = len(path) - 1  
                if path_length < RANDOM_TEST_CONFIG["min_path_length"]:
                    continue
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            except Exception as e:
                logger.warning(f"Path length calculation failed: {e}")
                continue
        else:
            start = random.choice(all_nodes)
            remaining_nodes = [n for n in all_nodes if n != start]
            if not remaining_nodes:
                continue
            goal = random.choice(remaining_nodes)
        
        start_info = base_algorithm.node_info.get(start, {})
        goal_info = base_algorithm.node_info.get(goal, {})
        
        start_title = start_info.get('title', f"Node{start}")[:20]
        goal_title = goal_info.get('title', f"Node{goal}")[:20]
        
        test_cases.append({
            "start": start,
            "goal": goal,
            "desc": f"Random Test ({start_title} → {goal_title})"
        })
    
    if not test_cases:
        logger.warning("Failed to generate enough test cases, using default nodes")
        test_cases = [{"start": 374, "goal": 641, "desc": "Default Test Case"}]
    
    logger.info(f"Generated {len(test_cases)} random test cases")
    return test_cases

# --------------------------
# 执行路径生成（含详细调试信息）
# --------------------------
def generate_paths(algorithms, test_cases):
    all_results = {}
    
    for case_idx, case in enumerate(test_cases):
        print(f"\n=== Test Case {case_idx+1}: {case['desc']} ({case['start']}→{case['goal']}) ===")
        
        case_results = {}
        for alg_name, algorithm in algorithms.items():
            start_exists = case["start"] in algorithm.G.nodes
            goal_exists = case["goal"] in algorithm.G.nodes
            
            print(f"\n  {alg_name} Algorithm:")
            print(f"    Node Existence: start={start_exists}, goal={goal_exists}")
            
            if not start_exists or not goal_exists:
                case_results[alg_name] = {"path": None, "cost": float('inf'), "algorithm": algorithm}
                print(f"    Skipped: Node does not exist")
                continue
        
            try:
                if alg_name == "Dijkstra":
                    path, cost = algorithm.dijkstra_search(case["start"], case["goal"])
                else:
                    # Ensure different algorithms use the correct method
                    if hasattr(algorithm, 'collaborative_iteration'):
                        path, cost = algorithm.collaborative_iteration(
                            start=case["start"],
                            goal=case["goal"],
                            max_iters=20
                        )
                    elif hasattr(algorithm, 'find_path'):
                        path, cost = algorithm.find_path(case["start"], case["goal"])
                    else:
                        raise NotImplementedError(f"{alg_name} has no path generation method")
                
                case_results[alg_name] = {
                    "path": path,
                    "cost": cost,
                    "algorithm": algorithm,
                    "case_info": case
                }
                
                print(f"    Path Generation Successful: Nodes={len(path) if path else 0}, Cost={cost:.2f}")
                if path and len(path) >= 2:
                    print(f"    Path Example: {' → '.join(map(str, path[:5]))}...")
                    
                    # Display relation type information of the path
                    if len(path) >= 3:
                        print(f"    Path Relation Type Examples:")
                        for i in range(min(3, len(path)-1)):
                            edge_data = algorithm.G.get_edge_data(path[i], path[i+1], {}) or {}
                            rel_type = None
                            for key in ['relation_type', 'type', 'rel_type']:
                                if key in edge_data:
                                    rel_type = edge_data[key]
                                    break
                            print(f"      {path[i]}→{path[i+1]}: {rel_type or 'Unknown'}")
                    
            except Exception as e:
                print(f"    Generation Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                case_results[alg_name] = {"path": None, "cost": float('inf'), "algorithm": algorithm, "case_info": case}
        
        all_results[case_idx] = case_results

    print("\n=== Path Generation Results Summary ===")
    for case_idx, case_results in all_results.items():
        print(f"\nTest Case {case_idx+1}:")
        for alg_name, data in case_results.items():
            path = data["path"]
            if path and len(path) >= 2:
                print(f"  {alg_name}: Valid Path, Nodes={len(path)}, Cost={data['cost']:.2f}")
            else:
                print(f"  {alg_name}: Invalid Path ({path})")
    
    return all_results

# --------------------------
# 核心知识点覆盖率计算
# --------------------------
def calculate_knowledge_coverage(all_results, algorithms):
    """Calculate core knowledge point coverage"""
    base_algorithm = algorithms["Dijkstra"]
    all_knowledge_points = set()
    
    # Collect all knowledge points (nodes)
    for node_id in base_algorithm.G.nodes():
        node_info = base_algorithm.node_info.get(node_id, {})
        if node_info.get('title'):
            all_knowledge_points.add(node_id)
    
    total_knowledge = len(all_knowledge_points)
    coverage_results = {}
    
    # Ensure all algorithms have coverage data (even 0)
    for alg_name in algorithms.keys():
        covered_points = set()
        case_coverage = []
        
        for case_idx, case_results in all_results.items():
            if alg_name in case_results:
                data = case_results[alg_name]
                path = data["path"]
                
                if path and len(path) >= 2:
                    # Calculate coverage for this test case
                    case_covered = set(path)
                    case_coverage_ratio = len(case_covered & all_knowledge_points) / total_knowledge if total_knowledge > 0 else 0
                    case_coverage.append(case_coverage_ratio)
                    
                    # Add to total covered set
                    covered_points.update(path)
        
        # Calculate overall coverage
        overall_coverage = len(covered_points & all_knowledge_points) / total_knowledge if total_knowledge > 0 else 0
        
        coverage_results[alg_name] = {
            'overall_coverage': overall_coverage,
            'average_case_coverage': np.mean(case_coverage) if case_coverage else 0,
            'covered_points_count': len(covered_points),
            'total_points': total_knowledge,
            'case_coverage_details': case_coverage
        }
    
    return coverage_results

# --------------------------
# 优化后的路径平滑度计算（整合章节、Bloom、关系连贯性）
# --------------------------
def calculate_path_smoothness(all_results, algorithms):
    """Calculate path smoothness (integrating chapter changes, Bloom level changes, relation coherence)"""
    smoothness_results = {}
    
    # Ensure all algorithms have smoothness data (even 0)
    for alg_name in algorithms.keys():
        smoothness_scores = []
        chapter_scores = []
        bloom_scores = []
        relation_scores = []  # New: relation coherence scores
        transition_analysis = {}
        
        print(f"\n=== {alg_name} Algorithm Smoothness Calculation ===")
        
        for case_idx, case_results in all_results.items():
            if alg_name in case_results:
                data = case_results[alg_name]
                path = data["path"]
                
                if path and len(path) >= 3:  # Need at least 3 nodes to calculate complete smoothness
                    algorithm = data["algorithm"]
                    
                    # Calculate scores for each dimension
                    chapter_score = calculate_chapter_coherence(path, algorithm)
                    bloom_score = calculate_bloom_coherence(path, algorithm)
                    relation_score = calculate_relation_coherence(path, algorithm)  # New
                    
                    # Comprehensive score (weighted sum)
                    total_smoothness = (chapter_score * 0.4) + (bloom_score * 0.3) + (relation_score * 0.3)
                    
                    smoothness_scores.append(total_smoothness)
                    chapter_scores.append(chapter_score)
                    bloom_scores.append(bloom_score)
                    relation_scores.append(relation_score)  # New
                    
                    transition_analysis[case_idx] = {
                        'total_smoothness': total_smoothness,
                        'chapter_coherence': chapter_score,
                        'bloom_coherence': bloom_score,
                        'relation_coherence': relation_score,  # New
                        'path_length': len(path)
                    }
                    
                    # Print detailed scores for each case
                    if case_idx < 3:  # Print only first 3 cases
                        print(f"  Case{case_idx+1}: Overall={total_smoothness:.3f}, Chapter={chapter_score:.3f}, Bloom={bloom_score:.3f}, Relation={relation_score:.3f}")
        
        # Calculate averages
        avg_chapter = np.mean(chapter_scores) if chapter_scores else 0
        avg_bloom = np.mean(bloom_scores) if bloom_scores else 0
        avg_relation = np.mean(relation_scores) if relation_scores else 0  # New
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
        
        print(f"  Average Scores: Overall={avg_smoothness:.3f}, Chapter={avg_chapter:.3f}, Bloom={avg_bloom:.3f}, Relation={avg_relation:.3f}")
        
        smoothness_results[alg_name] = {
            'average_smoothness': avg_smoothness,
            'smoothness_std': np.std(smoothness_scores) if smoothness_scores else 0,
            'average_chapter_coherence': avg_chapter,
            'average_bloom_coherence': avg_bloom,
            'average_relation_coherence': avg_relation,  # New
            'case_smoothness': smoothness_scores,
            'case_chapter': chapter_scores,
            'case_bloom': bloom_scores,
            'case_relation': relation_scores,  # New
            'transition_details': transition_analysis
        }
    
    return smoothness_results

# --------------------------
# 可视化功能（英文版本）
# --------------------------
def create_smoothness_waveform_chart(smoothness_results):
    """Create path smoothness module waveform comparison chart (including relation coherence)"""
    # Prepare data (use English dimensions)
    dimensions = [
        LABEL_MAPPING["章节连贯性"],
        LABEL_MAPPING["Bloom连贯性"],
        LABEL_MAPPING["关系连贯性"],
        LABEL_MAPPING["综合平滑度"]
    ]
    algorithms = list(smoothness_results.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set x-axis positions
    x = np.arange(len(dimensions))
    
    # Set professional color scheme
    colors = [COLORS.get(alg, plt.cm.Set1(i/len(algorithms))) for i, alg in enumerate(algorithms)]
    
    # Plot waveform for each algorithm
    for i, alg in enumerate(algorithms):
        smooth_data = smoothness_results[alg]
        
        values = [
            smooth_data['average_chapter_coherence'],
            smooth_data['average_bloom_coherence'],
            smooth_data['average_relation_coherence'],
            smooth_data['average_smoothness']
        ]
        
        # Use smooth curve to connect points
        x_smooth = np.linspace(0, len(dimensions)-1, 100)
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(x, values, k=2)
        y_smooth = spline(x_smooth)
        
        # Plot smooth waveform
        line, = ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=4, alpha=0.8, label=alg)
        
        # Fill area under waveform
        ax.fill_between(x_smooth, y_smooth, alpha=0.2, color=colors[i])
        
        # Add markers at original data points
        ax.scatter(x, values, color=colors[i], s=150, zorder=5, 
                  edgecolors='white', linewidth=2)
        
        # Add value labels at each data point
        for j, val in enumerate(values):
            ax.annotate(f'{val:.3f}', (x[j], val), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       color=colors[i])
    
    # Set chart style
    ax.set_xlabel(LABEL_MAPPING["平滑度维度"], fontsize=14, fontweight='bold')
    ax.set_ylabel(LABEL_MAPPING["得分"], fontsize=14, fontweight='bold')
    ax.set_title(LABEL_MAPPING["路径平滑度各模块性能波形对比（含关系连贯性）"], fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Set background color
    ax.set_facecolor('#F8F9FA')
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
             ncol=len(algorithms), fontsize=12, frameon=True,
             fancybox=True, shadow=True)
    
    # Add decorative borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'smoothness_waveform_chart.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
def create_radar_chart(coverage_results, smoothness_results):
    """Create radar chart to show algorithm multi-dimensional performance comparison (including relation coherence)"""
    # 1. 修正分类：将"覆盖率"改为核心知识点覆盖率的英文（Core Knowledge Point Coverage）
    categories = [
        "Core Knowledge Point Coverage",  # 直接使用英文，无需通过LABEL_MAPPING映射
        LABEL_MAPPING["章节连贯性"],
        LABEL_MAPPING["Bloom连贯性"],
        LABEL_MAPPING["关系连贯性"],
        LABEL_MAPPING["综合平滑度"]
    ]
    N = len(categories)
    
    # Get all algorithm names
    all_algorithms = set(coverage_results.keys()) | set(smoothness_results.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    # Create angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the figure
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))
    
    # Plot radar chart for each algorithm
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#577590']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, alg in enumerate(all_algorithms):
        cov_val = coverage_results.get(alg, {}).get('overall_coverage', 0)
        smooth_data = smoothness_results.get(alg, {})
        
        chap_val = smooth_data.get('average_chapter_coherence', 0)
        bloom_val = smooth_data.get('average_bloom_coherence', 0)
        relation_val = smooth_data.get('average_relation_coherence', 0)
        total_val = smooth_data.get('average_smoothness', 0)
        
        # Handle NaN values
        values = [
            cov_val if not np.isnan(cov_val) else 0,
            chap_val if not np.isnan(chap_val) else 0,
            bloom_val if not np.isnan(bloom_val) else 0,
            relation_val if not np.isnan(relation_val) else 0,
            total_val if not np.isnan(total_val) else 0
        ]
        
        values += values[:1]  # Close the figure
        
        color = COLORS.get(alg, colors[i % len(colors)])
        
        # Plot waveform
        ax.plot(angles, values, linewidth=3, label=alg, color=color, 
                marker=markers[i % len(markers)], markersize=8, 
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        
        # Fill area
        ax.fill(angles, values, alpha=0.2, color=color)
        
        # 2. 增大数值标签字体（从fontsize=9改为11）
        for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value + 0.05, f'{value:.3f}', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=color)
    
    # 3. 增大极轴分类标签字体（从fontsize=12改为14）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    
    # 4. 增大径向刻度字体（从fontsize=10改为12）
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # 5. 增大图表标题字体（从fontsize=18改为20）
    ax.set_title(LABEL_MAPPING["算法多维度性能对比雷达图"], fontsize=20, fontweight='bold', pad=20)
    # 增大图例字体（从fontsize=12改为13）
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13, 
              frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'radar_chart.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_parallel_coordinates(coverage_results, smoothness_results):
    """Create multi-dimensional parallel coordinates chart (including relation coherence)"""
    # Prepare data (use English categories)
    categories = [
        LABEL_MAPPING["覆盖率"],
        LABEL_MAPPING["章节连贯性"],
        LABEL_MAPPING["Bloom连贯性"],
        LABEL_MAPPING["关系连贯性"],
        LABEL_MAPPING["综合平滑度"]
    ]
    
    # Get all algorithm names
    all_algorithms = set(coverage_results.keys()) | set(smoothness_results.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    # Collect data
    data = []
    for alg in all_algorithms:
        cov_val = coverage_results.get(alg, {}).get('overall_coverage', 0)
        smooth_data = smoothness_results.get(alg, {})
        
        chap_val = smooth_data.get('average_chapter_coherence', 0)
        bloom_val = smooth_data.get('average_bloom_coherence', 0)
        relation_val = smooth_data.get('average_relation_coherence', 0)
        total_val = smooth_data.get('average_smoothness', 0)
        
        # Handle NaN values
        cov_val = cov_val if not np.isnan(cov_val) else 0
        chap_val = chap_val if not np.isnan(chap_val) else 0
        bloom_val = bloom_val if not np.isnan(bloom_val) else 0
        relation_val = relation_val if not np.isnan(relation_val) else 0
        total_val = total_val if not np.isnan(total_val) else 0
        
        data.append([cov_val, chap_val, bloom_val, relation_val, total_val])
    
    if not data:
        logger.warning("No valid data to generate parallel coordinates chart")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Set axes
    x = list(range(len(categories)))
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    
    # Plot each line
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#577590']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (alg, values) in enumerate(zip(all_algorithms, data)):
        color = COLORS.get(alg, colors[i % len(colors)])
        
        # Plot line
        ax.plot(x, values, marker=markers[i % len(markers)], linewidth=3, 
                markersize=10, label=alg, color=color, 
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        
        # Add value labels at each point
        for j, val in enumerate(values):
            ax.text(j, val + 0.02, f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color=color)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Add vertical separators
    for i in x[1:]:
        ax.axvline(x=i, color='gray', alpha=0.3, linestyle='--')
    
    # Set title and legend
    ax.set_title(LABEL_MAPPING["算法多维度性能对比平行坐标图贯性"], fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # Set background color
    ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'parallel_coordinates.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comparison_bar_chart(coverage_results, smoothness_results):
    """Create coverage, smoothness and relation coherence comparison bar chart"""
    algorithms = list(coverage_results.keys())
    
    # Prepare data
    coverage_values = []
    smoothness_values = []
    relation_values = []
    valid_algorithms = []
    
    for alg in algorithms:
        cov_val = coverage_results[alg]['overall_coverage']
        smooth_data = smoothness_results.get(alg, {})
        
        sm_val = smooth_data.get('average_smoothness', 0)
        rel_val = smooth_data.get('average_relation_coherence', 0)
        
        cov_val = cov_val if not np.isnan(cov_val) else 0
        sm_val = sm_val if not np.isnan(sm_val) else 0
        rel_val = rel_val if not np.isnan(rel_val) else 0
        
        coverage_values.append(cov_val)
        smoothness_values.append(sm_val)
        relation_values.append(rel_val)
        valid_algorithms.append(alg)
    
    if not valid_algorithms:
        logger.warning("No valid data to generate bar chart")
        return
    
    x = np.arange(len(valid_algorithms))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot bar charts
    bars1 = ax.bar(x - width, coverage_values, width, label=LABEL_MAPPING["知识点覆盖率"], 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars2 = ax.bar(x, smoothness_values, width, label=LABEL_MAPPING["综合平滑度"],
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars3 = ax.bar(x + width, relation_values, width, label=LABEL_MAPPING["关系连贯性"],
                   color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars1, coverage_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars2, smoothness_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars3, relation_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Set labels
    ax.set_xlabel(LABEL_MAPPING["算法类型"], fontsize=14)
    ax.set_ylabel(LABEL_MAPPING["得分"], fontsize=14)
    ax.set_title(LABEL_MAPPING["算法覆盖率、平滑度与关系连贯性对比"], fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_algorithms, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(coverage_values), max(smoothness_values), max(relation_values)) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'comparison_bar_chart.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_smoothness_heatmap(smoothness_results):
    """Create smoothness dimension heatmap analysis (including relation coherence)"""
    algorithms = list(smoothness_results.keys())
    dimensions = [
        LABEL_MAPPING["章节连贯性"],
        LABEL_MAPPING["Bloom连贯性"],
        LABEL_MAPPING["关系连贯性"],
        LABEL_MAPPING["综合平滑度"]
    ]
    
    # Prepare data matrix
    data_matrix = []
    valid_algorithms = []
    
    for alg in algorithms:
        smooth_data = smoothness_results[alg]
        row = [
            smooth_data.get('average_chapter_coherence', 0),
            smooth_data.get('average_bloom_coherence', 0),
            smooth_data.get('average_relation_coherence', 0),
            smooth_data.get('average_smoothness', 0)
        ]
        # Handle NaN values
        row = [0 if np.isnan(x) else x for x in row]
        data_matrix.append(row)
        valid_algorithms.append(alg)
    
    if not valid_algorithms or not data_matrix:
        logger.warning("No valid data to generate heatmap")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use seaborn heatmap
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set labels
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(valid_algorithms)))
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_yticklabels(valid_algorithms, fontsize=12)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Add value annotations
    for i in range(len(valid_algorithms)):
        for j in range(len(dimensions)):
            text_color = "white" if data_matrix[i][j] > 0.7 or data_matrix[i][j] < 0.3 else "black"
            text = ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                          ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(LABEL_MAPPING["得分"], rotation=-90, va="bottom", fontsize=12)
    
    ax.set_title(LABEL_MAPPING["平滑度各维度分析热力图（含关系连贯性）"], fontsize=18, fontweight='bold', pad=20)
    fig.tight_layout()
    
    plt.savefig(os.path.join(VISUAL_DIR, 'smoothness_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_relation_coherence_detail_chart(smoothness_results):
    """Create detailed relation coherence analysis chart"""
    algorithms = list(smoothness_results.keys())
    
    # Prepare data
    data_by_algorithm = {}
    for alg in algorithms:
        smooth_data = smoothness_results[alg]
        relation_scores = smooth_data.get('case_relation', [])
        if relation_scores:
            data_by_algorithm[alg] = relation_scores
    
    if not data_by_algorithm:
        logger.warning("No relation coherence data to generate detailed chart")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Subplot 1: Box plot
    data = []
    labels = []
    colors = []
    for i, (alg, scores) in enumerate(data_by_algorithm.items()):
        data.append(scores)
        labels.append(alg)
        colors.append(COLORS.get(alg, plt.cm.Set3(i / len(data_by_algorithm))))
    
    box_plot = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add scatter points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.08, size=len(d))
        ax1.scatter(x, d, alpha=0.4, s=30, c=colors[i], edgecolor='white', linewidth=0.5)
    
    ax1.set_ylabel(LABEL_MAPPING["关系连贯性"], fontsize=12)
    ax1.set_title(LABEL_MAPPING["各算法关系连贯性分布"], fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Subplot 2: Average comparison bar chart
    avg_scores = [np.mean(scores) for scores in data]
    bars = ax2.bar(labels, avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, avg_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel(f"Average {LABEL_MAPPING['关系连贯性']}", fontsize=12)
    ax2.set_title(LABEL_MAPPING["各算法平均关系连贯性对比"], fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(avg_scores) * 1.15)
    
    plt.suptitle(LABEL_MAPPING["关系连贯性详细分析"], fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'relation_coherence_detail.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_performance_boxplot(smoothness_results):
    """Create smoothness distribution box plot (including relation coherence)"""
    algorithms = list(smoothness_results.keys())
    
    # Prepare multiple data types
    data_types = {
        LABEL_MAPPING["综合平滑度"]: 'case_smoothness',
        LABEL_MAPPING["章节连贯性"]: 'case_chapter',
        LABEL_MAPPING["Bloom连贯性"]: 'case_bloom',
        LABEL_MAPPING["关系连贯性"]: 'case_relation'
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (title, key) in enumerate(data_types.items()):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        
        for i, alg in enumerate(algorithms):
            values = smoothness_results[alg].get(key, [])
            if values:
                data.append(values)
                labels.append(alg)
                colors.append(COLORS.get(alg, plt.cm.Set3(i / len(algorithms))))
        
        if data:
            box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add scatter points
            for i, d in enumerate(data):
                x = np.random.normal(i+1, 0.08, size=len(d))
                ax1.scatter(x, d, alpha=0.4, s=30, c=colors[i], edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel(LABEL_MAPPING["得分"], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.suptitle(LABEL_MAPPING["各维度平滑度分布对比"], fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'performance_boxplot.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comprehensive_visualization(coverage_results, smoothness_results, all_results):
    """Create comprehensive visualization charts (including relation coherence)"""
    if not os.path.exists(VISUAL_DIR):
        os.makedirs(VISUAL_DIR)
    
    try:
        # 1. Smoothness module waveform chart
        create_smoothness_waveform_chart(smoothness_results)
        print("✓ Smoothness waveform chart generated (including relation coherence)")
    except Exception as e:
        logger.error(f"Failed to create smoothness waveform chart: {e}")
        print(f"Smoothness waveform chart creation failed: {str(e)}")
    
    try:
        # 2. Radar chart
        create_radar_chart(coverage_results, smoothness_results)
        print("✓ Radar chart generated (including relation coherence)")
    except Exception as e:
        logger.error(f"Failed to create radar chart: {e}")
    
    try:
        # 3. Parallel coordinates chart
        create_parallel_coordinates(coverage_results, smoothness_results)
        print("✓ Parallel coordinates chart generated (including relation coherence)")
    except Exception as e:
        logger.error(f"Failed to create parallel coordinates chart: {e}")
    
    try:
        # 4. Comparison bar chart
        create_comparison_bar_chart(coverage_results, smoothness_results)
        print("✓ Comparison bar chart generated (including relation coherence)")
    except Exception as e:
        logger.error(f"Failed to create bar chart: {e}")
    
    try:
        # 5. Smoothness heatmap
        create_smoothness_heatmap(smoothness_results)
        print("✓ Smoothness heatmap generated (including relation coherence)")
    except Exception as e:
        logger.error(f"Failed to create heatmap: {e}")
    
    try:
        # 6. Detailed relation coherence chart
        create_relation_coherence_detail_chart(smoothness_results)
        print("✓ Detailed relation coherence chart generated")
    except Exception as e:
        logger.error(f"Failed to create detailed relation coherence chart: {e}")
    
    try:
        # 7. Performance boxplot
        create_performance_boxplot(smoothness_results)
        print("✓ Performance boxplot generated (including relation coherence)")
    except Exception as e:
        logger.error(f"Failed to create boxplot: {e}")
    
    print(f"\nAll visualization charts saved to: {VISUAL_DIR}")
    print(f"Generated charts include:")
    print(f"   - smoothness_waveform_chart.png (Smoothness module waveform chart)")
    print(f"   - radar_chart.png (Multi-dimensional performance radar chart)")
    print(f"   - parallel_coordinates.png (Parallel coordinates chart)")
    print(f"   - comparison_bar_chart.png (Coverage, smoothness & relation coherence comparison)")
    print(f"   - smoothness_heatmap.png (Smoothness dimension heatmap)")
    print(f"   - relation_coherence_detail.png (Detailed relation coherence analysis)")
    print(f"   - performance_boxplot.png (Smoothness distribution boxplot)")

# --------------------------
# 综合分析和报告生成（英文版本）
# --------------------------
def generate_comprehensive_report(all_results, coverage_results, smoothness_results, algorithms):
    """Generate comprehensive analysis report (including relation coherence)"""
    print("\n" + "="*80)
    print("                      " + LABEL_MAPPING["算法性能综合分析报告"])
    print("="*80)
    
    # 1. Core knowledge point coverage analysis
    print("\n1. " + LABEL_MAPPING["核心知识点覆盖率分析"])
    print("-" * 50)
    for alg_name, coverage in coverage_results.items():
        print(f"\n{alg_name} Algorithm:")
        print(f"   - Overall Coverage: {coverage['overall_coverage']:.2%}")
        print(f"   - Average Case Coverage: {coverage['average_case_coverage']:.2%}")
        print(f"   - Covered Knowledge Points: {coverage['covered_points_count']}/{coverage['total_points']}")
        print(f"   - Valid Cases: {len(coverage['case_coverage_details'])}")
    
    # 2. Path smoothness analysis
    print("\n\n2. " + LABEL_MAPPING["路径平滑度分析（整合章节/Bloom/关系连贯性）"])
    print("-" * 50)
    for alg_name, smoothness in smoothness_results.items():
        print(f"\n{alg_name} Algorithm:")
        print(f"   - Average Overall Smoothness: {smoothness['average_smoothness']:.3f}")
        print(f"   - Chapter Coherence Score: {smoothness['average_chapter_coherence']:.3f}")
        print(f"   - Bloom Level Coherence Score: {smoothness['average_bloom_coherence']:.3f}")
        print(f"   - Edge Relation Coherence Score: {smoothness['average_relation_coherence']:.3f}")
        print(f"   - Valid Calculation Cases: {len(smoothness['case_smoothness'])}")
    
    # 3. Algorithm comparison summary
    print("\n\n3. " + LABEL_MAPPING["算法对比总结"])
    print("-" * 50)
    
    # Find optimal algorithms
    coverage_ranking = sorted(coverage_results.items(), key=lambda x: x[1]['overall_coverage'], reverse=True)
    smoothness_ranking = sorted(smoothness_results.items(), key=lambda x: x[1]['average_smoothness'], reverse=True)
    relation_ranking = sorted(smoothness_results.items(), key=lambda x: x[1]['average_relation_coherence'], reverse=True)
    
    print(f"\n{LABEL_MAPPING['覆盖率排名']}:")
    for i, (alg_name, coverage) in enumerate(coverage_ranking, 1):
        print(f"   {i}. {alg_name}: {coverage['overall_coverage']:.2%}")
    
    print(f"\n{LABEL_MAPPING['综合平滑度排名']}:")
    for i, (alg_name, smoothness) in enumerate(smoothness_ranking, 1):
        print(f"   {i}. {alg_name}: {smoothness['average_smoothness']:.3f}")
    
    print(f"\n{LABEL_MAPPING['关系连贯性排名']}:")
    for i, (alg_name, smoothness) in enumerate(relation_ranking, 1):
        print(f"   {i}. {alg_name}: {smoothness['average_relation_coherence']:.3f}")
    
    # 4. Save analysis results to file
    save_analysis_results(coverage_results, smoothness_results, all_results)
    
    # 5. Create visualization charts
    print("\n\n5. Generating all visualization charts...")
    create_comprehensive_visualization(coverage_results, smoothness_results, all_results)

# --------------------------
# 保存分析结果（英文版本）
# --------------------------
def save_analysis_results(coverage_results, smoothness_results, all_results):
    """Save analysis results to file (including relation coherence)"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Create summary dataframe (enhanced version, including relation coherence)
    summary_data = []
    # Get all algorithm names
    all_algorithms = set(coverage_results.keys()) | set(smoothness_results.keys())
    
    for alg_name in all_algorithms:
        coverage_data = coverage_results.get(alg_name, {})
        smoothness_data = smoothness_results.get(alg_name, {})
        
        summary_data.append({
            'Algorithm': alg_name,
            'Overall_Coverage': f"{coverage_data.get('overall_coverage', 0):.2%}",
            'Average_Case_Coverage': f"{coverage_data.get('average_case_coverage', 0):.2%}",
            'Covered_Points': f"{coverage_data.get('covered_points_count', 0)}/{coverage_data.get('total_points', 0)}",
            'Average_Overall_Smoothness': f"{smoothness_data.get('average_smoothness', 0):.3f}",
            'Chapter_Coherence': f"{smoothness_data.get('average_chapter_coherence', 0):.3f}",
            'Bloom_Coherence': f"{smoothness_data.get('average_bloom_coherence', 0):.3f}",
            'Relation_Coherence': f"{smoothness_data.get('average_relation_coherence', 0):.3f}",
            'Smoothness_Std': f"{smoothness_data.get('smoothness_std', 0):.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(SAVE_DIR, 'algorithm_comparison_summary.csv'), index=False, encoding='utf-8-sig')
    
    # Save detailed relation coherence data
    relation_detail_data = []
    for alg_name in smoothness_results.keys():
        smooth_data = smoothness_results[alg_name]
        relation_scores = smooth_data.get('case_relation', [])
        for i, score in enumerate(relation_scores):
            relation_detail_data.append({
                'Algorithm': alg_name,
                'Case_Index': i+1,
                'Relation_Coherence_Score': score,
                'Average_Relation_Coherence': smooth_data.get('average_relation_coherence', 0)
            })
    
    if relation_detail_data:
        relation_df = pd.DataFrame(relation_detail_data)
        relation_df.to_csv(os.path.join(SAVE_DIR, 'relation_coherence_details.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\nAnalysis results saved to: {SAVE_DIR}")
    print(f"✓ Summary data (including relation coherence) saved successfully")
    print(f"✓ Detailed relation coherence data saved (relation_coherence_details.csv)")

# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    if not os.path.exists(VISUAL_DIR):
        os.makedirs(VISUAL_DIR)

    print("Initializing algorithms...")
    algorithms = init_algorithms()
    
    print("Generating test cases...")
    TEST_CASES = get_random_test_cases(algorithms)
    
    if not TEST_CASES:
        logger.error("Failed to generate test cases, program exiting")
        exit(1)
    
    print("Generating paths...")
    all_results = generate_paths(algorithms, TEST_CASES)
    
    if all_results:
        print("\nCalculating core knowledge point coverage...")
        coverage_results = calculate_knowledge_coverage(all_results, algorithms)
        
        print("Calculating path smoothness (integrating chapter/Bloom/relation coherence)...")
        smoothness_results = calculate_path_smoothness(all_results, algorithms)
        
        # Print results summary
        print("\n=== Results Summary ===")
        print(f"Algorithms included in coverage analysis: {list(coverage_results.keys())}")
        print(f"Algorithms included in smoothness analysis: {list(smoothness_results.keys())}")
        
        print("\nGenerating comprehensive analysis report...")
        generate_comprehensive_report(all_results, coverage_results, smoothness_results, algorithms)
        
        print("\n" + "="*80)
        print("                      Analysis Completed!")
        print("="*80)
        print(f"\nResult files saved to: {SAVE_DIR}")
        print(f"Visualization charts saved to: {VISUAL_DIR}")
        print(f"\n✨ New relation coherence metric and visualization analysis added")
        print(f"✨ New detailed relation coherence chart: relation_coherence_detail.png")
        print(f"📊 All charts updated to include relation coherence dimension")
    else:
        print("No valid paths generated, cannot perform analysis")