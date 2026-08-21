import heapq
import logging
from collections import defaultdict

import pandas as pd
from py2neo import Graph
import plotly.express as px
# Dash 相关
import dash
from dash import html, dcc
import dash_cytoscape as cyto

# ------------------ 配置与日志 ------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LearningPathComparison")
logging.getLogger("py2neo").setLevel(logging.WARNING)

# ------------------ 连接 Neo4j ------------------ #
graph = Graph("bolt://localhost:7687", auth=("neo4j", "123456789"))

# ------------------ 数据加载 ------------------ #
def fetch_graph_data():
    nodes = {}
    node_query = """
    MATCH (n)
    WHERE n:Chapter OR n:KnowledgePoint OR n:SubKnowledgePoint
    OPTIONAL MATCH (n)<-[:HAS_KNOWLEDGE]-(ch:Chapter)
    OPTIONAL MATCH (n)-[:BELONGS_TO_CHAPTER]->(ch2:Chapter)
    OPTIONAL MATCH (n:SubKnowledgePoint)-[:CHILD_OF]->(parent:KnowledgePoint)
    WITH n,
         coalesce(ch.order, ch2.order, 0) AS chapter_order,
         coalesce(n.bloom_level, 1) AS bloom,
         parent.id AS parent_kp_id,
         coalesce(n.is_core_kp, false) AS is_core  
    RETURN n, chapter_order, bloom, parent_kp_id, is_core  
    """
    results = graph.run(node_query)
    for record in results:
        n = record["n"]
        key = n.get("id","") or str(n.identity)
        props = dict(n)
        props["labels"] = list(n.labels)
        props["chapter_order"] = int(record["chapter_order"] or 0)
        props["bloom_level"] = int(record["bloom"]) if key.startswith(("KP","SUB")) else 0
        props["chapter_id"] = props.get("chapter_id","0")
        props["kp_id"] = record.get("parent_kp_id") or ""
        props["is_core"] = record["is_core"]
        nodes[n.identity] = props
    logger.info(f"加载节点数: {len(nodes)}")
    return nodes

# ------------------ 构造全局学习图 ------------------ #
def build_learning_graph(nodes):
    learning_edges = defaultdict(list)
    chapters = {}
    kps_by_chapter = defaultdict(list)
    subkps_by_kp = defaultdict(list)

    for nid, props in nodes.items():
        idstr = props.get('id','').strip()
        if idstr.startswith('CH'):
            chapters[props['id']] = (nid, props)
        elif idstr.startswith('KP'):
            ch_id = props.get('chapter_id','')
            kps_by_chapter[ch_id].append((nid, props))
        elif idstr.startswith('SUB'):
            parent = props.get('kp_id','').strip()
            if parent:
                subkps_by_kp[parent].append((nid, props))

    sorted_chapters = sorted(
        chapters.items(),
        key=lambda x: int(x[1][1].get('chapter_order',0))
    )

    for idx, (chap_id, (chap_nid, chap_props)) in enumerate(sorted_chapters):
        kps = kps_by_chapter.get(chap_id, [])
        if not kps: continue
        sorted_kps = sorted(kps, key=lambda x: x[1].get('bloom_level',0))

        first_kp = sorted_kps[0][0]
        learning_edges[chap_nid].append((first_kp, 0.1))

        for i, (curr_nid, curr_p) in enumerate(sorted_kps):
            for j in range(i+1, len(sorted_kps)):
                nxt_nid = sorted_kps[j][0]
                w = max(
                    sorted_kps[j][1].get('bloom_level',0) - curr_p.get('bloom_level',0),
                    0.5
                )
                learning_edges[curr_nid].append((nxt_nid, w))

        for kp_nid, kp_p in sorted_kps:
            kp_id = kp_p.get('id','')
            subs = subkps_by_kp.get(kp_id, [])
            if subs:
                for sid, sp in subs:
                    learning_edges[kp_nid].append((sid, 0.1))
                sorted_subs = sorted(subs, key=lambda x: x[1].get('bloom_level',0))
                for i in range(len(sorted_subs)):
                    for j in range(i+1, len(sorted_subs)):
                        s_src = sorted_subs[i][0]
                        s_tgt = sorted_subs[j][0]
                        w = max(
                            nodes[s_tgt].get('bloom_level',0) - nodes[s_src].get('bloom_level',0),
                            0.3
                        )
                        learning_edges[s_src].append((s_tgt, w))

        if idx < len(sorted_chapters)-1:
            next_chap = sorted_chapters[idx+1][0]
            next_kps = kps_by_chapter.get(next_chap, [])
            last_kp = sorted_kps[-1][0]
            exit_bloom = nodes[last_kp].get('bloom_level',0)
            for nid_n, p_n in next_kps:
                if p_n.get('bloom_level',0) <= exit_bloom + 3:
                    learning_edges[last_kp].append((nid_n, 0.5))
                    break

    return learning_edges, sorted_chapters

# ------------------ 传统 A* 算法 ------------------ #
def a_star_search(graph_edges, nodes, start, goal, heuristic_func):
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {nid: float('inf') for nid in nodes}
    g_score[start] = 0
    metrics = {
        'total_cost': 0.0,
        'core_covered': set(),
        'bloom_jumps': 0,
        'nodes_visited': 0
    }
    logger.info(f"传统A*算法开始搜索：起点 {start}，终点 {goal}")

    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        metrics['nodes_visited'] += 1

        if current == goal:
            path = reconstruct_path(came_from, current, nodes)
            metrics['total_cost'] = g_score[current]
            return path, metrics

        for neighbor, weight in graph_edges.get(current, []):
            tentative_g = g_score[current] + weight

            if nodes[neighbor].get('is_core'):
                metrics['core_covered'].add(neighbor)

            bloom_diff = abs(nodes[neighbor].get('bloom_level', 0) - nodes[current].get('bloom_level', 0))
            if bloom_diff >= 2:
                metrics['bloom_jumps'] += 1

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_func(neighbor, goal, nodes)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None, metrics

# ------------------ 改进的认知 A* 算法 ------------------ #
def a_star_cognitive_search(graph_edges, nodes, start, goal, heuristic_func):
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {nid: float('inf') for nid in nodes}
    g_score[start] = 0
    metrics = {
        'total_cost': 0.0,
        'core_covered': set(),
        'bloom_jumps': 0,
        'edu_influence': 0.0,
        'nodes_visited': 0
    }

    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        metrics['nodes_visited'] += 1

        if current == goal:
            path = reconstruct_path(came_from, current, nodes)
            metrics['total_cost'] = g_score[current]
            return path, metrics

        if current_f > g_score[current] + heuristic_func(current, goal, nodes):
            continue

        for neighbor, weight in graph_edges.get(current, []):
            edu_weight = 1 + (nodes[neighbor].get("edu_pagerank", 0) * 20)
            metrics['edu_influence'] += edu_weight

            bloom_diff = nodes[neighbor].get("bloom_level", 0) - nodes[current].get("bloom_level", 0)
            penalty = bloom_diff * 3 if bloom_diff > 2 else 0
            if nodes[neighbor].get("is_core"):
                penalty *= 0.2

            continuity_bonus = 0.5 if nodes[neighbor].get("chapter_id") == nodes[current].get("chapter_id") else 0

            raw_tentative = g_score[current] + (weight * edu_weight) + penalty - continuity_bonus
            tentative_g = max(raw_tentative, 0)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_func(neighbor, goal, nodes)
                heapq.heappush(open_heap, (f_score, neighbor))

                if nodes[neighbor].get('is_core'):
                    metrics['core_covered'].add(neighbor)
                if abs(bloom_diff) >= 2:
                    metrics['bloom_jumps'] += 1

    return None, metrics

def reconstruct_path(came_from, current, nodes):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def heuristic_traditional(nid, goal, nodes):
    return abs(nodes[goal].get("bloom_level", 0) - nodes[nid].get("bloom_level", 0))

def heuristic_cognitive(nid, goal, nodes):
    edu = nodes[nid].get("edu_pagerank", 0)
    bloom_diff = abs(nodes[goal].get("bloom_level", 0) - nodes[nid].get("bloom_level", 0))
    return max(bloom_diff * 0.5 - edu * 500, 0)

# ------------------ 生成展示元素 ------------------ #
def generate_path_elements(nodes, path, learning_edges):
    from collections import defaultdict

    chap_map = {p['id']: nid for nid, p in nodes.items() if 'Chapter' in p.get('labels', [])}
    kps_by_chap = defaultdict(list)
    for nid, p in nodes.items():
        if 'KnowledgePoint' in p.get('labels', []):
            chap_id = p.get('chapter_id', '')
            kps_by_chap[chap_id].append((nid, p))
    for chap_id in kps_by_chap:
        kps_by_chap[chap_id].sort(key=lambda x: x[1].get('bloom_level', 0))

    sub_by_kp = defaultdict(list)
    for nid, p in nodes.items():
        if 'SubKnowledgePoint' in p.get('labels', []):
            parent = p.get('kp_id', '')
            sub_by_kp[parent].append((nid, p))
    for kp in sub_by_kp:
        sub_by_kp[kp].sort(key=lambda x: x[1].get('bloom_level', 0))

    elements = []
    for nid, p in nodes.items():
        labels = p.get('labels', [])
        color = 'blue' if 'Chapter' in labels else 'pink' if 'KnowledgePoint' in labels else 'green'
        elements.append({
            'data': {
                'id': str(nid),
                'label': f"{p.get('id')} Bloom:{p.get('bloom_level', 0)}",
                'color': color
            }
        })

    for chap_id, chap_nid in chap_map.items():
        for kp_nid, _ in kps_by_chap.get(chap_id, []):
            elements.append({
                'data': {'source': str(chap_nid), 'target': str(kp_nid), 'weight': 0.1}
            })

    for kp_list in kps_by_chap.values():
        for i in range(len(kp_list) - 1):
            src = kp_list[i][0]
            tgt = kp_list[i+1][0]
            elements.append({
                'data': {'source': str(src), 'target': str(tgt), 'weight': 1}
            })

    for kp_id, subs in sub_by_kp.items():
        for sid, _ in subs:
            kp_nid = next(n for n, p in nodes.items() if p.get('id') == kp_id)
            elements.append({
                'data': {'source': str(kp_nid), 'target': str(sid), 'weight': 0.1}
            })

    for subs in sub_by_kp.values():
        for i in range(len(subs) - 1):
            src = subs[i][0]
            tgt = subs[i+1][0]
            elements.append({
                'data': {'source': str(src), 'target': str(tgt), 'weight': 0.3}
            })

    return elements

def generate_cyto_elements(nodes_subset, relationships_subset, path):
    elements = []
    for nid, props in nodes_subset.items():
        node_id = str(nid)
        title = props.get("title") or props.get("description") or node_id
        color = "gray"
        labels = props.get("labels", [])
        if "Chapter" in labels:
            color = "blue"
        elif "KnowledgePoint" in labels:
            color = "pink"
        elif "SubKnowledgePoint" in labels:
            color = "green"
        detail = f"Ch:{props.get('chapter_id','未知')}\nBloom:{props.get('bloom_level',0)}\n" \
                 f"{(title[:10] + '...') if len(title) > 10 else title}"
        elements.append({
            "data": {"id": node_id, "label": detail, "color": color, "bloom": props.get("bloom_level", 0)}
        })

    for src in relationships_subset:
        for tgt, weight in relationships_subset[src]:
            elements.append({
                "data": {"source": str(src), "target": str(tgt), "weight": weight}
            })

    return elements

# ------------------ Dash 可视化构建 ------------------ #
def run_dash_app(trad_elements, cog_elements, trad_metrics=None, cog_metrics=None):
    enhanced_highlight_stylesheet = [
        {
            'selector': '.cog-edge',
            'style': {
                'line-color': '#00FF00',
                'target-arrow-color': '#00FF00',
                'curve-style': 'unbundled-bezier',
                'width': 'mapData(weight, 0, 1, 2, 8)',
                'line-style': 'dashed'
            }
        },
        {
            'selector': '.cog-node',
            'style': {
                'border-width': 4,
                'border-color': '#FFD700',
                'border-opacity': 0.9
            }
        },
        {
            'selector': '[is_core = "true"]',
            'style': {
                'border-width': 6,
                'border-color': '#FF4500',
                'shape': 'star'
            }
        }
    ]

    enhanced_common_stylesheet = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'background-color': 'data(color)',
                'width': 100,
                'height': 80,
                'font-size': '12px',
                'text-wrap': 'wrap',
                'text-max-width': '90px',
                'border-width': 2,
                'border-color': '#2c3e50'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'line-color': '#666',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#666',
                'width': 'mapData(weight, 0, 1, 1, 5)'
            }
        }
    ]

    app = dash.Dash(__name__)

    layout = [
        html.H1("增强型学习路径对比", style={'textAlign': 'center', 'padding': '20px', 'color': '#2c3e50'}),
        html.Div([
            html.Div([
                html.H3("传统路径", style={'color': '#27ae60', 'borderBottom': '3px solid #27ae60'}),
                cyto.Cytoscape(
                    id='cyto-traditional',
                    layout={'name': 'cose', 'animate': True, 'nodeRepulsion': 5000},
                    style={'height': '70vh', 'border': '2px solid #27ae60'},
                    elements=trad_elements,
                    stylesheet=enhanced_common_stylesheet
                )
            ], className="six columns"),
            html.Div([
                html.H3("优化路径", style={'color': '#00FF00', 'borderBottom': '3px solid #00FF00'}),
                cyto.Cytoscape(
                    id='cyto-cognitive',
                    layout={'name': 'cose', 'animate': True, 'nodeRepulsion': 5000},
                    style={'height': '70vh', 'border': '2px solid #00FF00'},
                    elements=cog_elements,
                    stylesheet=enhanced_common_stylesheet + enhanced_highlight_stylesheet
                )
            ], className="six columns")
        ], className="row", style={'padding': '20px'}),
        html.Div([
            dcc.Graph(
                id='metrics-comparison',
                figure=create_enhanced_chart(trad_metrics, cog_metrics)
            )
        ], style={
            'width': '85%',
            'margin': '20px auto',
            'padding': '25px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '15px',
            'boxShadow': '0 4px 6px 0 rgba(0,0,0,0.1)'
        })
    ]

    app.layout = html.Div(layout)
    # Dash 2.x+: 用 app.run() 代替 app.run_server()
    app.run(debug=True, port=8050, use_reloader=False)

# ------------------ 全局路径生成 ------------------ #
def get_global_path(learning_edges, nodes):
    chapters = {props["id"]: (nid, props) for nid, props in nodes.items() if props.get("id", "").startswith("CH")}
    sorted_chaps = sorted(chapters.items(), key=lambda x: x[1][1].get("chapter_order", 0))
    if len(sorted_chaps) < 2:
        logger.error("至少需要两个章节才能生成路径")
        return None, None

    start_chap_id = sorted_chaps[0][0]
    start_kps = [nid for nid, props in nodes.items()
                 if props.get("chapter_id") == start_chap_id and props.get("id","").startswith("KP")]
    if not start_kps:
        logger.error(f"章节 {start_chap_id} 无知识点")
        return None, None
    global_start = start_kps[0]

    end_chap_id = sorted_chaps[-1][0]
    end_kps = [nid for nid, props in nodes.items()
               if props.get("chapter_id") == end_chap_id and props.get("id","").startswith("KP")]
    if not end_kps:
        return None, None
    last_kp = end_kps[-1]
    sub_kps = [nid for nid, p in nodes.items() if p.get("kp_id") == nodes[last_kp].get("id")]
    global_goal = sub_kps[-1] if sub_kps else last_kp

    trad_path, trad_metrics = a_star_search(learning_edges, nodes, global_start, global_goal, heuristic_traditional)
    cog_path, cog_metrics = a_star_cognitive_search(
        learning_edges, nodes, global_start, global_goal, heuristic_cognitive
    )

    logger.info("\n===== 路径对比指标 =====")
    logger.info(
        f"[传统A*] 路径成本: {trad_metrics['total_cost']:.2f} | 核心节点覆盖: {len(trad_metrics['core_covered'])}"
    )
    logger.info(
        f"[认知A*] 路径成本: {cog_metrics['total_cost']:.2f} | 核心节点覆盖: {len(cog_metrics['core_covered'])}"
    )
    logger.info(f"Bloom跃迁次数: 传统={trad_metrics['bloom_jumps']} vs 认知={cog_metrics['bloom_jumps']}")
    logger.info(
        f"教育影响值: {cog_metrics['edu_influence']:.2f} | 访问节点数: 传统={trad_metrics['nodes_visited']} vs 认知={cog_metrics['nodes_visited']}"
    )

    trad_metrics['continuity'] = calculate_continuity(nodes, trad_path)
    trad_metrics['cognitive_load'] = calculate_cognitive_load(nodes, trad_path)
    cog_metrics['continuity'] = calculate_continuity(nodes, cog_path)
    cog_metrics['cognitive_load'] = calculate_cognitive_load(nodes, cog_path)

    return (trad_path, cog_path), (trad_metrics, cog_metrics)

# ------------------ 生成全局展示元素 ------------------ #
def generate_global_view(nodes, learning_edges, trad_path, cog_path):
    trad_elements = generate_path_elements(nodes, trad_path, learning_edges) if trad_path else []
    cog_elements  = generate_path_elements(nodes, cog_path, learning_edges) if cog_path else []
    return trad_elements, cog_elements

# ------------------ 指标函数 ------------------ #
def calculate_continuity(nodes, path):
    if not path: return 0
    same_chapter = 0
    prev_chap = nodes[path[0]].get('chapter_id')
    for nid in path[1:]:
        curr_chap = nodes[nid].get('chapter_id')
        if curr_chap == prev_chap:
            same_chapter += 1
        prev_chap = curr_chap
    return round(same_chapter / (len(path) - 1), 2)

def calculate_cognitive_load(nodes, path):
    if len(path) < 2: return 0
    diffs = [
        abs(nodes[path[i]].get('bloom_level', 0) - nodes[path[i-1]].get('bloom_level', 0))
        for i in range(1, len(path))
    ]
    return round(sum(diffs) / len(diffs), 2)

# ------------------ 路径元素生成 ------------------ #
def generate_enhanced_elements(nodes, path, is_cognitive=False):
    elements = []
    path_set = set(path)

    for nid, props in nodes.items():
        node_type = 'Chapter' if 'Chapter' in props.get('labels', []) else \
                    'KP' if 'KnowledgePoint' in props.get('labels', []) else 'SubKP'
        color_map = {'Chapter': '#3498db', 'KP': '#e74c3c', 'SubKP': '#2ecc71'}
        node_data = {
            'id': str(nid),
            'label': f"{props.get('id')}\nBloom:{props.get('bloom_level', 0)}",
            'color': color_map[node_type],
            'is_core': 'true' if props.get('is_core') else 'false',
            'cognitive_path': 'true' if is_cognitive else 'false'
        }
        if nid in path_set:
            node_data.update({
                'borderWidth': 6,
                'borderColor': '#FFD700' if is_cognitive else '#27ae60',
                'fontWeight': 'bold'
            })
        elements.append({'data': node_data})

    for i in range(len(path) - 1):
        src, tgt = path[i], path[i+1]
        elements.append({
            'data': {
                'source': str(src),
                'target': str(tgt),
                'weight': nodes[tgt].get('bloom_level', 1),
                'cognitive': 'true'
            }
        })

    return elements

# ------------------ 对比可视化 ------------------ #
def create_enhanced_chart(trad_metrics, cog_metrics):
    df = pd.DataFrame({
        'Metric': ['路径成本', '核心覆盖', 'Bloom跃迁', '连续性', '认知负荷'],
        '传统算法': [
            trad_metrics['total_cost'],
            len(trad_metrics['core_covered']),
            trad_metrics['bloom_jumps'],
            trad_metrics.get('continuity', 0),
            trad_metrics.get('cognitive_load', 0)
        ],
        '改进算法': [
            cog_metrics['total_cost'],
            len(cog_metrics['core_covered']),
            cog_metrics['bloom_jumps'],
            cog_metrics.get('continuity', 0),
            cog_metrics.get('cognitive_load', 0)
        ]
    })
    fig = px.bar(df, x='Metric', y=['传统算法', '改进算法'],
                 barmode='group', text_auto='.2f',
                 color_discrete_map={'传统算法': '#27ae60', '改进算法': '#00FF00'},
                 template='plotly_white')
    fig.update_layout(
        title={'text': '算法性能对比', 'font': {'size': 24}},
        xaxis_title=None,
        yaxis_title='指标值',
        hovermode="x unified",
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02}
    )
    return fig

# ------------------ 主流程 ------------------ #
def main():
    nodes = fetch_graph_data()
    learning_edges, _ = build_learning_graph(nodes)
    (trad_path, cog_path), (trad_metrics, cog_metrics) = get_global_path(learning_edges, nodes)

    # 如果需要 Dash 可视化，请启用以下两行：
    trad_elements = generate_path_elements(nodes, trad_path, learning_edges)
    run_dash_app(trad_elements, generate_path_elements(nodes, cog_path, learning_edges),
                 trad_metrics, cog_metrics)

    # 调用增强版可视化
    from visualization_enhancement import EnhancedVisualizer
    visualizer = EnhancedVisualizer(nodes, trad_path, cog_path, trad_metrics, cog_metrics)
    visualizer.show_all_visualizations()

if __name__ == "__main__":
    main()
