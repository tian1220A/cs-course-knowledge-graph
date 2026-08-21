import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import networkx as nx
import os
import re
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
import logging
import random

# Font settings — serif for paper clarity, larger base size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PathComparison")

# --------------------------
# Translation utility — converts Chinese DB data to English
# --------------------------
TRANSLATE_ENABLED = True          # Set to False to skip translation
TRANSLATION_CACHE = {}            # In-memory cache: {zh_text: en_text}

# Try importing a translator backend (order of preference)
_translator = None
_translator_name = "none"

# Option 1: deep-translator (most stable)
try:
    from deep_translator import GoogleTranslator
    _translator = GoogleTranslator(source='zh-CN', target='en')
    _translator_name = "deep-translator (Google)"
except ImportError:
    pass

# Option 2: googletrans
if _translator is None:
    try:
        from googletrans import Translator as GTranslator
        _translator = GTranslator()
        _translator_name = "googletrans"
    except ImportError:
        pass

# Option 3: translate
if _translator is None:
    try:
        from translate import Translator as SimpleTranslator
        _translator = SimpleTranslator(from_lang='zh', to_lang='en')
        _translator_name = "translate"
    except ImportError:
        pass

if _translator is not None:
    logger.info(f"Translation enabled via {_translator_name}")
else:
    logger.warning("No translation library found. Install one: pip install deep-translator")
    logger.warning("Chinese text will be displayed as-is.")


def is_chinese(text):
    """Check if a string contains Chinese characters."""
    if not text:
        return False
    return bool(re.search(r'[一-鿿]', text))


def translate_text(text, cache=True):
    """
    Translate Chinese text to English.
    - Caches results to avoid repeated API calls.
    - Falls back to original text on failure.
    - Skips translation if TRANSLATE_ENABLED is False or text is not Chinese.
    """
    if not text or not isinstance(text, str):
        return text

    # Skip if already non-Chinese
    if not is_chinese(text):
        return text

    if not TRANSLATE_ENABLED or _translator is None:
        return text

    # Check cache
    if cache and text in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[text]

    try:
        if _translator_name == "deep-translator (Google)":
            result = _translator.translate(text)
        elif _translator_name == "googletrans":
            result = _translator.translate(text, src='zh-CN', dest='en').text
        elif _translator_name == "translate":
            result = _translator.translate(text)
        else:
            return text

        if cache:
            TRANSLATION_CACHE[text] = result
        return result
    except Exception as e:
        logger.warning(f"Translation failed for '{text[:30]}...': {e}")
        return text  # Fallback to original


def translate_node_info(node_info, key, default='Unknown'):
    """Get a node info field and translate it if needed."""
    value = node_info.get(key, default)
    if not value:
        return default
    return translate_text(str(value))


def translate_batch(texts):
    """
    Pre-translate a batch of Chinese texts. Useful to warm the cache
    before visualization so the chart renders quickly.
    """
    if not TRANSLATE_ENABLED or _translator is None:
        return
    for text in texts:
        if text and is_chinese(str(text)):
            translate_text(str(text))
            time.sleep(0.05)  # Small delay to avoid rate-limiting

# Repeat count for tests
REPEAT_TIMES = 1

# Neo4j connection info
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "123456789"
}

# Path color config (vivid colors)
PATH_COLORS = {
    "Dijkstra": "#FF4444",    # Bright red
    "Enhanced1": "#0099FF",   # Bright blue (most prominent)
    "RippleNet": "#33CC33"    # Bright green
}

# Path shape config
PATH_MARKERS = {
    "Dijkstra": 'o',  # Circle
    "Enhanced1": '*',  # Star
    "RippleNet": 's'   # Square
}

# Path visual weight
PATH_WEIGHTS = {
    "Dijkstra": 1.8,
    "Enhanced1": 2.0,
    "RippleNet": 1.5
}

# Edge offset for visual separation
EDGE_OFFSET = 0.03

# Path node spacing parameters
PATH_NODE_SPACING = 1.0
PATH_LAYOUT_FORCE = 1.5

# Output directory
SAVE_DIR = "path_visualization_results"

# Custom test case config
CUSTOM_TEST_CASES = [
    {"start": 374, "goal": 641, "desc": "Custom Test Case 1"},
    {"start": 100, "goal": 200, "desc": "Custom Test Case 2"},
]

# Uniform node size for all path nodes
UNIFORM_NODE_SIZE = 800

# Max characters for node title labels (longer titles are truncated)
MAX_TITLE_LENGTH = 20

# --------------------------
# Initialize algorithm instances
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
# Get custom test cases
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
            logger.warning(f"Node not found in case '{case['desc']}' (start_exists={start_exists}, goal_exists={goal_exists})")
            continue

        try:
            if G.is_directed():
                path = nx.shortest_path(G, source=start, target=goal)
            else:
                path = nx.shortest_path(G, source=start, target=goal)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning(f"No path from {start} to {goal} in case '{case['desc']}'")
            continue

        start_info = base_algorithm.node_info.get(start, {})
        goal_info = base_algorithm.node_info.get(goal, {})

        start_title_raw = start_info.get('title', f"Node{start}") or f"Node{start}"
        goal_title_raw = goal_info.get('title', f"Node{goal}") or f"Node{goal}"
        # Translate Chinese node titles to English
        start_title = translate_text(str(start_title_raw))[:20]
        goal_title = translate_text(str(goal_title_raw))[:20]

        valid_test_cases.append({
            "start": start,
            "goal": goal,
            "desc": f"{case['desc']} ({start_title} → {goal_title})"
        })

    if not valid_test_cases:
        logger.warning("No valid custom test cases, using defaults")
        valid_test_cases = [{"start": 374, "goal": 641, "desc": "Default Test Case"}]

    logger.info(f"Using {len(valid_test_cases)} custom test case(s)")
    return valid_test_cases

# --------------------------
# Interactive test case input
# --------------------------
def get_interactive_test_cases(algorithms):
    base_algorithm = algorithms["Dijkstra"]
    G = base_algorithm.G

    print("\n=== Interactive Test Case Configuration ===")
    print(f"Graph has {G.number_of_nodes()} nodes total")

    test_cases = []

    while True:
        try:
            start_input = input("\nEnter start node ID (or press Enter to finish): ")
            if not start_input:
                break

            start = int(start_input)
            if start not in G.nodes:
                print(f"Node {start} does not exist!")
                continue

            goal_input = input("Enter goal node ID: ")
            if not goal_input:
                continue

            goal = int(goal_input)
            if goal not in G.nodes:
                print(f"Node {goal} does not exist!")
                continue

            desc = input("Enter test case description (optional): ") or f"Custom Test ({start}→{goal})"

            try:
                if G.is_directed():
                    path = nx.shortest_path(G, source=start, target=goal)
                else:
                    path = nx.shortest_path(G, source=start, target=goal)
                print(f"✓ Found path from {start} to {goal}, length={len(path)-1}")
            except nx.NetworkXNoPath:
                print(f"✗ No path from {start} to {goal}")
                if input("Use this test case anyway? (y/N) ").lower() != 'y':
                    continue

            start_info = base_algorithm.node_info.get(start, {})
            goal_info = base_algorithm.node_info.get(goal, {})

            start_title_raw = start_info.get('title', f"Node{start}") or f"Node{start}"
            goal_title_raw = goal_info.get('title', f"Node{goal}") or f"Node{goal}"
            start_title = translate_text(str(start_title_raw))[:20]
            goal_title = translate_text(str(goal_title_raw))[:20]

            test_cases.append({
                "start": start,
                "goal": goal,
                "desc": f"{desc} ({start_title} → {goal_title})"
            })

        except ValueError:
            print("Please enter a valid node ID (number)")
        except Exception as e:
            print(f"Error: {e}")

    if not test_cases:
        print("Using default test case")
        test_cases = [{"start": 374, "goal": 641, "desc": "Default Test Case"}]

    return test_cases

# --------------------------
# Execute path generation
# --------------------------
def generate_paths(algorithms, test_cases):
    paths = {}

    for case in test_cases:
        print(f"\n=== Test Case: {case['desc']} ({case['start']}→{case['goal']}) ===")

        for alg_name, algorithm in algorithms.items():
            start_exists = case["start"] in algorithm.G.nodes
            goal_exists = case["goal"] in algorithm.G.nodes
            if not start_exists or not goal_exists:
                paths[alg_name] = {"path": None, "cost": float('inf'), "algorithm": algorithm}
                print(f"  {alg_name}: Node not found (start_exists={start_exists}, goal_exists={goal_exists})")
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
                print(f"  {alg_name}: Path found, cost={cost:.2f}, nodes={len(path) if path else 0}")
            except Exception as e:
                print(f"  {alg_name}: Failed - {str(e)}")
                paths[alg_name] = {"path": None, "cost": float('inf'), "algorithm": algorithm, "case": case}

    print("\n=== Path Generation Summary ===")
    for alg_name, data in paths.items():
        path = data["path"]
        if path and len(path) >= 2:
            print(f"{alg_name}: Valid path, nodes={len(path)}, cost={data['cost']:.2f}")
        else:
            print(f"{alg_name}: Invalid path ({path})")
    return paths

# --------------------------
# ✅ Key function: filter out BLOOM=0 nodes
# --------------------------
def clean_path_remove_bloom_zero(original_path, node_info_dict):
    """
    Clean path: auto-skip chapter nodes with Bloom=0
    """
    cleaned = []
    for node_id in original_path:
        bloom = node_info_dict.get(node_id, {}).get('bloom_level', 1)
        if bloom != 0:  # Keep only non-zero bloom nodes
            cleaned.append(node_id)
    return cleaned

# --------------------------
# Calculate edge offset positions
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
# Optimize path node layout
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
# Multi-path knowledge graph visualization (optimized: skip Bloom=0 nodes)
# --------------------------
def visualize_multi_path_knowledge_graph(algorithms, paths, case_desc, case_start, case_goal):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    base_algorithm = algorithms["Dijkstra"]
    G = base_algorithm.G
    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty, cannot visualize")
        return

    # Prepare path data
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
        
        # ✅ Core: clean path, skip Bloom=0 nodes
        if path:
            path = clean_path_remove_bloom_zero(path, algo.node_info)

        path_list.append(path)
        path_labels.append(alg_name if path else f"{alg_name} (No Path)")
        path_colors.append(PATH_COLORS[alg_name] if path else "#cccccc")
        path_markers.append(PATH_MARKERS[alg_name])
        path_weights.append(PATH_WEIGHTS.get(alg_name, 1.5))

        if path:
            for node in path:
                node_path_mapping[node].append(alg_name)

    # Prepare node style data
    node_types = {n['label'] for n in base_algorithm.node_info.values()}
    node_list = list(G.nodes())

    # === Pre-translate all Chinese node data (batch warm-up) ===
    if TRANSLATE_ENABLED and _translator is not None:
        logger.info("Pre-translating Chinese node titles/chapters for visualization...")
        texts_to_translate = set()
        for nid in node_list:
            info = base_algorithm.node_info.get(nid, {})
            title = str(info.get('title', '') or '')
            chapter = str(info.get('chapter_id', '') or '')
            if is_chinese(title):
                texts_to_translate.add(title)
            if is_chinese(chapter):
                texts_to_translate.add(chapter)
        if texts_to_translate:
            translate_batch(texts_to_translate)
            logger.info(f"Pre-translated {len(texts_to_translate)} unique Chinese strings")
    # =============================================================

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color_map = ListedColormap(base_colors[:len(node_types)])

    bloom_cmap = LinearSegmentedColormap.from_list(
        'bloom_cmap', ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c']
    )

    node_details = {}
    node_size = []

    for nid in node_list:
        info = base_algorithm.node_info.get(nid, {})
        type_idx = list(node_types).index(info['label']) if info.get('label') in node_types else 0
        base_color = color_map(type_idx)

        edu_pr = base_algorithm.node_attrs.get(nid, {}).get('edu_pr', 0.0)
        bloom_level = info.get('bloom_level', 0)
        size = 300 + 800 * edu_pr + 150 * bloom_level
        node_size.append(max(200, size))

        # Translate Chinese DB values to English for display
        raw_title = info.get('title', '') or ''
        raw_chapter = info.get('chapter_id', '') or ''
        translated_title = translate_text(str(raw_title)) if raw_title else 'Unknown'
        translated_chapter = translate_text(str(raw_chapter)) if raw_chapter else 'Unknown'

        node_details[nid] = {
            'title': translated_title if translated_title else 'Unknown',
            'chapter': translated_chapter if translated_chapter else 'Unknown',
            'bloom': bloom_level,
            'base_color': base_color,
            'size': size
        }

    # Prepare edge style data
    edge_styles = {}
    edge_colors = {}
    for u, v, data in G.edges(data=True):
        rel_type = data.get('rel_type', 'Unknown')
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

    # Compute initial layout
    pos = nx.spring_layout(G, k=0.5, iterations=200, seed=42, scale=2)

    # Optimize path node layout
    pos = optimize_path_layout(G, paths, pos)

    # Create figure
    plt.figure(figsize=(24, 20), dpi=600)
    ax = plt.gca()

    # Set background color
    ax.set_facecolor('#F8F9FA')
    plt.gcf().patch.set_facecolor('white')

    # Collect all path nodes and edges
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

    # 1. Draw non-path edges
    nx.draw_networkx_edges(
        G, pos, edgelist=non_path_edges,
        edge_color=[edge_colors.get(e, '#7f7f7f') for e in non_path_edges],
        style=[edge_styles.get(e, 'dashdot') for e in non_path_edges],
        width=0.8, alpha=0.2, ax=ax
    )

    # 2. Draw non-path nodes
    non_path_nodes_sorted = sorted(non_path_nodes, key=lambda x: node_details[x]['size'], reverse=True)
    non_path_sizes = [node_details[n]['size'] * 0.5 for n in non_path_nodes_sorted]
    non_path_colors = [node_details[n]['base_color'] for n in non_path_nodes_sorted]

    nx.draw_networkx_nodes(
        G, pos, nodelist=non_path_nodes_sorted,
        node_color=non_path_colors,
        node_size=non_path_sizes,
        alpha=0.15, ax=ax
    )

    # 3. Draw path edges
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

    # 4. Draw path nodes + sequence numbers (duplicate nodes auto-offset)
    node_drawing_order = [0, 1, 2]

    for path_idx in node_drawing_order:
        path = path_list[path_idx]
        color = path_colors[path_idx]
        marker = path_markers[path_idx]
        weight = path_weights[path_idx]

        if not path or len(path) < 2:
            continue

        # Three algorithms use fixed offset directions to avoid overlap
        if path_idx == 0:
            # Dijkstra (red) → top-left offset
            off_x, off_y = -0.06,  0.06
        elif path_idx == 1:
            # Enhanced1 (blue) → bottom offset
            off_x, off_y =  0.00, -0.08
        else:
            # RippleNet (green) → top-right offset
            off_x, off_y =  0.06,  0.06

        # Only draw nodes on this path
        for step_idx, nid in enumerate(path):
            if nid not in pos:
                continue

            x, y = pos[nid]
            px = x + off_x
            py = y + off_y

            # Draw node
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

            # Glow effect
            ax.scatter(
                px, py,
                s=UNIFORM_NODE_SIZE * 1.3,
                c=color,
                marker=marker,
                alpha=0.25,
                edgecolors='none',
                zorder=19 + path_idx
            )

            # Path sequence number
            ax.text(
                px, py + 0.08,
                str(step_idx + 1),
                fontsize=16,
                fontweight='bold',
                color=color,
                ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=color, alpha=0.9),
                zorder=25 + path_idx
            )
    # 5. Draw labels
    displayed_nodes = set()

    for nid in all_path_nodes:
        if nid in displayed_nodes:
            continue

        details = node_details.get(nid, {})
        x, y = pos[nid]

        # Node title — truncate if too long
        raw_title = details.get('title', 'Unknown')
        if len(raw_title) > MAX_TITLE_LENGTH:
            display_title = raw_title[:MAX_TITLE_LENGTH - 1] + '…'
        else:
            display_title = raw_title

        ax.text(
            x, y + 0.28,
            display_title,
            fontsize=20, color='black', fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.95, pad=4, boxstyle='round,pad=0.5', edgecolor='gray'),
            zorder=100
        )

        # Chapter & Bloom info — truncate chapter name
        raw_chapter = details.get('chapter', 'Unknown')
        if len(str(raw_chapter)) > 15:
            short_chapter = str(raw_chapter)[:14] + '…'
        else:
            short_chapter = raw_chapter
        info_text = f"Ch: {short_chapter} | Bloom: {details.get('bloom', 0)}"
        ax.text(
            x, y - 0.28,
            info_text,
            fontsize=15, ha='center', va='top', color='darkblue',
            bbox=dict(facecolor='white', alpha=0.9, pad=3, boxstyle='round,pad=0.3'),
            zorder=99
        )

        path_names = node_path_mapping.get(nid, [])
        if len(path_names) > 1:
            path_text = " | ".join(path_names)
            ax.text(
                x, y - 0.38,
                f"Paths: {path_text}",
                fontsize=14, ha='center', va='top', color='gray',
                bbox=dict(facecolor='white', alpha=0.8, pad=2),
                zorder=98
            )

        displayed_nodes.add(nid)

    # Start/Goal node markers
    if case_start in pos:
        x, y = pos[case_start]
        ax.scatter(x, y, s=3000, marker='*', c='gold', alpha=0.95,
                  edgecolors='darkorange', linewidths=6, zorder=120)
        ax.text(x, y + 0.5, 'START', fontsize=24, fontweight='bold',
               ha='center', color='darkred',
               bbox=dict(facecolor='white', alpha=0.9, pad=5, boxstyle='round,pad=0.5'),
               zorder=121)
        ax.text(x, y - 0.5, f'ID: {case_start}', fontsize=16, fontweight='bold',
               ha='center', color='black',
               bbox=dict(facecolor='white', alpha=0.8, pad=3),
               zorder=120)

    if case_goal in pos:
        x, y = pos[case_goal]
        ax.scatter(x, y, s=3000, marker='*', c='purple', alpha=0.95,
                  edgecolors='darkviolet', linewidths=6, zorder=120)
        ax.text(x, y + 0.5, 'GOAL', fontsize=24, fontweight='bold',
               ha='center', color='darkblue',
               bbox=dict(facecolor='white', alpha=0.9, pad=5, boxstyle='round,pad=0.5'),
               zorder=121)
        ax.text(x, y - 0.5, f'ID: {case_goal}', fontsize=16, fontweight='bold',
               ha='center', color='black',
               bbox=dict(facecolor='white', alpha=0.8, pad=3),
               zorder=120)

    plt.title(f'Multi-Path Knowledge Graph Comparison\n{case_desc}', fontsize=30, pad=45, fontweight='bold', color='#2C3E50')

    # Legend
    legend_elements = []
    for i, (alg_name, color, marker) in enumerate(zip(["Dijkstra (Red)", "Enhanced1 (Blue)", "RippleNet (Green)"],
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
        mpatches.Patch(color='gold', label='Start Node', alpha=0.8, ec='darkorange', lw=2),
        mpatches.Patch(color='purple', label='Goal Node', alpha=0.8, ec='darkviolet', lw=2),
        mpatches.Patch(color='#1f77b4', alpha=0.3, label='Non-path Nodes'),
        mpatches.Patch(color='#7f7f7f', alpha=0.3, label='Non-path Edges'),
    ])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=18, framealpha=0.95, fancybox=True, shadow=True)

    plt.axis('off')
    plt.tight_layout()

    save_filename = f"multi_path_comparison_{case_desc.replace(' ', '_').replace('(', '').replace(')', '').replace('→', '_')}.png"
    save_path = os.path.join(SAVE_DIR, save_filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    logger.info(f"Multi-path knowledge graph saved to: {save_path}")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    algorithms = init_algorithms()

    print("Select test case mode:")
    print("1. Use predefined custom test cases")
    print("2. Interactive test case input")
    print("3. Use random test cases")

    choice = input("\nEnter choice (1/2/3, default=1): ").strip() or "1"

    if choice == "1":
        TEST_CASES = get_custom_test_cases(algorithms)
    elif choice == "2":
        TEST_CASES = get_interactive_test_cases(algorithms)
    else:
        TEST_CASES = [{"start": 374, "goal": 641, "desc": "Default Test Case"}]

    if not TEST_CASES:
        logger.error("No test cases generated, exiting")
        exit(1)

    for case in TEST_CASES:
        paths = generate_paths(algorithms, [case])
        visualize_multi_path_knowledge_graph(algorithms, paths, case["desc"], case["start"], case["goal"])

        print(f"\n=== Detailed Path Info: {case['desc']} ===")
        for alg_name, data in paths.items():
            path = data["path"]
            if path and len(path) >= 2:
                print(f"\n{alg_name}: Valid path, nodes={len(path)}, cost={data['cost']:.2f}")
                print("Path details:")
                base_algorithm = algorithms[alg_name]
                for i, node_id in enumerate(path):
                    node_info = base_algorithm.node_info.get(node_id, {})
                    raw_title = node_info.get('title', '') or 'Unknown'
                    raw_chapter = node_info.get('chapter_id', '') or 'Unknown'
                    title_en = translate_text(str(raw_title))
                    chapter_en = translate_text(str(raw_chapter))
                    print(f"  {i+1}. {title_en} (ID: {node_id})")
                    print(f"     Chapter: {chapter_en}")
                    print(f"     Bloom Level: {node_info.get('bloom_level', 0)}")
            else:
                print(f"\n{alg_name}: Invalid path ({path})")