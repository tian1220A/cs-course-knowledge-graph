import logging
import heapq
import numpy as np
import matplotlib.pyplot as plt
from py2neo import Graph
import networkx as nx

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FlexibleAStarSearch")


class FlexibleAStarSearch:
    def __init__(self, neo4j_uri, user, password):
        self.graphdb = Graph(neo4j_uri, auth=(user, password))
        self.G = nx.DiGraph()
        self.node_info = {}  # 存储节点详情（标题、章节等）

    def load_graph(self):
        """加载所有边，通过权重调整降低章节顺序边的优先级"""
        self.G = nx.DiGraph()
        self.node_info = {}

        # 加载所有节点（保留完整属性）
        node_query = """
        MATCH (n) 
        WHERE n:KnowledgePoint OR n:SubKnowledgePoint OR n:Chapter
        RETURN id(n) as nid, labels(n)[0] as label, coalesce(n.title, '未命名') as title, 
               coalesce(n.chapter_id, 0) as chapter_id, coalesce(n.bloom_level, 0) as bloom_level,
               coalesce(n.kp_id, '') as kp_id, coalesce(n.sub_kp_id, '') as sub_kp_id
        """
        nodes = list(self.graphdb.run(node_query))
        for node in nodes:
            nid = node['nid']
            self.G.add_node(nid)
            self.node_info[nid] = {
                'label': node['label'],
                'title': node['title'],
                'chapter_id': node['chapter_id'],
                'bloom_level': node['bloom_level'],
                'kp_id': node['kp_id'],
                'sub_kp_id': node['sub_kp_id']
            }

        # 加载所有边，对章节顺序边设置高权重（降低优先级）
        edge_query = """
        MATCH (a)-[r]->(b)
        RETURN id(a) as src, id(b) as tgt, type(r) as rel_type, coalesce(r.weight, 1.0) as weight
        """
        edges = list(self.graphdb.run(edge_query))
        for edge in edges:
            src = edge['src']
            tgt = edge['tgt']
            rel_type = edge['rel_type']
            base_weight = edge['weight']

            # 关键调整：章节顺序边（NEXT_CHAPTER）权重翻倍，降低算法选择优先级
            if rel_type == 'NEXT_CHAPTER':
                final_weight = base_weight * 3.0  # 高权重 → 高成本 → 算法尽量避开
            # 章节内顺序边（NEXT_KNOWLEDGE/NEXT_SUB_KNOWLEDGE）权重提高50%
            elif rel_type in ['NEXT_KNOWLEDGE', 'NEXT_SUB_KNOWLEDGE']:
                final_weight = base_weight * 1.5
            # 知识点逻辑边（依赖、父子、关联）保持原权重（低成本 → 优先选择）
            else:
                final_weight = base_weight

            self.G.add_edge(src, tgt, weight=final_weight, rel_type=rel_type)

        logger.info(f"图加载完成：节点数 {self.G.number_of_nodes()}, 边数 {self.G.number_of_edges()}")

    def _get_node_chapter(self, node_id):
        """获取节点所属章节"""
        return self.node_info.get(node_id, {}).get('chapter_id', 0)

    def _heuristic(self, node, goal):
        """启发式函数：优先考虑知识点逻辑关联，弱化章节影响"""
        node_info = self.node_info.get(node, {})
        goal_info = self.node_info.get(goal, {})

        # 1. 知识难度差异（核心权重：0.4）
        bloom_diff = abs(node_info.get('bloom_level', 0) - goal_info.get('bloom_level', 0))
        
        # 2. 知识点ID关联性（如果是子知识点/知识点，优先ID关联：0.3）
        id_sim = 0
        if node_info['label'] == 'SubKnowledgePoint' and goal_info['label'] == 'SubKnowledgePoint':
            # 子知识点：父KP ID是否相同
            node_kp = node_info.get('kp_id', '')
            goal_kp = goal_info.get('kp_id', '')
            id_sim = 0 if node_kp == goal_kp else 1
        elif node_info['label'] == 'KnowledgePoint' and goal_info['label'] == 'KnowledgePoint':
            # 知识点：是否属于同一章节（弱化处理，权重0.1）
            id_sim = 0 if self._get_node_chapter(node) == self._get_node_chapter(goal) else 0.1

        # 3. 章节差异（最低权重：0.2）→ 弱化章节影响
        chapter_diff = 0 if self._get_node_chapter(node) == self._get_node_chapter(goal) else 1

        # 总启发式成本：难度差异 > ID关联 > 章节差异
        return bloom_diff * 0.4 + id_sim * 0.3 + chapter_diff * 0.2

    def a_star(self, start, goal):
        """优化后的A*算法：优先选择低权重的逻辑关联边，避开高权重的章节顺序边"""
        if start == goal:
            return [start], 0.0
        if start not in self.G or goal not in self.G:
            logger.error(f"起始节点{start}或目标节点{goal}不存在于图中")
            return None, float('inf')

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = {}  # 记录路径回溯
        g_score = {start: 0.0}  # 实际成本
        f_score = {start: self._heuristic(start, goal)}  # 实际+启发式成本
        visited = set()

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            # 找到目标，回溯路径
            if current == goal:
                return self._reconstruct_path(came_from, goal), g_score[current]

            # 遍历所有邻居，优先选择低权重边
            for neighbor in self.G.neighbors(current):
                if neighbor in visited:
                    continue
                edge_weight = self.G[current][neighbor]['weight']
                tentative_g = g_score[current] + edge_weight

                # 优先选择成本更低的路径（逻辑关联边因权重低，更容易被选中）
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        logger.warning(f"未找到从{start}到{goal}的有效路径")
        # 调试信息：输出起始节点的邻居及边类型/权重
        if start in self.G:
            neighbors = list(self.G.neighbors(start))
            logger.info(f"起始节点{start}的邻居（{len(neighbors)}个）：")
            for n in neighbors:
                rel_type = self.G[start][n]['rel_type']
                weight = self.G[start][n]['weight']
                logger.info(f"  - 邻居{n}（{self.node_info[n]['title']}）：边类型{rel_type}，权重{weight}")
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        """回溯构建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def validate_path(self, path):
        """验证路径连通性，并输出边类型"""
        valid = True
        print("\n=== 路径边详情（验证连通性）===")
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            if not self.G.has_edge(src, tgt):
                logger.warning(f"路径无效：{src} -> {tgt} 无直接边")
                valid = False
                continue
            rel_type = self.G[src][tgt]['rel_type']
            weight = self.G[src][tgt]['weight']
            print(f"  {src} -> {tgt}：边类型[{rel_type}]，权重{weight:.1f}")
        return valid

    def run_search(self, start, goal):
        """执行搜索并输出详细结果"""
        self.load_graph()
        # 节点存在性检查
        if start not in self.node_info:
            logger.error(f"起始节点{start}不存在于数据库中")
            return None, float('inf')
        if goal not in self.node_info:
            logger.error(f"目标节点{goal}不存在于数据库中")
            return None, float('inf')
        
        # 执行A*搜索
        path, total_cost = self.a_star(start, goal)
        if not path:
            logger.error("路径搜索失败")
            return None, float('inf')
        
        # 验证路径并输出
        self.validate_path(path)
        print("\n=== 最终路径结果（不强制章节顺序）===")
        print(f"  路径节点ID序列: {path}")
        print("  路径节点详情:")
        for idx, nid in enumerate(path, 1):
            detail = self.node_info[nid]
            print(f"    第{idx}步：{detail['title']}")
            print(f"        类型: {detail['label']} | 章节ID: {detail['chapter_id']} | 难度等级: {detail['bloom_level']}")
        print(f"  路径总成本: {total_cost:.4f}")
        print(f"  路径长度（节点数）: {len(path)}")
        print(f"  章节变化序列: {[self._get_node_chapter(nid) for nid in path]}")
        
        return path, total_cost


if __name__ == "__main__":
    # 初始化搜索实例
    a_star_search = FlexibleAStarSearch(
        neo4j_uri="bolt://localhost:7687",
        user="neo4j",
        password="123456789"
    )

    # 定义起止节点（替换为实际节点ID）
    start_node = 15
    goal_node = 282

    # 执行搜索
    a_star_search.run_search(start_node, goal_node)