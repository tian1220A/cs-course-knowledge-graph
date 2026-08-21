import logging
from collections import deque
from py2neo import Graph
import heapq
import numpy as np

from confirmation.visualization import PathVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AdaptiveLearningPath")


class AdaptiveLearningPathGenerator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._validate_connection()
        self.node_cache = {}
        self.last_path_ids = None

    def _validate_connection(self):
        try:
            self.graph.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.critical(f"Neo4j连接失败: {str(e)}")
            raise

    def _get_node_id(self, title: str) -> int:
        """通过title获取节点ID（支持知识点和子知识点）"""
        if title in self.node_cache:
            return self.node_cache[title]

        query = """
        MATCH (n)
        WHERE (n:KnowledgePoint OR n:SubKnowledgePoint) 
        AND (n.title = $title OR n.description = $title)
        RETURN id(n) as nid LIMIT 1
        """
        result = self.graph.run(query, title=title).data()
        if not result:
            raise ValueError(f"未找到标题为'{title}'的节点")
        node_id = result[0]['nid']
        self.node_cache[title] = node_id
        return node_id

    def _build_chapter_map(self):
        """构建章节顺序映射"""
        query = "MATCH (c:Chapter) RETURN c.chapter_id, c.order ORDER BY c.order"
        chapters = self.graph.run(query).data()
        return {rec['c.chapter_id']: rec['c.order'] for rec in chapters}

    def _get_chapter_order(self, node_id: int) -> int:
        """获取节点所属章节的order"""
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        OPTIONAL MATCH (n)-[:BELONGS_TO_CHAPTER|HAS_KNOWLEDGE]-(c:Chapter)
        RETURN COALESCE(c.order, 0) AS chapter_order LIMIT 1
        """
        result = self.graph.run(query, node_id=node_id).data()
        return result[0]['chapter_order'] if result else 0

    def _get_heuristic(self, current: int, goal: int) -> float:
        """改进的启发函数：结合几何距离和PageRank系数"""
        # 获取教育权重
        pr_query = "MATCH (n) WHERE id(n) = $nid RETURN n.edu_pagerank as pr"
        current_pr = self.graph.run(pr_query, nid=current).data()[0]['pr'] or 0.01
        goal_pr = self.graph.run(pr_query, nid=goal).data()[0]['pr'] or 0.01

        # 几何距离归一化
        distance = abs(current - goal)
        max_node = max(abs(current), abs(goal), 1)
        norm_distance = distance / max_node

        # 组合启发值
        return 0.7 * norm_distance + 0.3 * (1 - goal_pr)

    def _validate_chapter_sequence(self, path: list[int]) -> list[int]:
        """优化章节顺序，确保章节不重复且顺序正确"""
        optimized_path = []
        encountered_orders = set()  # 记录已处理的章节顺序
        last_order = -1  # 跟踪当前最大章节顺序

        for node in path:
            current_order = self._get_chapter_order(node)
            is_chapter = self._is_chapter_node(node)

            # 处理章节跳跃
            if last_order != -1 and current_order > last_order + 1:
                # 插入缺失的中间章节节点
                for missing_order in range(last_order + 1, current_order):
                    if missing_order not in encountered_orders:
                        chapter_node = self._get_chapter_node(missing_order)
                        optimized_path.append(chapter_node)
                        encountered_orders.add(missing_order)
                        last_order = missing_order  # 更新为插入的章节顺序

            # 处理当前节点
            if is_chapter:
                # 章节节点：确保唯一性
                if current_order not in encountered_orders:
                    optimized_path.append(node)
                    encountered_orders.add(current_order)
                    last_order = max(last_order, current_order)
            else:
                # 知识点节点：直接添加并更新章节顺序
                optimized_path.append(node)
                if current_order > last_order:
                    last_order = current_order

        return optimized_path

    def _is_chapter_node(self, node_id: int) -> bool:
        """判断节点是否为章节节点"""
        query = "MATCH (n) WHERE id(n) = $node_id RETURN 'Chapter' IN labels(n) as is_chapter"
        result = self.graph.run(query, node_id=node_id).data()
        return result[0]['is_chapter'] if result else False

    def find_learning_path(self, start_title: str, goal_title: str) -> list[str]:
        """主入口：通过标题查找学习路径"""
        try:
            start = self._get_node_id(start_title)
            goal = self._get_node_id(goal_title)

            # A*算法核心
            open_heap = []
            heapq.heappush(open_heap, (0, start))
            came_from = {}
            g_score = {start: 0}

            while open_heap:
                current_cost, current = heapq.heappop(open_heap)

                if current == goal:
                    path = self._reconstruct_path(came_from, goal)
                    optimized_path = self._validate_chapter_sequence(path)
                    self.last_path_ids = optimized_path
                    return self._convert_to_titles(optimized_path)

                # 遍历后继节点
                query = """
                MATCH (a)-[r]->(b)
                WHERE id(a) = $current
                RETURN id(b) as neighbor, coalesce(r.weight, 1.0) as weight
                """
                for rec in self.graph.run(query, current=current):
                    neighbor = rec['neighbor']
                    edge_weight = rec['weight']

                    tentative_g = g_score[current] + edge_weight
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._get_heuristic(neighbor, goal)
                        heapq.heappush(open_heap, (f_score, neighbor))

            return []
        except Exception as e:
            logger.error(f"路径查找失败: {str(e)}", exc_info=True)
            return []

    def _reconstruct_path(self, came_from: dict, current: int) -> list[int]:
        """回溯路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _convert_to_titles(self, path: list[int]) -> list[str]:
        """动态查询并附加推荐知识点"""
        titles = []
        current_chapter = None

        for node_id in path:
            node_info = self.graph.run("""
            MATCH (n) WHERE id(n)=$node_id 
            RETURN 
                coalesce(n.title, n.description, '未命名节点') as title,
                labels(n)[0] as type
            """, node_id=node_id).data()[0]

            chap_info = self._get_chapter_info(node_id)

            # 章节节点处理
            if node_info['type'] == 'Chapter':
                current_chapter = f"第{chap_info['order']}章 {chap_info['title']}"
                titles.append(f"[章节] {current_chapter}")

                # 查询并附加前10知识点
                bridge_query = """
                MATCH (c:Chapter {order: $order})<-[]-(k)
                WHERE k:KnowledgePoint OR k:SubKnowledgePoint
                RETURN k.title as title
                ORDER BY k.edu_pagerank DESC 
                LIMIT 10
                """
                bridges = self.graph.run(bridge_query, order=chap_info['order']).data()
                if bridges:
                    titles.append(f"[系统推荐] {current_chapter} 前10核心知识点：")
                    titles.extend([f"  → {b['title']}" for b in bridges])
                continue

            # 正常知识点标注
            title_str = f"[{node_info['type']}] {node_info['title']}"
            if chap_info['order'] != -1 and current_chapter:
                title_str += f" (属于{current_chapter})"
            titles.append(title_str)

        return titles

    def _get_chapter_info(self, node_id: int) -> dict:
        """获取完整的章节信息"""
        # 先检查是否是章节节点
        node_type = self.graph.run("""
        MATCH (n) WHERE id(n)=$node_id 
        RETURN labels(n)[0] as type
        """, node_id=node_id).data()[0]['type']

        if node_type == 'Chapter':
            # 直接获取章节属性
            result = self.graph.run("""
            MATCH (c:Chapter) WHERE id(c)=$node_id
            RETURN c.order as order, c.title as title
            """, node_id=node_id).data()
        else:
            # 知识点/子知识点通过更灵活的关系查询
            result = self.graph.run("""
            MATCH (n)-[:HAS_KNOWLEDGE|BELONGS_TO_CHAPTER|CHILD_OF|PARENT_OF*..3]-(c:Chapter)
            WHERE id(n)=$node_id
            RETURN c.order as order, c.title as title 
            ORDER BY c.order DESC LIMIT 1
            """, node_id=node_id).data()

        return {
            'order': result[0]['order'] if result else -1,
            'title': result[0]['title'] if result else '未关联章节'
        } if result else {'order': -1, 'title': '未关联章节'}

    def _get_chapter_node(self, order: int) -> int:
        """获取指定order的章节节点ID"""
        query = "MATCH (c:Chapter {order: $order}) RETURN id(c) as id LIMIT 1"
        return self.graph.run(query, order=order).data()[0]['id']


# 测试用例
if __name__ == "__main__":
    generator = AdaptiveLearningPathGenerator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="123456789"
    )

    visualizer = PathVisualizer(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="123456789"
    )

    test_cases = [
        ("Hadoop架构演进与核心组件", "YARN与Kubernetes调度器对比"),
        ("HDFS读写流程与数据分块策略", "元数据版本控制机制"),
        ("Hadoop架构演进与核心组件", "背压失效场景诊断"),
        ("Shuffle阶段工作机制", "Shuffle阶段工作机制"),
        ("不存在的知识点", "Shuffle阶段工作机制"),
        ("Shuffle阶段工作机制", "不存在的知识点")
    ]

    for start, goal in test_cases:
        print(f"\n=== 测试案例：{start} → {goal} ===")
        try:
            # 前置验证（统一错误处理）
            start_id = generator._get_node_id(start)
            goal_id = generator._get_node_id(goal)

            # 检查章节有效性（改为警告而非阻断）
            start_chap = generator._get_chapter_info(start_id)
            goal_chap = generator._get_chapter_info(goal_id)
            if start_chap['order'] == -1:
                logger.warning(f"起始节点'{start}'未关联有效章节")
            if goal_chap['order'] == -1:
                logger.warning(f"目标节点'{goal}'未关联有效章节")

            path = generator.find_learning_path(start, goal)
            if path:
                print("生成路径：")
                # 调整输出顺序为章节在前，知识点在后
                chapter_nodes = [n for n in path if n.startswith("[章节]")]
                knowledge_nodes = [n for n in path if not n.startswith("[章节]")]

                # 先输出章节信息
                if chapter_nodes:
                    print("\n章节顺序:")
                    for chap in chapter_nodes:
                        print(f"  {chap.split('] ')[1]}")

                # 输出详细学习路径
                print("\n详细学习路径:")
                for i, title in enumerate(knowledge_nodes, 1):
                    print(f"{i}. {title}")

                output_filename = f"path_{start[:3]}_to_{goal[:3]}.html"
                visualizer.visualize_path(
                    generator.last_path_ids,
                    output_file=output_filename
                )
                print(f"可视化结果已保存至：{output_filename}")
            else:
                print("未找到有效路径")
        except Exception as e:
            logger.error(f"路径生成失败: {str(e)}")
            print(f"错误：{str(e)}")
