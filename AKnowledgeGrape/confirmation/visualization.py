# visualization.py
from py2neo import Graph
from pyvis.network import Network
import webbrowser


class PathVisualizer:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def visualize_path(self, node_ids: list, output_file: str = "learning_path.html"):
        """可视化学习路径"""
        # 查询节点详细信息
        nodes = self._get_nodes_info(node_ids)
        # 创建网络对象
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
            notebook=False
        )
        # 添加原始节点和路径边
        self._add_nodes(net, nodes)
        self._add_path_edges(net, node_ids)
        # 添加章节推荐知识点
        self._add_recommended_knowledge(net, node_ids)
        # 生成并打开网页
        net.save_graph(output_file)
        webbrowser.open(output_file)

    def _get_nodes_info(self, node_ids):
        """获取节点的标签、标题和PageRank值"""
        nodes = []
        for nid in node_ids:
            query = """
            MATCH (n) WHERE id(n) = $nid
            RETURN 
                labels(n)[0] as label,
                coalesce(n.title, n.description, '未命名节点') as title,
                coalesce(n.edu_pagerank, 0.1) as pagerank
            """
            result = self.graph.run(query, nid=nid).data()[0]
            nodes.append({
                "id": nid,
                "label": result["label"],
                "title": result["title"],
                "pagerank": result["pagerank"]
            })
        return nodes

    def _add_nodes(self, net, nodes):
        """添加节点到网络图"""
        for node in nodes:
            net.add_node(
                n_id=node["id"],
                label=node["title"],
                title=self._get_node_tooltip(node),
                color=self._get_node_color(node["label"]),
                size=node["pagerank"] * 20 + 10,
                shape=self._get_node_shape(node["label"])
            )

    def _add_path_edges(self, net, node_ids):
        """添加路径顺序边"""
        for i in range(len(node_ids) - 1):
            net.add_edge(
                node_ids[i],
                node_ids[i + 1],
                color="#FF0000",
                width=2,
                arrows="to",
                title=f"学习步骤 {i + 1}"
            )

    def _add_recommended_knowledge(self, net, node_ids):
        """为每个章节节点添加推荐的知识点"""
        for node_id in node_ids:
            # 判断是否为章节节点
            is_chapter = self._is_chapter_node(node_id)
            if not is_chapter:
                continue
            # 查询推荐知识点
            recommended = self._get_recommended_knowledge(node_id)
            # 添加节点和边
            for rec_id in recommended:
                # 确保知识点节点已存在
                if not any(n['id'] == rec_id for n in net.nodes):
                    rec_node = self._get_nodes_info([rec_id])[0]
                    self._add_nodes(net, [rec_node])
                # 添加章节到知识点的边
                net.add_edge(
                    node_id,
                    rec_id,
                    color="#00FF00",
                    title="推荐知识点",
                    arrows="to",
                    dashes=True
                )

    def _is_chapter_node(self, node_id: int) -> bool:
        """判断节点是否为章节节点"""
        query = "MATCH (n) WHERE id(n)=$node_id RETURN 'Chapter' IN labels(n) as is_chapter"
        result = self.graph.run(query, node_id=node_id).data()
        return result[0]['is_chapter'] if result else False

    def _get_recommended_knowledge(self, chapter_id: int) -> list[int]:
        """获取该章节的前10知识点节点ID"""
        query = """
        MATCH (c:Chapter)-[]->(k)
        WHERE id(c) = $chapter_id 
        AND (k:KnowledgePoint OR k:SubKnowledgePoint)
        RETURN id(k) as kid
        ORDER BY k.edu_pagerank DESC 
        LIMIT 10
        """
        result = self.graph.run(query, chapter_id=chapter_id).data()
        return [rec['kid'] for rec in result]

    def _get_node_tooltip(self, node):
        """生成节点悬停提示"""
        return (
            f"类型: {node['label']}\n"
            f"PageRank: {node['pagerank']:.2f}\n"
            f"ID: {node['id']}"
        )

    def _get_node_color(self, label_type: str) -> str:
        """根据节点类型设置颜色"""
        colors = {
            "Chapter": "#FF6B6B",  # 红色
            "KnowledgePoint": "#4ECDC4",  # 青色
            "SubKnowledgePoint": "#45B7D1"  # 深青色
        }
        return colors.get(label_type, "#999999")

    def _get_node_shape(self, label_type: str) -> str:
        """根据节点类型设置形状"""
        shapes = {
            "Chapter": "star",
            "KnowledgePoint": "dot",
            "SubKnowledgePoint": "triangle"
        }
        return shapes.get(label_type, "square")