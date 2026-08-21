import logging
from py2neo import Graph
import networkx as nx
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

# 禁用可能的警告信息
warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PageRankCalculator")


class PageRankCalculator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.graph.run("RETURN 1")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.critical(f"Neo4j连接失败: {e}")
            raise
        
        # 存储节点详细数据（用于可视化）
        self.node_details = {}
        self.G = None  # 存储NetworkX图对象

    def fetch_graph_data(self):
        """加载所有核心知识点节点和边，同时保存节点详细属性"""
        # 节点查询：获取课时、提及次数、Bloom等级等属性
        node_query = """
        MATCH (n)
        WHERE n:KnowledgePoint OR n:SubKnowledgePoint
        RETURN 
            CASE 
                WHEN n:KnowledgePoint THEN n.kp_id
                WHEN n:SubKnowledgePoint THEN n.sub_kp_id
            END AS business_id,
            n,
            coalesce(n.class_hours, 1) AS class_hours,
            coalesce(n.syllabus_mentions, 1) AS syllabus_mentions,
            coalesce(n.bloom_level, 1) AS bloom_level,
            coalesce(n.title, n.name, n.kp_id, n.sub_kp_id) AS name
        """
        nodes = {}
        self.node_details.clear()  # 清空历史数据
        for record in self.graph.run(node_query):
            business_id = record["business_id"]
            nodes[business_id] = record["n"]
            # 保存节点详细属性
            self.node_details[business_id] = {
                "class_hours": float(record["class_hours"]),
                "syllabus_mentions": int(record["syllabus_mentions"]),
                "bloom_level": min(int(record["bloom_level"]), 6),
                "name": record["name"]
            }

        # 边查询：只匹配两端都是 KnowledgePoint 或 SubKnowledgePoint 的关系
        edge_query = """
        MATCH (a)-[r]->(b)
        WHERE (a:KnowledgePoint OR a:SubKnowledgePoint)
          AND (b:KnowledgePoint OR b:SubKnowledgePoint)
        RETURN 
            CASE 
                WHEN a:KnowledgePoint THEN a.kp_id
                WHEN a:SubKnowledgePoint THEN a.sub_kp_id
            END AS source,
            CASE 
                WHEN b:KnowledgePoint THEN b.kp_id
                WHEN b:SubKnowledgePoint THEN b.sub_kp_id
            END AS target
        ORDER BY source, target 
        """
        edges = []
        for record in self.graph.run(edge_query):
            edges.append((record["source"], record["target"]))

        logger.info(f"稳定加载节点数: {len(nodes)}, 边数: {len(edges)}")
        return nodes, edges

    def compute_pagerank(self, top_n: int = 10, damping_factor: float = 0.85, visualize: bool = True):
        """
        构建 NetworkX 图并计算 PageRank，
        同时将结果写入 Neo4j（使用业务ID匹配），
        可选生成可视化图表。
        """
        nodes, edges = self.fetch_graph_data()
        self.G = nx.DiGraph()
        self.G.add_nodes_from(nodes.keys())
        self.G.add_edges_from(edges)

        # 修正：使用self.G而不是G
        initial_pr = {node: 1.0 / len(nodes) for node in self.G.nodes()} if len(nodes) > 0 else {}
        pr = nx.pagerank(self.G, alpha=damping_factor, max_iter=5000, tol=1e-15, nstart=initial_pr)
        
        # 将PageRank结果存入node_details
        for node_id, score in pr.items():
            self.node_details[node_id]["pagerank"] = score

        self.write_pagerank_to_neo4j(pr)

        # 生成可视化图表
        if visualize:
            self.visualize_results()

        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        result = []
        for node_id, score in sorted_pr[:top_n]:
            node = nodes[node_id]
            name = node.get("title") or node.get("name") or node.get("kp_id") or node.get("sub_kp_id") or str(node_id)
            result.append((name, score))
        return result

    def write_pagerank_to_neo4j(self, pagerank_dict):
        """
        将计算得到的 PageRank 值写入 Neo4j，
        使用 kp_id 或 sub_kp_id 作为业务ID匹配节点。
        """
        try:
            query = """
            UNWIND $pagerankData AS row
            MATCH (n)
            WHERE ( (n:KnowledgePoint AND n.kp_id = row.id)
                    OR (n:SubKnowledgePoint AND n.sub_kp_id = row.id) )
            SET n.pagerank = row.score
            """
            data = [{"id": node_id, "score": score} for node_id, score in pagerank_dict.items()]
            self.graph.run(query, pagerankData=data)
            logger.info("PageRank 值已成功写入 Neo4j 数据库中")
        except Exception as e:
            logger.error(f"写入 PageRank 值失败: {e}")
            raise

    def visualize_results(self, save_path="./basic_pagerank_visualizations/"):
        """
        生成5类可视化图表：
        1. PageRank分值分布直方图
        2. 核心知识点vs非核心知识点散点图（课时vs提及次数）
        3. 各指标相关性热力图
        4. 节点入度vs PageRank散点图
        5. 核心vs非核心知识点PageRank箱线图
        """
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"开始生成可视化图表，保存路径：{save_path}")

        # 检查数据完整性
        if not self.node_details or "pagerank" not in next(iter(self.node_details.values())):
            logger.error("无节点数据或未计算PageRank，无法生成可视化")
            return

        # 准备数据
        node_ids = list(self.node_details.keys())
        pagerank = [self.node_details[node]["pagerank"] for node in node_ids]
        class_hours = [self.node_details[node]["class_hours"] for node in node_ids]
        syllabus_mentions = [self.node_details[node]["syllabus_mentions"] for node in node_ids]
        bloom_level = [self.node_details[node]["bloom_level"] for node in node_ids]
        in_degrees = [self.G.in_degree(node) for node in node_ids]  # 节点入度
        
        # 核心知识点标记（前20%）
        pr_threshold = sorted(pagerank, reverse=True)[max(int(len(pagerank)*0.2)-1, 0)] if pagerank else 0
        contr_is_core = [1 if x >= pr_threshold else 0 for x in pagerank]

        # 1. PageRank分值分布直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(pagerank, bins=30, kde=True, color="#2E86AB", edgecolor="black")
        plt.axvline(pr_threshold, color="red", linestyle="--", linewidth=2, label=f"核心阈值: {pr_threshold:.6f}")
        plt.title("基础版PageRank分值分布", fontsize=14, fontweight="bold")
        plt.xlabel("PageRank分值", fontsize=12)
        plt.ylabel("节点数量", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "pagerank_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 2. 课时vs提及次数散点图（按核心知识点着色）
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(class_hours, syllabus_mentions, c=contr_is_core, 
                            cmap=LinearSegmentedColormap.from_list("core", ["#A23B72", "#F18F01"]),
                            alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.set_label("是否核心知识点")
        cbar.set_ticklabels(["非核心", "核心"])
        plt.title("课时 vs 大纲提及次数（按核心知识点分类）", fontsize=14, fontweight="bold")
        plt.xlabel("课时", fontsize=12)
        plt.ylabel("大纲提及次数", fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "class_hours_vs_mentions.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 3. 各指标相关性热力图
        corr_data = np.array([
            pagerank,
            class_hours,
            syllabus_mentions,
            bloom_level,
            in_degrees
        ]).T
        corr_labels = ["PageRank", "课时", "提及次数", "Bloom等级", "节点入度"]
        
        plt.figure(figsize=(9, 7))
        corr_matrix = np.corrcoef(corr_data.T)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                    xticklabels=corr_labels, yticklabels=corr_labels,
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("各指标相关性热力图", fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(save_path, "indicators_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 4. 节点入度vs PageRank散点图（按Bloom等级着色）
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(in_degrees, pagerank, c=bloom_level, 
                            cmap="viridis", alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        plt.colorbar(scatter, label="Bloom认知等级")
        plt.title("节点入度 vs 基础版PageRank（按Bloom等级着色）", fontsize=14, fontweight="bold")
        plt.xlabel("节点入度", fontsize=12)
        plt.ylabel("PageRank分值", fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "in_degree_vs_pagerank.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 5. 核心vs非核心知识点PageRank箱线图
        core_pr = [pagerank[i] for i, val in enumerate(contr_is_core) if val == 1]
        non_core_pr = [pagerank[i] for i, val in enumerate(contr_is_core) if val == 0]
        
        plt.figure(figsize=(10, 6))
        box_data = [non_core_pr, core_pr]
        box_plot = plt.boxplot(box_data, labels=["非核心知识点", "核心知识点"], 
                              patch_artist=True, notch=True)
        
        # 设置箱体颜色
        colors = ["#A23B72", "#F18F01"]
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title("核心vs非核心知识点PageRank分值对比", fontsize=14, fontweight="bold")
        plt.ylabel("PageRank分值", fontsize=12)
        plt.grid(alpha=0.3, axis="y")
        plt.savefig(os.path.join(save_path, "core_vs_noncore_boxplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("所有可视化图表生成完成！")


if __name__ == "__main__":
    # 请根据实际Neo4j配置修改以下参数
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "123456789"

    pr_calc = PageRankCalculator(neo4j_uri, neo4j_user, neo4j_password)
    top_results = pr_calc.compute_pagerank(top_n=10, visualize=True)

    print("PageRank 计算结果（前10）：")
    for name, score in top_results:
        print(f"节点: {name}, PageRank: {score:.6f}")