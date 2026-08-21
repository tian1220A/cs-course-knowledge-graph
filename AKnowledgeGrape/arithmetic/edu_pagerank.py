import logging
import numpy as np
import networkx as nx
from py2neo import Graph
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # 新增：三维绘图支持

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EduPageRankCalculator")


class EduPageRankCalculator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.graph.run("RETURN 1")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.critical(f"Neo4j连接失败: {e}")
            raise

        self.nodes = {}
        self.edges = []
        self.G = None  # 保存NetworkX图对象，用于可视化
        self.in_degree = {}  # 新增：存储节点入度

    def fetch_data(self):
        """加载所有核心知识点节点和边，新增计算节点入度"""
        node_query = """
        MATCH (n)
        WHERE n:KnowledgePoint OR n:SubKnowledgePoint
        RETURN 
            CASE 
                WHEN n:KnowledgePoint THEN n.kp_id
                WHEN n:SubKnowledgePoint THEN n.sub_kp_id
            END AS id,
            coalesce(n.class_hours, 1) AS class_hours,
            coalesce(n.syllabus_mentions, 1) AS syllabus_mentions,
            coalesce(n.bloom_level, 1) AS bloom_level,
            coalesce(n.title, n.name, n.kp_id, n.sub_kp_id, "节点"+toString(id(n))) AS name
        """
        node_records = self.graph.run(node_query)
        for record in node_records:
            node_id = record["id"]
            self.nodes[node_id] = {
                "class_hours": float(record["class_hours"]),
                "syllabus_mentions": int(record["syllabus_mentions"]),
                "bloom_level": min(int(record["bloom_level"]), 6),
                "name": record["name"]
            }

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
        """
        edge_records = self.graph.run(edge_query)
        for record in edge_records:
            self.edges.append((record["source"], record["target"]))
        logger.info(f"加载节点数: {len(self.nodes)}, 边数: {len(self.edges)}")

        # 新增：计算节点入度（有多少条边指向该节点）
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes.keys())
        self.G.add_edges_from(self.edges)
        self.in_degree = dict(self.G.in_degree())  # 入度字典：{节点ID: 入度值}
        # 将入度存入nodes字典
        for node_id in self.nodes:
            self.nodes[node_id]["in_degree"] = self.in_degree.get(node_id, 0)
        logger.info("节点入度计算完成")

    def compute_edu_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        """计算教育版PageRank（复用原有逻辑）"""
        if self.G is None:
            self.G = nx.DiGraph()
            self.G.add_nodes_from(self.nodes.keys())
            self.G.add_edges_from(self.edges)
            
        N = self.G.number_of_nodes()
        if N == 0:
            logger.error("无节点数据")
            return

        max_syllabus = max(data["syllabus_mentions"] for data in self.nodes.values()) or 1
        max_hours = max(data["class_hours"] for data in self.nodes.values()) or 1

        # 计算静态权重
        for node in self.G.nodes:
            data = self.nodes[node]
            syllabus_norm = np.log1p(data["syllabus_mentions"]) / np.log1p(max_syllabus)
            hours_norm = data["class_hours"] / max_hours if max_hours else 0
            bloom_norm = (7 - data["bloom_level"]) / 6.0
            static_weight = float(0.5 * hours_norm + 0.3 * syllabus_norm + 0.2 * bloom_norm)
            self.G.nodes[node]["static_weight"] = static_weight
            self.G.nodes[node]["edu_pr"] = 1.0 / N
            self.nodes[node]["static_weight"] = static_weight

        # 迭代计算PageRank
        for iter_count in range(max_iter):
            new_pr = {n: (1 - alpha) / N + alpha * sum(
                self.G.nodes[pred]["static_weight"] * self.G.nodes[pred]["edu_pr"] /
                (self.G.out_degree(pred) if self.G.out_degree(pred) > 0 else 1)
                for pred in self.G.predecessors(n)) for n in self.G.nodes}
            total = sum(new_pr.values())
            for n in new_pr:
                new_pr[n] /= total
            diff = max(abs(new_pr[n] - self.G.nodes[n]["edu_pr"]) for n in self.G.nodes)
            for n in self.G.nodes:
                self.G.nodes[n]["edu_pr"] = new_pr[n]
            logger.info(f"Iter {iter_count + 1} | Diff: {diff:.8f}")
            if diff < tol:
                logger.info("算法收敛")
                break

        # 保存结果到nodes字典
        for node in self.G.nodes:
            self.nodes[node]["edu_pr"] = self.G.nodes[node]["edu_pr"]

    def write_results_to_neo4j(self):
        """将结果写入Neo4j（新增入度存储）"""
        pr_values = [v["edu_pr"] for v in self.nodes.values()]
        pr_values.sort(reverse=True)
        threshold_index = max(int(len(pr_values) * 0.2) - 1, 0)
        threshold_value = pr_values[threshold_index] if pr_values else 0
        logger.info(f"核心知识点阈值: {threshold_value:.6f}")

        data = []
        for node_id, info in self.nodes.items():
            data.append({
                "id": node_id,
                "edu_pr": round(info["edu_pr"], 6),
                "in_degree": info["in_degree"],  # 新增：存储入度
                "is_core_kp": info["edu_pr"] >= threshold_value
            })

        query = """
        UNWIND $data AS row
        MATCH (n)
        WHERE ( (n:KnowledgePoint AND n.kp_id = row.id)
                OR (n:SubKnowledgePoint AND n.sub_kp_id = row.id) )
        SET n.edu_pagerank = row.edu_pr,
            n.in_degree = row.in_degree,  
            n.is_core_kp = row.is_core_kp
        """
        self.graph.run(query, data=data)
        logger.info("教育PageRank、入度及核心标志写入完成")

    def visualize_results(self, save_path="./edu_pagerank_visualizations/"):
        """生成优化后的可视化图表集合"""
        import os
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"开始生成可视化图表，保存路径：{save_path}")

        # 提取所有指标数据
        node_ids = list(self.nodes.keys())
        edu_pr = [self.nodes[node]["edu_pr"] for node in node_ids]
        class_hours = [self.nodes[node]["class_hours"] for node in node_ids]
        syllabus_mentions = [self.nodes[node]["syllabus_mentions"] for node in node_ids]
        bloom_level = [self.nodes[node]["bloom_level"] for node in node_ids]
        static_weight = [self.nodes[node]["static_weight"] for node in node_ids]
        in_degree = [self.nodes[node]["in_degree"] for node in node_ids]
        
        # 核心知识点标记
        pr_threshold = sorted(edu_pr, reverse=True)[max(int(len(edu_pr)*0.2)-1, 0)] if edu_pr else 0
        is_core = [1 if x >= pr_threshold else 0 for x in edu_pr]

        # ========== 1. PageRank分值分布直方图 ==========
        plt.figure(figsize=(10, 6))
        sns.histplot(edu_pr, bins=30, kde=True, color="#2E86AB", edgecolor="black")
        plt.axvline(pr_threshold, color="red", linestyle="--", linewidth=2, label=f"核心阈值: {pr_threshold:.6f}")
        plt.title("教育版PageRank分值分布", fontsize=14, fontweight="bold")
        plt.xlabel("PageRank分值", fontsize=12)
        plt.ylabel("节点数量", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "pagerank_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ========== 2. 三维散点图（PageRank + 静态权重 + 入度，按Bloom等级着色） ==========
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            edu_pr,               # x轴：PageRank值
            static_weight,        # y轴：静态权重
            in_degree,            # z轴：入度数
            c=bloom_level,        # 颜色：Bloom认知等级
            cmap="viridis",       # 渐变色彩映射
            alpha=0.7, 
            s=60, 
            edgecolors="black", 
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Bloom认知等级", rotation=270, labelpad=20)
        
        ax.set_title("PageRank vs 静态权重 vs 入度 三维分布", fontsize=14, fontweight="bold")
        ax.set_xlabel("PageRank分值", fontsize=12)
        ax.set_ylabel("静态权重", fontsize=12)
        ax.set_zlabel("节点入度数", fontsize=12)
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "3d_pagerank_static_indegree.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ========== 3. 各指标相关性热力图（含入度） ==========
        corr_data = np.array([
            edu_pr,
            class_hours,
            syllabus_mentions,
            bloom_level,
            static_weight,
            in_degree
        ]).T
        corr_labels = ["PageRank", "课时", "提及次数", "Bloom等级", "静态权重", "节点入度"]
        
        plt.figure(figsize=(10, 8))
        corr_matrix = np.corrcoef(corr_data.T)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".3f", 
            cmap="RdBu_r",
            xticklabels=corr_labels, 
            yticklabels=corr_labels,
            center=0, 
            square=True, 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8}
        )
        plt.title("各指标相关性热力图", fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(save_path, "indicators_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ========== 4. 静态权重与PageRank散点图（按Bloom等级着色） ==========
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(static_weight, edu_pr, c=bloom_level, 
                            cmap="viridis", alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        plt.colorbar(scatter, label="Bloom认知等级")
        plt.title("静态权重 vs 教育版PageRank", fontsize=14, fontweight="bold")
        plt.xlabel("静态权重", fontsize=12)
        plt.ylabel("PageRank分值", fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "static_weight_vs_pagerank.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ========== 5. 核心vs非核心知识点PageRank箱线图 ==========
        core_pr = [edu_pr[i] for i, val in enumerate(is_core) if val == 1]
        non_core_pr = [edu_pr[i] for i, val in enumerate(is_core) if val == 0]
        
        plt.figure(figsize=(10, 6))
        box_data = [non_core_pr, core_pr]
        box_plot = plt.boxplot(box_data, labels=["非核心知识点", "核心知识点"], 
                            patch_artist=True, notch=True)
        
        colors = ["#A23B72", "#F18F01"]
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title("核心vs非核心知识点PageRank分值对比", fontsize=14, fontweight="bold")
        plt.ylabel("PageRank分值", fontsize=12)
        plt.grid(alpha=0.3, axis="y")
        plt.savefig(os.path.join(save_path, "core_vs_noncore_boxplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ========== 6. 入度与PageRank散点图（按核心标记着色） ==========
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(in_degree, edu_pr, c=is_core, 
                            cmap=LinearSegmentedColormap.from_list("core", ["#A23B72", "#F18F01"]),
                            alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        plt.colorbar(scatter, ticks=[0, 1], label="是否核心知识点")
        plt.colorbar(scatter).set_ticklabels(["非核心", "核心"])
        plt.title("节点入度 vs 教育版PageRank", fontsize=14, fontweight="bold")
        plt.xlabel("节点入度数", fontsize=12)
        plt.ylabel("PageRank分值", fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, "indegree_vs_pagerank.png"), dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("所有可视化图表生成完成！")

    def run(self, visualize=True):
        self.fetch_data()
        self.compute_edu_pagerank()
        self.write_results_to_neo4j()
        if visualize:
            self.visualize_results()


if __name__ == "__main__":
    # 请根据实际Neo4j配置修改以下参数
    calc = EduPageRankCalculator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="123456789"
    )
    # 运行并生成可视化（默认保存到当前目录的edu_pagerank_visualizations文件夹）
    calc.run(visualize=True)