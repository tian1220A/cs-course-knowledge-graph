import logging
import matplotlib.pyplot as plt
import numpy as np
from py2neo import Graph
import warnings
import os
from matplotlib import cm
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("LinePlot")

class Chapter6LinePlot:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.uri = neo4j_uri
        self.user = neo4j_user
        self.pwd = neo4j_password
        self.names = []
        self.pr_scores = []
        self.edu_pr_scores = []

    def fetch_chapter6_data(self):
        try:
            self.graph = Graph(self.uri, auth=(self.user, self.pwd))
            query = """
            MATCH (n)
            WHERE (n:KnowledgePoint OR n:SubKnowledgePoint)
              AND n.chapter_id = 'CH06'
              AND exists(n.pagerank)
              AND exists(n.edu_pagerank)
            RETURN n.title AS name, n.pagerank AS pr, n.edu_pagerank AS edu_pr
            ORDER BY n.order
            """
            records = self.graph.run(query).data()

            if not records:
                logger.warning("No data found, using demo data")
                self._use_demo_data()
            else:
                self.names = [r["name"] for r in records]
                self.pr_scores = [float(r["pr"]) for r in records]
                self.edu_pr_scores = [float(r["edu_pr"]) for r in records]
                logger.info(f"✅ Loaded {len(self.names)} knowledge points")
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            self._use_demo_data()
            return True

    def _use_demo_data(self):
        self.names = [
            "HBase Architecture & Components",
            "HBase Data Model",
            "HBase Read/Write Process",
            "HBase Distributed Deployment",
            "HBase Performance Tuning",
            "Hybrid Load Elastic Scaling"
        ]
        self.pr_scores = [0.014, 0.012, 0.011, 0.009, 0.008, 0.014]
        self.edu_pr_scores = [0.015, 0.013, 0.012, 0.010, 0.009, 0.012]

    def draw_chapter6_line_chart(self):
        n = len(self.names)
        x = np.arange(n)
        xticks = [f"f{i+1}" for i in range(n)]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        all_vals = np.concatenate([self.pr_scores, self.edu_pr_scores])
        vmin, vmax = all_vals.min(), all_vals.max()
        norm = Normalize(vmin=vmin, vmax=vmax)

        # --------------------------
        # Traditional PageRank (Red)
        # --------------------------
        cmap_red = cm.Reds
        ax.plot(x, self.pr_scores, marker='^', color='#C82423',
                linewidth=2, markersize=8, label='Traditional PageRank')
        
        for i in range(n-1):
            color_start = cmap_red(norm(self.pr_scores[i]))
            ax.fill_between(
                [x[i], x[i+1]], 0, [self.pr_scores[i], self.pr_scores[i+1]],
                color=color_start, alpha=0.7
            )

        # --------------------------
        # Edu-link PageRank (Blue)
        # --------------------------
        cmap_blue = cm.Blues
        ax.plot(x, self.edu_pr_scores, marker='*', color='#236BC8',
                linewidth=2, markersize=8, label='Edu-link PageRank')
        
        for i in range(n-1):
            color_start = cmap_blue(norm(self.edu_pr_scores[i]))
            ax.fill_between(
                [x[i], x[i+1]], 0, [self.edu_pr_scores[i], self.edu_pr_scores[i+1]],
                color=color_start, alpha=0.5
            )

        # --------------------------
        # Color bars 优化：更长、更近、更紧凑
        # --------------------------
        sm_red = cm.ScalarMappable(norm=norm, cmap=cmap_red)
        sm_red.set_array([])
        cbar1 = fig.colorbar(sm_red, ax=ax, shrink=0.75, pad=0.015, aspect=40)
        cbar1.set_label('Traditional PageRank Weight', fontsize=7)

        sm_blue = cm.ScalarMappable(norm=norm, cmap=cmap_blue)
        sm_blue.set_array([])
        cbar2 = fig.colorbar(sm_blue, ax=ax, shrink=0.75, pad=0.045, aspect=40)
        cbar2.set_label('Edu-link PageRank Weight', fontsize=7)

        # --------------------------
        # 图例优化：和折线标记一致
        # --------------------------
        ax.legend(handles=[
            plt.Line2D([], [], color='#C82423', marker='^', linestyle='-', linewidth=2, markersize=8, label='Traditional PageRank'),
            plt.Line2D([], [], color='#236BC8', marker='*', linestyle='-', linewidth=2, markersize=8, label='Edu-link PageRank')
        ], fontsize=10, loc='upper left')

        # --------------------------
        # English labels
        # --------------------------
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, fontsize=8)
        ax.set_xlabel("Knowledge Points", fontsize=12)
        ax.set_ylabel("Weight Score", fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig("chapter6_dual_heatmap_english_final.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

if __name__ == "__main__":
    plot = Chapter6LinePlot("bolt://localhost:7687", "neo4j", "123456789")
    if plot.fetch_chapter6_data():
        plot.draw_chapter6_line_chart()