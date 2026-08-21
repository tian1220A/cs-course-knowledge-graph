import logging
import matplotlib.pyplot as plt
import numpy as np
from py2neo import Graph
import os
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("RaincloudPlot")

class PageRankRaincloudPlot:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.graph.run("RETURN 1")
        self.data = {
            "traditional_noncore": [],
            "traditional_core": [],
            "improved_noncore": [],
            "improved_core": []
        }

    def fetch_and_split_data(self):
        query = """
        MATCH (n)
        WHERE (n:KnowledgePoint OR n:SubKnowledgePoint) AND exists(n.pagerank)
        RETURN n.pagerank AS traditional_pr, n.edu_pagerank AS improved_pr
        """
        records = self.graph.run(query)
        all_records = []
        for r in records:
            t = float(r["traditional_pr"])
            i = float(r["improved_pr"])
            all_records.append((t,i))
        if not all_records:
            logger.error("No data")
            return
        n = len(all_records)
        core_num = max(1,int(n*0.2))
        tr = sorted([x[0] for x in all_records], reverse=True)
        ir = sorted([x[1] for x in all_records], reverse=True)
        t_thr = tr[core_num-1]
        i_thr = ir[core_num-1]
        for t,i in all_records:
            if t>=t_thr:
                self.data["traditional_core"].append(t)
            else:
                self.data["traditional_noncore"].append(t)
            if i>=i_thr:
                self.data["improved_core"].append(i)
            else:
                self.data["improved_noncore"].append(i)

    def draw_raincloud(self):
        # 画布尺寸微调：刚好放下标题 + 底部文字
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

        ax.spines[:].set_visible(False)

        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=True, length=4, width=0.8, labelsize=10, color='#555')
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.002))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

        ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#aaa')
        ax.set_axisbelow(True)

        ax.set_ylim(-0.0015, 0.018)
        ax.set_xlim(0, 2.8)
        ax.yaxis.label.set_color('#333')
        ax.yaxis.label.set_fontweight('bold')
        COLOR_NONCORE = "#102C57"
        COLOR_CORE = "#990000"

        POS_PRE, POS_POST = 1.0, 1.8
        pre_cloud = POS_PRE - 0.02
        pre_box   = POS_PRE + 0.06
        post_cloud= POS_POST + 0.02
        post_box  = POS_POST - 0.06

        cloud_w = 0.12
        scatter_sp = 0.007
        box_w = 0.05

        pre_nc = np.array(self.data["traditional_noncore"])
        pre_co = np.array(self.data["traditional_core"])
        post_nc = np.array(self.data["improved_noncore"])
        post_co = np.array(self.data["improved_core"])
        means = {}

        def plot_group(arr, color, cloud_x, box_x, side, min_offset=0):
            if len(arr)==0: return
            med= np.median(arr)
            mi = arr.min()
            ma = arr.max()
            mu = arr.mean()

            kde = stats.gaussian_kde(arr)
            y = np.linspace(mi,ma,200)
            xk = kde(y)/kde(y).max()*cloud_w
            if side == 'L':
                ax.fill_betweenx(y, cloud_x-xk, cloud_x, alpha=0.3, color=color)
                ax.plot(cloud_x-xk, y, lw=1, color=color)
                txt_x = cloud_x - cloud_w - 0.03
                ha = 'right'
            else:
                ax.fill_betweenx(y, cloud_x, cloud_x+xk, alpha=0.3, color=color)
                ax.plot(cloud_x+xk, y, lw=1, color=color)
                txt_x = cloud_x + cloud_w + 0.03
                ha = 'left'

            xj = np.random.normal((cloud_x+box_x)/2, scatter_sp, len(arr))
            ax.scatter(xj, arr, s=2, alpha=0.6, color=color)

            bp = ax.boxplot(arr, positions=[box_x], widths=box_w, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.8)
            ax.scatter(box_x, mu, marker='D', s=25, color=color, ec='k', zorder=5)

            # 只保留 Max / Med / Min，无Q1/Q3，底部错开不重叠
            ax.text(txt_x, ma, f'Max {ma:.4f}', ha=ha, va='top', fontsize=6.5, color=color)
            ax.text(txt_x, med,f'Med {med:.4f}', ha=ha, va='center', fontsize=6.5, color=color)
            ax.text(txt_x, mi + min_offset, f'Min {mi:.4f}', ha=ha, va='bottom', fontsize=6.5, color=color)

            return mu

        # 绘制四组
        if len(pre_nc)>0: means['pre0'] = plot_group(pre_nc, COLOR_NONCORE, pre_cloud, pre_box, 'L', 0.0001)
        if len(pre_co)>0: means['pre1'] = plot_group(pre_co, COLOR_CORE, pre_cloud, pre_box, 'L', -0.0001)
        if len(post_nc)>0: means['post0'] = plot_group(post_nc, COLOR_NONCORE, post_cloud, post_box, 'R', 0.0001)
        if len(post_co)>0: means['post1'] = plot_group(post_co, COLOR_CORE, post_cloud, post_box, 'R', -0.0001)

        # 均值连线
        if 'pre1' in means and 'post1' in means:
            ax.plot([pre_box,post_box],[means['pre1'],means['post1']], color=COLOR_CORE, lw=2, alpha=0.8)
        if 'pre0' in means and 'post0' in means:
            ax.plot([pre_box,post_box],[means['pre0'],means['post0']], color=COLOR_NONCORE, lw=2, alpha=0.8)

        # ========= 恢复表头 + 底部标注 =========
        ##ax.set_title("PageRank Score Distribution", fontsize=12, weight='bold')   # 表头回来
        ax.set_ylabel("Score", fontsize=12)
        ax.text(POS_PRE, -0.0012, 'PageRank', ha='center', fontsize=11, fontweight='bold')
        ax.text(POS_POST, -0.0012, 'Edu-link PageRank', ha='center', fontsize=11, fontweight='bold')

        # 图例
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=COLOR_CORE, label="Core Node"),
            Patch(color=COLOR_NONCORE, label="NonCore Node")],
            title="Node Type", loc="upper right", fontsize=9)

        # 保证整张图完全显示，不裁切、不多留白
        plt.tight_layout(pad=0.3)
        plt.savefig("final_raincloud.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

if __name__=="__main__":
    plotter = PageRankRaincloudPlot("bolt://localhost:7687","neo4j","123456789")
    plotter.fetch_and_split_data()
    plotter.draw_raincloud()