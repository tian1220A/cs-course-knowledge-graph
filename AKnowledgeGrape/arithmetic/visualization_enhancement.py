# visualization_enhancement.py
import os
import webbrowser
import matplotlib

matplotlib.use('Agg')  # 必须在其他matplotlib导入前设置
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from plotly.subplots import make_subplots
import webbrowser
import os
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedVisualizer:
    def __init__(self, nodes, trad_path, cog_path, trad_metrics, cog_metrics):
        self.nodes = nodes
        self.trad_path = trad_path
        self.cog_path = cog_path
        self.trad_metrics = trad_metrics
        self.cog_metrics = cog_metrics

        # 补充Bloom序列数据
        self.trad_metrics['bloom_sequence'] = [nodes[nid]['bloom_level'] for nid in trad_path]
        self.cog_metrics['bloom_sequence'] = [nodes[nid]['bloom_level'] for nid in cog_path]

    def _save_and_open_html(self, network, filename):
        """保存并自动打开HTML文件"""
        output_path = os.path.abspath(filename)
        network.save_graph(output_path)
        webbrowser.open(f'file://{output_path}')
        print(f"已生成交互式图表: {output_path}")

    def show_network_comparison(self):
        """生成聚焦路径的交互式网络图"""
        G = nx.DiGraph()
        edge_weights = defaultdict(float)

        # 收集所有路径节点
        all_path_nodes = set(self.trad_path) | set(self.cog_path)

        # 添加路径节点
        for nid in all_path_nodes:
            props = self.nodes[nid]
            G.add_node(nid,
                       label=f"{props['id']}\nBloom:{props['bloom_level']}",
                       group=props['labels'][0],
                       title=f"章节:{props.get('chapter_id', '')}\n核心:{props['is_core']}",
                       size=20 + 5 * props['bloom_level'])

        # 添加传统路径边
        for i in range(len(self.trad_path) - 1):
            src, tgt = self.trad_path[i], self.trad_path[i + 1]
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt,
                           color='#F5A623',
                           width=5,
                           title=f'传统路径 | Bloom差: {abs(self.nodes[tgt]["bloom_level"] - self.nodes[src]["bloom_level"])}')

        # 添加认知路径边
        for i in range(len(self.cog_path) - 1):
            src, tgt = self.cog_path[i], self.cog_path[i + 1]
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt,
                           color='#50E3C2',
                           width=7,
                           title=f'改进路径 | Bloom差: {abs(self.nodes[tgt]["bloom_level"] - self.nodes[src]["bloom_level"])}')

        # 设置高级布局参数
        nt = Network(height="800px", width="100%", directed=True)
        nt.from_nx(G)
        nt.set_options("""
        {
          "physics": {
            "stabilization": {"iterations": 200},
            "hierarchical": {
              "enabled": true,
              "levelSeparation": 150,
              "nodeSpacing": 200
            }
          },
          "nodes": {
            "scaling": {
              "min": 20,
              "max": 50
            }
          }
        }
        """)
        self._save_and_open_html(nt, 'path_focus.html')

    def show_metrics_radar(self):
        """单图合并展示传统 vs 改进雷达，并在图下方居中显示分析结论"""

        # 1. 准备原始数据
        raw = {
            'cost': self.trad_metrics['total_cost'],
            'core': len(self.trad_metrics['core_covered']),
            'jump': self.trad_metrics['bloom_jumps'],
            'cont': self.trad_metrics['continuity'] * 100,
            'load': self.trad_metrics['cognitive_load'],
        }
        raw_c = {
            'cost': self.cog_metrics['total_cost'],
            'core': len(self.cog_metrics['core_covered']),
            'jump': self.cog_metrics['bloom_jumps'],
            'cont': self.cog_metrics['continuity'] * 100,
            'load': self.cog_metrics['cognitive_load'],
        }

        labels = ['路径成本', '核心覆盖', 'Bloom跃迁', '连续性(%)', '认知负荷']
        trad_vals = [raw[k] for k in ['cost', 'core', 'jump', 'cont', 'load']]
        cog_vals = [raw_c[k] for k in ['cost', 'core', 'jump', 'cont', 'load']]

        # 2. 单一 polar 子图
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=trad_vals, theta=labels,
            name='传统算法',
            fill='toself', opacity=0.5,
            line=dict(color='#F5A623', width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatterpolar(
            r=cog_vals, theta=labels,
            name='改进算法',
            fill='toself', opacity=0.5,
            line=dict(color='#50E3C2', width=2),
            marker=dict(size=6)
        ))

        # 3. 配置雷达轴 & 布局
        # radialaxis.range 手动设上下限，保证两者都在可视范围内
        max_val = max(max(trad_vals), max(cog_vals)) * 1.1
        fig.update_layout(
            title=dict(text='算法性能对比雷达图（原始值）', x=0.5, y=0.95, font_size=20),
            polar=dict(
                radialaxis=dict(range=[0, max_val], showticklabels=True, ticks=''),
                angularaxis=dict(direction='clockwise')
            ),
            showlegend=True,
            legend=dict(orientation='h', x=0.5, xanchor='center', y=1.05),
            margin=dict(t=120, b=260, l=80, r=80),
            height=700, width=700,
        )

        # 4. 构造并添加居中注释
        analysis = [
            f"路径成本: {raw['cost']:.2f} → {raw_c['cost']:.2f}  (↓{(raw['cost'] - raw_c['cost']) / raw['cost'] * 100:.1f}%)",
            f"核心覆盖: {raw['core']} → {raw_c['core']}  (Δ{raw_c['core'] - raw['core']})",
            f"Bloom跃迁: {raw['jump']} → {raw_c['jump']}  (↓{raw['jump'] - raw_c['jump']} 次)",
            f"连续性: {raw['cont']:.1f}% → {raw_c['cont']:.1f}%  (↑{raw_c['cont'] - raw['cont']:.1f}%)",
            f"认知负荷: {raw['load']:.2f} → {raw_c['load']:.2f}  (↓{raw['load'] - raw_c['load']:.2f})"
        ]
        fig.add_annotation(
            x=0.5, y=-0.35, xref='paper', yref='paper',
            text="<br>".join(analysis),
            showarrow=False, align='center',
            font=dict(size=14)
        )

        # 5. 保存并打开
        out = "radar_analysis.html"
        fig.write_html(out)
        webbrowser.open(f"file://{os.path.abspath(out)}")
    def show_cognitive_analysis(self):
        """生成带统计分析的认知负荷图"""
        plt.figure(figsize=(14, 6))

        # 左图：分布统计
        plt.subplot(121)
        bins = np.arange(0, max(max(self.trad_metrics['bloom_sequence']),
                                max(self.cog_metrics['bloom_sequence'])) + 2) - 0.5
        plt.hist(self.trad_metrics['bloom_sequence'], bins=bins,
                 alpha=0.5, label='传统', density=True)
        plt.hist(self.cog_metrics['bloom_sequence'], bins=bins,
                 alpha=0.5, label='改进', density=True)
        plt.xlabel("Bloom层级", fontsize=12)
        plt.ylabel("概率密度", fontsize=12)
        plt.title("Bloom层级分布对比", fontsize=14)
        plt.legend()

        # 右图：趋势分析
        plt.subplot(122)
        window_size = max(3, len(self.trad_path) // 10)
        trad_smooth = np.convolve(self.trad_metrics['bloom_sequence'],
                                  np.ones(window_size) / window_size, mode='valid')
        cog_smooth = np.convolve(self.cog_metrics['bloom_sequence'],
                                 np.ones(window_size) / window_size, mode='valid')
        plt.plot(trad_smooth, color='#F5A623', label='传统', linewidth=2)
        plt.plot(cog_smooth, color='#50E3C2', label='改进', linewidth=2)
        plt.xlabel("路径步数", fontsize=12)
        plt.ylabel(f"{window_size}点移动平均", fontsize=12)
        plt.title("认知负荷变化趋势", fontsize=14)
        plt.legend()

        # 添加统计结论
        stats_text = [
            f"传统路径长度: {len(self.trad_path)}",
            f"改进路径长度: {len(self.cog_path)}",
            f"最大Bloom差（传统）: {max(np.abs(np.diff(self.trad_metrics['bloom_sequence'])))}",
            f"最大Bloom差（改进）: {max(np.abs(np.diff(self.cog_metrics['bloom_sequence'])))}"
        ]
        plt.figtext(0.5, 0.01, "\n".join(stats_text),
                    ha='center', fontsize=11,
                    bbox=dict(facecolor='#f0f0f0', alpha=0.8))

        output_path = os.path.abspath("cognitive_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存认知分析图: {output_path}")

    def show_all_visualizations(self):
        """执行全量可视化"""
        self.show_network_comparison()
        self.show_metrics_radar()
        self.show_cognitive_analysis()
        print("\n=== 可视化结果 ===")
        print("1. 路径对比: path_focus.html")
        print("2. 指标分析: radar_analysis.html")
        print("3. 认知分析: cognitive_analysis.png")