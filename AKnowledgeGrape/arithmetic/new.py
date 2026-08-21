import logging
from collections import defaultdict
import math
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from py2neo import Graph
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import beta

from typing import List, Dict, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedCollaborativeLearning")


class EnhancedCollaborativeLearning12:
    def __init__(self, neo4j_uri, user, password, alpha=0.85, feedback_strength=0.1, dynamic_threshold=0.008,
                 lambda_bayes=10.0, beta_smooth=(2, 2), core_kp_weight=1.5, smoothness_weight=1.8):
        # 基础初始化
        self.graphdb = Graph(neo4j_uri, auth=(user, password))
        self.alpha = alpha
        self.feedback_strength = feedback_strength
        self.dynamic_threshold = dynamic_threshold
        self.G = nx.DiGraph()
        self.node_attrs = {}
        self.node_info = {}
        self.pr_history = defaultdict(list)
        self.metrics = {
            'cost': [],
            'path_length': [],
            'pr_std': [],
            'pr_max_diff': [],
            'core_kp_coverage': [],  # 新增：核心知识点覆盖率
            'path_smoothness': []     # 新增：路径平滑度
        }
        
        # 贝叶斯模块参数
        self.lambda_bayes = lambda_bayes  # 贝叶斯惩罚权重
        self.beta_smooth = beta_smooth    # Beta平滑参数
        self.prior_prob: dict[tuple, float] = {}  # (from_node, to_node) -> 先验概率
        self.likelihood_cache: dict[tuple, dict] = {}  # 似然缓存
        self.node_chapter_order: dict[int, tuple] = {}  # node_id -> (chapter_id, chapter_order)
        
        # 核心知识点和路径平滑度权重
        self.core_kp_weight = core_kp_weight
        self.smoothness_weight_base = smoothness_weight  # 基础平滑度权重
        self.smoothness_weight = smoothness_weight       # 动态调整的平滑度权重
        self.core_kp_threshold = 0.2  # 核心知识点阈值（PageRank前20%）
        
        # 平滑度优化参数（类内统一管理）
        self.smoothness_params = {
            'chapter_transition_penalty': 0.8,        # 跨章节惩罚
            'chapter_reverse_penalty': 1.5,           # 章节逆序惩罚倍数
            'bloom_jump_penalty': 1.0,                # Bloom跳跃惩罚
            'bloom_level_reward': {                   # Bloom层级递进奖励
                0: 0.7,                               # 同层级
                1: 1.0,                               # 递进1级
                2: 0.3,                               # 递进2级
                'default': 0.1                        # 更多级
            },
            'consecutive_core_bonus': 0.5,            # 连续核心知识点奖励
            'consecutive_core_extra': 0.3,            # 连续核心知识点额外奖励
            'intra_chapter_reward': 1.2,              # 章节内奖励
            'intra_chapter_position_bonus': 0.3,      # 章节内位置连续奖励
            'prerequisite_edge_reward': 0.8,          # 先修关系奖励
            'history_path_bonus': 1.0,                # 历史平滑路径奖励
            'smoothness_weight_growth': 0.1,          # 平滑度权重增长系数
            'early_stop_smoothness_threshold': 0.9    # 提前终止的平滑度阈值
        }
        
        # 历史平滑路径记录
        self.smooth_paths = set()
        self.current_iteration = 0  # 当前迭代次数

    def pairwise(self, iterable):
        import itertools
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def _calculate_core_kp_coverage(self, path: List[int]) -> float:
        """计算路径中的核心知识点覆盖率"""
        if not path:
            return 0.0
        
        core_kps = [n for n in self.G.nodes() if self.node_attrs[n].get('edu_pr', 0) >= self.core_kp_threshold]
        if not core_kps:
            return 0.0
        
        path_core_kps = [n for n in path if n in core_kps]
        coverage = len(path_core_kps) / len(core_kps)
        
        # 额外奖励：连续核心知识点
        consecutive_bonus = 0
        for i in range(len(path)-1):
            if path[i] in core_kps and path[i+1] in core_kps:
                consecutive_bonus += self.smoothness_params['consecutive_core_bonus']
        
        return min(coverage + consecutive_bonus, 1.0)

    def _calculate_path_smoothness(self, path: List[int]) -> float:
        """增强版：计算路径平滑度（章节连续性和Bloom层级递进）"""
        if len(path) < 2:
            return 0.0
        
        smoothness_score = 0
        chapter_changes = 0
        bloom_jumps = 0
        consecutive_core = 0
        
        # 历史路径奖励
        path_tuple = tuple(path)
        if path_tuple in self.smooth_paths:
            smoothness_score += self.smoothness_params['history_path_bonus']
        
        for i in range(len(path)-1):
            node1, node2 = path[i], path[i+1]
            info1 = self.node_info[node1]
            info2 = self.node_info[node2]
            
            # 章节连续性（强化版）
            if info1['chapter_id'] == info2['chapter_id']:
                smoothness_score += self.smoothness_params['intra_chapter_reward']
                # 章节内位置连续性
                if abs(info1.get('node_order', 0) - info2.get('node_order', 0)) <= 2:
                    smoothness_score += self.smoothness_params['intra_chapter_position_bonus']
            else:
                chapter_changes += 1
                # 章节顺序合理性（更严格）
                if info1['chapter_order'] < info2['chapter_order']:
                    smoothness_score += 0.5
                else:
                    smoothness_score += 0.1  # 逆序惩罚加重
            
            # Bloom层级递进（更精细）
            bloom_diff = info2['bloom_level'] - info1['bloom_level']
            if bloom_diff in self.smoothness_params['bloom_level_reward']:
                smoothness_score += self.smoothness_params['bloom_level_reward'][bloom_diff]
            elif bloom_diff > 2:
                bloom_jumps += bloom_diff - 2
                smoothness_score += self.smoothness_params['bloom_level_reward']['default']
            else:
                smoothness_score += 0.2  # 层级下降惩罚
            
            # 边关系强度（优先官方先修关系）
            if self.G.has_edge(node1, node2):
                edge_data = self.G[node1][node2]
                if edge_data.get('rel_type') == 'REQUIRES_PREREQUISITE':
                    smoothness_score += self.smoothness_params['prerequisite_edge_reward']
                else:
                    smoothness_score += edge_data.get('weight', 1.0) * 0.4
            
            # 连续核心知识点奖励
            if info1['is_core'] and info2['is_core']:
                consecutive_core += 1
                smoothness_score += self.smoothness_params['consecutive_core_bonus']
        
        # 连续核心知识点额外奖励
        if consecutive_core >= 2:
            smoothness_score += consecutive_core * self.smoothness_params['consecutive_core_extra']
        
        # 惩罚计算（更严格）
        chapter_penalty = chapter_changes * self.smoothness_params['chapter_transition_penalty']
        bloom_penalty = bloom_jumps * self.smoothness_params['bloom_jump_penalty']
        total_penalty = chapter_penalty + bloom_penalty
        
        max_possible = (len(path)-1) * (self.smoothness_params['intra_chapter_reward'] + 
                                       self.smoothness_params['bloom_level_reward'][1] + 
                                       self.smoothness_params['prerequisite_edge_reward']) + \
                       consecutive_core * (self.smoothness_params['consecutive_core_bonus'] + 
                                           self.smoothness_params['consecutive_core_extra']) + \
                       self.smoothness_params['history_path_bonus']
        
        smoothness = max(0, (smoothness_score - total_penalty) / max_possible) if max_possible > 0 else 0
        
        # 记录高质量平滑路径
        if smoothness > self.smoothness_params['early_stop_smoothness_threshold']:
            self.smooth_paths.add(path_tuple)
        
        return smoothness

    def _record_metrics(self, path, cost):
        self.metrics['cost'].append(cost)
        self.metrics['path_length'].append(len(path))
        
        # 新增：记录核心知识点覆盖率和平滑度
        self.metrics['core_kp_coverage'].append(self._calculate_core_kp_coverage(path))
        self.metrics['path_smoothness'].append(self._calculate_path_smoothness(path))

        pr_values = [attrs.get('edu_pr', 0.0) for attrs in self.node_attrs.values() if attrs]

        if len(pr_values) >= 2:
            self.metrics['pr_std'].append(np.std(pr_values))
        else:
            self.metrics['pr_std'].append(0.0)
            logger.warning("权威值不足2个，标准差设为0")

        if pr_values:
            self.metrics['pr_max_diff'].append(max(pr_values) - min(pr_values))
        else:
            self.metrics['pr_max_diff'].append(0.0)
            logger.warning("无有效权威值，最大差异设为0")

    def _plot_metrics(self):
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.metrics['cost'], 'o-')
        plt.title('路径成本迭代变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('成本')
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(self.metrics['path_length'], 's-')
        plt.title('路径长度迭代变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('节点数')
        plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(self.metrics['pr_std'], '^-')
        plt.title('权威值标准差变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('标准差')
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(self.metrics['core_kp_coverage'], 'o-', color='green')
        plt.title('核心知识点覆盖率变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('覆盖率')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(self.metrics['path_smoothness'], 's-', color='orange')
        plt.title('路径平滑度变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('平滑度')
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        plt.plot(self.metrics['core_kp_coverage'], 'o-', label='核心知识点覆盖率')
        plt.plot(self.metrics['path_smoothness'], 's-', label='路径平滑度')
        plt.title('核心指标综合对比')
        plt.xlabel('迭代轮次')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('collaborative_metrics_enhanced.png', dpi=300)
        plt.close()

    def _generate_metric_report(self):
        if not self.metrics['cost']:
            logger.warning("无有效指标数据，跳过报告生成")
            return

        initial_cost = self.metrics['cost'][0]
        final_cost = self.metrics['cost'][-1]
        cost_reduction = (initial_cost - final_cost) / initial_cost if initial_cost else 0.0
        is_self_loop = (initial_cost == 0 and final_cost == 0)
        cost_stability = np.std(self.metrics['cost'][-5:]) if len(self.metrics['cost']) >= 5 else 0.0
        pr_std_mean = np.mean(self.metrics['pr_std'])
        pr_max_diff = self.metrics['pr_max_diff'][-1]
        
        # 新增指标
        final_core_coverage = self.metrics['core_kp_coverage'][-1] if self.metrics['core_kp_coverage'] else 0.0
        final_smoothness = self.metrics['path_smoothness'][-1] if self.metrics['path_smoothness'] else 0.0
        avg_core_coverage = np.mean(self.metrics['core_kp_coverage']) if self.metrics['core_kp_coverage'] else 0.0
        avg_smoothness = np.mean(self.metrics['path_smoothness']) if self.metrics['path_smoothness'] else 0.0

        report = {
            '最终成本': final_cost,
            '成本下降率': cost_reduction,
            '路径收敛稳定性': cost_stability,
            '权威值集中度': pr_std_mean,
            '权威值最大差异': pr_max_diff,
            '最终核心知识点覆盖率': final_core_coverage,
            '平均核心知识点覆盖率': avg_core_coverage,
            '最终路径平滑度': final_smoothness,
            '平均路径平滑度': avg_smoothness
        }
        
        conclusions = []
        if is_self_loop:
            conclusions.append("自循环路径无需优化，直接返回最优解")
        else:
            if report['成本下降率'] > 0.9:
                conclusions.append("算法优化显著：成本下降率超过 90%，路径搜索效率极高。")
            elif report['成本下降率'] > 0.5:
                conclusions.append("算法有效但可优化：成本下降率 50%~90%，建议调整反馈强度或迭代次数。")
            else:
                conclusions.append("优化效果不足：成本下降率低于 50%，需检查数据连通性或参数设置。")
            if report['路径收敛稳定性'] < 1.0:
                conclusions.append("路径收敛稳定：最后5次迭代成本波动小（标准差 <1.0），结果可靠。")
            else:
                conclusions.append("收敛不稳定：成本波动较大，建议增加迭代次数或降低学习率。")
        
        # 新增结论
        if report['最终核心知识点覆盖率'] > 0.7:
            conclusions.append("核心知识点覆盖优秀：覆盖率超过70%，路径包含大部分关键知识点。")
        elif report['最终核心知识点覆盖率'] > 0.4:
            conclusions.append("核心知识点覆盖良好：覆盖率40%~70%，路径包含部分关键知识点。")
        else:
            conclusions.append("核心知识点覆盖不足：覆盖率低于40%，建议调整核心知识点权重。")
        
        if report['最终路径平滑度'] > 0.8:
            conclusions.append("路径平滑度优秀：章节切换和Bloom层级递进合理，学习路径流畅。")
        elif report['最终路径平滑度'] > 0.5:
            conclusions.append("路径平滑度良好：章节切换较少，Bloom层级递进基本合理。")
        else:
            conclusions.append("路径平滑度不足：存在频繁章节切换或层级跳跃，建议优化路径规划。")
        
        if report['权威值集中度'] < 0.1:
            conclusions.append("权威值分布均匀：节点权威值差异小，系统处于平衡状态。")
        else:
            conclusions.append("权威值差异显著：存在核心知识点聚集现象，符合教学预期。")
        if report['权威值最大差异'] < self.dynamic_threshold:
            conclusions.append("权威值区分不足：建议提高反馈强度或调整参数设置。")

        pr_changes = {nid: abs(hist[-1] - hist[0]) for nid, hist in self.pr_history.items() if len(hist) >= 2}
        top_nodes = sorted(pr_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n=== TOP5关键节点权威值变化 ===")
        for nid, delta in top_nodes:
            detail = self.get_node_details(nid)
            init_pr = self.pr_history[nid][0]
            final_pr = self.pr_history[nid][-1]
            rel = (delta / init_pr) if init_pr else 0
            is_core = "✓" if final_pr >= self.core_kp_threshold else "✗"
            print(
                f"节点{nid} ({detail['title']}) {is_core}: 初始 {init_pr:.4f} → 最终 {final_pr:.4f}, 绝对 {delta:.4f}, 相对 {rel:.2%}")

        print("\n=== 指标分析报告 ===")
        for k, v in report.items(): print(f"{k}: {v:.4f}")
        print("\n=== 分析结论 ===")
        for c in conclusions: print(c)

        with open('metric_report_enhanced.txt', 'w') as f:
            f.write("=== Collaborative Learning Metrics (Enhanced) ===\n")
            for k, v in report.items(): f.write(f"{k}: {v}\n")
            f.write("\n=== Conclusions ===\n")
            f.write("\n".join(conclusions))

    def _get_chapter_order(self, chapter_id):
        if not chapter_id:
            return 0
        q = f"MATCH (c:Chapter {{chapter_id: '{chapter_id}'}}) RETURN c.order as order"
        res = self.graphdb.run(q).data()
        return res[0]['order'] if res else 0

    def _init_bayesian_prior(self, edges_result):
        """重构后的先验概率初始化：基于全局结构特征"""
        for rec in edges_result:
            src = rec['src']
            tgt = rec['tgt']
            rel_type = rec['rel_type']
            
            u_info = self.node_info[src]
            v_info = self.node_info[tgt]
            
            # 特征1：章节顺序合理性
            chapter_score = 1.0 if u_info['chapter_order'] < v_info['chapter_order'] else \
                            0.8 if u_info['chapter_order'] == v_info['chapter_order'] else 0.2
            
            # 特征2：Bloom层级递进合理性
            bloom_diff = v_info['bloom_level'] - u_info['bloom_level']
            bloom_score = 1.0 if bloom_diff == 1 else \
                          0.8 if bloom_diff == 0 else \
                          0.3 if bloom_diff > 1 else 0.2
            
            # 特征3：关系类型权重
            rel_score = 1.0 if rel_type == 'REQUIRES_PREREQUISITE' else \
                        0.9 if rel_type == 'CONTAINS' else 0.7
            
            # 特征4：核心知识点关联
            src_is_core = self.node_attrs[src].get('edu_pr', 0) >= self.core_kp_threshold
            tgt_is_core = self.node_attrs[tgt].get('edu_pr', 0) >= self.core_kp_threshold
            core_score = 1.0 if (src_is_core and tgt_is_core) else \
                         0.8 if (src_is_core or tgt_is_core) else 0.7
            
            # 加权融合先验特征
            raw_prior = (0.3 * chapter_score + 0.3 * bloom_score + 0.2 * rel_score + 0.2 * core_score)
            
            # Beta分布平滑
            alpha_prior, beta_prior = self.beta_smooth
            smoothed_prior = beta.cdf(raw_prior, alpha_prior, beta_prior)
            
            self.prior_prob[(src, tgt)] = smoothed_prior

    def _calculate_likelihood(self, from_node, to_node, path_context: list):
        """重构后的似然函数：基于路径局部特征"""
        if (from_node, to_node) in self.likelihood_cache:
            return self.likelihood_cache[(from_node, to_node)]
        
        u_info = self.node_info[from_node]
        v_info = self.node_info[to_node]
        likelihood = {}
        
        # 特征1：路径中连续核心知识点
        prev_node = path_context[-2] if len(path_context) >= 2 else None
        if prev_node and self.node_info[prev_node]['is_core'] and u_info['is_core'] and v_info['is_core']:
            likelihood['consecutive_core'] = 0.9
        else:
            likelihood['consecutive_core'] = 0.5
        
        # 特征2：Bloom递进一致性
        if len(path_context) >= 3:
            recent_blooms = [self.node_info[n]['bloom_level'] for n in path_context[-2:]] + [v_info['bloom_level']]
            bloom_trend = np.diff(recent_blooms)
            if all(bt == 1 for bt in bloom_trend):
                likelihood['bloom_consistent'] = 0.9
            elif all(bt == 0 for bt in bloom_trend):
                likelihood['bloom_consistent'] = 0.8
            else:
                likelihood['bloom_consistent'] = 0.3
        else:
            likelihood['bloom_consistent'] = 0.6
        
        # 特征3：章节内连续跳转
        if len(path_context) >= 3:
            recent_chapters = [self.node_info[n]['chapter_id'] for n in path_context[-2:]] + [v_info['chapter_id']]
            if all(c == recent_chapters[0] for c in recent_chapters):
                likelihood['intra_chapter_consec'] = 0.9
            else:
                likelihood['intra_chapter_consec'] = 0.5
        else:
            likelihood['intra_chapter_consec'] = 0.7
        
        # 似然融合（几何平均）
        total_likelihood = np.prod(list(likelihood.values())) ** (1/len(likelihood))
        self.likelihood_cache[(from_node, to_node)] = {
            'feature_likelihood': likelihood,
            'total_likelihood': total_likelihood
        }
        return self.likelihood_cache[(from_node, to_node)]

    def _calculate_bayesian_posterior(self, from_node, to_node, path_context: list):
        """重构后的贝叶斯后验概率计算"""
        # 先验概率P(合理跳转)
        prior = self.prior_prob.get((from_node, to_node), 0.5)
        
        # 似然P(特征|合理跳转)
        likelihood_data = self._calculate_likelihood(from_node, to_node, path_context)
        likelihood = likelihood_data['total_likelihood']
        
        # 证据P(特征)：加权平均近似
        evidence = likelihood * prior + (1 - likelihood) * (1 - prior)
        
        # 后验概率
        posterior = (likelihood * prior) / (evidence + 1e-8)
        
        return posterior, prior, likelihood

    def load_graph(self):
        self.G = nx.DiGraph()
        self.node_attrs = dict()
        self.node_info = dict()
        self.prior_prob.clear()
        self.likelihood_cache.clear()
        self.node_chapter_order.clear()

        # 加载节点数据（含章节顺序）
        q_nodes = """
        MATCH (n) 
        WHERE n:KnowledgePoint OR n:SubKnowledgePoint OR n:Chapter
        RETURN 
            id(n) as nid,
            labels(n)[0] as label,
            coalesce(n.title, '未命名节点') as title,
            coalesce(n.chapter_id, 0) as chapter_id,
            coalesce(n.order, 0) as chapter_order,
            coalesce(n.edu_pagerank, 0.0) as edu_pr,
            coalesce(n.bloom_level, 0) as bloom_level,
            coalesce(n.syllabus_mentions, 0) as syllabus_mentions,
            coalesce(n.class_hours, 0) as class_hours
        """
        nodes_result = list(self.graphdb.run(q_nodes))
        for rec in nodes_result:
            nid = rec['nid']
            self.G.add_node(nid)
            
            # 解析章节顺序
            chapter_id = rec['chapter_id']
            chapter_order = rec['chapter_order'] if rec['label'] == 'Chapter' else self._get_chapter_order(chapter_id)
            self.node_chapter_order[nid] = (chapter_id, int(chapter_order) if chapter_order else 0)
            
            self.node_info[nid] = {
                'label': rec['label'],
                'title': rec['title'],
                'chapter_id': chapter_id,
                'bloom_level': rec['bloom_level'],
                'chapter_order': self.node_chapter_order[nid][1],
                'syllabus_mentions': rec['syllabus_mentions'],
                'class_hours': rec['class_hours'],
                'node_order': rec.get('node_order', 0),  # 节点在章节内的顺序
                'is_core': False  # 后续更新
            }
            self.node_attrs[nid] = {'edu_pr': max(rec['edu_pr'], 0.03)}

        # 识别核心知识点
        pr_values = [self.node_attrs[n]['edu_pr'] for n in self.G.nodes()]
        if pr_values:
            self.core_kp_threshold = np.percentile(pr_values, 80)  # 前20%为核心知识点
            for nid in self.G.nodes():
                self.node_info[nid]['is_core'] = self.node_attrs[nid]['edu_pr'] >= self.core_kp_threshold

        # 加载边数据（含关系类型）
        q_edges = """
        MATCH (a)-[r]->(b)
        WHERE (a:KnowledgePoint OR a:SubKnowledgePoint OR a:Chapter)
            AND (b:KnowledgePoint OR b:SubKnowledgePoint OR b:Chapter)
        RETURN 
            id(a) as src, 
            id(b) as tgt, 
            coalesce(r.weight, 1.0) as weight,
            type(r) as rel_type
        """
        edges_result = list(self.graphdb.run(q_edges))
        edges_to_add = [(rec['src'], rec['tgt'], {'weight': rec['weight'] or 1.0, 'rel_type': rec['rel_type']}) for rec in edges_result]
        self.G.add_edges_from(edges_to_add)

        # 初始化贝叶斯先验概率
        self._init_bayesian_prior(edges_result)

        logger.info(f"加载节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}")
        logger.info(f"核心知识点数: {sum(1 for n in self.G.nodes() if self.node_info[n]['is_core'])} (阈值: {self.core_kp_threshold:.4f})")
        logger.info(f"贝叶斯先验概率初始化完成，共{len(self.prior_prob)}条关系")

    def get_node_details(self, node_id):
        if node_id not in self.node_info:
            return None
        return {
            'id': node_id,
            'label': self.node_info[node_id]['label'],
            'title': self.node_info[node_id]['title'],
            'chapter_id': self.node_info[node_id]['chapter_id'],
            'bloom_level': self.node_info[node_id]['bloom_level'],
            'edu_pr': self.node_attrs[node_id].get('edu_pr', 0.0),
            'is_core': self.node_info[node_id]['is_core']
        }

    def compute_edu_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        G = nx.DiGraph()
        G.add_nodes_from(self.G.nodes())
        G.add_edges_from([(u, v, data) for u, v, data in self.G.edges(data=True)])
        N = G.number_of_nodes()
        
        if N == 0:
            logger.error("无节点数据")
            return

        max_syllabus = max(self.node_info[node]['syllabus_mentions'] for node in self.G.nodes()) if self.node_info else 1
        max_hours = max(self.node_info[node]['class_hours'] for node in self.G.nodes()) if self.node_info else 1

        for node in G.nodes:
            data = self.node_info[node]
            syllabus_norm = np.log1p(data.get("syllabus_mentions", 0)) / np.log1p(max_syllabus) if max_syllabus > 0 else 0
            hours_norm = data.get("class_hours", 0) / max_hours if max_hours > 0 else 0
            bloom_norm = (7 - data["bloom_level"]) / 6.0 if data["bloom_level"] > 0 else 0
            
            static_weight = float(0.5 * hours_norm + 0.3 * syllabus_norm + 0.2 * bloom_norm)
            G.nodes[node]["static_weight"] = static_weight
            G.nodes[node]["edu_pr"] = 1.0 / N

        for iter_count in range(max_iter):
            new_pr = {}
            for n in G.nodes:
                pr_value = (1 - alpha) / N
                
                for pred in G.predecessors(n):
                    out_weight = sum(G.edges[pred, succ]['weight'] for succ in G.successors(pred))
                    if out_weight > 0:
                        pr_value += alpha * (
                            G.nodes[pred]["static_weight"] * 
                            G.nodes[pred]["edu_pr"] * 
                            G.edges[pred, n]['weight'] / out_weight
                        )
                
                new_pr[n] = pr_value

            total = sum(new_pr.values())
            for n in new_pr:
                new_pr[n] /= total if total > 0 else 1

            diff = max(abs(new_pr[n] - G.nodes[n]["edu_pr"]) for n in G.nodes)
            
            for n in G.nodes:
                G.nodes[n]["edu_pr"] = new_pr[n]

            logger.info(f"Iter {iter_count + 1} | Diff: {diff:.8f}")
            
            if diff < tol:
                logger.info("算法收敛")
                break

        for node in G.nodes:
            self.node_attrs[node]['edu_pr'] = G.nodes[node]["edu_pr"]

        self.write_results_to_neo4j()

    def write_results_to_neo4j(self):
        pr_values = [self.node_attrs[node]['edu_pr'] for node in self.G.nodes()]
        pr_values.sort(reverse=True)

        threshold_index = max(int(len(pr_values) * 0.2) - 1, 0)
        threshold_value = pr_values[threshold_index] if pr_values else 0
        self.core_kp_threshold = threshold_value

        logger.info(f"核心知识点阈值: {threshold_value:.6f}")

        data = []
        for node_id in self.G.nodes():
            is_core = self.node_attrs[node_id]['edu_pr'] >= threshold_value
            self.node_info[node_id]['is_core'] = is_core
            data.append({
                "nid": node_id,
                "edu_pr": round(self.node_attrs[node_id]['edu_pr'], 6),
                "is_core_kp": is_core
            })

        query = """
        UNWIND $data AS row
        MATCH (n)
        WHERE id(n) = row.nid
        SET n.edu_pagerank = row.edu_pr,
            n.is_core_kp = row.is_core_kp
        """

        self.graphdb.run(query, data=data)
        logger.info("教育PageRank及核心标志写入完成")

    def update_pagerank(self):
        N = self.G.number_of_nodes()
        if N == 0:
            logger.warning("图中无节点，跳过PageRank更新")
            return

        edge_decay = {(u, v): self.G[u][v]['weight'] for u, v in self.G.edges()}
        out_edges = {u: list(self.G.successors(u)) for u in self.G.nodes()}
        
        pr = {node: 1.0 / N for node in self.G.nodes()}
        gamma = 0.8

        for iter_round in range(8):
            new_pr = {}
            for n in self.G.nodes():
                pr_value = (1 - self.alpha) / N
                for pred in self.G.predecessors(n):
                    if (pred, n) not in edge_decay:
                        continue
                    w_ij = edge_decay[(pred, n)]
                    out_deg = sum(edge_decay[(pred, succ)] for succ in out_edges[pred]) if out_edges[pred] else 1.0
                    if out_deg > 0:
                        pr_value += self.alpha * (w_ij * pr[pred]) / out_deg
                
                # 核心知识点奖励
                if self.node_info[n]['is_core']:
                    pr_value *= self.core_kp_weight
                
                new_pr[n] = pr_value
            
            total = sum(new_pr.values())
            if total > 0:
                for n in new_pr:
                    new_pr[n] /= total
            
            pr = new_pr.copy()

        for node in self.G.nodes():
            self.node_attrs[node]['edu_pr'] = pr[node]

        self.write_results_to_neo4j()
        logger.info("教育版PageRank权威值（含边衰减）更新完成")

    def _calculate_smoothness_penalty(self, current, neighbor):
        """增强版：计算路径平滑度惩罚（使用类内参数）"""
        current_info = self.node_info[current]
        neighbor_info = self.node_info[neighbor]
        
        penalty = 0
        
        # 章节切换惩罚（动态权重）
        if current_info['chapter_id'] != neighbor_info['chapter_id']:
            # 检查章节顺序是否合理
            if current_info['chapter_order'] > neighbor_info['chapter_order']:
                penalty += (self.smoothness_params['chapter_transition_penalty'] * 
                           self.smoothness_params['chapter_reverse_penalty'])
            else:
                penalty += self.smoothness_params['chapter_transition_penalty']
            
            # 跨章节距离惩罚
            chapter_diff = abs(current_info['chapter_order'] - neighbor_info['chapter_order'])
            if chapter_diff > 1:
                penalty += chapter_diff * 0.3
        
        # Bloom层级跳跃惩罚（更精细）
        bloom_diff = neighbor_info['bloom_level'] - current_info['bloom_level']
        if bloom_diff > 2:
            penalty += (bloom_diff - 1) * self.smoothness_params['bloom_jump_penalty']
        elif bloom_diff < 0:
            penalty += abs(bloom_diff) * self.smoothness_params['bloom_jump_penalty'] * 0.8
        
        # 应用动态平滑度权重
        return penalty * self.smoothness_weight

    def a_star_search(self, start, goal):
        if start == goal:
            return [start], 0
            
        if start not in self.G or goal not in self.G:
            logger.error("起始或目标节点不存在")
            return None, float('inf')

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = dict()
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal)}
        visited = set()
        heap_max_size = 800
        prune_threshold = 2.0

        # 节点缓存
        node_cache = {}
        for nid in self.G.nodes():
            node_info = self.node_info.get(nid, {})
            node_attrs = self.node_attrs.get(nid, {})
            node_cache[nid] = {
                'chapter_id': node_info.get('chapter_id', 0),
                'bloom_level': node_info.get('bloom_level', 0),
                'is_core': node_info.get('is_core', False),
                'edu_pr': node_attrs.get('edu_pr', 0.0),
                'chapter_order': node_info.get('chapter_order', 0)
            }
        adjacency_list = defaultdict(list)
        for u, v, attrs in self.G.edges(data=True):
            adjacency_list[u].append((v, attrs.get('weight', 1.0), attrs.get('rel_type', 'UNKNOWN')))

        goal_chapter = node_cache[goal]['chapter_id']
        goal_bloom = node_cache[goal]['bloom_level']

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return self._reconstruct_path(came_from, goal), g_score[current]

            best_known = g_score.get(goal, float('inf'))
            if g_score[current] > best_known * prune_threshold:
                continue

            current_data = node_cache[current]
            current_g = g_score[current]
            current_chapter = current_data['chapter_id']
            current_bloom = current_data['bloom_level']

            for neighbor, edge_weight, rel_type in adjacency_list.get(current, []):
                if neighbor in visited:
                    continue

                # 构建路径上下文
                path_context = list(self._reconstruct_path(came_from, current)) + [current]
                
                # 基础代价计算
                neighbor_data = node_cache[neighbor]
                delta_bloom = abs(current_bloom - neighbor_data['bloom_level'])
                
                # Bloom层级惩罚优化
                if delta_bloom == 0:
                    bloom_penalty = 0
                elif delta_bloom == 1:
                    bloom_penalty = 0.5
                else:
                    bloom_penalty = delta_bloom * 2
                
                # 核心知识点奖励增强
                core_bonus = -1.0 if neighbor_data['is_core'] else 0
                if current_data['is_core'] and neighbor_data['is_core']:
                    core_bonus = -1.5  # 连续核心知识点额外奖励
                
                # 章节内奖励
                chapter_bonus = -0.8 if current_chapter == neighbor_data['chapter_id'] else 0
                
                # 重构后的贝叶斯惩罚计算
                posterior, prior, likelihood = self._calculate_bayesian_posterior(current, neighbor, path_context)
                bayes_penalty = -math.log(posterior + 1e-8) * self.lambda_bayes
                
                # 平滑度惩罚（使用类内参数）
                smoothness_penalty = self._calculate_smoothness_penalty(current, neighbor)
                
                # 其他代价项
                w_prime = edge_weight * (1.0 + 20.0 * neighbor_data['edu_pr'])
                edu_pr_inverse = 1.0 / max(neighbor_data['edu_pr'], 0.01)
                continuity_reward = -0.5 if current_chapter == neighbor_data['chapter_id'] else 0.0
                penalty = 3 * delta_bloom * 0.2 if neighbor_data['is_core'] else (3 * delta_bloom if delta_bloom > 2 else 0.0)

                # 总代价
                tentative_g = (current_g + w_prime + penalty + continuity_reward + 
                              bayes_penalty + smoothness_penalty + core_bonus + chapter_bonus)
                tentative_f = tentative_g + self._heuristic(current, goal) + edu_pr_inverse

                if tentative_f < f_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_f
                    heapq.heappush(open_heap, (tentative_f, neighbor))

                if len(open_heap) > heap_max_size:
                    max_idx = max(range(len(open_heap)), key=lambda i: open_heap[i][0])
                    del open_heap[max_idx]
                    heapq.heapify(open_heap)

        return None, float('inf')

    def _heuristic(self, node, goal):
        """改进的启发函数，考虑核心知识点和平滑度"""
        node_data = self.node_info.get(node, {})
        goal_data = self.node_info.get(goal, {})
        
        # 基础启发
        delta_bloom = abs(node_data.get('bloom_level', 0) - goal_data.get('bloom_level', 0))
        node_pr = self.node_attrs.get(node, {}).get('edu_pr', 0.0)
        
        # 核心知识点奖励
        if node_data.get('is_core', False):
            core_bonus = -0.7  # 增强核心知识点奖励
        else:
            core_bonus = 0
        
        # 章节距离惩罚（更精细）
        chapter_diff = abs(node_data.get('chapter_order', 0) - goal_data.get('chapter_order', 0))
        if chapter_diff == 0:
            chapter_penalty = 0
        elif chapter_diff == 1:
            chapter_penalty = 0.1
        else:
            chapter_penalty = chapter_diff * 0.2
        
        # Bloom距离惩罚
        if delta_bloom <= 1:
            bloom_penalty = 0.1 * delta_bloom
        else:
            bloom_penalty = 0.3 * delta_bloom
        
        return max(0.2 * bloom_penalty - 120 * node_pr + core_bonus + chapter_penalty, 0.0)

    def validate_path(self, path):
        if not path:
            return False
        for i in range(len(path) - 1):
            if not self.G.has_edge(path[i], path[i + 1]):
                logger.warning(f"非法路径段：{path[i]} -> {path[i + 1]}")
                return False
        return True

    def evaluate_and_adjust(self, path):
        iteration = len(self.metrics['cost'])
        dynamic_feedback = max(0.2, 0.5 * (0.8 ** iteration))

        decay_rate = 0.5 if iteration < 5 else 0.7
        for u, v in self.G.edges():
            if u in path or v in path:
                # 核心知识点边权重衰减较慢
                if self.node_info[u]['is_core'] or self.node_info[v]['is_core']:
                    self.G[u][v]['weight'] *= (decay_rate + 0.2)
                else:
                    self.G[u][v]['weight'] *= decay_rate

        self.update_pagerank()
        logger.info("完成权重修正")

    def _dynamic_alpha(self, current_iter, max_iters, base_alpha):
        progress = current_iter / max_iters
        return max(0.6, base_alpha * (1 - progress * 0.3))

    def _update_smoothness_weight(self, current_iter):
        """动态更新平滑度权重"""
        self.smoothness_weight = (self.smoothness_weight_base * 
                                 (1 + current_iter * self.smoothness_params['smoothness_weight_growth']))
        logger.info(f"平滑度权重更新为: {self.smoothness_weight:.2f}")

    def _check_convergence(self, cost, unstable_count):
        if len(self.metrics['cost']) >= 4:
            last_3 = self.metrics['cost'][-3:]
            if max(last_3) - min(last_3) > 1.0:
                return unstable_count + 1
        return 0

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _handle_self_loop(self, node_id):
        self.load_graph()
        self._record_metrics([node_id], 0)
        self.update_pagerank()
        self._record_metrics([node_id], 0)
        for nid in self.G.nodes():
            self.pr_history[nid].append(self.node_attrs[nid].get('edu_pr', 0))
        if node_id not in self.G.nodes:
            logger.error("自循环节点不存在于图谱中")
            return None, float('inf')
        print(f"\n路径搜索：{self.get_node_details(node_id)['label']} [自循环路径]")
        return [node_id], 0

    def _validate_and_update(self, path, cost, best_path, min_cost):
        if not path:
            logger.warning("无法找到有效路径，终止迭代")
            return False
        if cost < min_cost and self.validate_path(path):
            best_path, min_cost = path.copy(), cost
        return True

    def collaborative_iteration(self, start, goal, max_iters=10):
        self.pr_history = defaultdict(list)
        self.current_iteration = 0
        self.smooth_paths.clear()  # 清空历史路径
        
        if start == goal:
            best_path, min_cost = self._handle_self_loop(start)
            if best_path is not None:
                self._record_metrics(best_path, min_cost)
                self._finalize_iteration()
            return best_path, min_cost

        self.load_graph()
        best_path, min_cost = None, float('inf')
        unstable_count = 0
        original_alpha = self.alpha

        node_exists = set(self.G.nodes())
        if start not in node_exists or goal not in node_exists:
            logger.error("起始或目标节点不存在")
            return None, float('inf')

        for iteration in range(max_iters):
            self.current_iteration = iteration
            # 动态更新平滑度权重
            self._update_smoothness_weight(iteration)
            
            self.alpha = self._dynamic_alpha(iteration, max_iters, original_alpha)
            logger.info(f"=== 迭代第 {iteration + 1} 轮 (alpha={self.alpha:.2f}, 平滑度权重={self.smoothness_weight:.2f}) ===")
            
            path, cost = self.a_star_search(start, goal)

            if not path:
                logger.warning("无法找到有效路径，终止迭代")
                break
            
            # 检查是否达到高平滑度阈值，提前终止
            current_smoothness = self._calculate_path_smoothness(path)
            if current_smoothness > self.smoothness_params['early_stop_smoothness_threshold']:
                logger.info(f"找到高平滑度路径（{current_smoothness:.3f}），提前终止迭代")
                best_path, min_cost = path.copy(), cost
                break
            
            if self.validate_path(path) and cost < min_cost:
                best_path, min_cost = path.copy(), cost

            pr_snapshot = {nid: self.node_attrs[nid].get('edu_pr', 0) for nid in self.G.nodes()}
            for nid, pr_val in pr_snapshot.items():
                self.pr_history[nid].append(pr_val)

            unstable_count = self._check_convergence(cost, unstable_count)
            if unstable_count >= 3:
                logger.info("检测到持续不稳定，延长迭代次数至40")
                max_iters = 40

            self.evaluate_and_adjust(path)
            self._record_metrics(path, cost)

            # 检查核心指标是否达标
            if len(self.metrics['core_kp_coverage']) >= 3:
                recent_coverage = self.metrics['core_kp_coverage'][-3:]
                recent_smoothness = self.metrics['path_smoothness'][-3:]
                if (np.mean(recent_coverage) > 0.7 and 
                    np.mean(recent_smoothness) > 0.8 and
                    max(self.metrics['cost'][-3:]) - min(self.metrics['cost'][-3:]) < 0.1):
                    logger.info("核心指标已达标且收敛稳定，提前终止迭代")
                    break

        self._finalize_iteration()
        return best_path, min_cost

    def _finalize_iteration(self):
        self._plot_metrics()
        self._generate_metric_report()
        self.pr_history = {}
        # 重置平滑度权重
        self.smoothness_weight = self.smoothness_weight_base

    def print_path_details(self, path):
        """增强版：打印路径的详细属性，包括核心知识点标记"""
        if not path or len(path) < 2:
            print("路径无效或长度不足，无法打印详情")
            return

        print("\n=== 路径详细属性 ===")
        print(f"核心知识点覆盖率: {self._calculate_core_kp_coverage(path):.2%}")
        print(f"路径平滑度: {self._calculate_path_smoothness(path):.2%}")
        print("-" * 60)
        
        for i in range(len(path)):
            node = path[i]
            details = self.get_node_details(node)
            if not details:
                continue
                
            core_mark = "★" if details['is_core'] else " "
            print(f"节点 {i+1}{core_mark}: {details['title']}")
            print(f"  类型: {details['label']}")
            print(f"  章节ID: {details['chapter_id']} (顺序: {self.node_info[node]['chapter_order']})")
            print(f"  Bloom层级: {details['bloom_level']}")
            print(f"  教育权威值: {details['edu_pr']:.4f} {'(核心知识点)' if details['is_core'] else ''}")
            
            if i < len(path) - 1:
                edge_data = self.G.get_edge_data(path[i], path[i+1])
                rel_type = edge_data.get('rel_type', '未知关系')
                weight = edge_data.get('weight', 1.0)
                print(f"  → 边关系: {rel_type}, 权重: {weight:.2f}")
                
                # 平滑度分析
                next_node = path[i+1]
                if self.node_info[node]['chapter_id'] != self.node_info[next_node]['chapter_id']:
                    print(f"    章节切换: {self.node_info[node]['chapter_id']} → {self.node_info[next_node]['chapter_id']}")
                
                bloom_diff = self.node_info[next_node]['bloom_level'] - self.node_info[node]['bloom_level']
                if bloom_diff > 1:
                    print(f"    Bloom跳跃: +{bloom_diff} (建议平滑过渡)")
                elif bloom_diff < 0:
                    print(f"    Bloom下降: {bloom_diff} (层级倒退)")
            
            print("-" * 60)

    def visualize_knowledge_graph(self, path=None, figsize=(16, 14), dpi=300):
        """可视化知识图谱，突出显示核心知识点和路径平滑度"""
        if self.G.number_of_nodes() == 0:
            logger.warning("图谱为空，无法可视化")
            return

        # 1. 准备节点样式数据
        node_types = {n['label'] for n in self.node_info.values()}
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 基础节点颜色（章节/知识点/子知识点）
        highlight_color = '#ff3333'  # 路径节点高亮色
        core_color = '#ffd700'       # 核心知识点颜色
        color_map = ListedColormap(base_colors[:len(node_types)])
        
        # 为Bloom层级创建渐变色映射
        bloom_cmap = LinearSegmentedColormap.from_list(
            'bloom_cmap', ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c']
        )

        node_color = []
        node_size = []
        node_details = {}  # 存储节点详细信息（用于标注）
        node_list = list(self.G.nodes())
        
        for nid in node_list:
            info = self.node_info[nid]
            # 基础颜色（按节点类型）
            type_idx = list(node_types).index(info['label'])
            base_color = color_map(type_idx)
            
            # 核心知识点使用金色
            if info['is_core']:
                base_color = core_color
            
            # 节点大小（权威值+层级+核心标记）
            size = 300 + 800 * self.node_attrs[nid]['edu_pr'] + 150 * info['bloom_level']
            if info['is_core']:
                size *= 1.3  # 核心节点更大
            node_size.append(max(200, size))
            
            # 存储节点详细信息
            node_details[nid] = {
                'title': info['title'][:12] + '...' if len(info['title']) > 12 else info['title'],
                'chapter': info['chapter_id'],
                'bloom': info['bloom_level'],
                'is_core': info['is_core'],
                'base_color': base_color
            }

        # 2. 准备边样式数据
        edge_styles = {}
        edge_colors = {}
        edge_labels = {}  # 存储边关系标签
        for u, v, data in self.G.edges(data=True):
            rel_type = data['rel_type']
            if rel_type == 'REQUIRES_PREREQUISITE':
                edge_styles[(u, v)] = 'solid'
                edge_colors[(u, v)] = '#d62728'  # 红色
                edge_labels[(u, v)] = '先修'
            elif rel_type == 'NEXT_CHAPTER':
                edge_styles[(u, v)] = 'dashed'
                edge_colors[(u, v)] = '#2ca02c'  # 绿色
                edge_labels[(u, v)] = '后续章节'
            elif rel_type == 'CONTAINS':
                edge_styles[(u, v)] = 'dotted'
                edge_colors[(u, v)] = '#9467bd'  # 紫色
                edge_labels[(u, v)] = '包含'
            else:
                edge_styles[(u, v)] = 'dashdot'
                edge_colors[(u, v)] = '#7f7f7f'  # 灰色
                edge_labels[(u, v)] = '关联'

        # 3. 布局计算
        pos = nx.spring_layout(self.G, k=0.4, iterations=100, seed=42)  # 固定种子确保布局一致

        # 4. 创建绘图
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # 4.1 绘制非路径节点和边（底层）
        if path:
            non_path_nodes = [n for n in self.G.nodes() if n not in path]
            non_path_edges = [(u, v) for u, v in self.G.edges() if u not in path or v not in path]
        else:
            non_path_nodes = self.G.nodes()
            non_path_edges = self.G.edges()

        # 绘制非路径节点（半透明）
        nx.draw_networkx_nodes(
            self.G, pos, nodelist=non_path_nodes,
            node_color=[node_details[n]['base_color'] for n in non_path_nodes],
            node_size=[node_size[node_list.index(n)] for n in non_path_nodes],
            alpha=0.4, ax=ax
        )

        # 绘制非路径边（半透明）
        nx.draw_networkx_edges(
            self.G, pos, edgelist=non_path_edges,
            edge_color=[edge_colors[e] for e in non_path_edges],
            style=[edge_styles[e] for e in non_path_edges],
            width=1, alpha=0.3, ax=ax
        )

        # 4.2 突出显示路径节点和边（顶层）
        if path and len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # 绘制路径边（加粗高亮）
            nx.draw_networkx_edges(
                self.G, pos, edgelist=path_edges,
                edge_color='#ff3333', width=3, alpha=0.9,
                ax=ax, arrowstyle='->', arrowsize=20
            )
            
            # 绘制路径节点（高亮色+放大）
            path_node_colors = []
            path_node_sizes = []
            for n in path:
                if node_details[n]['is_core']:
                    path_node_colors.append('#ffaa00')  # 核心节点用橙色高亮
                    path_node_sizes.append(node_size[node_list.index(n)] * 1.8)
                else:
                    path_node_colors.append(highlight_color)
                    path_node_sizes.append(node_size[node_list.index(n)] * 1.5)
            
            nx.draw_networkx_nodes(
                self.G, pos, nodelist=path,
                node_color=path_node_colors, node_size=path_node_sizes,
                alpha=0.9, ax=ax, edgecolors='black', linewidths=2
            )

            # 4.3 标注路径节点的详细信息
            for i, nid in enumerate(path):
                details = node_details[nid]
                # 节点标签（标题）
                nx.draw_networkx_labels(
                    self.G, pos, {nid: details['title']},
                    font_size=10, font_color='black', font_weight='bold',
                    ax=ax
                )
                # 详细信息标注（偏移位置避免重叠）
                x, y = pos[nid]
                offset = 0.05 * (i % 2 * 2 - 1)  # 交替偏移避免重叠
                core_mark = "★核心" if details['is_core'] else ""
                plt.text(
                    x + offset, y - 0.08,
                    f"章节: {details['chapter']}\nBloom: {details['bloom']} {core_mark}",
                    fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.8, pad=2)
                )

            # 4.4 标注路径边的关系类型和平滑度
            for i, (u, v) in enumerate(path_edges):
                mid_pos = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
                
                # 检查平滑度问题
                smoothness_note = ""
                if self.node_info[u]['chapter_id'] != self.node_info[v]['chapter_id']:
                    smoothness_note = "章节切换"
                bloom_diff = self.node_info[v]['bloom_level'] - self.node_info[u]['bloom_level']
                if abs(bloom_diff) > 1:
                    smoothness_note += f" Bloom{bloom_diff:+d}"
                
                label_text = edge_labels.get((u, v), '关联')
                if smoothness_note:
                    label_text += f"\n({smoothness_note})"
                
                plt.text(
                    mid_pos[0], mid_pos[1] + 0.03,
                    label_text,
                    fontsize=9, color='#ff3333', fontweight='bold',
                    ha='center', bbox=dict(facecolor='white', alpha=0.7, pad=1)
                )

        # 4.5 非路径节点的简化标签
        if path:
            nx.draw_networkx_labels(
                self.G, pos, 
                {n: (node_details[n]['title'][:6] + '...') + ('★' if node_details[n]['is_core'] else '') 
                 for n in non_path_nodes},
                font_size=7, font_color='gray', alpha=0.7, ax=ax
            )

        # 5. 图例和标题
        plt.title('知识图谱路径可视化（红色为路径，金色为核心知识点）', fontsize=15, pad=20)
        
        # 添加自定义图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='路径节点',
                   markerfacecolor=highlight_color, markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='核心知识点',
                   markerfacecolor=core_color, markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='非路径节点',
                   markerfacecolor='#1f77b4', markersize=10, alpha=0.4),
            Line2D([0], [0], color='#ff3333', lw=3, label='路径边'),
            Line2D([0], [0], color='#7f7f7f', lw=1, label='非路径边'),
            Line2D([0], [0], color='w', label='Bloom层级越高',
                   marker='o', markerfacecolor='#e31a1c', markersize=10),
            Line2D([0], [0], color='w', label='Bloom层级越低',
                   marker='o', markerfacecolor='#ffffcc', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig('knowledge_graph_path_visualization_enhanced.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info("增强版知识图谱已保存至 knowledge_graph_path_visualization_enhanced.png")


if __name__ == "__main__":
    # 初始化系统（增强版参数）
    system = EnhancedCollaborativeLearning12(
        neo4j_uri="bolt://localhost:7687",
        user="neo4j",
        password="123456789",
        alpha=0.8,
        feedback_strength=0.5,
        lambda_bayes=10.0,
        beta_smooth=(2, 2),
        core_kp_weight=1.5,
        smoothness_weight=1.8  # 提高平滑度权重
    )
    
    # 加载图数据
    system.load_graph()
    
    # 从数据库随机获取测试用例
    query = """
    MATCH (n)
    WHERE n:KnowledgePoint OR n:SubKnowledgePoint
    WITH collect(id(n)) as nodes
    RETURN nodes
    """
    result = system.graphdb.run(query).data()
    if not result or not result[0]['nodes']:
        logger.error("未找到有效的知识点节点")
        exit(1)
    
    node_ids = result[0]['nodes']
    num_test_cases = 20  # 设置测试用例数量
    
    # 随机生成测试用例
    import random
    test_cases = []
    for i in range(num_test_cases):
        start = random.choice(node_ids)
        goal = random.choice(node_ids)
        while goal == start:  # 确保起点和终点不同
            goal = random.choice(node_ids)
            
        test_cases.append({
            "start": start,
            "goal": goal,
            "desc": f"随机测试用例 {i+1}"
        })
    
    final_results = []
    
    for case in test_cases:
        print(f"\n=== 测试用例 [{case['desc']}] 起止节点 ({case['start']}->{case['goal']}) ===")
        best_path, final_cost = system.collaborative_iteration(
            start=case["start"],
            goal=case["goal"],
            max_iters=30
        )
        
        if best_path is None or len(best_path) == 0:
            logger.warning(f"测试用例 {case['desc']} 未找到有效路径")
            continue
        
        final_results.append({
            "desc": case["desc"],
            "path": best_path,
            "cost": final_cost,
            "core_coverage": system._calculate_core_kp_coverage(best_path),
            "smoothness": system._calculate_path_smoothness(best_path),
            "metrics": system.metrics.copy()
        })
        
        print("\n=== 路径详细属性 ===")
        system.print_path_details(best_path)
        # system.visualize_knowledge_graph(path=best_path)
    
    print("\n=== 跨测试用例综合分析 ===")
    valid_cases = [
        res for res in final_results
        if res["path"] is not None
           and len(res["path"]) > 1
           and res["metrics"]['cost'][0] > 0
    ]
    
    if valid_cases:
        avg_core_coverage = np.mean([res["core_coverage"] for res in valid_cases])
        avg_smoothness = np.mean([res["smoothness"] for res in valid_cases])
        avg_cost_reduction = np.mean([
            (res["metrics"]['cost'][0] - res["metrics"]['cost'][-1]) / res["metrics"]['cost'][0]
            for res in valid_cases
        ])
        
        print(f"成功用例数: {len(valid_cases)}/{len(test_cases)}")
        print(f"平均成本优化率: {avg_cost_reduction:.2%}")
        print(f"平均核心知识点覆盖率: {avg_core_coverage:.2%}")
        print(f"平均路径平滑度: {avg_smoothness:.2%}")
        
        # 优秀案例分析
        excellent_cases = [res for res in valid_cases 
                          if res["core_coverage"] > 0.7 and res["smoothness"] > 0.7]
        print(f"\n优秀案例数（高覆盖率+高平滑度）: {len(excellent_cases)}/{len(valid_cases)}")
    else:
        print("无有效测试用例结果")