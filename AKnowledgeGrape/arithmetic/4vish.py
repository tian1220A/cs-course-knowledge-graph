import logging
from collections import defaultdict
import math
import json
from flask import Flask, jsonify, render_template_string

import networkx as nx
from py2neo import Graph
import heapq
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedCollaborativeLearning")

# 初始化Flask应用
app = Flask(__name__)

class EnhancedCollaborativeLearning:
    def __init__(self, neo4j_uri, user, password, alpha=0.85, feedback_strength=0.1, dynamic_threshold=0.008,
                 lambda_bayes=10.0, beta_smooth=(2, 2)):
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
            'pr_max_diff': []
        }
        
        # 贝叶斯模块参数
        self.lambda_bayes = lambda_bayes  # 贝叶斯惩罚权重
        self.beta_smooth = beta_smooth    # Beta平滑参数
        self.prior_prob: dict[tuple, float] = {}  # (from_node, to_node) -> 先验概率
        self.node_chapter_order: dict[int, tuple] = {}  # node_id -> (chapter_id, chapter_order)

    def _plot_metrics(self):
        plt.figure(figsize=(14, 8))
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['cost'], 'o-')
        plt.title('路径成本迭代变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('成本')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['path_length'], 's-')
        plt.title('路径长度迭代变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('节点数')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.metrics['pr_std'], '^-')
        plt.title('权威值标准差变化')
        plt.xlabel('迭代轮次')
        plt.ylabel('标准差')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('collaborative_metrics.png', dpi=300)
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

        report = {
            '最终成本': final_cost,
            '成本下降率': cost_reduction,
            '路径收敛稳定性': cost_stability,
            '权威值集中度': pr_std_mean,
            '权威值最大差异': pr_max_diff
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
            print(
                f"节点{nid} ({detail['title']}): 初始 {init_pr:.4f} → 最终 {final_pr:.4f}, 绝对 {delta:.4f}, 相对 {rel:.2%}")

        print("\n=== 指标分析报告 ===")
        for k, v in report.items(): print(f"{k}: {v:.4f}")
        print("\n=== 分析结论 ===")
        for c in conclusions: print(c)

        with open('metric_report.txt', 'w') as f:
            f.write("=== Collaborative Learning Metrics ===\n")
            for k, v in report.items(): f.write(f"{k}: {v}\n")
            f.write("\n=== Conclusions ===\n")
            f.write("\n".join(conclusions))

    def _record_metrics(self, path, cost):
        self.metrics['cost'].append(cost)
        self.metrics['path_length'].append(len(path))

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

    def _get_chapter_order(self, chapter_id):
        if not chapter_id:
            return 0
        q = f"MATCH (c:Chapter {{chapter_id: '{chapter_id}'}}) RETURN c.order as order"
        res = self.graphdb.run(q).data()
        return res[0]['order'] if res else 0

    def _init_bayesian_prior(self, edges_result):
        for rec in edges_result:
            src = rec['src']
            tgt = rec['tgt']
            rel_type = rec['rel_type']
            src_chapter_order = self.node_chapter_order[src][1]
            tgt_chapter_order = self.node_chapter_order[tgt][1]

            # 初始化先验概率
            if rel_type == 'REQUIRES_PREREQUISITE':
                prior = 0.9
            elif src_chapter_order < tgt_chapter_order:
                prior = 0.7
            elif src_chapter_order == tgt_chapter_order:
                prior = 0.6
            else:
                prior = 0.1

            # Beta平滑处理
            alpha, beta = self.beta_smooth
            smoothed = (prior * (alpha + beta - 2) + alpha - 1) / (alpha + beta - 1)
            self.prior_prob[(src, tgt)] = smoothed

    def load_graph(self):
        self.G = nx.DiGraph()
        self.node_attrs = dict()
        self.node_info = dict()
        self.prior_prob.clear()
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
            coalesce(n.bloom_level, 0) as bloom_level
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
                'chapter_order': self.node_chapter_order[nid][1]
            }
            self.node_attrs[nid] = {'edu_pr': max(rec['edu_pr'], 0.03)}

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
        edges_to_add = [(rec['src'], rec['tgt'], {'weight': rec['weight'] or 1.0}) for rec in edges_result]
        self.G.add_edges_from(edges_to_add)

        # 初始化贝叶斯先验概率
        self._init_bayesian_prior(edges_result)

        logger.info(f"加载节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}")
        logger.info(f"贝叶斯先验概率初始化完成，共{len(self.prior_prob)}条关系")

    def get_node_details(self, node_id):
        if node_id not in self.node_info:
            return None
        return {
            'id': node_id,
            'label': self.node_info[node_id]['label'],
            'title': self.node_info[node_id]['title'],
            'chapter_id': self.node_info[node_id]['chapter_id']
        }

    def compute_edu_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        G = nx.DiGraph()
        G.add_nodes_from(self.node_attrs.keys())
        G.add_edges_from([(u, v) for u, v, _ in self.G.edges(data=True)])
        N = G.number_of_nodes()
        
        if N == 0:
            logger.error("无节点数据")
            return

        max_syllabus = 1  # 简化处理，实际应从节点属性获取
        max_hours = 1      # 简化处理
        nodes = {nid: {'syllabus_mentions': 1, 'class_hours': 1, 'bloom_level': 1} for nid in G.nodes()}

        for node in G.nodes:
            data = nodes[node]
            syllabus_norm = np.log1p(data["syllabus_mentions"]) / np.log1p(max_syllabus)
            hours_norm = data["class_hours"] / max_hours if max_hours else 0
            bloom_norm = (7 - data["bloom_level"]) / 6.0
            
            static_weight = float(0.5 * hours_norm + 0.3 * syllabus_norm + 0.2 * bloom_norm)
            G.nodes[node]["static_weight"] = static_weight
            G.nodes[node]["edu_pr"] = 1.0 / N

        for iter_count in range(max_iter):
            new_pr = {}
            for n in G.nodes:
                pr_value = (1 - alpha) / N
                
                for pred in G.predecessors(n):
                    out_weight = sum(G.edges[pred, succ].get('weight', 1) 
                                for succ in G.successors(pred))
                    if out_weight > 0:
                        pr_value += alpha * (
                            G.nodes[pred]["static_weight"] * 
                            G.nodes[pred]["edu_pr"] * 
                            G.edges[pred, n].get('weight', 1) / out_weight
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
                self.node_attrs[node]["edu_pr"] = G.nodes[node]["edu_pr"]

    def write_results_to_neo4j(self):
        pr_values = [v["edu_pr"] for v in self.node_attrs.values()]
        pr_values.sort(reverse=True)

        threshold_index = max(int(len(pr_values) * 0.2) - 1, 0)
        threshold_value = pr_values[threshold_index] if pr_values else 0

        logger.info(f"核心知识点阈值: {threshold_value:.6f}")

        data = []
        for node_id, info in self.node_attrs.items():
            data.append({
                "id": node_id,
                "edu_pr": round(info["edu_pr"], 6),
                "is_core_kp": info["edu_pr"] >= threshold_value
            })

        query = """
        UNWIND $data AS row
        MATCH (n)
        WHERE id(n) = row.id
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
            out_deg_total = {}
            for u in self.G.nodes():
                out_deg_total[u] = sum(edge_decay[(u, succ)] for succ in out_edges[u]) if out_edges[u] else 1.0

            for node in self.G.nodes():
                pr_value = (1 - self.alpha) / N
                for pred in self.G.predecessors(node):
                    if (pred, node) not in edge_decay:
                        continue
                    w_ij = edge_decay[(pred, node)]
                    out_deg = out_deg_total[pred]
                    if out_deg > 0:
                        pr_value += self.alpha * (w_ij * pr[pred]) / out_deg
                new_pr[node] = pr_value
            
            total = sum(new_pr.values())
            if total > 0:
                for node in new_pr:
                    new_pr[node] /= total
            
            pr = new_pr.copy()

            for u, v in edge_decay.keys():
                edge_decay[(u, v)] *= gamma

        update_queries = []
        for nid in pr:
            edu_pr = pr[nid]
            self.node_attrs[nid]['edu_pr'] = edu_pr
            update_queries.append({
                'nid': nid,
                'pr': edu_pr
            })
        
        if update_queries:
            self.graphdb.run("""
                UNWIND $data AS row
                MATCH (n) WHERE id(n) = row.nid
                SET n.edu_pagerank = row.pr
            """, data=update_queries)
        
        logger.info("教育版PageRank权威值（含边衰减）更新完成")

    # 贝叶斯核心方法
    def _bayes_likelihood(self, from_node, to_node):
        """似然函数：评估节点顺序合理性"""
        from_info = self.node_info[from_node]
        to_info = self.node_info[to_node]
        from_chapter_order = from_info['chapter_order']
        to_chapter_order = to_info['chapter_order']

        # 规则1：官方先修关系
        rel_query = self.graphdb.run(f"""
            MATCH (a)-[r]->(b) 
            WHERE id(a)={from_node} AND id(b)={to_node} AND type(r)='REQUIRES_PREREQUISITE'
            RETURN count(r) > 0 as is_prerequisite
        """).data()
        if rel_query and rel_query[0]['is_prerequisite']:
            return 1.0
        
        # 规则2：符合章节顺序
        if from_chapter_order < to_chapter_order or from_info['chapter_id'] == to_info['chapter_id']:
            return 0.8
        
        # 规则3：逆序或无关
        return 0.1

    def _calculate_bayesian_posterior(self, from_node, to_node):
        """计算后验概率：先验 * 似然"""
        prior = self.prior_prob.get((from_node, to_node), 0.1)
        likelihood = self._bayes_likelihood(from_node, to_node)
        return prior * likelihood + 1e-8  # 避免log(0)

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
            adjacency_list[u].append((v, attrs.get('weight', 1.0)))

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

            for neighbor, edge_weight in adjacency_list.get(current, []):
                if neighbor in visited:
                    continue

                # 基础代价计算
                neighbor_data = node_cache[neighbor]
                delta_bloom = abs(current_bloom - neighbor_data['bloom_level'])
                h = max(0.3 * delta_bloom - 100 * neighbor_data['edu_pr'], 0.0)
                penalty = 3 * delta_bloom * 0.2 if neighbor_data['is_core'] else (3 * delta_bloom if delta_bloom > 2 else 0.0)
                continuity_reward = -0.5 if current_chapter == neighbor_data['chapter_id'] else 0.0
                w_prime = edge_weight * (1.0 + 20.0 * neighbor_data['edu_pr'])
                edu_pr_inverse = 1.0 / max(neighbor_data['edu_pr'], 0.01)

                # 贝叶斯惩罚计算
                posterior = self._calculate_bayesian_posterior(current, neighbor)
                bayes_penalty = -math.log(posterior) * self.lambda_bayes

                # 总代价（含贝叶斯惩罚）
                tentative_g = current_g + w_prime + penalty + continuity_reward + bayes_penalty
                tentative_f = tentative_g + h + edu_pr_inverse

                if tentative_f >= f_score.get(neighbor, float('inf')):
                    continue

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
        node_data = self.node_info.get(node, {})
        goal_data = self.node_info.get(goal, {})
        delta_bloom = abs(node_data.get('bloom_level', 0) - goal_data.get('bloom_level', 0))
        node_pr = self.node_attrs.get(node, {}).get('edu_pr', 0.0)
        return max(0.3 * delta_bloom - 100 * node_pr, 0.0)

    def validate_path(self, path):
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
                self.G[u][v]['weight'] *= decay_rate

        self.update_pagerank()
        logger.info("完成权重修正")

    def _dynamic_alpha(self, current_iter, max_iters, base_alpha):
        progress = current_iter / max_iters
        return max(0.6, base_alpha * (1 - progress * 0.3))

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
            best_path, min_cost = path, cost
        return True

    def collaborative_iteration(self, start, goal, max_iters=10):
        self.pr_history = defaultdict(list)

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
            self.alpha = self._dynamic_alpha(iteration, max_iters, original_alpha)
            logger.info(f"=== 迭代第 {iteration + 1} 轮 (alpha={self.alpha:.2f}) ===")
            path, cost = self.a_star_search(start, goal)

            if not path:
                logger.warning("无法找到有效路径，终止迭代")
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

            if len(self.metrics['cost']) >= 5:
                recent_costs = self.metrics['cost'][-5:]
                if max(recent_costs) - min(recent_costs) < 0.1:
                    logger.info("成本已收敛，提前终止迭代")
                    break

        self._finalize_iteration()
        return best_path, min_cost

    def _finalize_iteration(self):
        self._plot_metrics()
        self._generate_metric_report()
        self.pr_history = {}

    def get_path_with_attributes(self, start, goal):
        """获取路径及所有节点/边属性，用于前端可视化"""
        self.load_graph()
        path, cost = self.collaborative_iteration(start, goal, max_iters=30)
        if not path:
            return None, None, None

        # 生成章节颜色映射（固定色系，便于识别）
        chapter_ids = list({self.node_info[nid]['chapter_id'] for nid in path})
        chapter_colors = {
            cid: f"hsl({(i * 60) % 360}, 70%, 70%)"  # 每60度一个颜色
            for i, cid in enumerate(chapter_ids)
        }

        # 提取路径节点属性
        nodes = []
        for nid in path:
            info = self.node_info[nid]
            attrs = self.node_attrs[nid]
            # 节点大小：bloom_level * 10（认知属性可视化）
            size = max(20, info['bloom_level'] * 10)  # 最小20，避免过小
            # 章节颜色：使用预定义的颜色映射
            color = chapter_colors[info['chapter_id']]
            nodes.append({
                'id': nid,
                'label': info['title'][:15] + '...' if len(info['title']) > 15 else info['title'],
                'title': (f"节点ID: {nid}\n名称: {info['title']}\n"
                          f"章节ID: {info['chapter_id']}\n章节顺序: {info['chapter_order']}\n"
                          f"认知层级(Bloom): {info['bloom_level']}\n权威值: {attrs['edu_pr']:.4f}"),
                'size': size,
                'color': color,
                'chapter_id': info['chapter_id'],
                'chapter_order': info['chapter_order'],
                'bloom_level': info['bloom_level'],
                'edu_pr': attrs['edu_pr']
            })

        # 提取路径边属性
        edges = []
        for i in range(len(path)-1):
            from_nid = path[i]
            to_nid = path[i+1]
            # 获取边权重
            weight = self.G[from_nid][to_nid]['weight'] if self.G.has_edge(from_nid, to_nid) else 1.0
            # 查询关系类型
            rel_query = self.graphdb.run(f"""
                MATCH (a)-[r]->(b) WHERE id(a)={from_nid} AND id(b)={to_nid} RETURN type(r) as rel_type
            """).data()
            rel_type = rel_query[0]['rel_type'] if rel_query else 'UNKNOWN'
            # 贝叶斯先验概率
            prior = self.prior_prob.get((from_nid, to_nid), 0.1)
            # 边标签
            label = f"{rel_type}\n权重:{weight:.2f}\n先验:{prior:.2f}"
            edges.append({
                'from': from_nid,
                'to': to_nid,
                'label': label,
                'title': (f"从节点: {from_nid} → 到节点: {to_nid}\n关系类型: {rel_type}\n"
                          f"权重: {weight:.4f}\n贝叶斯先验概率: {prior:.4f}"),
                'weight': weight,
                'rel_type': rel_type,
                'prior_prob': prior,
                'color': '#666'
            })

        return nodes, edges, cost, chapter_colors

# 初始化系统实例
system = EnhancedCollaborativeLearning(
    neo4j_uri="bolt://localhost:7687",
    user="neo4j",
    password="123456789",
    alpha=0.8,
    feedback_strength=0.5,
    lambda_bayes=10.0,
    beta_smooth=(2, 2)
)

# Flask接口：获取路径数据
@app.route('/api/path/<int:start>/<int:goal>')
def get_path(start, goal):
    nodes, edges, cost, chapter_colors = system.get_path_with_attributes(start, goal)
    if not nodes:
        return jsonify({"error": "未找到有效路径"}), 404
    # 格式化章节颜色为前端可展示的格式
    chapter_legend = [
        {"chapter_id": cid, "color": color} 
        for cid, color in chapter_colors.items()
    ]
    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "cost": cost,
        "path_length": len(nodes),
        "chapter_legend": chapter_legend
    })

# Flask接口：前端页面
@app.route('/visualize/<int:start>/<int:goal>')
def visualize(start, goal):
    return render_template_string('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>学习路径可视化（章节+认知属性）</title>
    <script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: "Microsoft YaHei", sans-serif; background-color: #f5f5f5; }
        .container { margin: 20px auto; max-width: 1400px; }
        .header { margin-bottom: 20px; text-align: center; }
        .graph-container { height: 700px; border: 1px solid #ddd; border-radius: 8px; background-color: #fff; }
        .info-panel { margin-top: 20px; padding: 20px; background-color: #fff; border-radius: 8px; border: 1px solid #ddd; }
        .attribute-tag { display: inline-block; margin: 0 5px 5px 0; padding: 3px 8px; border-radius: 4px; font-size: 12px; }
        .chapter-tag { background-color: #e8f4f8; color: #2d3748; }
        .bloom-tag { background-color: #fdf2f8; color: #2d3748; }
        .legend-item { display: inline-flex; align-items: center; margin: 0 10px 5px 0; }
        .legend-color { width: 15px; height: 15px; border-radius: 3px; margin-right: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>学习路径可视化</h1>
            <p>起点：{{ start }} → 终点：{{ goal }} | 路径成本：<span id="cost">加载中...</span> | 节点数：<span id="node-count">加载中...</span></p>
        </div>
        <div class="graph-container" id="network"></div>
        <div class="info-panel">
            <h4>属性说明</h4>
            <div>
                <div>
                    <strong>章节图例：</strong>
                    <div id="chapter-legend"></div>
                </div>
                <br>
                <span class="attribute-tag bloom-tag">认知属性：节点大小 = Bloom层级（越大层级越高）</span>
                <span class="attribute-tag">边标签：关系类型 + 权重 + 贝叶斯先验概率</span>
                <span class="attribute-tag">交互：拖拽节点 | 缩放视图 | 点击节点高亮关联边</span>
            </div>
        </div>
    </div>

    <script>
        // 初始化网络图表
        const container = document.getElementById('network');
        const start = {{ start }};
        const goal = {{ goal }};

        // 加载路径数据
        fetch(`/api/path/${start}/${goal}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }

                // 更新页面信息
                document.getElementById('cost').textContent = data.cost.toFixed(4);
                document.getElementById('node-count').textContent = data.path_length;

                // 生成章节图例
                const legendContainer = document.getElementById('chapter-legend');
                data.chapter_legend.forEach(item => {
                    const legendItem = document.createElement('div');
                    legendItem.className = 'legend-item';
                    legendItem.innerHTML = `
                        <div class="legend-color" style="background-color: ${item.color};"></div>
                        <span>章节ID: ${item.chapter_id}</span>
                    `;
                    legendContainer.appendChild(legendItem);
                });

                // 配置Vis.js网络
                const options = {
                    nodes: {
                        shape: 'ellipse',
                        font: { size: 12, face: 'Microsoft YaHei' },
                        borderWidth: 2,
                        hover: { size: 5 }
                    },
                    edges: {
                        arrows: 'to',
                        arrowStrikethrough: false,
                        font: { size: 10, face: 'Microsoft YaHei' },
                        length: 150,
                        width: 2
                    },
                    layout: {
                        hierarchical: {
                            direction: 'LR',  // 从左到右布局（符合学习路径顺序）
                            sortMethod: 'directed',
                            levelSeparation: 200,
                            nodeSpacing: 100
                        }
                    },
                    interaction: {
                        dragNodes: true,
                        zoomView: true,
                        panView: true,
                        hover: true,
                        selectConnectedEdges: true
                    },
                    physics: {
                        hierarchicalRepulsion: {
                            centralGravity: 0.3,
                            springLength: 200,
                            springConstant: 0.01
                        }
                    },
                    tooltip: {
                        enabled: true,
                        fontSize: 12,
                        fontFace: 'Microsoft YaHei',
                        delay: 300
                    }
                };

                // 创建网络实例
                const network = new vis.Network(container, { nodes: data.nodes, edges: data.edges }, options);

                // 点击节点高亮关联边
                network.on('click', function(params) {
                    if (params.nodes.length > 0) {
                        const nodeId = params.nodes[0];
                        const connectedEdges = network.getConnectedEdges(nodeId);
                        network.setSelection({ nodes: [nodeId], edges: connectedEdges });
                    }
                });
            })
            .catch(error => console.error('加载路径失败:', error));
    </script>
</body>
</html>
''', start=start, goal=goal)

if __name__ == "__main__":
    # 启动Flask服务，访问 http://localhost:5000/visualize/374/641 查看可视化结果
    app.run(debug=True, host='0.0.0.0', port=5000)