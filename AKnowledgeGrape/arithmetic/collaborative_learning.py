import logging
from collections import defaultdict

import networkx as nx
from py2neo import Graph
import heapq
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（需确保系统中存在该字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedCollaborativeLearning")


class EnhancedCollaborativeLearning:
    def __init__(self, neo4j_uri, user, password, alpha=0.85, feedback_strength=0.1, dynamic_threshold=0.008):
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

    def _plot_metrics(self):
        """可视化关键指标"""
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
        """生成指标分析报告"""
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
        """记录单次迭代指标"""
        # 基础指标记录
        self.metrics['cost'].append(cost)
        self.metrics['path_length'].append(len(path))

        # 权威值数据准备
        pr_values = [attrs.get('edu_pr', 0.0) for attrs in self.node_attrs.values() if attrs]

        # 标准差计算保护
        if len(pr_values) >= 2:
            self.metrics['pr_std'].append(np.std(pr_values))
        else:
            self.metrics['pr_std'].append(0.0)
            logger.warning("权威值不足2个，标准差设为0")

        # 最大差异计算保护
        if pr_values:
            self.metrics['pr_max_diff'].append(max(pr_values) - min(pr_values))
        else:
            self.metrics['pr_max_diff'].append(0.0)
            logger.warning("无有效权威值，最大差异设为0")

    def load_graph(self):
        """动态加载图数据（优化：缓存节点/边数据，避免重复查询）"""
        self.G = nx.DiGraph()
        self.node_attrs = dict()  # 替换defaultdict，减少哈希开销
        self.node_info = dict()

        # 一次性查询所有节点数据，避免循环中多次调用
        q_nodes = """
        MATCH (n) 
        WHERE n:KnowledgePoint OR n:SubKnowledgePoint OR n:Chapter
        RETURN 
            id(n) as nid,
            labels(n)[0] as label,
            coalesce(n.title, '未命名节点') as title,
            coalesce(n.chapter_id, 0) as chapter_id,
            coalesce(n.edu_pagerank, 0.0) as edu_pr,
            coalesce(n.bloom_level, 0) as bloom_level,
            coalesce(n.is_core_kp, false) as is_core_kp
        """
        nodes_result = list(self.graphdb.run(q_nodes))  # 一次性获取所有结果
        for rec in nodes_result:
            nid = rec['nid']
            self.G.add_node(nid)
            self.node_info[nid] = {
                'label': rec['label'],
                'title': rec['title'],
                'chapter_id': rec['chapter_id'],
                'bloom_level': rec['bloom_level'],
                'is_core': rec['is_core_kp']
            }
            self.node_attrs[nid] = {'edu_pr': max(rec['edu_pr'], 0.03)}

        # 一次性查询所有边数据
        q_edges = """
        MATCH (a)-[r]->(b)
        WHERE (a:KnowledgePoint OR a:SubKnowledgePoint OR a:Chapter)
            AND (b:KnowledgePoint OR b:SubKnowledgePoint OR b:Chapter)
        RETURN 
            id(a) as src, 
            id(b) as tgt, 
            coalesce(r.weight, 1.0) as weight
        """
        edges_result = list(self.graphdb.run(q_edges))
        edges_to_add = [(rec['src'], rec['tgt'], {'weight': rec['weight'] or 1.0}) for rec in edges_result]
        self.G.add_edges_from(edges_to_add)  # 批量添加边，比循环添加快

        logger.info(f"加载节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}")
        
    def get_node_details(self, node_id):
        """获取节点详细信息"""
        if node_id not in self.node_info:
            return None
        return {
            'id': node_id,
            'label': self.node_info[node_id]['label'],
            'title': self.node_info[node_id]['title'],
            'chapter_id': self.node_info[node_id]['chapter_id'],
            'bloom_level': self.node_info[node_id]['bloom_level'],
            'is_core': self.node_info[node_id]['is_core'],
            'edu_pr': self.node_attrs.get(node_id, {}).get('edu_pr', 0.0)
        }

    def compute_edu_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        """
        计算教育版PageRank，考虑教学指标：
        1. 静态权重计算：
        - class_hours 与 syllabus_mentions 归一化到 [0,1]
        - bloom_level 归一化为 (7 - bloom_level)/6
        - 加权求和：0.5*hours + 0.3*syllabus + 0.2*bloom
        2. PageRank迭代：
        - 考虑静态权重和图结构
        - 使用阻尼因子控制随机游走
        """
        # 创建图结构
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        G.add_edges_from(self.edges)
        N = G.number_of_nodes()
        
        if N == 0:
            logger.error("无节点数据")
            return

        # 计算最大值用于归一化
        max_syllabus = max(data["syllabus_mentions"] for data in self.nodes.values()) or 1
        max_hours = max(data["class_hours"] for data in self.nodes.values()) or 1

        # 初始化节点属性
        for node in G.nodes:
            data = self.nodes[node]
            # 归一化处理
            syllabus_norm = np.log1p(data["syllabus_mentions"]) / np.log1p(max_syllabus)
            hours_norm = data["class_hours"] / max_hours if max_hours else 0
            bloom_norm = (7 - data["bloom_level"]) / 6.0
            
            # 计算静态权重
            static_weight = float(0.5 * hours_norm + 0.3 * syllabus_norm + 0.2 * bloom_norm)
            G.nodes[node]["static_weight"] = static_weight
            G.nodes[node]["edu_pr"] = 1.0 / N  # 初始均匀分布

        # PageRank迭代
        for iter_count in range(max_iter):
            new_pr = {}
            for n in G.nodes:
                # 基础随机游走概率
                pr_value = (1 - alpha) / N
                
                # 累加入度节点的贡献
                for pred in G.predecessors(n):
                    # 获取出度权重和
                    out_weight = sum(G.edges[pred, succ].get('weight', 1) 
                                for succ in G.successors(pred))
                    if out_weight > 0:
                        # 考虑静态权重和边权重
                        pr_value += alpha * (
                            G.nodes[pred]["static_weight"] * 
                            G.nodes[pred]["edu_pr"] * 
                            G.edges[pred, n].get('weight', 1) / out_weight
                        )
                
                new_pr[n] = pr_value

            # 归一化
            total = sum(new_pr.values())
            for n in new_pr:
                new_pr[n] /= total if total > 0 else 1

            # 计算变化量
            diff = max(abs(new_pr[n] - G.nodes[n]["edu_pr"]) for n in G.nodes)
            
            # 更新值
            for n in G.nodes:
                G.nodes[n]["edu_pr"] = new_pr[n]

                logger.info(f"Iter {iter_count + 1} | Diff: {diff:.8f}")
                
                # 检查收敛
                if diff < tol:
                    logger.info("算法收敛")
                    break

            # 更新节点数据
            for node in G.nodes:
                self.nodes[node]["edu_pr"] = G.nodes[node]["edu_pr"]

            # 写入数据库
            self.write_results_to_neo4j()

    def write_results_to_neo4j(self):
        """
        将计算结果写入Neo4j：
        1. 更新教育版PageRank值
        2. 标记前20%为核心知识点
        """
        # 提取并排序PageRank值
        pr_values = [v["edu_pr"] for v in self.nodes.values()]
        pr_values.sort(reverse=True)

        # 计算核心知识点阈值
        threshold_index = max(int(len(pr_values) * 0.2) - 1, 0)
        threshold_value = pr_values[threshold_index] if pr_values else 0

        logger.info(f"核心知识点阈值: {threshold_value:.6f}")

        # 准备更新数据
        data = []
        for node_id, info in self.nodes.items():
            data.append({
                "id": node_id,
                "edu_pr": round(info["edu_pr"], 6),
                "is_core_kp": info["edu_pr"] >= threshold_value
            })

        # 执行Neo4j更新
        query = """
        UNWIND $data AS row
        MATCH (n)
        WHERE ( (n:KnowledgePoint AND n.kp_id = row.id)
                OR (n:SubKnowledgePoint AND n.sub_kp_id = row.id) )
        SET n.edu_pagerank = row.edu_pr,
            n.is_core_kp = row.is_core_kp
        """

        self.graph.run(query, data=data)
        logger.info("教育PageRank及核心标志写入完成")

    def update_pagerank(self):
        """教育版PageRank计算（优化：减少循环嵌套，预计算出度权重）"""
        N = self.G.number_of_nodes()
        if N == 0:
            logger.warning("图中无节点，跳过PageRank更新")
            return

        # 预缓存边权重
        edge_decay = {(u, v): self.G[u][v]['weight'] for u, v in self.G.edges()}
        # 预计算每个节点的出边列表
        out_edges = {u: list(self.G.successors(u)) for u in self.G.nodes()}
        
        pr = {node: 1.0 / N for node in self.G.nodes()}
        gamma = 0.8

        # 减少迭代轮次（原10轮可优化为8轮，通过验证收敛性）
        for iter_round in range(8):
            new_pr = {}
            # 预计算所有节点的出度总权重（避免重复计算）
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
            
            # 归一化
            total = sum(new_pr.values())
            if total > 0:
                for node in new_pr:
                    new_pr[node] /= total
            
            pr = new_pr.copy()

            # 批量更新边衰减，避免循环中重复操作
            for u, v in edge_decay.keys():
                edge_decay[(u, v)] *= gamma

        # 批量写入数据库（减少Neo4j交互次数）
        update_queries = []
        for nid in pr:
            edu_pr = pr[nid]
            self.node_attrs[nid]['edu_pr'] = edu_pr
            update_queries.append({
                'nid': nid,
                'pr': edu_pr
            })
        
        # 批量执行更新
        if update_queries:
            self.graphdb.run("""
                UNWIND $data AS row
                MATCH (n) WHERE id(n) = row.nid
                SET n.edu_pagerank = row.pr
            """, data=update_queries)
        
        logger.info("教育版PageRank权威值（含边衰减）更新完成")
    
    def a_star_search(self, start, goal):
        if start == goal:
            return [start], 0
            
        if start not in self.G or goal not in self.G:
            logger.error("起始或目标节点不存在")
            return None, float('inf')

        # 1. 数据结构优化：用优先级队列+节点状态缓存，减少重复查询
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))  # 存储 (f_score, 节点ID)
        came_from = dict()
        
        # 缓存g_score（已走路径成本）和f_score（预估总成本），避免重复计算
        g_score = {start: 0.0}
        f_score = {start: self._heuristic(start, goal)}  # 预计算初始启发值
        
        visited = set()
        heap_max_size = 800  # 缩小队列容量（优化前1000），减少堆操作开销
        prune_threshold = 2.0  # 剪枝阈值：当前成本超过最优成本2倍直接跳过（优化前1.5倍，平衡剪枝率和精度）

        # 2. 预缓存全局数据（比原优化更彻底，覆盖所有计算依赖）
        # 节点属性缓存（一次性提取所有需要的属性）
        node_cache = {}
        for nid in self.G.nodes():
            node_info = self.node_info.get(nid, {})
            node_attrs = self.node_attrs.get(nid, {})
            node_cache[nid] = {
                'chapter_id': node_info.get('chapter_id', 0),
                'bloom_level': node_info.get('bloom_level', 0),
                'is_core': node_info.get('is_core', False),
                'edu_pr': node_attrs.get('edu_pr', 0.0)
            }
        # 边权重缓存（直接构建邻接表，避免遍历Graph）
        adjacency_list = defaultdict(list)
        for u, v, attrs in self.G.edges(data=True):
            adjacency_list[u].append((v, attrs.get('weight', 1.0)))

        # 3. 目标节点属性预取（避免循环中重复查询）
        goal_chapter = node_cache[goal]['chapter_id']
        goal_bloom = node_cache[goal]['bloom_level']

        while open_heap:
            # 弹出当前最优节点（f_score最小）
            current_f, current = heapq.heappop(open_heap)
            
            # 4. 剪枝优化：已访问节点直接跳过（避免重复处理）
            if current in visited:
                continue
            visited.add(current)

            # 5. 终止条件优化：找到目标直接返回（无需处理后续节点）
            if current == goal:
                return self._reconstruct_path(came_from, goal), g_score[current]

            # 6. 成本剪枝：超过最优已知成本的2倍直接跳过（减少无效搜索）
            best_known = g_score.get(goal, float('inf'))
            if g_score[current] > best_known * prune_threshold:
                continue

            # 7. 当前节点属性预取（从全局缓存中获取，O(1)查询）
            current_data = node_cache[current]
            current_g = g_score[current]
            current_chapter = current_data['chapter_id']
            current_bloom = current_data['bloom_level']

            # 8. 邻接节点遍历优化：从预构建邻接表获取，避免Graph查询
            for neighbor, edge_weight in adjacency_list.get(current, []):
                if neighbor in visited:
                    continue  # 已访问节点跳过，减少重复计算

                # 9. 代价计算简化（合并重复逻辑，减少数值运算）
                neighbor_data = node_cache[neighbor]
                delta_bloom = abs(current_bloom - neighbor_data['bloom_level'])
                
                # 启发值简化：保留核心逻辑，移除冗余系数（原-500*pr改为-100*pr，减少数值偏差）
                h = max(0.3 * delta_bloom - 100 * neighbor_data['edu_pr'], 0.0)
                
                # 惩罚项简化：合并条件判断，减少分支开销
                penalty = 3 * delta_bloom * 0.2 if neighbor_data['is_core'] else (3 * delta_bloom if delta_bloom > 2 else 0.0)
                
                # 连续性奖励：直接判断章节是否一致，无冗余逻辑
                continuity_reward = -0.5 if current_chapter == neighbor_data['chapter_id'] else 0.0

                # 边权重调整：合并乘法运算，减少计算步骤
                w_prime = edge_weight * (1.0 + 20.0 * neighbor_data['edu_pr'])

                # 计算教育权重的倒数（避免除零错误）
                edu_pr_inverse = 1.0 / max(neighbor_data['edu_pr'], 0.01)

                # 预估成本计算（g_score+w_prime+penalty+reward+h）
                tentative_g = current_g + w_prime + penalty + continuity_reward
                tentative_f = tentative_g + h + edu_pr_inverse  # 修改为加上教育权重倒数

                # 10. 缓存更新优化：仅当新成本更优时才更新（避免无效写入）
                if tentative_f >= f_score.get(neighbor, float('inf')):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_f
                heapq.heappush(open_heap, (tentative_f, neighbor))

                # 11. 队列大小控制优化：超过容量时移除最大f_score节点（更精准剪枝）
                if len(open_heap) > heap_max_size:
                    # 找到f_score最大的索引并删除（比原del最后一个更精准）
                    max_idx = max(range(len(open_heap)), key=lambda i: open_heap[i][0])
                    del open_heap[max_idx]
                    heapq.heapify(open_heap)

        # 无路径时返回
        return None, float('inf')

    def _heuristic(self, node, goal):
        """辅助方法：简化启发函数计算（独立提取，方便后续优化）"""
        node_data = self.node_info.get(node, {})
        goal_data = self.node_info.get(goal, {})
        delta_bloom = abs(node_data.get('bloom_level', 0) - goal_data.get('bloom_level', 0))
        node_pr = self.node_attrs.get(node, {}).get('edu_pr', 0.0)
        return max(0.3 * delta_bloom - 100 * node_pr, 0.0)

    def validate_path(self, path):
        """检查路径中相邻节点是否实际连通"""
        for i in range(len(path) - 1):
            if not self.G.has_edge(path[i], path[i + 1]):
                logger.warning(f"非法路径段：{path[i]} -> {path[i + 1]}")
                return False
        return True

    def evaluate_and_adjust(self, path):
        """评估调整逻辑增强权威值区分度"""
        iteration = len(self.metrics['cost'])
        dynamic_feedback = max(0.2, 0.5 * (0.8 ** iteration))

        # 动态衰减率调整
        decay_rate = 0.5 if iteration < 5 else 0.7
        for u, v in self.G.edges():
            if u in path or v in path:
                self.G[u][v]['weight'] *= decay_rate

        self.update_pagerank()
        logger.info("完成权重修正")

    def _dynamic_alpha(self, current_iter, max_iters, base_alpha):
        """模拟退火参数调整"""
        progress = current_iter / max_iters
        # 初期高alpha加速收敛，后期降低提升稳定性
        return max(0.6, base_alpha * (1 - progress * 0.3))

    def _check_convergence(self, cost, unstable_count):
        """收敛稳定性检测"""
        if len(self.metrics['cost']) >= 4:
            last_3 = self.metrics['cost'][-3:]
            if max(last_3) - min(last_3) > 1.0:
                return unstable_count + 1
        return 0

    def _reconstruct_path(self, came_from, current):
        """路径回溯"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _handle_self_loop(self, node_id):
        """自循环路径处理"""
        self.load_graph()
        self._record_metrics([node_id], 0)  # 确保自循环路径的权威值被记录
        self.update_pagerank()  # 强制更新权威值
        self._record_metrics([node_id], 0)
        for nid in self.G.nodes():
            self.pr_history[nid].append(self.node_attrs[nid].get('edu_pr', 0))
        if node_id not in self.G.nodes:
            logger.error("自循环节点不存在于图谱中")
            return None, float('inf')
        details = self.get_node_details(node_id)
        print(f"\n路径搜索：{details['label']} [自循环路径]")
        print(f"节点详情：{details['title']} (Bloom层级: {details['bloom_level']})")
        return [node_id], 0

    def _validate_and_update(self, path, cost, best_path, min_cost):
        """路径验证与更新逻辑"""
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

        # 只加载一次图，避免迭代中重复加载（原逻辑每次迭代都load_graph，导致大量重复IO）
        self.load_graph()
        best_path, min_cost = None, float('inf')
        unstable_count = 0
        original_alpha = self.alpha

        # 预缓存节点存在性，避免重复查询
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

            # 批量记录权威值历史，避免循环中重复查询
            pr_snapshot = {nid: self.node_attrs[nid].get('edu_pr', 0) for nid in self.G.nodes()}
            for nid, pr_val in pr_snapshot.items():
                self.pr_history[nid].append(pr_val)

            unstable_count = self._check_convergence(cost, unstable_count)
            if unstable_count >= 3:
                logger.info("检测到持续不稳定，延长迭代次数至40")
                max_iters = 40

            self.evaluate_and_adjust(path)
            self._record_metrics(path, cost)

            # 提前收敛判断：如果成本变化小于阈值，直接终止迭代
            if len(self.metrics['cost']) >= 5:
                recent_costs = self.metrics['cost'][-5:]
                if max(recent_costs) - min(recent_costs) < 0.1:
                    logger.info("成本已收敛，提前终止迭代")
                    break

        self._finalize_iteration()
        return best_path, min_cost

    def _finalize_iteration(self):
        """迭代收尾工作"""
        self._plot_metrics()
        self._generate_metric_report()
        # 重置临时存储
        self.pr_history = {}

# 修改后的主程序部分
if __name__ == "__main__":
    # 定义多组测试用例（不同起止节点组合）
    test_cases = [
        {"start": 15, "goal": 282, "desc": "常规知识点路径"}
    ]

    final_results = []

    # 每次测试独立初始化系统
    for case in test_cases:
        # 每次测试创建新实例
        system = EnhancedCollaborativeLearning(
            neo4j_uri="bolt://localhost:7687",
            user="neo4j",
            password="123456789",
            alpha=0.8,  # 调整初始alpha
            feedback_strength=0.5  # 增强反馈强度
        )
        system.load_graph()

        # 节点存在性检查
        start_exists = case["start"] in system.G.nodes
        goal_exists = case["goal"] in system.G.nodes
        if not (start_exists and goal_exists):
            print(f"\n测试用例 {case['desc']} 无效：")
            if not start_exists:
                print(f"  起始节点 {case['start']} 不存在于知识图谱")
            if not goal_exists:
                print(f"  目标节点 {case['goal']} 不存在于知识图谱")
            continue

        print(f"\n=== 测试用例 [{case['desc']}] 起止节点 ({case['start']}->{case['goal']}) ===")
        best_path, final_cost = system.collaborative_iteration(
            start=case["start"],
            goal=case["goal"],
            max_iters=30
        )
        if best_path is None or len(best_path) == 0:
            logger.warning(f"测试用例 {case['desc']} 未找到有效路径")
            continue
        
        # 记录结果，包含bloom层级信息
        path_details = []
        for nid in best_path:
            details = system.get_node_details(nid)
            path_details.append({
                "node_id": nid,
                "title": details["title"],
                "chapter_id": details["chapter_id"],
                "bloom_level": details["bloom_level"],
                "is_core": details["is_core"],
                "edu_pr": details["edu_pr"]
            })
        
        # 记录结果
        final_results.append({
            "desc": case["desc"],
            "path": best_path,
            "cost": final_cost,
            "metrics": system.metrics.copy(),
            "path_details": path_details
        })

    print("\n=== 各测试用例最终路径详情 ===")
    for res in final_results:
        print(f"\n【{res['desc']}】")
        print(f"  路径节点ID序列: {res['path']}")
        print(f"  最终成本: {res['cost']:.4f}")
        print("  路径详情:")
        print("  " + "-"*100)
        print(f"  {'节点ID':<8} {'标题':<40} {'章节ID':<8} {'Bloom层级':<10} {'是否核心':<8} {'权威值':<10}")
        print("  " + "-"*100)
        for detail in res["path_details"]:
            print(f"  {detail['node_id']:<8} {detail['title'][:38]:<40} {detail['chapter_id']:<8} "
                  f"{detail['bloom_level']:<10} {detail['is_core']!s:<8} {detail['edu_pr']:.4f}")
        print("  " + "-"*100)

    # 跨测试用例分析
    print("\n=== 跨测试用例综合分析 ===")
    valid_cases = [
        res for res in final_results
        if res["path"] is not None
           and len(res["path"]) > 1  # 排除自循环路径
           and res["metrics"]['cost'][0] > 0  # 排除零初始成本
    ]

    if valid_cases:
        analysis = {
            "成功用例数": len(valid_cases),
            "平均成本下降率": np.mean([
                (case["metrics"]['cost'][0] - case["metrics"]['cost'][-1]) / case["metrics"]['cost'][0]
                for case in valid_cases
            ]),
            "权威值集中趋势": np.mean([
                case["metrics"]['pr_std'][-1]
                for case in valid_cases
                if len(case["metrics"]['pr_std']) >= 1
            ]),
            "平均Bloom层级跨度": np.mean([
                max(d['bloom_level'] for d in case['path_details']) - min(d['bloom_level'] for d in case['path_details'])
                for case in valid_cases
            ])
        }

        print(f"成功路径比例: {analysis['成功用例数']}/{len(test_cases)}")
        print(f"平均成本优化率: {analysis['平均成本下降率']:.2%}")
        print(f"权威值最终集中度: {analysis['权威值集中趋势']:.4f}")
        print(f"平均Bloom层级跨度: {analysis['平均Bloom层级跨度']:.2f}")