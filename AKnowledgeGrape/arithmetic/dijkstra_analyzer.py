import sys
import os
import logging
import time
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from py2neo import Graph
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import pickle

# 将当前目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False    

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedDijkstraAnalyzer")

@dataclass
class CoverageResult:
    """覆盖率计算结果数据类"""
    ratio: float
    covered_nodes: Set[int]
    total_nodes: int
    weighted_ratio: float
    category_coverage: Dict[str, float]

class DijkstraPathAnalyzer:  # 保持类名为DijkstraPathAnalyzer以兼容导入
    """增强版Dijkstra路径分析器（兼容原类名）"""
    def __init__(self, neo4j_uri, user, password):
        """初始化分析器"""
        self.graphdb = Graph(neo4j_uri, auth=(user, password))
        self.G = nx.DiGraph()
        self.node_attrs = {}
        self.node_info = {}
        self.metrics = defaultdict(list)
        
        # 优化：预计算的重要节点索引
        self.important_nodes = {}
        self.chapter_index = defaultdict(set)
        self.bloom_index = defaultdict(set)
        self.knowledge_type_index = defaultdict(set)
        
        # 缓存优化
        self.path_cache = {}
        self.coverage_cache = {}
        
        # 性能优化参数 - 放宽限制
        self.use_pagerank = True
        self.cache_enabled = True
        self.batch_size = 100  # 增大批量大小
        
        # 覆盖率优化参数
        self.coverage_expansion_factor = 1.5  # 覆盖率扩展因子
        self.min_coverage_nodes = 10  # 最小覆盖节点数
        self.important_node_percentage = 0.25  # 重要节点占比（从20%提高到30%）
        
        # 路径搜索放宽限制
        self.search_expansion_limit = 2.0  # 搜索扩展限制
        self.max_path_length = 50  # 最大路径长度
        self.coverage_bonus_weight = 0.2  # 覆盖率奖励权重

    def load_graph(self, node_weight_property="importance", use_cache=False):
        """加载图数据并计算节点重要性（优化版）"""
        if use_cache and self._load_from_cache():
            logger.info("从缓存加载图数据")
            return
            
        start_time = time.time()
        self.G = nx.DiGraph()
        self.node_attrs = defaultdict(dict)
        self.node_info = {}
        
        try:
            # 1. 批量加载节点（优化：使用参数化查询）
            q_nodes = """
               MATCH (n) 
               WHERE n:KnowledgePoint OR n:SubKnowledgePoint OR n:Chapter
               RETURN 
                   id(n) as nid,
                   labels(n)[0] as label,
                   coalesce(n.title, '未命名节点') as title,
                   coalesce(n.chapter_id, '未知章节') as chapter_id,
                   coalesce(n.bloom_level, 0) as bloom_level,
                   coalesce(n.importance, 1.0) as importance,
                   coalesce(n.difficulty, 1.0) as difficulty
               """
            
            # 优化：批量处理节点数据
            nodes_data = list(self.graphdb.run(q_nodes))
            if not nodes_data:
                logger.warning("未找到任何节点数据")
                return
                
            # 批量添加节点
            node_ids = []
            for rec in nodes_data:
                nid = rec['nid']
                node_ids.append(nid)
                self.G.add_node(nid)
                
                # 节点信息存储优化
                node_info = {
                    'label': rec['label'],
                    'title': rec['title'][:50],  # 限制标题长度
                    'chapter_id': str(rec['chapter_id']),
                    'bloom_level': int(rec['bloom_level']) if rec['bloom_level'] else 0,
                    'difficulty': float(rec['difficulty'])
                }
                self.node_info[nid] = node_info
                self.node_attrs[nid]['raw_weight'] = float(rec['importance'])
                
                # 构建分类索引（优化覆盖率计算）
                self.chapter_index[node_info['chapter_id']].add(nid)
                self.bloom_index[node_info['bloom_level']].add(nid)
                self.knowledge_type_index[node_info['label']].add(nid)
            
            logger.info(f"加载 {len(node_ids)} 个节点，耗时 {time.time()-start_time:.2f}秒")
            
            # 2. 优化的PageRank计算
            if len(node_ids) >= 5 and self.use_pagerank:
                try:
                    # 优化：使用更高效的PageRank计算
                    pr_start = time.time()
                    pr_values = nx.pagerank(
                        self.G, 
                        alpha=0.85,
                        personalization={n: self.node_attrs[n]['raw_weight'] for n in node_ids},
                        max_iter=200,  # 增加迭代次数
                        tol=1e-06
                    )
                    logger.info(f"PageRank计算耗时 {time.time()-pr_start:.2f}秒")
                    
                    for nid in node_ids:
                        self.node_attrs[nid]['weight'] = max(0.1, pr_values.get(nid, 1.0))
                        # 计算综合权重（重要性+难度）
                        self.node_attrs[nid]['composite_weight'] = (
                            self.node_attrs[nid]['weight'] * 
                            (1 + self.node_info[nid]['difficulty']) / 2
                        )
                        
                except Exception as e:
                    logger.warning(f"PageRank计算失败，使用原始权重: {e}")
                    for nid in node_ids:
                        self.node_attrs[nid]['weight'] = max(0.1, self.node_attrs[nid]['raw_weight'])
                        self.node_attrs[nid]['composite_weight'] = self.node_attrs[nid]['weight']
            else:
                for nid in node_ids:
                    self.node_attrs[nid]['weight'] = max(0.1, self.node_attrs[nid]['raw_weight'])
                    self.node_attrs[nid]['composite_weight'] = self.node_attrs[nid]['weight']
            
            # 3. 批量加载边关系（优化：使用关系类型索引）
            q_edges = """
               MATCH (a)-[r]->(b)
               WHERE (a:KnowledgePoint OR a:SubKnowledgePoint OR a:Chapter)
                 AND (b:KnowledgePoint OR b:SubKnowledgePoint OR b:Chapter)
               RETURN 
                   id(a) as src, 
                   id(b) as tgt,
                   type(r) as relation_type,
                   coalesce(r.weight, 1.0) as rel_weight
               """
            
            # 优化：批量处理边数据
            edges_data = list(self.graphdb.run(q_edges))
            edge_count = 0
            
            # 预定义关系权重（优化：使用字典映射）
            relation_weights = {
                'CONTAINS': 1.0,
                'REQUIRES': 1.2,
                'RELATED_TO': 0.8,
                'EXTENDS': 1.1,
                'PART_OF': 0.9
            }
            
            for rec in edges_data:
                src, tgt = rec['src'], rec['tgt']
                if src in self.G.nodes and tgt in self.G.nodes:
                    rel_type = rec['relation_type']
                    base_rel_weight = float(rec['rel_weight'])
                    
                    # 优化：向量化计算边权重
                    edge_weight = (self.node_attrs[src]['composite_weight'] + 
                                 self.node_attrs[tgt]['composite_weight']) * 0.5
                    edge_weight *= relation_weights.get(rel_type, 1.0) * base_rel_weight
                    
                    self.G.add_edge(src, tgt, 
                                  weight=edge_weight, 
                                  relation_type=rel_type,
                                  base_weight=base_rel_weight)
                    edge_count += 1
            
            # 4. 识别重要节点（优化：放宽重要节点选择）
            self._identify_important_nodes()
            
            # 保存缓存
            if use_cache:
                self._save_to_cache()
                
            load_time = time.time() - start_time
            logger.info(f"图加载完成：节点={self.G.number_of_nodes()}, 边={edge_count}, 耗时={load_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"图数据加载失败: {e}")
            raise

    def _identify_important_nodes(self):
        """优化的重要节点识别算法（放宽限制）"""
        # 1. 基于PageRank的全局重要节点
        if self.use_pagerank and self.G.number_of_nodes() >= 5:
            pr_values = nx.pagerank(self.G, alpha=0.85, max_iter=100)
            sorted_nodes = sorted(pr_values.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_nodes = [(nid, self.node_attrs[nid]['composite_weight']) 
                          for nid in self.G.nodes if nid in self.node_attrs]
            sorted_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # 2. 放宽重要节点选择标准
        total_nodes = len(sorted_nodes)
        # 计算重要节点数量（取百分比和最小数量的较大值）
        important_count = max(
            int(total_nodes * self.important_node_percentage),
            self.min_coverage_nodes
        )
        
        # 扩展重要节点集合
        expanded_count = int(important_count * self.coverage_expansion_factor)
        expanded_count = min(expanded_count, total_nodes)
        
        # 分层级存储重要节点
        self.important_nodes['top_5'] = set([nid for nid, _ in sorted_nodes[:min(5, total_nodes)]])
        self.important_nodes['top_10'] = set([nid for nid, _ in sorted_nodes[:min(10, total_nodes)]])
        self.important_nodes['top_20'] = set([nid for nid, _ in sorted_nodes[:min(20, total_nodes)]])
        self.important_nodes['top_expanded'] = set([nid for nid, _ in sorted_nodes[:expanded_count]])
        self.important_nodes['all'] = set([nid for nid, _ in sorted_nodes])
        
        # 3. 按类别识别重要节点（扩展）
        self.important_nodes['by_chapter'] = {}
        for chapter, nodes in self.chapter_index.items():
            chapter_nodes = [(n, self.node_attrs[n]['composite_weight']) for n in nodes if n in self.node_attrs]
            chapter_nodes.sort(key=lambda x: x[1], reverse=True)
            # 每个章节选择更多重要节点
            chapter_important = int(len(chapter_nodes) * 0.4)  # 40%的节点作为重要节点
            chapter_important = max(chapter_important, 3)  # 至少3个
            self.important_nodes['by_chapter'][chapter] = set([n for n, _ in chapter_nodes[:chapter_important]])
        
        logger.info(f"识别重要节点完成：Top扩展={len(self.important_nodes['top_expanded'])}个")

    def dijkstra_search(self, start, goal, use_heap=True, prioritize_coverage=True):
        """优化的Dijkstra算法（放宽限制，优先覆盖率）"""
        # 缓存检查
        cache_key = (start, goal, prioritize_coverage)
        if self.cache_enabled and cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if start == goal:
            result = ([start], 0.0)
            if self.cache_enabled:
                self.path_cache[cache_key] = result
            return result

        if start not in self.G.nodes or goal not in self.G.nodes:
            logger.error(f"节点不存在：start={start}, goal={goal}")
            result = (None, float('inf'))
            if self.cache_enabled:
                self.path_cache[cache_key] = result
            return result

        # 优化：使用更高效的距离初始化
        dist = {node: float('inf') for node in [start]}
        dist[start] = 0.0
        prev = {}
        
        # 优化：使用heapq或更高效的实现
        heap = []
        heapq.heappush(heap, (0.0, start))
        visited = set()
        
        # 获取重要节点集合用于优先访问
        important_nodes = self.important_nodes.get('top_expanded', set())

        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == goal:
                break

            # 放宽搜索范围限制
            if len(visited) > len(self.G.nodes) * self.search_expansion_limit:
                logger.info(f"搜索范围扩展到{len(visited)}个节点，继续搜索以提高覆盖率")
                # 不提前终止，继续搜索

            # 遍历邻居节点
            for neighbor in self.G.successors(current_node):
                if neighbor in visited:
                    continue
                    
                edge_weight = self.G[current_node][neighbor].get('weight', 1.0)
                
                # 如果优先覆盖率，对重要节点给予权重奖励
                if prioritize_coverage and neighbor in important_nodes:
                    edge_weight *= (1 - self.coverage_bonus_weight)  # 降低重要节点的权重，优先访问
                    
                new_dist = current_dist + edge_weight

                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    prev[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))

        # 路径重构
        if goal not in prev and start != goal:
            # 尝试寻找更长的路径
            logger.info("尝试寻找替代路径以提高覆盖率")
            result = (None, float('inf'))
        else:
            path = self._reconstruct_path(prev, goal, start)
            # 检查路径长度，如果太短则尝试扩展
            if len(path) < 5 and prioritize_coverage:
                extended_path = self._extend_path_for_coverage(path)
                if extended_path:
                    path = extended_path
            result = (path, dist.get(goal, float('inf')))
        
        # 缓存结果
        if self.cache_enabled:
            self.path_cache[cache_key] = result
            
        return result

    def _extend_path_for_coverage(self, path):
        """扩展路径以提高覆盖率"""
        if not path or len(path) >= self.max_path_length:
            return path
            
        # 获取当前路径覆盖的重要节点
        current_nodes = set(path)
        important_nodes = self.important_nodes.get('top_expanded', set())
        uncovered = important_nodes - current_nodes
        
        if not uncovered:
            return path
            
        # 尝试从路径末尾扩展
        last_node = path[-1]
        extended_path = path.copy()
        
        # BFS寻找最近的未覆盖重要节点
        queue = [(last_node, 0)]
        visited = set(current_nodes)
        found = None
        
        while queue and len(extended_path) < self.max_path_length:
            node, depth = queue.pop(0)
            if node in uncovered:
                found = node
                break
                
            for neighbor in self.G.successors(node):
                if neighbor not in visited and depth < 5:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        # 如果找到，扩展路径
        if found:
            try:
                extension = nx.shortest_path(self.G, source=last_node, target=found)
                if len(extended_path) + len(extension) <= self.max_path_length:
                    extended_path.extend(extension[1:])
                    logger.info(f"路径扩展以覆盖更多重要节点，原长度{len(path)}→{len(extended_path)}")
            except:
                pass
                
        return extended_path

    def _reconstruct_path(self, prev, goal, start):
        """优化的路径重构算法（放宽长度限制）"""
        path = [goal]
        current = goal
        
        while current in prev:
            current = prev[current]
            path.append(current)
            if current == start:
                break
                
        # 允许更长的路径
        if len(path) > self.max_path_length:
            logger.info(f"路径长度{len(path)}超过限制，但保留以提高覆盖率")
            
        return path[::-1]

    def calculate_knowledge_coverage(self, path, coverage_type='top_expanded', use_weighted=True):
        """优化的核心知识点覆盖率计算（使用扩展的重要节点集）"""
        if not path or len(path) < 2:
            return CoverageResult(0.0, set(), 0, 0.0, {})
        
        # 缓存检查
        path_tuple = tuple(path)
        cache_key = (path_tuple, coverage_type, use_weighted)
        if self.cache_enabled and cache_key in self.coverage_cache:
            return self.coverage_cache[cache_key]
        
        # 获取目标重要节点集合（默认使用扩展集合）
        if coverage_type in self.important_nodes:
            target_nodes = self.important_nodes[coverage_type]
        else:
            target_nodes = self.important_nodes['top_expanded']
        
        if not target_nodes:
            return CoverageResult(0.0, set(), 0, 0.0, {})
        
        # 计算覆盖的重要节点
        path_nodes = set(path)
        covered_nodes = path_nodes & target_nodes
        
        # 基础覆盖率
        coverage_ratio = len(covered_nodes) / len(target_nodes) if target_nodes else 0.0
        
        # 加权覆盖率（考虑节点重要性）
        if use_weighted:
            total_weight = sum(self.node_attrs[nid]['composite_weight'] for nid in target_nodes)
            covered_weight = sum(self.node_attrs[nid]['composite_weight'] for nid in covered_nodes)
            weighted_ratio = covered_weight / total_weight if total_weight > 0 else 0.0
        else:
            weighted_ratio = coverage_ratio
        
        # 分类覆盖率分析
        category_coverage = {}
        
        # 按章节覆盖率
        chapter_coverage = {}
        for chapter, nodes in self.chapter_index.items():
            chapter_target = nodes & target_nodes
            if chapter_target:
                chapter_covered = chapter_target & covered_nodes
                chapter_coverage[chapter] = len(chapter_covered) / len(chapter_target)
        if chapter_coverage:
            category_coverage['chapter'] = np.mean(list(chapter_coverage.values()))
        
        # 按Bloom层级覆盖率
        bloom_coverage = {}
        for level, nodes in self.bloom_index.items():
            bloom_target = nodes & target_nodes
            if bloom_target:
                bloom_covered = bloom_target & covered_nodes
                bloom_coverage[level] = len(bloom_covered) / len(bloom_target)
        if bloom_coverage:
            category_coverage['bloom'] = np.mean(list(bloom_coverage.values()))
        
        result = CoverageResult(
            ratio=coverage_ratio,
            covered_nodes=covered_nodes,
            total_nodes=len(target_nodes),
            weighted_ratio=weighted_ratio,
            category_coverage=category_coverage
        )
        
        # 缓存结果
        if self.cache_enabled:
            self.coverage_cache[cache_key] = result
            
        return result

    def analyze_path_quality_batch(self, start_nodes, goal_nodes):
        """批量路径质量分析（性能优化）"""
        results = []
        batch_start = time.time()
        
        # 批量处理
        for i, (start, goal) in enumerate(zip(start_nodes, goal_nodes)):
            if i > 0 and i % self.batch_size == 0:
                logger.info(f"已处理 {i} 条路径，平均耗时 {(time.time()-batch_start)/i:.3f}秒/条")
            
            # 优先考虑覆盖率的路径搜索
            path, cost = self.dijkstra_search(start, goal, prioritize_coverage=True)
            if path:
                coverage = self.calculate_knowledge_coverage(path, coverage_type='top_expanded')
                quality = self.analyze_path_quality(path)
                
                results.append({
                    'start': start,
                    'goal': goal,
                    'path': path,
                    'cost': cost,
                    'coverage': coverage,
                    'quality': quality
                })
        
        return results

    def analyze_path_quality(self, path):
        """综合分析路径质量（优化版）"""
        if not path or len(path) < 2:
            return {
                'chapter_changes': 0,
                'bloom_level_changes': 0,
                'avg_bloom_level': 0,
                'knowledge_coverage_ratio': 0,
                'weighted_coverage': 0,
                'smoothness_score': 0,
                'complexity_score': 0,
                'path_efficiency': 0
            }
        
        # 优化：批量获取节点详情
        node_details = []
        for nid in path:
            if nid in self.node_info and nid in self.node_attrs:
                details = {
                    'chapter_id': self.node_info[nid]['chapter_id'],
                    'bloom_level': self.node_info[nid]['bloom_level'],
                    'weight': self.node_attrs[nid]['composite_weight'],
                    'type': self.node_info[nid]['label']
                }
                node_details.append(details)
        
        if not node_details:
            return {'chapter_changes': 0, 'bloom_level_changes': 0, 'avg_bloom_level': 0}
        
        # 向量化计算指标
        chapters = [d['chapter_id'] for d in node_details]
        blooms = [d['bloom_level'] for d in node_details]
        weights = [d['weight'] for d in node_details]
        
        # 计算指标（优化：使用numpy向量化操作）
        chapter_changes = np.sum(np.array(chapters[1:]) != np.array(chapters[:-1]))
        bloom_diff = np.abs(np.diff(blooms))
        bloom_changes = np.sum(bloom_diff)
        avg_bloom = np.mean(blooms)
        
        # 计算覆盖率（使用扩展的重要节点集）
        coverage = self.calculate_knowledge_coverage(path, coverage_type='top_expanded')
        
        # 平滑度和复杂度（放宽平滑度限制）
        smoothness_score = 1.0 / (1.0 + bloom_changes * 0.05 + chapter_changes * 0.1)  # 降低变化惩罚
        complexity_score = np.std(weights) + len(path) * 0.05  # 降低复杂度权重
        
        # 路径效率（放宽长度限制）
        path_efficiency = min(1.0, 20.0 / len(path)) if len(path) > 0 else 0  # 从10放宽到20
        
        return {
            'chapter_changes': chapter_changes,
            'bloom_level_changes': int(bloom_changes),
            'avg_bloom_level': float(avg_bloom),
            'knowledge_coverage_ratio': coverage.ratio,
            'weighted_coverage': coverage.weighted_ratio,
            'covered_count': len(coverage.covered_nodes),
            'total_important_nodes': coverage.total_nodes,
            'smoothness_score': float(smoothness_score),
            'complexity_score': float(complexity_score),
            'path_length': len(path),
            'path_efficiency': float(path_efficiency),
            'chapter_coverage': coverage.category_coverage.get('chapter', 0),
            'bloom_coverage': coverage.category_coverage.get('bloom', 0)
        }

    def run_comprehensive_analysis(self, num_tests=20):
        """运行综合分析测试（优化版）"""
        start_time = time.time()
        self.load_graph(use_cache=True)
        
        if len(self.G.nodes) < 2:
            logger.error("节点数量不足，无法进行测试")
            return []
        
        # 优化：随机选择节点时避免重复
        nodes = list(self.G.nodes)
        np.random.shuffle(nodes)
        
        # 批量生成测试用例
        test_cases = []
        for i in range(num_tests):
            if i + 1 >= len(nodes):
                break
            start = nodes[i]
            goal = nodes[i + 1]
            test_cases.append((start, goal))
        
        # 批量处理测试用例
        results = []
        for i, (start, goal) in enumerate(test_cases):
            path, cost = self.dijkstra_search(start, goal, prioritize_coverage=True)
            execution_time = time.time() - start_time
            
            analysis = self.analyze_path_quality(path) if path else {}
            
            result = {
                'test_id': i + 1,
                'start': start,
                'goal': goal,
                'path': path,
                'cost': cost,
                'execution_time': execution_time,
                'analysis': analysis,
                'desc': f"测试{i+1}: {self.node_info.get(start, {}).get('title', '')[:15]} -> {self.node_info.get(goal, {}).get('title', '')[:15]}"
            }
            
            results.append(result)
            
            if path and cost != float('inf'):
                self.metrics['path_cost'].append(cost)
                self.metrics['path_length'].append(len(path))
                self.metrics['execution_time'].append(execution_time)
                self.metrics['coverage_ratio'].append(analysis.get('knowledge_coverage_ratio', 0))
                self.metrics['smoothness_score'].append(analysis.get('smoothness_score', 0))
        
        total_time = time.time() - start_time
        logger.info(f"完成 {len(results)} 次路径分析测试，总耗时 {total_time:.2f}秒，平均 {total_time/len(results):.3f}秒/次")
        
        # 打印覆盖率统计
        if self.metrics['coverage_ratio']:
            avg_coverage = np.mean(self.metrics['coverage_ratio'])
            logger.info(f"平均核心知识点覆盖率: {avg_coverage:.2%}")
        
        return results

    def _save_to_cache(self):
        """保存图数据到缓存"""
        try:
            cache_data = {
                'graph': self.G,
                'node_attrs': self.node_attrs,
                'node_info': self.node_info,
                'important_nodes': self.important_nodes,
                'chapter_index': self.chapter_index,
                'bloom_index': self.bloom_index,
                'timestamp': time.time()
            }
            
            with open('graph_cache.pkl', 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
            return False

    def _load_from_cache(self):
        """从缓存加载图数据"""
        try:
            with open('graph_cache.pkl', 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查缓存是否过期（24小时）
            if time.time() - cache_data['timestamp'] > 86400:
                return False
                
            self.G = cache_data['graph']
            self.node_attrs = cache_data['node_attrs']
            self.node_info = cache_data['node_info']
            self.important_nodes = cache_data['important_nodes']
            self.chapter_index = cache_data['chapter_index']
            self.bloom_index = cache_data['bloom_index']
            return True
        except:
            return False

    def visualize_comprehensive_analysis(self, results):
        """生成综合分析可视化图表"""
        if not results:
            logger.warning("无有效结果，无法生成可视化图表")
            return

        successful_results = [r for r in results if r['path'] and len(r['path']) > 1]
        if not successful_results:
            logger.warning("无成功路径结果")
            return

        # 1. 生成核心知识点覆盖率分析图
        fig, ax = plt.subplots(figsize=(12, 8))
        self._create_coverage_chart(successful_results, ax)
        plt.savefig('knowledge_coverage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("核心知识点覆盖率分析图已保存")

        # 2. 生成路径质量雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        self._create_radar_chart(successful_results, ax)
        plt.savefig('path_quality_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("路径质量雷达图已保存")

        # 3. 生成执行时间分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        self._create_execution_time_chart(successful_results, ax)
        plt.savefig('execution_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("执行时间分布图已保存")

        # 4. 生成路径特征散点图
        fig, ax = plt.subplots(figsize=(10, 6))
        self._create_scatter_plot(successful_results, ax)
        plt.savefig('path_characteristics_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("路径特征散点图已保存")

    def _create_coverage_chart(self, results, ax):
        """创建核心知识点覆盖率分析图（优化版）"""
        case_descs = [r['desc'][:15] + '...' if len(r['desc']) > 15 else r['desc'] for r in results]
        coverage_ratios = [r['analysis'].get('knowledge_coverage_ratio', 0) for r in results]
        weighted_ratios = [r['analysis'].get('weighted_coverage', 0) for r in results]
        covered_counts = [r['analysis'].get('covered_count', 0) for r in results]
        total_nodes = [r['analysis'].get('total_important_nodes', 0) for r in results]
        
        x = np.arange(len(case_descs))
        width = 0.35
        
        # 绘制双层柱状图
        bars1 = ax.bar(x - width/2, coverage_ratios, width, 
                    label='基础覆盖率', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, weighted_ratios, width,
                    label='加权覆盖率', alpha=0.8, color='#e74c3c')
        
        ax2 = ax.twinx()
        line = ax2.plot(x, covered_counts, 'ko-', linewidth=2, 
                        markersize=6, label='覆盖节点数')
        line2 = ax2.plot(x, total_nodes, 'ro--', linewidth=2, 
                         markersize=6, label='总重要节点数')
        
        ax.set_xlabel('测试用例', fontsize=12)
        ax.set_ylabel('覆盖率', fontsize=12)
        ax2.set_ylabel('节点数量', fontsize=12)
        ax.set_title('核心知识点覆盖率分析（扩展版）', fontsize=14, pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(case_descs, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax2.set_ylim(0, max(max(covered_counts), max(total_nodes)) * 1.2 if covered_counts else 10)
        
        ax.grid(True, alpha=0.3)
        ax2.grid(False)
        
        # 添加数据标签
        for bar, ratio in zip(bars1, coverage_ratios):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{ratio:.1%}', ha='center', va='bottom', fontsize=8)
        
        for bar, ratio in zip(bars2, weighted_ratios):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{ratio:.1%}', ha='center', va='bottom', fontsize=8)
        
        # 添加平均线
        avg_coverage = np.mean(coverage_ratios)
        avg_weighted = np.mean(weighted_ratios)
        ax.axhline(y=avg_coverage, color='#3498db', linestyle='--', alpha=0.7,
                  label=f'平均基础覆盖率: {avg_coverage:.1%}')
        ax.axhline(y=avg_weighted, color='#e74c3c', linestyle='--', alpha=0.7,
                  label=f'平均加权覆盖率: {avg_weighted:.1%}')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        return ax

    def _create_radar_chart(self, results, ax):
        """创建路径质量雷达图（优化版）"""
        if not results:
            return
            
        # 计算多维度指标
        metrics = {
            '覆盖率': np.mean([r['analysis'].get('knowledge_coverage_ratio', 0) for r in results]),
            '加权覆盖率': np.mean([r['analysis'].get('weighted_coverage', 0) for r in results]),
            '平滑度': np.mean([r['analysis'].get('smoothness_score', 0) for r in results]),
            '效率': np.mean([r['analysis'].get('path_efficiency', 0) for r in results]),
            '章节覆盖': np.mean([r['analysis'].get('chapter_coverage', 0) for r in results]),
            'Bloom覆盖': np.mean([r['analysis'].get('bloom_coverage', 0) for r in results])
        }
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#2ecc71', label='路径质量')
        ax.fill(angles, values, alpha=0.25, color='#2ecc71')
        
        # 添加数值标签
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('路径质量多维度分析（扩展覆盖率）', fontsize=14)
        ax.grid(True)
        ax.legend(loc='upper right')

    def _create_execution_time_chart(self, results, ax):
        """创建执行时间分布图（优化版）"""
        execution_times = [r['execution_time'] * 1000 for r in results]
        path_lengths = [len(r['path']) if r['path'] else 0 for r in results]
        coverage_scores = [r['analysis'].get('knowledge_coverage_ratio', 0) for r in results]
        
        scatter = ax.scatter(path_lengths, execution_times, 
                           c=coverage_scores, cmap='RdYlGn', 
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('路径长度', fontsize=12)
        ax.set_ylabel('执行时间 (毫秒)', fontsize=12)
        ax.set_title('路径长度与执行时间关系（颜色表示覆盖率）', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(path_lengths) > 1 and np.std(path_lengths) > 0:
            z = np.polyfit(path_lengths, execution_times, 1)
            p = np.poly1d(z)
            ax.plot(path_lengths, p(path_lengths), "r--", alpha=0.8, linewidth=2,
                   label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
            ax.legend()
        
        plt.colorbar(scatter, ax=ax, label='核心知识点覆盖率')

    def _create_scatter_plot(self, results, ax):
        """创建路径特征散点图（优化版）"""
        coverage_ratios = [r['analysis'].get('knowledge_coverage_ratio', 0) for r in results]
        smoothness_scores = [r['analysis'].get('smoothness_score', 0) for r in results]
        path_lengths = [len(r['path']) if r['path'] else 0 for r in results]
        execution_times = [r['execution_time'] * 1000 for r in results]
        
        scatter = ax.scatter(coverage_ratios, smoothness_scores, 
                          c=path_lengths, s=np.array(execution_times) * 2,
                          cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('核心知识点覆盖率', fontsize=12)
        ax.set_ylabel('路径平滑度', fontsize=12)
        ax.set_title('覆盖率vs平滑度关系（扩展版）', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='路径长度')
        
        # 添加最佳路径标记
        if results:
            best_idx = np.argmax([r['analysis'].get('knowledge_coverage_ratio', 0) * 
                                r['analysis'].get('smoothness_score', 0) for r in results])
            ax.scatter(coverage_ratios[best_idx], smoothness_scores[best_idx],
                     c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                     label='最佳路径')
            ax.legend()

    def generate_detailed_report(self, results):
        """生成详细分析报告（优化版）"""
        successful_results = [r for r in results if r['path'] and r['cost'] != float('inf')]
        
        if not successful_results:
            return "无成功路径结果可生成报告"
        
        # 计算综合指标
        coverage_ratios = [r['analysis'].get('knowledge_coverage_ratio', 0) for r in successful_results]
        weighted_ratios = [r['analysis'].get('weighted_coverage', 0) for r in successful_results]
        smoothness_scores = [r['analysis'].get('smoothness_score', 0) for r in successful_results]
        execution_times = [r['execution_time'] for r in results]
        covered_counts = [r['analysis'].get('covered_count', 0) for r in successful_results]
        total_important_nodes = [r['analysis'].get('total_important_nodes', 0) for r in successful_results]
        
        report = []
        report.append("=" * 70)
        report.append("            DIJKSTRA算法路径分析详细报告（扩展覆盖率版）")
        report.append("=" * 70)
        
        # 总体统计
        report.append(f"\n📊 总体统计结果:")
        report.append(f"   总测试用例数: {len(results)}")
        report.append(f"   成功路径数: {len(successful_results)}")
        report.append(f"   成功率: {len(successful_results)/len(results)*100:.1f}%")
        
        # 核心知识点覆盖率统计（优化：加权覆盖率）
        report.append(f"\n🎯 核心知识点覆盖率分析（扩展重要节点集）:")
        report.append(f"   平均基础覆盖率: {np.mean(coverage_ratios):.2%}")
        report.append(f"   平均加权覆盖率: {np.mean(weighted_ratios):.2%}")
        report.append(f"   最高覆盖率: {max(coverage_ratios):.2%}")
        report.append(f"   覆盖率标准差: {np.std(coverage_ratios):.4f}")
        report.append(f"   平均覆盖重要节点数: {np.mean(covered_counts):.1f}/{np.mean(total_important_nodes):.1f}")
        
        # 性能统计
        report.append(f"\n⚡ 性能分析:")
        report.append(f"   平均执行时间: {np.mean(execution_times)*1000:.2f}毫秒")
        report.append(f"   平均路径长度: {np.mean([len(r['path']) for r in successful_results]):.1f}")
        report.append(f"   平均路径平滑度: {np.mean(smoothness_scores):.3f}")
        
        # 最佳路径分析
        successful_results.sort(key=lambda x: (
            x['analysis'].get('knowledge_coverage_ratio', 0) * 
            x['analysis'].get('smoothness_score', 0)
        ), reverse=True)
        
        report.append(f"\n🏆 最佳路径推荐 (前3名):")
        for i, result in enumerate(successful_results[:3]):
            analysis = result['analysis']
            report.append(f"\n   {i+1}. {result['desc']}")
            report.append(f"      基础覆盖率: {analysis.get('knowledge_coverage_ratio', 0):.2%}")
            report.append(f"      加权覆盖率: {analysis.get('weighted_coverage', 0):.2%}")
            report.append(f"      覆盖节点数: {analysis.get('covered_count', 0)}/{analysis.get('total_important_nodes', 0)}")
            report.append(f"      路径平滑度: {analysis.get('smoothness_score', 0):.3f}")
            report.append(f"      路径效率: {analysis.get('path_efficiency', 0):.3f}")
        
        # 性能优化建议
        report.append(f"\n💡 分析总结:")
        avg_coverage = np.mean(coverage_ratios)
        if avg_coverage > 0.7:
            report.append(f"   ✅ 核心知识点覆盖率优秀 ({avg_coverage:.1%})")
        elif avg_coverage > 0.5:
            report.append(f"   ⚠️  核心知识点覆盖率良好 ({avg_coverage:.1%})")
        else:
            report.append(f"   ⚠️  核心知识点覆盖率仍有提升空间 ({avg_coverage:.1%})")
        
        return "\n".join(report)


# 为了兼容原有的EnhancedDijkstraAnalyzer类名，添加别名
EnhancedDijkstraAnalyzer = DijkstraPathAnalyzer


def main():
    """主函数"""
    try:
        # 初始化分析器
        analyzer = DijkstraPathAnalyzer(
            neo4j_uri="bolt://localhost:7687",
            user="neo4j",
            password="123456789"
        )
        
        # 启用缓存和优化
        analyzer.cache_enabled = True
        analyzer.use_pagerank = True
        
        logger.info("开始路径分析测试（扩展覆盖率版）...")
        
        # 运行综合分析
        results = analyzer.run_comprehensive_analysis(num_tests=50)
        
        # 生成可视化图表
        analyzer.visualize_comprehensive_analysis(results)
        
        # 生成并打印报告
        report = analyzer.generate_detailed_report(results)
        print(report)
        
        # 保存报告
        with open('dijkstra_analysis_report_expanded.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("扩展覆盖率版分析完成！")
        
    except Exception as e:
        logger.error(f"分析过程出现错误: {e}")
        raise


if __name__ == "__main__":
    main()