from py2neo import Graph
from typing import Dict, List, Tuple, Union, Set, Optional
import logging
import warnings
import time
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

# 禁用Py2neo弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KnowledgePathPlanner")

class CoverageType(Enum):
    """覆盖率类型枚举"""
    OVERALL = "overall"  # 总体覆盖率
    CHAPTER = "chapter"  # 章节覆盖率
    BLOOM_LEVEL = "bloom"  # Bloom层级覆盖率
    KNOWLEDGE_TYPE = "knowledge_type"  # 知识点类型覆盖率

@dataclass
class CoverageResult:
    """覆盖率计算结果数据类"""
    coverage_ratio: float
    covered_points: Set[int]
    total_points: int
    details: Dict[str, float]
    coverage_by_category: Dict[str, Dict[str, float]]

class KnowledgePathPlanner:
    def __init__(self, neo4j_uri: str, user: str, password: str):
        """初始化路径规划器，支持任意节点类型"""
        self.graph = Graph(neo4j_uri, auth=(user, password))
        self.G = nx.DiGraph()  # 初始化NetworkX图对象
        self.node_info = {}
        self.node_attrs = {}
        
        # 覆盖率分析相关属性
        self.knowledge_points_index = {}  # 知识点索引
        self.chapter_index = defaultdict(set)  # 章节索引
        self.bloom_level_index = defaultdict(set)  # Bloom层级索引
        self.knowledge_type_index = defaultdict(set)  # 知识点类型索引
        
        # 算法配置
        self.RIPPLE_DEPTH = 20  # 增加涟漪深度
        self.RIPPLE_ATTENUATION = 0.85  # 降低衰减系数
        self.TOPO_SORT_MAX_ITER = 2000  # 增加拓扑排序迭代次数
        self.NODE_COLLECT_DEPTH = 20  # 增加节点收集深度
        self.BFS_MAX_DEPTH = 30  # 增加BFS最大深度
        
        # 平滑度优化参数
        self.SMOOTHNESS_CHAPTER_WEIGHT = 0.3  # 降低章节变化权重
        self.SMOOTHNESS_BLOOM_WEIGHT = 0.2   # 降低Bloom层级变化权重
        self.SMOOTHNESS_SMALL_CHANGE_THRESHOLD = 1  # Bloom层级小变化阈值
        self.SMOOTHNESS_PENALTY_FACTOR = 0.5  # 小变化惩罚因子
        
        self._validate_graph_connection()
        self._cache_all_relations()

    def _validate_graph_connection(self):
        """验证连接并检查核心节点/关系"""
        try:
            supported_nodes = ["KnowledgePoint", "SubKnowledgePoint", "Chapter"]
            for node_type in supported_nodes:
                count = self.graph.run(f"MATCH (n:{node_type}) RETURN count(n) AS cnt").data()[0]["cnt"]
                logger.info(f"检测到{node_type}节点：{count}个")

            # 查询所有关系类型
            rel_types = self.graph.run("""
                MATCH ()-[r]->() 
                RETURN DISTINCT type(r) AS rel_type, count(r) AS cnt
                ORDER BY cnt DESC
            """).data()
            
            for rel in rel_types:
                logger.info(f"检测到关系类型 {rel['rel_type']}: {rel['cnt']}个")

        except Exception as e:
            logger.critical(f"知识图谱连接失败：{str(e)}")
            raise

    def _cache_all_relations(self):
        """缓存所有关系（包括所有类型），并同步到NetworkX图"""
        self.relation_cache = defaultdict(lambda: defaultdict(list))
        
        # 获取所有关系类型
        rel_types = self.graph.run("""
            MATCH ()-[r]->() 
            RETURN DISTINCT type(r) AS rel_type
        """).data()
        
        # 加载所有关系
        for rel in rel_types:
            rel_type = rel['rel_type']
            query = f"""
            MATCH (n)-[r:{rel_type}]->(m)
            RETURN id(n) AS src_id, labels(n)[0] AS src_label,
                   id(m) AS tgt_id, labels(m)[0] AS tgt_label,
                   coalesce(r.weight, 1.0) AS weight
            """
            results = self.graph.run(query).data()
            
            for res in results:
                src_id = res["src_id"]
                src_label = res["src_label"]
                tgt_id = res["tgt_id"]
                tgt_label = res["tgt_label"]
                weight = float(res["weight"])
                
                self.relation_cache[src_label][rel_type].append((src_id, tgt_id, weight, tgt_label))
                self.G.add_node(src_id, label=src_label)
                self.G.add_node(tgt_id, label=tgt_label)
                self.G.add_edge(src_id, tgt_id, weight=weight, rel_type=rel_type)
        
        logger.info("所有关系缓存完成")

    def load_graph(self):
        """加载图数据（包含节点详情）并构建覆盖率分析索引"""
        query = """
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
        for rec in self.graph.run(query):
            nid = rec['nid']
            label = rec['label']
            chapter_id = str(rec['chapter_id'])
            bloom_level = int(rec['bloom_level']) if rec['bloom_level'] else 0
            
            self.node_info[nid] = {
                'label': label,
                'title': rec['title'],
                'chapter_id': chapter_id,
                'bloom_level': bloom_level,
                'importance': float(rec['importance']),
                'difficulty': float(rec['difficulty'])
            }
            self.node_attrs[nid] = {
                'raw_weight': float(rec['importance'])
            }
            
            # 构建覆盖率分析索引
            if label in ["KnowledgePoint", "SubKnowledgePoint"]:
                self.knowledge_points_index[nid] = {
                    'type': label,
                    'chapter': chapter_id,
                    'bloom_level': bloom_level,
                    'importance': float(rec['importance'])
                }
                self.chapter_index[chapter_id].add(nid)
                self.bloom_level_index[bloom_level].add(nid)
                self.knowledge_type_index[label].add(nid)
        
        logger.info(f"图数据加载完成，共{len(self.node_info)}个节点")
        logger.info(f"知识点索引构建完成，共{len(self.knowledge_points_index)}个知识点")
        logger.info(f"覆盖章节数：{len(self.chapter_index)}")
        logger.info(f"覆盖Bloom层级数：{len(self.bloom_level_index)}")

    def calculate_smoothness_score(self, path: List[int]) -> float:
        """优化平滑度计算，放宽限制"""
        if len(path) < 2:
            return 1.0  # 单节点路径平滑度为1
        
        # 获取路径节点的章节和Bloom层级
        chapters = []
        blooms = []
        for nid in path:
            if nid in self.node_info:
                chapters.append(self.node_info[nid]['chapter_id'])
                blooms.append(self.node_info[nid]['bloom_level'])
        
        if len(chapters) < 2 or len(blooms) < 2:
            return 0.8  # 信息不足时给予较高平滑度
        
        # 计算变化次数（放宽小变化的限制）
        chapter_changes = sum(1 for i in range(1, len(chapters)) if chapters[i] != chapters[i-1])
        
        # 对小变化降低权重
        bloom_diff = np.abs(np.diff(blooms))
        bloom_changes = sum(
            d * self.SMOOTHNESS_PENALTY_FACTOR 
            if d <= self.SMOOTHNESS_SMALL_CHANGE_THRESHOLD 
            else d 
            for d in bloom_diff
        )
        
        # 平滑度得分（变化越小得分越高，放宽后得分更稳定）
        total_changes = chapter_changes * self.SMOOTHNESS_CHAPTER_WEIGHT + bloom_changes * self.SMOOTHNESS_BLOOM_WEIGHT
        smoothness_score = 1.0 / (1.0 + total_changes)
        
        return min(1.0, smoothness_score)  # 限制最大得分为1

    def calculate_knowledge_coverage(self, paths: List[List[int]], 
                                    coverage_type: CoverageType = CoverageType.OVERALL) -> CoverageResult:
        """
        计算知识点覆盖率（增强版）
        
        Args:
            paths: 路径列表，每个路径是节点ID的列表
            coverage_type: 覆盖率计算类型
            
        Returns:
            CoverageResult: 详细的覆盖率计算结果
        """
        # 收集所有覆盖的节点
        covered_points = set()
        for path in paths:
            if path and len(path) > 0:
                covered_points.update(path)
        
        # 根据覆盖率类型筛选目标知识点
        if coverage_type == CoverageType.OVERALL:
            target_points = set(self.knowledge_points_index.keys())
        elif coverage_type == CoverageType.CHAPTER:
            target_points = set()
            for chapter_points in self.chapter_index.values():
                target_points.update(chapter_points)
        elif coverage_type == CoverageType.BLOOM_LEVEL:
            target_points = set()
            for level_points in self.bloom_level_index.values():
                target_points.update(level_points)
        elif coverage_type == CoverageType.KNOWLEDGE_TYPE:
            target_points = set()
            for type_points in self.knowledge_type_index.values():
                target_points.update(type_points)
        else:
            target_points = set(self.knowledge_points_index.keys())
        
        # 计算总体覆盖率
        total_points = len(target_points)
        if total_points == 0:
            coverage_ratio = 0.0
        else:
            # 只计算知识点类型的节点覆盖率
            covered_knowledge_points = covered_points & target_points
            coverage_ratio = len(covered_knowledge_points) / total_points
        
        # 计算详细覆盖率指标
        details = {
            'coverage_ratio': coverage_ratio,
            'covered_count': len(covered_knowledge_points),
            'total_count': total_points,
            'coverage_percentage': coverage_ratio * 100
        }
        
        # 按类别计算覆盖率
        coverage_by_category = {}
        
        # 按章节覆盖率
        chapter_coverage = {}
        for chapter_id, chapter_points in self.chapter_index.items():
            covered_in_chapter = len(covered_points & chapter_points)
            total_in_chapter = len(chapter_points)
            chapter_coverage[chapter_id] = {
                'covered': covered_in_chapter,
                'total': total_in_chapter,
                'ratio': covered_in_chapter / total_in_chapter if total_in_chapter > 0 else 0
            }
        coverage_by_category['chapters'] = chapter_coverage
        
        # 按Bloom层级覆盖率
        bloom_coverage = {}
        for level, level_points in self.bloom_level_index.items():
            covered_in_level = len(covered_points & level_points)
            total_in_level = len(level_points)
            bloom_coverage[str(level)] = {
                'covered': covered_in_level,
                'total': total_in_level,
                'ratio': covered_in_level / total_in_level if total_in_level > 0 else 0
            }
        coverage_by_category['bloom_levels'] = bloom_coverage
        
        # 按知识点类型覆盖率
        type_coverage = {}
        for ktype, type_points in self.knowledge_type_index.items():
            covered_in_type = len(covered_points & type_points)
            total_in_type = len(type_points)
            type_coverage[ktype] = {
                'covered': covered_in_type,
                'total': total_in_type,
                'ratio': covered_in_type / total_in_type if total_in_type > 0 else 0
            }
        coverage_by_category['knowledge_types'] = type_coverage
        
        # 计算加权覆盖率（考虑知识点重要性）
        total_importance = sum(self.node_info[nid]['importance'] for nid in target_points)
        covered_importance = sum(self.node_info[nid]['importance'] 
                               for nid in covered_knowledge_points if nid in self.node_info)
        weighted_coverage = covered_importance / total_importance if total_importance > 0 else 0
        
        details['weighted_coverage'] = weighted_coverage
        details['weighted_percentage'] = weighted_coverage * 100
        
        # 计算路径平滑度指标
        if paths:
            avg_smoothness = np.mean([self.calculate_smoothness_score(path) for path in paths if len(path) > 1])
            details['average_smoothness'] = avg_smoothness
        
        return CoverageResult(
            coverage_ratio=coverage_ratio,
            covered_points=covered_knowledge_points,
            total_points=total_points,
            details=details,
            coverage_by_category=coverage_by_category
        )
    
    def calculate_path_coverage_metrics(self, paths: List[List[int]]) -> Dict[str, float]:
        """
        计算路径覆盖的详细指标
        
        Args:
            paths: 路径列表
            
        Returns:
            包含多种覆盖率指标的字典
        """
        # 计算各种类型的覆盖率
        overall_coverage = self.calculate_knowledge_coverage(paths, CoverageType.OVERALL)
        chapter_coverage = self.calculate_knowledge_coverage(paths, CoverageType.CHAPTER)
        bloom_coverage = self.calculate_knowledge_coverage(paths, CoverageType.BLOOM_LEVEL)
        
        # 计算路径多样性指标
        unique_nodes = set()
        for path in paths:
            unique_nodes.update(path)
        
        # 计算路径覆盖的均匀性
        chapter_distribution = defaultdict(int)
        for node_id in unique_nodes:
            if node_id in self.node_info:
                chapter_id = self.node_info[node_id]['chapter_id']
                chapter_distribution[chapter_id] += 1
        
        # 计算熵值作为均匀性指标
        if chapter_distribution:
            total = sum(chapter_distribution.values())
            entropy = -sum((count/total) * np.log2(count/total) for count in chapter_distribution.values() if count > 0)
            max_entropy = np.log2(len(chapter_distribution))
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uniformity = 0
        
        # 计算Bloom层级覆盖广度
        bloom_levels_covered = set()
        for node_id in unique_nodes:
            if node_id in self.node_info:
                bloom_levels_covered.add(self.node_info[node_id]['bloom_level'])
        bloom_coverage_breadth = len(bloom_levels_covered) / len(self.bloom_level_index) if self.bloom_level_index else 0
        
        # 计算平均平滑度
        avg_smoothness = np.mean([self.calculate_smoothness_score(path) for path in paths if len(path) > 1]) if paths else 0
        
        return {
            'overall_coverage': overall_coverage.coverage_ratio,
            'weighted_coverage': overall_coverage.details['weighted_coverage'],
            'chapter_coverage': chapter_coverage.coverage_ratio,
            'bloom_level_coverage': bloom_coverage.coverage_ratio,
            'coverage_uniformity': uniformity,
            'bloom_coverage_breadth': bloom_coverage_breadth,
            'average_smoothness': avg_smoothness,
            'unique_nodes_covered': len(unique_nodes),
            'total_knowledge_points': len(self.knowledge_points_index),
            'chapters_covered': len(chapter_distribution),
            'total_chapters': len(self.chapter_index),
            'bloom_levels_covered': len(bloom_levels_covered),
            'total_bloom_levels': len(self.bloom_level_index)
        }
    
    def get_coverage_report(self, paths: List[List[int]]) -> str:
        """生成覆盖率分析报告"""
        metrics = self.calculate_path_coverage_metrics(paths)
        coverage_result = self.calculate_knowledge_coverage(paths)
        
        report = []
        report.append("=" * 60)
        report.append("知识点覆盖率分析报告")
        report.append("=" * 60)
        report.append(f"\n总体覆盖率: {metrics['overall_coverage']:.2%}")
        report.append(f"加权覆盖率(考虑重要性): {metrics['weighted_coverage']:.2%}")
        report.append(f"章节覆盖率: {metrics['chapter_coverage']:.2%}")
        report.append(f"Bloom层级覆盖率: {metrics['bloom_level_coverage']:.2%}")
        report.append(f"\n覆盖节点数: {metrics['unique_nodes_covered']}/{metrics['total_knowledge_points']}")
        report.append(f"覆盖章节数: {metrics['chapters_covered']}/{metrics['total_chapters']}")
        report.append(f"覆盖Bloom层级数: {metrics['bloom_levels_covered']}/{metrics['total_bloom_levels']}")
        report.append(f"\n覆盖均匀性指数: {metrics['coverage_uniformity']:.3f}")
        report.append(f"Bloom层级覆盖广度: {metrics['bloom_coverage_breadth']:.3f}")
        report.append(f"平均路径平滑度: {metrics['average_smoothness']:.3f} (已放宽限制)")
        
        # 按章节详细统计
        report.append("\n\n按章节覆盖率统计:")
        report.append("-" * 40)
        for chapter_id, stats in coverage_result.coverage_by_category['chapters'].items():
            if stats['total'] > 0:
                report.append(f"章节 {chapter_id}: {stats['covered']}/{stats['total']} ({stats['ratio']:.2%})")
        
        # 按Bloom层级详细统计
        report.append("\n\n按Bloom层级覆盖率统计:")
        report.append("-" * 40)
        for level, stats in sorted(coverage_result.coverage_by_category['bloom_levels'].items()):
            if stats['total'] > 0:
                report.append(f"Bloom层级 {level}: {stats['covered']}/{stats['total']} ({stats['ratio']:.2%})")
        
        return "\n".join(report)

    def collaborative_iteration(self, start: int, goal: int, max_iters: int = 20) -> Tuple[List[int], float]:
        """统一接口：生成路径，返回（路径节点ID列表，路径成本）"""
        try:
            # 尝试多种路径查找方法
            path_ids = self.find_path_bfs(start, goal)
            if path_ids:
                smoothness = self.calculate_smoothness_score(path_ids)
                logger.info(f"BFS路径平滑度: {smoothness:.3f}")
            
            if not path_ids or smoothness < 0.5:  # 平滑度不足时尝试其他算法
                path_ids, total_weight = self._ripple_cross_type_search_fixed(start, goal)
                if path_ids:
                    smoothness = self.calculate_smoothness_score(path_ids)
                    logger.info(f"RippleNet路径平滑度: {smoothness:.3f}")
            
            if not path_ids:
                path_ids = self._topo_cross_type_sort_fixed(start, goal)
                total_weight = -1.0
                if path_ids:
                    smoothness = self.calculate_smoothness_score(path_ids)
                    logger.info(f"拓扑排序路径平滑度: {smoothness:.3f}")
            else:
                total_weight = self.calculate_path_weight(path_ids)
            
            return path_ids, total_weight
        except Exception as e:
            logger.error(f"路径生成失败: {str(e)}")
            return [], float('inf')

    def find_path_bfs(self, start_id: int, target_id: int) -> List[int]:
        """使用BFS查找最短路径，优化平滑度"""
        if start_id == target_id:
            return [start_id]
            
        visited = {start_id: None}
        queue = deque([start_id])
        found = False
        
        while queue and len(visited) < self.BFS_MAX_DEPTH * 10:
            current = queue.popleft()
            
            if current == target_id:
                found = True
                break
                
            # 获取所有邻居
            neighbors = []
            if current in self.G:
                neighbors = list(self.G.successors(current))
            
            # 优先选择章节和Bloom层级相近的节点（平滑度优化）
            if current in self.node_info:
                current_chapter = self.node_info[current]['chapter_id']
                current_bloom = self.node_info[current]['bloom_level']
                
                # 按平滑度排序邻居
                def smoothness_score(neighbor_id):
                    if neighbor_id not in self.node_info:
                        return 0
                    chapter_diff = 0 if self.node_info[neighbor_id]['chapter_id'] == current_chapter else 1
                    bloom_diff = abs(self.node_info[neighbor_id]['bloom_level'] - current_bloom)
                    if bloom_diff <= self.SMOOTHNESS_SMALL_CHANGE_THRESHOLD:
                        bloom_diff *= self.SMOOTHNESS_PENALTY_FACTOR
                    return -(chapter_diff * self.SMOOTHNESS_CHAPTER_WEIGHT + bloom_diff * self.SMOOTHNESS_BLOOM_WEIGHT)
                
                neighbors.sort(key=smoothness_score, reverse=True)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        if not found:
            logger.debug("BFS未找到路径")
            return []
        
        # 回溯路径
        path = []
        current = target_id
        while current is not None:
            path.append(current)
            current = visited[current]
        
        return path[::-1]

    def _ripple_cross_type_search_fixed(self, start_id: int, target_id: int) -> Tuple[List[int], float]:
        """修复版RippleNet跨类型搜索，优化平滑度"""
        if start_id == target_id:
            return [start_id], 1.0
            
        # 获取起始节点类型
        start_type = self.G.nodes[start_id].get('label') if start_id in self.G else None
        if not start_type:
            start_type = self._get_node_basic_info(start_id)[0]
        
        # 初始化传播
        ripple_sets = {0: {start_id: (1.0, None)}}
        visited = {start_id: (1.0, None)}
        
        logger.debug(f"RippleNet开始：起始{start_id}({start_type}) → 目标{target_id}")

        for depth in range(1, self.RIPPLE_DEPTH + 1):
            current_ripple = {}
            prev_nodes = ripple_sets[depth - 1]
            
            for node_id, (node_weight, prev_node) in prev_nodes.items():
                # 获取节点类型
                node_type = self.G.nodes[node_id].get('label') if node_id in self.G else None
                if not node_type:
                    node_type = self._get_node_basic_info(node_id)[0]
                
                # 遍历所有关系类型
                if node_type in self.relation_cache:
                    for rel_type, rel_list in self.relation_cache[node_type].items():
                        for src_id, tgt_id, rel_weight, tgt_label in rel_list:
                            if src_id == node_id:
                                # 计算传播权重
                                propagation_weight = node_weight * (self.RIPPLE_ATTENUATION ** depth) * rel_weight
                                
                                # 计算到当前节点的路径平滑度
                                temp_path = []
                                current_node = node_id
                                temp_path.append(current_node)
                                temp_prev = prev_node
                                while temp_prev is not None and temp_prev in visited:
                                    temp_path.append(temp_prev)
                                    temp_prev = visited[temp_prev][1]
                                temp_path.reverse()
                                temp_path.append(tgt_id)
                                
                                # 加入平滑度权重
                                smoothness = self.calculate_smoothness_score(temp_path)
                                propagation_weight *= (0.8 + smoothness * 0.2)  # 平滑度奖励
                                
                                # 更新访问记录
                                if tgt_id not in visited or propagation_weight > visited[tgt_id][0]:
                                    visited[tgt_id] = (propagation_weight, node_id)
                                    current_ripple[tgt_id] = (propagation_weight, node_id)
                                    
                                    # 找到目标节点
                                    if tgt_id == target_id:
                                        logger.debug(f"深度{depth}：找到目标节点{target_id}，权重{propagation_weight:.2f}")
                                        # 立即回溯路径
                                        path = self._reconstruct_path(visited, start_id, target_id)
                                        return path, propagation_weight
            
            if current_ripple:
                ripple_sets[depth] = current_ripple
            else:
                logger.debug(f"深度{depth}：无新节点传播，终止")
                break

        # 检查是否到达目标
        if target_id in visited:
            path = self._reconstruct_path(visited, start_id, target_id)
            total_weight = visited[target_id][0]
            return path, total_weight
        else:
            logger.warning("RippleNet未找到路径")
            return [], 0.0

    def _reconstruct_path(self, visited: dict, start_id: int, target_id: int) -> List[int]:
        """路径重构"""
        path = [target_id]
        current_node = target_id
        
        while current_node != start_id:
            if current_node not in visited:
                logger.error(f"回溯断裂：节点{current_node}不在访问记录中")
                return []
            
            prev_node = visited[current_node][1]
            if prev_node is None:
                logger.error(f"回溯断裂：节点{current_node}无前驱")
                return []
                
            path.insert(0, prev_node)
            current_node = prev_node
            
            # 放宽路径长度限制
            if len(path) > self.RIPPLE_DEPTH * 3:
                logger.warning("路径较长，但继续尝试重构")
                # 不直接返回，继续尝试
            
        logger.debug(f"路径回溯完成，长度{len(path)}")
        return path

    def _topo_cross_type_sort_fixed(self, start_id: int, target_id: int) -> List[int]:
        """修复版拓扑排序，放宽限制"""
        if start_id == target_id:
            return [start_id]
            
        # 使用NetworkX的最短路径
        try:
            path = nx.shortest_path(self.G, source=start_id, target=target_id)
            logger.debug(f"NetworkX最短路径长度：{len(path)}")
            return path
        except nx.NetworkXNoPath:
            logger.warning("NetworkX未找到最短路径")
        
        # 使用DFS，放宽深度限制
        path = self._dfs_search(start_id, target_id, max_depth=self.BFS_MAX_DEPTH * 2)
        if path:
            return path
        
        # 原始拓扑排序作为最后手段
        related_nodes = self._collect_related_nodes(start_id, depth=self.NODE_COLLECT_DEPTH * 2)
        if target_id not in related_nodes:
            logger.warning("目标节点不在相关节点中")
            return []
        
        # 使用NetworkX的拓扑排序
        try:
            subgraph = self.G.subgraph(related_nodes)
            if nx.is_directed_acyclic_graph(subgraph):
                topo_order = list(nx.topological_sort(subgraph))
            else:
                # 处理有环图，使用DFS排序
                topo_order = list(nx.dfs_postorder_nodes(subgraph, source=start_id))
            
            # 找到起始和目标节点的位置
            start_idx = topo_order.index(start_id) if start_id in topo_order else -1
            target_idx = topo_order.index(target_id) if target_id in topo_order else -1
            
            if start_idx >= 0 and target_idx >= 0:
                if start_idx < target_idx:
                    return topo_order[start_idx:target_idx+1]
                else:
                    # 允许反向路径
                    return topo_order[target_idx:start_idx+1][::-1]
            elif start_idx >= 0:
                # 从起始节点开始找路径
                return self._find_path_in_topo(topo_order, start_id, target_id)
        except Exception as e:
            logger.warning(f"拓扑排序失败：{e}")
        
        return []

    def _dfs_search(self, start_id: int, target_id: int, visited: set = None, 
                   path: list = None, current_depth: int = 0, max_depth: int = 60) -> List[int]:
        """DFS搜索路径，放宽深度限制"""
        if visited is None:
            visited = set()
        if path is None:
            path = []
            
        if current_depth > max_depth:
            return []
            
        path.append(start_id)
        visited.add(start_id)
        
        if start_id == target_id:
            return path.copy()
            
        # 获取所有邻居，优先选择平滑度高的
        neighbors = []
        if start_id in self.G:
            neighbors = list(self.G.successors(start_id)) + list(self.G.predecessors(start_id))
        
        # 去重并过滤已访问节点
        neighbors = [n for n in neighbors if n not in visited]
        
        # 按潜在平滑度排序
        if start_id in self.node_info:
            current_chapter = self.node_info[start_id]['chapter_id']
            current_bloom = self.node_info[start_id]['bloom_level']
            
            def neighbor_smoothness_score(neighbor_id):
                if neighbor_id not in self.node_info:
                    return 0
                chapter_diff = 0 if self.node_info[neighbor_id]['chapter_id'] == current_chapter else 1
                bloom_diff = abs(self.node_info[neighbor_id]['bloom_level'] - current_bloom)
                return -(chapter_diff + bloom_diff * 0.5)
            
            neighbors.sort(key=neighbor_smoothness_score, reverse=True)
        
        for neighbor in neighbors:
            result = self._dfs_search(neighbor, target_id, visited, path, current_depth + 1, max_depth)
            if result:
                return result
        
        path.pop()
        visited.remove(start_id)
        return []

    def _find_path_in_topo(self, topo_order: list, start_id: int, target_id: int) -> List[int]:
        """在拓扑序列中查找路径，放宽限制"""
        try:
            # 构建更宽松的可达性字典
            reachable = defaultdict(list)
            for i, node in enumerate(topo_order):
                if node in self.G:
                    for neighbor in self.G.successors(node):
                        if neighbor in topo_order:
                            reachable[node].append(neighbor)
                    # 也考虑前驱节点，增加路径可能性
                    for neighbor in self.G.predecessors(node):
                        if neighbor in topo_order:
                            reachable[node].append(neighbor)
            
            # BFS查找路径，允许更长路径
            queue = deque([start_id])
            prev = {start_id: None}
            
            while queue and len(prev) < self.TOPO_SORT_MAX_ITER:
                current = queue.popleft()
                if current == target_id:
                    break
                for neighbor in reachable.get(current, []):
                    if neighbor not in prev:
                        prev[neighbor] = current
                        queue.append(neighbor)
            
            # 重构路径
            if target_id in prev:
                path = []
                current = target_id
                while current is not None:
                    path.append(current)
                    current = prev[current]
                return path[::-1]
                
        except Exception as e:
            logger.debug(f"在拓扑序列中查找路径失败：{e}")
        
        return []

    def calculate_path_weight(self, path_ids: List[int]) -> float:
        """计算路径总权重"""
        if len(path_ids) <= 1:
            return 1.0
            
        total_weight = 0.0
        smoothness = self.calculate_smoothness_score(path_ids)
        
        for i in range(len(path_ids) - 1):
            src = path_ids[i]
            tgt = path_ids[i + 1]
            
            if src in self.G and tgt in self.G[src]:
                weight = self.G[src][tgt].get('weight', 1.0)
                # 加入平滑度调整
                weight *= (0.9 + smoothness * 0.1)
                total_weight += weight
        
        return total_weight

    def _collect_related_nodes(self, start_id: int, depth: int) -> Set[int]:
        """收集相关节点，放宽深度限制"""
        related = set()
        queue = deque([(start_id, 0)])
        
        while queue:
            node_id, current_depth = queue.popleft()
            
            if node_id in related or current_depth > depth:
                continue
                
            related.add(node_id)
            
            # 获取所有邻居，包括前驱和后继
            if node_id in self.G:
                for neighbor in self.G.successors(node_id):
                    queue.append((neighbor, current_depth + 1))
                for neighbor in self.G.predecessors(node_id):
                    queue.append((neighbor, current_depth + 1))

        logger.debug(f"收集相关节点完成，共{len(related)}个节点")
        return related

    def _get_node_basic_info(self, internal_id: int) -> Tuple[str, str]:
        """获取节点基本信息（类型+标题）"""
        query = """
        MATCH (n) 
        WHERE id(n) = $internal_id
        RETURN labels(n)[0] AS node_type, coalesce(n.title, '未知节点') AS title
        """
        result = self.graph.run(query, internal_id=internal_id).data()
        if not result:
            logger.error(f"节点ID {internal_id} 不存在")
            return "Unknown", "未知节点"
        return result[0]["node_type"], result[0]["title"]

    def _enrich_path_info(self, path_ids: List[int]) -> List[Tuple[int, str, str]]:
        """补充路径信息"""
        path_info = []
        for node_id in path_ids:
            node_type, title = self._get_node_basic_info(node_id)
            path_info.append((node_id, title, node_type))
        return path_info

    def get_node_details(self, node_id):
        """统一节点详情获取方法"""
        if node_id in self.node_info:
            return {
                'id': node_id,
                'label': self.node_info[node_id]['label'],
                'title': self.node_info[node_id]['title'],
                'chapter_id': self.node_info[node_id]['chapter_id'],
                'bloom_level': self.node_info[node_id]['bloom_level'],
                'edu_pr': self.node_attrs.get(node_id, {}).get('edu_pr', 0.0),
                'importance': self.node_info[node_id]['importance'],
                'difficulty': self.node_info[node_id]['difficulty']
            }
        else:
            # 从数据库获取
            node_type, title = self._get_node_basic_info(node_id)
            return {
                'id': node_id,
                'label': node_type,
                'title': title,
                'chapter_id': '未知章节',
                'bloom_level': 0,
                'edu_pr': 0.0,
                'importance': 1.0,
                'difficulty': 1.0
            }

    def generate_multiple_paths(self, start: int, goal: int, num_paths: int = 3) -> List[List[int]]:
        """生成多条路径并选择平滑度最高的"""
        all_paths = []
        
        # 尝试不同算法生成路径
        path_bfs = self.find_path_bfs(start, goal)
        if path_bfs:
            all_paths.append((path_bfs, self.calculate_smoothness_score(path_bfs)))
        
        path_ripple, _ = self._ripple_cross_type_search_fixed(start, goal)
        if path_ripple and path_ripple not in [p[0] for p in all_paths]:
            all_paths.append((path_ripple, self.calculate_smoothness_score(path_ripple)))
        
        path_topo = self._topo_cross_type_sort_fixed(start, goal)
        if path_topo and path_topo not in [p[0] for p in all_paths]:
            all_paths.append((path_topo, self.calculate_smoothness_score(path_topo)))
        
        # 按平滑度排序并返回
        all_paths.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in all_paths[:num_paths]]


# 使用示例
if __name__ == "__main__":
    # 初始化路径规划器
    planner = KnowledgePathPlanner(
        neo4j_uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password"
    )
    
    # 加载图数据
    planner.load_graph()
    
    # 生成路径（示例）
    start_node = 1  # 替换为实际起始节点ID
    end_node = 10   # 替换为实际目标节点ID
    
    # 生成多条路径
    paths = planner.generate_multiple_paths(start_node, end_node, num_paths=3)
    
    # 分析覆盖率
    if paths:
        print(planner.get_coverage_report(paths))
        
        # 输出每条路径的详细信息
        for i, path in enumerate(paths):
            print(f"\n路径 {i+1}:")
            print(f"长度: {len(path)}")
            print(f"平滑度: {planner.calculate_smoothness_score(path):.3f}")
            print(f"节点序列: {path}")