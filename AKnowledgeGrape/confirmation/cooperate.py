import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
# 注意：需确保 EnhancedCollaborativeLearning 类可正常导入
from arithmetic.collaborative_learning import EnhancedCollaborativeLearning

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 虚拟学生模型 =====================
class VirtualStudent:
    def __init__(self, student_id, start_node, target_node, learning_ability=0.5):
        self.student_id = student_id
        self.start_node = start_node
        self.current_node = start_node
        self.target_node = target_node
        self.learning_ability = np.clip(learning_ability, 0.1, 0.9)
        self.mastered_nodes = set([start_node])
        self.learning_history = []
        self.path_progress = 0

    def study_node(self, node_id, difficulty):
        """引入非线性概率和保底成功率"""
        base_prob = self.learning_ability * (1 - 0.8 * difficulty)  # 降低难度影响
        min_prob = 0.1  # 保底10%成功率
        success_prob = max(base_prob, min_prob)
        return random.random() < success_prob

# ===================== 智能教学实验组 =====================
class TeachingSimulator:
    def __init__(self, neo4j_config):
        self.ta = EnhancedCollaborativeLearning(**neo4j_config)
        self.ta.load_graph()
        self.knowledge_difficulty = self._init_knowledge_difficulty()
        self.students = []
        self.teaching_logs = []
        self.debug_info = defaultdict(list)
        
        # 获取最大连通分量
        from networkx import strongly_connected_components
        scc = list(strongly_connected_components(self.ta.G))
        if scc:
            self.largest_scc = max(scc, key=len)
            self.available_nodes = list(self.largest_scc)
        else:
            self.available_nodes = []

    def _init_knowledge_difficulty(self):
        """初始化知识点难度（基于权威值）"""
        difficulties = {}
        for nid in self.ta.G.nodes:
            pr = self.ta.node_attrs[nid]['edu_pr']
            # 调整难度计算公式，pr越高难度越低
            difficulties[nid] = np.clip(1.0 - pr, 0.1, 0.9)
        return difficulties

    def generate_students(self, num_students=10):
        """生成虚拟学生群体，确保起点和终点在最大连通分量中"""
        if not self.available_nodes:
            raise ValueError("图中没有可用的连通节点")
        nodes = self.available_nodes
        max_attempts = 100
        for i in range(num_students):
            attempts = 0
            while True:
                start, target = random.sample(nodes, 2)
                path, _ = self.ta.a_star_search(start, target)
                if path and len(path) >= 2:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    # 处理方式：选择相邻节点
                    start = random.choice(nodes)
                    neighbors = list(self.ta.G.neighbors(start))
                    target = random.choice(neighbors) if neighbors else start
                    break
            ability = random.gauss(0.6, 0.15)
            self.students.append(
                VirtualStudent(
                    student_id=i + 1,
                    start_node=start,
                    target_node=target,
                    learning_ability=ability
                )
            )

    def _update_difficulty(self, node_id, success):
        current_diff = self.knowledge_difficulty[node_id]
        if success:
            delta = -0.1 * (current_diff ** 1.5)  # 高难度时降幅更大
        else:
            delta = 0.03 * (1 - current_diff ** 0.5)  # 高难度时增幅更小
        self.knowledge_difficulty[node_id] = np.clip(current_diff + delta, 0.1, 0.9)

    def single_learning_process(self, student, max_steps=15):
        """单个学生的学习过程模拟"""
        current_path = []
        attempts = 0

        while student.current_node != student.target_node and attempts < max_steps:
            # 获取推荐路径
            path, _ = self.ta.a_star_search(student.current_node, student.target_node)
            if not path or len(path) < 2:
                break  # 无有效路径时终止

            next_node = path[1]
            difficulty = self.knowledge_difficulty[next_node]

            # 模拟学习
            success = student.study_node(next_node, difficulty)
            self._update_difficulty(next_node, success)

            # 记录学习过程
            record = {
                'step': attempts,
                'current': student.current_node,
                'next': next_node,
                'difficulty': difficulty,
                'success': success,
                'path_length': len(path)
            }
            student.learning_history.append(record)

            if success:
                student.current_node = next_node
                student.mastered_nodes.add(next_node)
                current_path = []
            else:
                current_path.append(next_node)
                if len(current_path) > 2:  # 减少失败容忍次数
                    # 回退到已掌握节点中离目标最近的点
                    mastered_nodes = student.mastered_nodes
                    if mastered_nodes:
                        best_fallback = min(
                            mastered_nodes,
                            key=lambda n: len(self.ta.a_star_search(n, student.target_node)[0] or [])
                        )
                        student.current_node = best_fallback
                    current_path = []

            attempts += 1

        # 记录调试信息
        if student.current_node == student.target_node:
            self.debug_info['success_students'].append(student.student_id)
        else:
            self.debug_info['fail_students'].append({
                'id': student.student_id,
                'last_node': student.current_node,
                'attempts': attempts
            })
        return student

    def batch_teaching_experiment(self, num_cycles=10):
        self.debug_info = {
            'difficulty_changes': defaultdict(list),
            'common_fail_nodes': defaultdict(int),
            'cycle_details': []
        }
        metrics = {
            'success_rate': [],
            'avg_steps': [],
            'path_efficiency': []
        }

        print("\n【智能教学实验组 - 实时进度】:")
        for cycle in range(num_cycles):
            # 生成新一批学生(每周期重置)
            self.generate_students(num_students=20)
            cycle_success = []
            cycle_steps = []
            path_efficiencies = []
            cycle_fail_nodes = defaultdict(int)

            # 本周期教学过程
            for student in self.students:
                self.single_learning_process(student)

                # 记录学习结果
                final_success = student.current_node == student.target_node
                cycle_success.append(final_success)
                cycle_steps.append(len(student.learning_history))

                # 计算路径效率(包含失败尝试)
                if final_success:
                    start_node = student.start_node
                    target_node = student.target_node
                    optimal_path, _ = self.ta.a_star_search(start_node, target_node)
                    if optimal_path and len(optimal_path) >= 2:
                        optimal_steps = len(optimal_path) - 1
                        actual_steps = len(student.learning_history)  # 包含所有尝试
                        if actual_steps > 0:
                            efficiency = optimal_steps / actual_steps
                            path_efficiencies.append(efficiency)

                # 记录失败节点
                if not final_success and student.learning_history:
                    last_fail_node = student.learning_history[-1]['next']
                    cycle_fail_nodes[last_fail_node] += 1

            # 记录全局调试信息
            self.debug_info['common_fail_nodes'].update(cycle_fail_nodes)
            self.debug_info['cycle_details'].append({
                'cycle': cycle,
                'fail_nodes': dict(cycle_fail_nodes)
            })

            # 计算本周期指标
            metrics['success_rate'].append(
                np.mean(cycle_success) if cycle_success else 0
            )
            metrics['avg_steps'].append(
                np.mean(cycle_steps) if cycle_steps else 0
            )
            metrics['path_efficiency'].append(
                np.nanmean(path_efficiencies) if path_efficiencies else 0
            )

            # 记录难度变化趋势
            for node in self.knowledge_difficulty:
                self.debug_info['difficulty_changes'][node].append(
                    self.knowledge_difficulty[node]
                )

            # 动态调整教学策略参数
            self.ta.alpha *= 0.95  # 逐步降低随机探索
            self.teaching_logs.append(metrics.copy())

            # 实时输出关键信息
            top_difficult = sorted(
                self.knowledge_difficulty.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            top_fails = sorted(
                cycle_fail_nodes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            print(f"\n周期 {cycle + 1}/{num_cycles}")
            print(f"成功率: {metrics['success_rate'][-1]:.2%} | "
                  f"平均步数: {metrics['avg_steps'][-1]:.1f} | "
                  f"效率: {metrics['path_efficiency'][-1]:.2f}")
            print(f"当前难点: " +
                  " | ".join([f"节点{nid}(难度:{diff:.2f})" for nid, diff in top_difficult]))
            print(f"失败热点: " +
                  " | ".join([f"节点{nid}({count}次)" for nid, count in top_fails]))
            
            # 重置学生群体
            self.students = []

        return metrics

    def visualize_results(self, exp_metrics, ctrl_metrics):
        """可视化教学效果（实验组+对照组对比）"""
        plt.figure(figsize=(12, 6))

        # 主坐标系设置
        ax = plt.gca()
        ax.set_xlabel('教学周期', fontsize=11, labelpad=8)
        ax.spines['top'].set_visible(False)

        # 绘制成功率
        ax.plot(exp_metrics['success_rate'], 'o-', color='#2ca02c', markersize=6, linewidth=2, label='智能教学-成功率')
        ax.plot(ctrl_metrics['success_rate'], 'o--', color='#d62728', markersize=6, linewidth=2, label='传统教学-成功率')

        # 绘制路径效率
        ax.plot(exp_metrics['path_efficiency'], 'D-', color='#ff7f0e', markersize=5, linewidth=2, label='智能教学-路径效率')
        ax.plot(ctrl_metrics['path_efficiency'], 'D--', color='#9467bd', markersize=5, linewidth=2, label='传统教学-路径效率')

        # 左轴配置
        ax.set_ylabel('比率值', fontsize=11, labelpad=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.grid(True, axis='y', alpha=0.3)

        # 创建右Y轴（平均步数）
        ax2 = ax.twinx()
        ax2.plot(exp_metrics['avg_steps'], 's-', color='#1f77b4', markersize=5, linewidth=2, alpha=0.9, label='智能教学-平均步数')
        ax2.plot(ctrl_metrics['avg_steps'], 's--', color='#8c564b', markersize=5, linewidth=2, alpha=0.9, label='传统教学-平均步数')

        # 右轴配置
        ax2.set_ylabel('学习步数', fontsize=11, labelpad=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_linewidth(0.5)
        ax2.grid(True, axis='y', alpha=0.3)

        # 智能设置Y轴范围
        max_steps = max(max(exp_metrics['avg_steps']), max(ctrl_metrics['avg_steps'])) * 1.1
        ax2.set_ylim(0, max_steps if max_steps > 5 else 10)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=3, fontsize=10, frameon=False)

        # 全局样式优化
        plt.title('智能教学 vs 传统线性教学 效果对比', fontsize=13, pad=50)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax2.tick_params(axis='both', which='major', labelsize=9)

        # 对齐刻度线
        plt.setp(ax.get_xticklabels(), ha='center')

        plt.tight_layout()
        plt.savefig('teaching_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

# ===================== 传统线性教学对照组 =====================
class TraditionalLinearTeaching:
    def __init__(self, neo4j_config):
        self.ta = EnhancedCollaborativeLearning(**neo4j_config)
        self.ta.load_graph()
        self.knowledge_difficulty = self._init_knowledge_difficulty()
        self.students = []
        self.teaching_logs = []
        
        # 获取最大连通分量（和实验组保持一致）
        from networkx import strongly_connected_components
        scc = list(strongly_connected_components(self.ta.G))
        self.largest_scc = max(scc, key=len) if scc else []
        self.available_nodes = list(self.largest_scc)

    def _init_knowledge_difficulty(self):
        """初始化知识点难度（和实验组完全一致）"""
        difficulties = {}
        for nid in self.ta.G.nodes:
            pr = self.ta.node_attrs[nid]['edu_pr']
            difficulties[nid] = np.clip(1.0 - pr, 0.1, 0.9)
        return difficulties

    def generate_students(self, num_students=20):
        """生成和实验组完全相同的学生群体"""
        if not self.available_nodes:
            raise ValueError("图中没有可用的连通节点")
        nodes = self.available_nodes
        max_attempts = 100
        for i in range(num_students):
            attempts = 0
            while True:
                start, target = random.sample(nodes, 2)
                path, _ = self.ta.a_star_search(start, target)
                if path and len(path) >= 2:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    start = random.choice(nodes)
                    neighbors = list(self.ta.G.neighbors(start))
                    target = random.choice(neighbors) if neighbors else start
                    break
            ability = random.gauss(0.6, 0.15)
            self.students.append(
                VirtualStudent(
                    student_id=i + 1,
                    start_node=start,
                    target_node=target,
                    learning_ability=ability
                )
            )

    def single_linear_learning(self, student, max_steps=15):
        """传统线性学习过程：固定路径+固定难度+无回退"""
        current_path = []
        attempts = 0

        while student.current_node != student.target_node and attempts < max_steps:
            # 核心差异1：固定线性路径（按节点ID排序）
            reachable_nodes = [n for n in self.available_nodes if self.ta.a_star_search(student.current_node, n)[0]]
            if not reachable_nodes:
                break
            # 线性路径：优先选ID最小的下一个节点
            reachable_nodes.sort()
            next_node = reachable_nodes[0] if reachable_nodes else student.current_node
            difficulty = self.knowledge_difficulty[next_node]  # 难度固定，不调整

            # 核心差异2：学习成功率模型和实验组一致
            success = student.study_node(next_node, difficulty)

            # 记录过程（和实验组格式一致）
            record = {
                'step': attempts,
                'current': student.current_node,
                'next': next_node,
                'difficulty': difficulty,
                'success': success,
                'path_length': len(reachable_nodes)
            }
            student.learning_history.append(record)

            # 核心差异3：失败无回退，重复尝试
            if success:
                student.current_node = next_node
                student.mastered_nodes.add(next_node)

            attempts += 1
        return student

    def batch_linear_experiment(self, num_cycles=10):
        """传统线性教学的多周期实验"""
        metrics = {
            'success_rate': [],
            'avg_steps': [],
            'path_efficiency': []
        }

        print("\n【传统线性教学对照组 - 实时进度】:")
        for cycle in range(num_cycles):
            self.generate_students(num_students=20)
            cycle_success = []
            cycle_steps = []
            path_efficiencies = []

            for student in self.students:
                self.single_linear_learning(student)
                # 记录和实验组相同的指标
                final_success = student.current_node == student.target_node
                cycle_success.append(final_success)
                cycle_steps.append(len(student.learning_history))

                if final_success:
                    start_node = student.start_node
                    target_node = student.target_node
                    optimal_path, _ = self.ta.a_star_search(start_node, target_node)
                    if optimal_path and len(optimal_path) >= 2:
                        optimal_steps = len(optimal_path) - 1
                        actual_steps = len(student.learning_history)
                        if actual_steps > 0:
                            efficiency = optimal_steps / actual_steps
                            path_efficiencies.append(efficiency)

            # 计算对照组指标
            metrics['success_rate'].append(np.mean(cycle_success) if cycle_success else 0)
            metrics['avg_steps'].append(np.mean(cycle_steps) if cycle_steps else 0)
            metrics['path_efficiency'].append(np.nanmean(path_efficiencies) if path_efficiencies else 0)

            self.teaching_logs.append(metrics.copy())

            # 输出对照组进度
            print(f"周期 {cycle + 1}/{num_cycles} | 成功率: {metrics['success_rate'][-1]:.2%} | 平均步数: {metrics['avg_steps'][-1]:.1f}")
            self.students = []

        return metrics

# ===================== 教学效果分析模块 =====================
class TeachingAnalysis:
    """教学效果分析模块"""

    @staticmethod
    def analyze_learning_patterns(logs, simulator):
        pattern_report = {}
        # 计算平均指标，过滤NaN值
        pattern_report['avg_success_rate'] = np.nanmean([x['success_rate'][-1] for x in logs])
        pattern_report['avg_efficiency'] = np.nanmean([x['path_efficiency'][-1] for x in logs])

        # 识别最佳实践周期
        success_rates = [x['success_rate'][-1] for x in logs]
        best_cycle = np.nanargmax(success_rates)
        pattern_report['best_params'] = {
            'alpha': 0.85 * (0.95 ** best_cycle),
            'feedback_strength': 0.1 + best_cycle * 0.02
        }

        # 使用simulator访问调试信息
        diff_changes = simulator.debug_info['difficulty_changes']
        pattern_report['difficulty_analysis'] = {
            'most_improved': TeachingAnalysis._find_most_changed(diff_changes, threshold=0.1),
            'most_difficult': TeachingAnalysis._find_hardest_nodes(simulator)
        }

        # 学习路径分析
        pattern_report['path_analysis'] = {
            'common_failures': TeachingAnalysis._find_common_failures(simulator),
            'avg_retry_attempts': np.nanmean([s['attempts'] for s in simulator.debug_info.get('fail_students', [])]) if simulator.debug_info.get('fail_students') else 0
        }

        return pattern_report

    @staticmethod
    def _find_most_changed(diff_data, threshold=0.1):
        """找出难度变化最大的知识点"""
        changes = {}
        for node, history in diff_data.items():
            if len(history) >= 2:
                delta = history[-1] - history[0]
                if abs(delta) > threshold:
                    changes[node] = delta
        return dict(sorted(changes.items(), key=lambda x: x[1], reverse=True)[:3])

    @staticmethod
    def _find_hardest_nodes(simulator):
        """找出最终难度最高的节点"""
        return dict(sorted(simulator.knowledge_difficulty.items(),
                           key=lambda x: x[1], reverse=True)[:3])

    @staticmethod
    def _find_common_failures(simulator):
        """找出常见失败节点"""
        failures = simulator.debug_info['common_fail_nodes']
        return dict(sorted(failures.items(), key=lambda x: x[1], reverse=True)[:3])

# ===================== 辅助函数：计算提升率 =====================
def calculate_improvement(exp_metrics, ctrl_metrics):
    """计算实验组相对对照组的提升比例"""
    # 取最后3个周期的平均值（稳定值）
    def get_stable_average(metrics_list):
        if len(metrics_list) >= 3:
            return np.mean(metrics_list[-3:])
        return np.mean(metrics_list)
    
    # 成功率提升
    exp_success = get_stable_average(exp_metrics['success_rate'])
    ctrl_success = get_stable_average(ctrl_metrics['success_rate'])
    success_improvement = (exp_success - ctrl_success) / ctrl_success * 100 if ctrl_success != 0 else 0
    
    # 平均步数优化
    exp_steps = get_stable_average(exp_metrics['avg_steps'])
    ctrl_steps = get_stable_average(ctrl_metrics['avg_steps'])
    steps_reduction = (ctrl_steps - exp_steps) / ctrl_steps * 100 if ctrl_steps != 0 else 0
    
    # 路径效率提升
    exp_efficiency = get_stable_average(exp_metrics['path_efficiency'])
    ctrl_efficiency = get_stable_average(ctrl_metrics['path_efficiency'])
    efficiency_improvement = (exp_efficiency - ctrl_efficiency) / ctrl_efficiency * 100 if ctrl_efficiency != 0 else 0
    
    return {
        'success_improvement': success_improvement,
        'steps_reduction': steps_reduction,
        'efficiency_improvement': efficiency_improvement,
        'exp_success': exp_success,
        'ctrl_success': ctrl_success,
        'exp_steps': exp_steps,
        'ctrl_steps': ctrl_steps
    }

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # Neo4j 配置
    neo4j_config = {
        "neo4j_uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "123456789",
        "alpha": 0.85,
        "feedback_strength": 0.1
    }

    # ========== 1. 运行智能教学方案（实验组） ==========
    print("===== 开始运行智能教学方案（实验组） =====")
    exp_simulator = TeachingSimulator(neo4j_config)
    exp_metrics = exp_simulator.batch_teaching_experiment(num_cycles=10)

    # ========== 2. 运行传统线性教学（对照组） ==========
    print("\n===== 开始运行传统线性教学（对照组） =====")
    ctrl_simulator = TraditionalLinearTeaching(neo4j_config)
    ctrl_metrics = ctrl_simulator.batch_linear_experiment(num_cycles=10)

    # ========== 3. 计算提升比例 ==========
    improvement = calculate_improvement(exp_metrics, ctrl_metrics)

    # ========== 4. 可视化对比结果 ==========
    exp_simulator.visualize_results(exp_metrics, ctrl_metrics)

    # ========== 5. 生成分析报告 ==========
    analysis = TeachingAnalysis.analyze_learning_patterns(exp_simulator.teaching_logs, exp_simulator)

    # ========== 6. 输出对比结果 ==========
    print("\n===== 实验组 vs 对照组 对比结果 =====")
    print(f"传统线性教学稳定成功率：{improvement['ctrl_success']:.2%}")
    print(f"智能教学方案稳定成功率：{improvement['exp_success']:.2%}")
    print(f"较传统线性教学提升：{improvement['success_improvement']:.1f}%")
    print(f"\n传统线性教学平均步数：{improvement['ctrl_steps']:.1f}步")
    print(f"智能教学方案平均步数：{improvement['exp_steps']:.1f}步")
    print(f"学习步长缩短：{improvement['steps_reduction']:.1f}%")

    # ========== 7. 保存详细报告 ==========
    with open('teaching_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("===== 智能教学 vs 传统线性教学 对比报告 =====\n\n")
        f.write("1. 成功率对比\n")
        f.write(f"   传统线性教学稳定成功率：{improvement['ctrl_success']:.2%}\n")
        f.write(f"   智能教学方案稳定成功率：{improvement['exp_success']:.2%}\n")
        f.write(f"   相对提升比例：{improvement['success_improvement']:.1f}%\n\n")
        f.write("2. 平均步数对比\n")
        f.write(f"   传统线性教学平均步数：{improvement['ctrl_steps']:.1f}步\n")
        f.write(f"   智能教学方案平均步数：{improvement['exp_steps']:.1f}步\n")
        f.write(f"   步长缩短比例：{improvement['steps_reduction']:.1f}%\n\n")
        f.write("3. 难点节点分析\n")
        for node, diff in analysis['difficulty_analysis']['most_difficult'].items():
            f.write(f"   节点 {node}：最终难度值 {diff:.2f}\n")
        f.write("\n4. 常见学习障碍\n")
        for node, count in analysis['path_analysis']['common_failures'].items():
            f.write(f"   节点 {node}：累计失败 {count} 次\n")

    # 保存详细日志
    with open('teaching_logs.txt', 'w', encoding='utf-8') as f:
        f.write("===== 智能教学实验组日志 =====\n")
        for idx, log in enumerate(exp_simulator.teaching_logs):
            f.write(f"教学周期 {idx + 1}:\n")
            f.write(f"  成功率: {log['success_rate'][-1]:.2%}\n")
            f.write(f"  平均步数: {log['avg_steps'][-1]:.1f}\n")
            f.write(f"  路径效率: {log['path_efficiency'][-1]:.2f}\n\n")
        
        f.write("===== 传统线性教学对照组日志 =====\n")
        for idx, log in enumerate(ctrl_simulator.teaching_logs):
            f.write(f"教学周期 {idx + 1}:\n")
            f.write(f"  成功率: {log['success_rate'][-1]:.2%}\n")
            f.write(f"  平均步数: {log['avg_steps'][-1]:.1f}\n")
            f.write(f"  路径效率: {log['path_efficiency'][-1]:.2f}\n\n")

    print("\n===== 实验完成 =====")
    print("1. 对比可视化图已保存：teaching_comparison.png")
    print("2. 详细对比报告已保存：teaching_comparison_report.txt")
    print("3. 完整日志已保存：teaching_logs.txt")