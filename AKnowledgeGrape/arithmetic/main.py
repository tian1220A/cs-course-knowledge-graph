import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from tqdm import tqdm
from matplotlib.patches import Patch

# ===================== Global Style =====================
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# 优化配色方案：使用专业的Matplotlib colormap，区分度更高
COLOR_MAP = cm.get_cmap('viridis', 12)  # 专业渐变色彩
COLOR_PALETTE = {
    "Basic_Sequential": "#E67E22",
    "Medium_Sequential": "#9B59B6",
    "Advanced_Sequential": "#27AE60",
    "Basic_Astar": "#1E88E5",
    "Medium_Astar": "#00BFFF",
    "Advanced_Astar": "#00C853"
}

# ===================== Knowledge Graph =====================
class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        for i in range(10):
            self.G.add_node(f"KP_{i}", name=f"Chapter {i}")
        for i in range(9):
            self.G.add_edge(f"KP_{i}", f"KP_{i+1}", weight=1.0)

    def get_sequential_path(self, start, end):
        nodes = sorted(self.G.nodes())
        try:
            s_idx = nodes.index(start)
            e_idx = nodes.index(end)
            return nodes[s_idx:e_idx+1] if s_idx <= e_idx else [start, end]
        except:
            return [start, end]

    def get_optimal_path(self, start, end):
        try:
            return nx.shortest_path(self.G, start, end)
        except:
            return [start, end]

# ===================== Algorithm Class =====================
class EnhancedCollaborativeLearning12:
    def __init__(self, neo4j_uri, user, password, alpha, feedback_strength, lambda_bayes, beta_smooth, core_kp_weight, smoothness_weight):
        self.knowledge_graph = KnowledgeGraph()
        self.algorithm_improvement = 0.0

    def a_star_search(self, start, goal):
        path = self.knowledge_graph.get_optimal_path(start, goal)
        return path, len(path)

    def collaborative_update(self, rate):
        improvement = rate * random.uniform(0.2, 0.25)
        self.algorithm_improvement = min(self.algorithm_improvement + improvement, 0.25)

# ===================== Experiment Config =====================
NUM_TRAIN_CYCLES = 3
STUDENTS_PER_GROUP = 80
MAX_STEPS = 20

ABILITY_GROUPS = {
    "Basic": {"mu": 0.60, "sigma": 0.03, "lower": 0.57, "upper": 0.63},
    "Medium": {"mu": 0.75, "sigma": 0.03, "lower": 0.72, "upper": 0.78},
    "Advanced": {"mu": 0.88, "sigma": 0.02, "lower": 0.86, "upper": 0.90}
}

# Success Rate Model
def dynamic_success_model(ability, is_path_learning=False):
    if ability < 0.70:
        return 0.72 if is_path_learning else 0.58
    elif ability < 0.85:
        return 0.92 if is_path_learning else 0.68
    else:
        return 0.95 if is_path_learning else 0.78

# ===================== Virtual Student =====================
class VirtualStudent:
    def __init__(self, student_id, group_name, start_node, goal_node, ability):
        self.id = student_id
        self.group_name = group_name
        self.start_node = start_node
        self.goal_node = goal_node
        self.initial_ability = ability
        self.current_ability = ability
        self.reset()

    def reset(self):
        self.current_node = self.start_node
        self.mastered = set()
        self.history = []

    def learn_step(self, target_node, is_path_learning=False):
        prob = dynamic_success_model(self.current_ability, is_path_learning)
        success = random.random() < prob
        if success:
            self.current_node = target_node
            self.mastered.add(target_node)
        return success

# ===================== Learning Strategies =====================
class LearningStrategy:
    @staticmethod
    def path_learning(student, algo_system, max_steps, cycle):
        path, _ = algo_system.a_star_search(student.start_node, student.goal_node)
        success_count = 0
        total_steps = min(len(path)-1, max_steps)

        for i in range(total_steps):
            next_node = path[i+1]
            if student.learn_step(next_node, is_path_learning=True):
                success_count += 1

        return {
            "Learning Method": "Cognition-Oriented A*",
            "Success Rate": success_count / total_steps if total_steps > 0 else 0,
            "Ability Group": student.group_name,
            "Cycle": cycle,
            "Start": student.start_node,
            "Goal": student.goal_node
        }

    @staticmethod
    def sequential_learning(student, kg, max_steps, cycle):
        path = kg.get_sequential_path(student.start_node, student.goal_node)
        success_count = 0
        total_steps = min(len(path)-1, max_steps)

        for i in range(total_steps):
            next_node = path[i+1]
            if student.learn_step(next_node, is_path_learning=False):
                success_count += 1

        return {
            "Learning Method": "Sequential Learning",
            "Success Rate": success_count / total_steps if total_steps > 0 else 0,
            "Ability Group": student.group_name,
            "Cycle": cycle,
            "Start": student.start_node,
            "Goal": student.goal_node
        }

# ===================== Main Simulation =====================
class TeachingSimulation:
    def __init__(self, neo_config):
        self.algo_system = EnhancedCollaborativeLearning12(**neo_config)
        self.kg = self.algo_system.knowledge_graph
        self.results = []

    def generate_students(self):
        students = []
        sid = 1
        fixed_pairs = [("KP_0", "KP_9"), ("KP_1", "KP_8"), ("KP_2", "KP_7")]

        for group, cfg in ABILITY_GROUPS.items():
            abilities = np.clip(np.random.normal(cfg["mu"], cfg["sigma"], STUDENTS_PER_GROUP), cfg["lower"], cfg["upper"])
            for ab in abilities:
                s, g = random.choice(fixed_pairs)
                students.append(VirtualStudent(sid, group, s, g, ab))
                sid +=1
        return students

    def run(self):
        students = self.generate_students()
        for cycle in range(1, NUM_TRAIN_CYCLES+1):
            self.algo_system.collaborative_update(0.15)
            for stu in tqdm(students, desc=f"Cycle {cycle}"):
                res1 = LearningStrategy.path_learning(stu, self.algo_system, MAX_STEPS, cycle)
                stu.reset()
                res2 = LearningStrategy.sequential_learning(stu, self.kg, MAX_STEPS, cycle)
                stu.reset()
                self.results.extend([res1, res2])
        self.analyze()

    def analyze(self):
        df = pd.DataFrame(self.results)
        self.df_detail = df
        summary = df.groupby(["Ability Group", "Learning Method"], as_index=False)["Success Rate"].mean()
        summary = summary.sort_values(by=["Ability Group", "Learning Method"])
        self.df_summary = summary
        print("📊 Experiment Results (Mean Success Rate):")
        print(summary.round(3))

    # ===================== 3D Plot (Optimized) =====================
    def plot_3d_bar_optimized(self):
        df = self.df_summary
        groups = ["Basic", "Medium", "Advanced"]
        methods = ["Sequential Learning", "Cognition-Oriented A*"]
        
        # 数据准备
        x_labels = list(df["Ability Group"].unique())
        y_labels = list(df["Learning Method"].unique())
        x_indices = {g: i for i, g in enumerate(groups)}
        y_indices = {m: i for i, m in enumerate(methods)}
        
        # 柱子尺寸
        dx = dy = 0.3
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 自定义颜色
        colors = []
        for _, row in df.iterrows():
            g = row["Ability Group"]
            m = row["Learning Method"]
            if m == "Sequential Learning":
                colors.append(COLOR_PALETTE[f"{g}_Sequential"])
            else:
                colors.append(COLOR_PALETTE[f"{g}_Astar"])

        # 绘制3D柱状图
        bars = ax.bar3d(
            [x_indices[g] for g in df["Ability Group"]],
            [y_indices[m] for m in df["Learning Method"]],
            np.zeros(len(df)),
            dx, dy, df["Success Rate"],
            color=colors, alpha=0.9, edgecolor='white', linewidth=0.8
        )

        # 优化数值标注（避免重叠）
        for i, (_, row) in enumerate(df.iterrows()):
            z_pos = row["Success Rate"] + 0.01
            ax.text(
                x_indices[row["Ability Group"]] + dx/2,
                y_indices[row["Learning Method"]] + dy/2,
                z_pos,
                f"{z_pos-0.01:.2f}",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=1)
            )

        # 坐标轴优化 —— 标签远离坐标轴，不重叠
        ax.set_xlabel("Ability Group", fontsize=14, labelpad=25, fontweight='bold')
        ax.set_ylabel("Learning Method", fontsize=14, labelpad=30, fontweight='bold')
        ax.set_zlabel("Success Rate", fontsize=14, labelpad=20, fontweight='bold')
        
        ax.set_xticks([i + dx/2 for i in range(len(groups))])
        ax.set_xticklabels(groups, fontsize=12)
        ax.set_yticks([i + dy/2 for i in range(len(methods))])
        ax.set_yticklabels(["Sequential", "Cog.-Oriented A*"], fontsize=12)
        ax.set_zlim(0, 1.0)

        # 标题
        ax.set_title(
            "Learning Success Rate: Sequential vs Cognition-Oriented A*",
            fontsize=16, pad=25, fontweight='bold'
        )

        # 图例放在右上角
        legend_elements = [
            Patch(facecolor=COLOR_PALETTE[f"{g}_Sequential"], label=f"{g} - Sequential") for g in groups
        ] + [
            Patch(facecolor=COLOR_PALETTE[f"{g}_Astar"], label=f"{g} - Cognition-Oriented A*") for g in groups
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

        # 视角调整
        ax.view_init(elev=25, azim=60)
        plt.tight_layout()
        
        # 保存高清图片
        plt.savefig(
            "optimized_3d_bar.png",
            dpi=600, bbox_inches='tight', facecolor='white'
        )
        plt.show()

    def save(self):
        with pd.ExcelWriter("optimized_results.xlsx") as w:
            self.df_detail.to_excel(w, "Detail", index=False)
            self.df_summary.to_excel(w, "Summary", index=False)

# ===================== Run =====================
if __name__ == "__main__":
    config = {
        "neo4j_uri": "bolt://localhost:7687", "user": "neo4j", "password": "123456",
        "alpha":0.8,"feedback_strength":0.5,"lambda_bayes":10.0,"beta_smooth":(2,2),
        "core_kp_weight":1.5,"smoothness_weight":1.8
    }
    sim = TeachingSimulation(config)
    sim.run()
    sim.plot_3d_bar_optimized()
    sim.save()
    print("✅ Experiment finished successfully!")