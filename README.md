<div align="center">

# 📚 cs-course-knowledge-graph

**面向计算机专业大学课程的教育知识图谱框架**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph-4B8BBE?style=for-the-badge)](https://networkx.org/)
[![Py2neo](https://img.shields.io/badge/Py2neo-Driver-5C6BC0?style=for-the-badge)](https://py2neo.org/)
[![License](https://img.shields.io/badge/License-Academic-9cf?style=for-the-badge)](#license)

*多层知识本体建模 · 改进 Edu-link PageRank 权重评估 · 认知自适应学习路径 · 闭环动态更新*

</div>

> 💡 当前数据与案例基于「大数据处理架构 Hadoop」课程构建，**方法论可迁移到其他计算机专业课程**。

---

## 📑 目录

- [✨ 核心思想](#-核心思想)
- [🏗️ 系统架构](#-系统架构)
- [🗂️ 目录结构](#-目录结构)
- [🧩 功能模块](#-功能模块)
- [📦 环境依赖](#-环境依赖)
- [🚀 快速开始](#-快速开始)
- [📊 数据说明](#-数据说明)
- [🎨 可视化结果](#-可视化结果)
- [⚠️ 注意事项](#-注意事项)

---

## ✨ 核心思想

传统课程知识点之间天然存在「先修—后继」依赖关系，但不同知识点对达成教学目标的重要性并不相同。本项目将课程知识体系建模为**有向图**，并引入**教育语义因子**对经典 PageRank 进行改造，从而得到更能反映教学价值的知识点权重；在此基础上，结合学习者认知水平，生成自适应的个性化学习路径，并通过学习反馈实现图谱权重的闭环更新。

| 步骤 | 技术路线 | 说明 |
| :--: | --- | --- |
| 1️⃣ | **多层本体建模** | 章节（Chapter）→ 知识点（KnowledgePoint）→ 子知识点（SubKnowledgePoint），并关联实验（Lab） |
| 2️⃣ | **改进 Edu-link PageRank** | 在标准 PageRank 基础上引入 `class_hours`（课时）、`syllabus_mentions`（大纲提及次数）、`bloom_level`（布鲁姆认知层级）等教育因子 |
| 3️⃣ | **认知自适应学习路径** | 基于增强 Dijkstra / 认知导向 A\* 算法，结合知识点权重与学习者能力生成个性化路径 |
| 4️⃣ | **闭环动态更新** | 采集学习反馈，动态调整知识点权重与学习路径 |

---

## 🏗️ 系统架构

```
                ┌──────────────────────────┐
                │   📁 O-Data（原始 Excel）  │
                └────────────┬─────────────┘
                             │  Conversion.py · 布鲁姆层级映射
                             ▼
                ┌──────────────────────────┐
                │   📁 D-Data（规范 CSV）    │
                └────────────┬─────────────┘
                             │  graph_builder.py / grabuilder.py
                             ▼
                ┌──────────────────────────┐
                │      🗄️ Neo4j 知识图谱     │
                │  Chapter → KP → SubKP    │
                └────────────┬─────────────┘
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌────────────┐    ┌────────────┐    ┌──────────────────┐
   │  PageRank  │    │ Edu-link   │    │  核心知识点验证    │
   │ (标准基线)  │    │  PageRank  │    │ (Spearman 相关)   │
   └─────┬──────┘    └─────┬──────┘    └──────────────────┘
         └──────────┬───────┘
                    ▼
          ┌──────────────────┐
          │  认知自适应学习路径 │ ← 增强 Dijkstra / A* + 学习者能力
          └─────────┬────────┘
                    ▼
          ┌──────────────────┐
          │   闭环动态更新     │ ← 协作学习反馈 → 权重 / 路径调整
          └──────────────────┘
```

---

## 🗂️ 目录结构

```
cs-course-knowledge-graph/
├── AKnowledgeGrape/                      # 主工程
│   ├── D-Data/                           # 规范化数据（CSV）
│   │   ├── chapters.csv                  #   章节
│   │   ├── knowledge_points.csv          #   知识点
│   │   ├── sub_knowledge_points.csv      #   子知识点
│   │   └── labs.csv                      #   实验
│   ├── O-Data/                           # 原始数据（Excel）
│   │   └── *.xlsx                        #   章节 / 知识点 / 实验 / 子知识点
│   ├── Practice/
│   │   ├── Builder/
│   │   │   ├── graph_builder.py          #   EduKGBuilder：单课程知识图谱构建
│   │   │   └── grabuilder.py             #   CrossDomainEduKGBuilder：跨学科图谱构建
│   │   └── Data/
│   │       └── Conversion.py             #   Excel → CSV 转换 + 布鲁姆层级映射
│   ├── arithmetic/                       # 核心算法
│   │   ├── pagerank.py                   #   标准 PageRank（基线）
│   │   ├── edu_pagerank.py               #   改进 Edu-link PageRank
│   │   ├── dijkstra_analyzer.py          #   认知自适应学习路径（增强 Dijkstra）
│   │   ├── collaborative_learning.py     #   协作学习 + 动态权重更新
│   │   ├── core_knowledge_validation.py  #   核心知识点验证（Spearman 相关）
│   │   ├── main.py                       #   教学仿真：顺序学习 vs 认知导向 A*
│   │   ├── data.py                       #   MOOC 课程信息爬虫
│   │   ├── 1path.py / 3path*.py / 4vish.py  # 路径可视化脚本
│   │   └── lib/                          #   复用组件（page / pagerankcom 等）
│   └── confirmation/                     # 验证与闭环
│       ├── dynamic_syllabus_annotation.py#   动态大纲标注（生成 PDF）
│       ├── cooperate.py                  #   协作教学仿真
│       ├── visualization.py              #   学习路径交互式可视化（pyvis）
│       └── *.html / *.pdf / *.png        #   验证产物
├── basic_pagerank_visualizations/        # 标准 PageRank 可视化结果
├── edu_pagerank_visualizations/          # Edu-link PageRank 可视化结果
└── path_visualization_results/           # 学习路径生成可视化结果
```

---

## 🧩 功能模块

### 📂 数据层（`Practice/Data/` 与 `D-Data`、`O-Data`）

- `Conversion.py` 将 `O-Data` 中的 Excel 原始数据转换为 `D-Data` 中的规范 CSV，并将布鲁姆认知层级（记忆 / 理解 / 应用 / 分析 / 评价 / 创造）映射为 1–6 的数值。
- `data.py` 提供 MOOC 课程信息爬虫，用于自动采集课程实体与关系。

### 🏗️ 知识图谱构建（`Practice/Builder/`）

- `graph_builder.py`（`EduKGBuilder`）：将 CSV 数据批量写入 Neo4j，构建「章节 → 知识点 → 子知识点」的多层本体结构及依赖关系。
- `grabuilder.py`（`CrossDomainEduKGBuilder`）：扩展支持跨学科知识图谱的构建。

### ⚖️ 权重评估算法（`arithmetic/`）

- `pagerank.py`（`PageRankCalculator`）：标准 PageRank，作为权重评估的**基线**。
- `edu_pagerank.py`（`EduPageRankCalculator`）：**改进 Edu-link PageRank**，在转移概率中融入课时、大纲提及次数与布鲁姆层级等教育语义因子，得到更能反映教学价值的知识点权重。
- `core_knowledge_validation.py`：通过 Spearman 秩相关等方法，验证改进算法与传统 PageRank 在核心知识点识别上的一致性。

### 🗺️ 学习路径生成（`arithmetic/`）

- `dijkstra_analyzer.py`（`EnhancedDijkstraAnalyzer`）：基于增强 Dijkstra 算法，结合知识点权重与覆盖率，生成目标导向的认知自适应学习路径。
- `main.py`：构建虚拟学生模型，对比「顺序学习」与「认知导向 A\* 路径学习」在不同能力分组下的成功率。
- `1path.py` / `3path.py` / `3pathd.py` / `3pathEng.py`：单路径 / 多路径可视化对比脚本。

### 🔄 闭环动态更新（`arithmetic/` 与 `confirmation/`）

- `collaborative_learning.py`（`EnhancedCollaborativeLearning`）：采集学习反馈，动态调整知识点权重，实现闭环更新。
- `cooperate.py`：协作式教学仿真，验证闭环更新对学习效果的影响。
- `dynamic_syllabus_annotation.py`：将权重结果反哺教学大纲，生成带标注的 PDF。
- `visualization.py`：基于 pyvis 的交互式学习路径可视化。

---

## 📦 环境依赖

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| Python | 3.9+ | 运行环境 |
| [Neo4j](https://neo4j.com/) | 3.x / 4.x / 5.x | 图数据库 |
| `py2neo` | — | Neo4j Python 驱动 |
| `networkx` | — | 图算法与结构 |
| `numpy` / `pandas` | — | 数值计算与数据处理 |
| `matplotlib` / `seaborn` | — | 可视化 |
| `scipy` | — | 相关性检验（Spearman） |
| `pyvis` | — | 交互式网络图 |
| `flask` | — | Web 可视化服务 |
| `reportlab` | — | PDF 大纲生成 |
| `tqdm` | — | 进度条 |
| `requests` / `lxml` | — | MOOC 数据爬取 |

一键安装：

```bash
pip install py2neo networkx numpy pandas matplotlib seaborn scipy pyvis flask reportlab tqdm requests lxml
```

---

## 🚀 快速开始

1. **准备数据**（可选，`D-Data` 已提供现成 CSV）：

   ```bash
   python AKnowledgeGrape/Practice/Data/Conversion.py
   ```

2. **启动 Neo4j**，确保 `bolt://localhost:7687` 可访问，用户名为 `neo4j`。

3. **构建知识图谱**：

   ```bash
   python AKnowledgeGrape/Practice/Builder/graph_builder.py
   ```

4. **计算知识点权重**：

   ```bash
   # 标准 PageRank（基线）
   python AKnowledgeGrape/arithmetic/pagerank.py

   # 改进 Edu-link PageRank
   python AKnowledgeGrape/arithmetic/edu_pagerank.py
   ```

5. **生成认知自适应学习路径**：

   ```bash
   python AKnowledgeGrape/arithmetic/dijkstra_analyzer.py
   ```

6. **验证与闭环更新**：

   ```bash
   python AKnowledgeGrape/arithmetic/core_knowledge_validation.py
   python AKnowledgeGrape/confirmation/cooperate.py
   ```

> ⚠️ 运行脚本时，请将工作目录设为脚本所在目录（部分脚本依赖相对导入，如 `core_knowledge_validation.py` 导入 `pagerank` / `edu_pagerank`）。

---

## 📊 数据说明

| 文件 | 关键字段 |
| --- | --- |
| `chapters.csv` | `chapter_id`、`title`、`order`、`class_hours`、`parent_id` |
| `knowledge_points.csv` | `kp_id`、`chapter_id`、`description`、`bloom_level`、`class_hours`、`syllabus_mentions` |
| `sub_knowledge_points.csv` | `sub_kp_id`、`kp_id`、`title`、`bloom_level`、`dependency_sub_kp`、`class_hours`、`syllabus_mentions` |
| `labs.csv` | `lab_id`、`lab_name`、`chapter_id`、`difficulty`、`hours`、`related_kp` |

其中 `bloom_level` 对应布鲁姆认知层级：

| 数值 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 层级 | 记忆 | 理解 | 应用 | 分析 | 评价 | 创造 |

---

## 🎨 可视化结果

仓库中随附三类可视化结果，便于快速了解方法效果：

| 目录 | 内容 |
| --- | --- |
| `basic_pagerank_visualizations/` | 标准 PageRank 的知识点权重分布、入度相关性等 |
| `edu_pagerank_visualizations/` | 改进 Edu-link PageRank 的三维权重分布、核心 vs 非核心知识点对比等 |
| `path_visualization_results/` | 认知自适应学习路径（单路径 / 多路径对比）、雷达图等 |

---

## ⚠️ 注意事项

- **Neo4j 连接信息**：当前部分脚本中硬编码了本地连接信息（`bolt://localhost:7687`，用户 `neo4j`，默认密码）。建议改用环境变量读取：

  ```python
  import os
  NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
  NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
  NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
  ```

- **中文字体**：绘图脚本依赖 `SimHei` / `Microsoft YaHei` 等中文字体，Linux 环境下需手动安装对应字体，否则图表中文可能显示为方框。
- **Python 虚拟环境**：本仓库未包含 `.venv`（虚拟环境）与 `.idea`（IDE 配置），开发时请自行 `python -m venv .venv` 创建。

---

## 📄 License

本项目仅用于学术研究与教育用途。
