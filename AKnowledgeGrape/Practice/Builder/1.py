import pandas as pd
from py2neo import Graph, Node, Relationship, Transaction
import logging
import os
import chardet

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KnowledgeGraphImporter")


class Neo4jKnowledgeImporter:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, extend_data_dir):
        self.graph = self._connect_neo4j(neo4j_uri, neo4j_user, neo4j_password)
        self.extend_data_dir = extend_data_dir
        self.cache = {
            "courses": {},               # 课程缓存（含跨学科课程）
            "knowledge_points": {},      # 知识点缓存（含跨学科知识点）
            "sub_knowledge_points": {},  # 子知识点缓存（含跨学科子知识点）
            "original_kps": {},          # 原学科知识点ID映射
            "original_sub_kps": {}       # 原学科子知识点ID映射
        }
        self.original_domain = "big_data_technology"  # 原学科标识

    def _connect_neo4j(self, uri, user, password):
        try:
            graph = Graph(uri, auth=(user, password))
            graph.run("RETURN 1")
            logger.info("Neo4j 连接成功")
            return graph
        except Exception as e:
            logger.critical(f"Neo4j 连接失败: {str(e)}")
            raise

    def _detect_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(1024 * 1024))
        return result["encoding"] or "utf-8-sig"

    def _load_csv(self, file_path):
        encoding = self._detect_encoding(file_path)
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            na_values=["", "nan", "None"],
            keep_default_na=True
        )
        # 清理字符串字段
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": "", "None": ""})
        # 处理数值字段
        numeric_cols = ["bloom_level", "class_hours", "syllabus_mentions", "relevance", "cross_correlation", "link_weight"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        logger.info(f"加载 {os.path.basename(file_path)}: {len(df)} 条数据")
        return df

    def _preload_original_nodes(self):
        logger.info("预加载原学科节点到缓存")
        # 预加载原学科知识点
        original_kps = self.graph.run(f"""
            MATCH (n:KnowledgePoint)
            WHERE n.domain = '{self.original_domain}'
            RETURN n.kp_id AS kp_id, id(n) AS node_id
        """).data()
        for item in original_kps:
            self.cache["original_kps"][item["kp_id"]] = item["node_id"]
        logger.info(f"原学科知识点缓存完成: {len(self.cache['original_kps'])} 个")

        # 预加载原学科子知识点
        original_sub_kps = self.graph.run(f"""
            MATCH (n:SubKnowledgePoint)
            WHERE n.domain = '{self.original_domain}'
            RETURN n.sub_kp_id AS sub_kp_id, id(n) AS node_id
        """).data()
        for item in original_sub_kps:
            self.cache["original_sub_kps"][item["sub_kp_id"]] = item["node_id"]
        logger.info(f"原学科子知识点缓存完成: {len(self.cache['original_sub_kps'])} 个")

    # ------------------------------
    # 1. 导入跨学科课程（Course标签）
    # ------------------------------
    def import_extend_courses(self, file_path):
        logger.info("导入跨学科课程（Course标签）")
        df = self._load_csv(file_path)
        tx = self.graph.begin()

        for _, row in df.iterrows():
            # 跨学科课程ID：EXT_+cross_domain_id（避免与原课程冲突）
            course_id = f"EXT_{row['cross_domain_id']}"
            # 从CSV的domain_name列获取课程名称（此处CSV确有该列，直接读取）
            course_name = row["domain_name"]
            # 创建跨学科课程节点
            course_node = Node(
                "Course",
                course_id=course_id,
                id=course_id,
                name=course_name,  # 课程名称=CSV的domain_name
                description=row["description"],
                relevance=row["relevance"],
                domain=course_name,  # domain=课程名称（区分原学科）
                credit=2,
                semester="2024春季"
            )
            tx.create(course_node)
            self.cache["courses"][course_id] = course_node
            logger.debug(f"创建跨学科课程: {course_id} - {course_name}")

        tx.commit()
        logger.info(f"跨学科课程导入完成: {len(self.cache['courses'])} 个")

    # ------------------------------
    # 2. 导入跨学科知识点（KnowledgePoint标签）
    # ------------------------------
    def import_extend_knowledge_points(self, file_path):
        logger.info("导入跨学科知识点（KnowledgePoint标签）")
        if not self.cache["courses"]:
            raise ValueError("请先导入跨学科课程（调用import_extend_courses）")
        
        df = self._load_csv(file_path)
        tx = self.graph.begin()

        for _, row in df.iterrows():
            # 跨学科知识点ID：EXT_+cross_kp_id
            kp_id = f"EXT_{row['cross_kp_id']}"
            # 关联跨学科课程：课程ID=EXT_+CSV的cross_domain_id
            course_id = f"EXT_{row['cross_domain_id']}"
            if course_id not in self.cache["courses"]:
                logger.warning(f"跨学科课程 {course_id} 不存在，跳过知识点 {kp_id}")
                continue

            # 关键修复：从课程节点的name属性获取领域名称（而非读CSV的domain_name）
            domain_name = self.cache["courses"][course_id]["name"]
            # 创建跨学科知识点节点
            kp_node = Node(
                "KnowledgePoint",
                kp_id=kp_id,
                id=kp_id,
                title=row["title"],
                description=row["description"],
                bloom_level=int(row["bloom_level"]),
                cross_correlation=row["cross_correlation"],
                class_hours=row["class_hours"],
                domain=domain_name,  # 继承课程的领域名称
                chapter_id=f"EXT_CH_{row['cross_domain_id']}"  # 虚拟章节ID
            )
            tx.create(kp_node)
            self.cache["knowledge_points"][kp_id] = kp_node

            # 创建虚拟章节（过渡课程→知识点的关联）
            virtual_chapter_id = f"EXT_CH_{row['cross_domain_id']}"
            virtual_chapter = Node(
                "Chapter",
                chapter_id=virtual_chapter_id,
                id=virtual_chapter_id,
                title=f"{domain_name} - 基础章节",  # 用领域名称命名章节
                order=1,
                class_hours=0,
                domain=domain_name
            )
            tx.create(virtual_chapter)
            # 课程→虚拟章节（HAS_CHAPTER）
            tx.create(Relationship(self.cache["courses"][course_id], "HAS_CHAPTER", virtual_chapter))
            # 虚拟章节→知识点（HAS_KNOWLEDGE）
            tx.create(Relationship(virtual_chapter, "HAS_KNOWLEDGE", kp_node))

            logger.debug(f"创建跨学科知识点: {kp_id} - {row['title']}（关联课程: {course_id}）")

        tx.commit()
        logger.info(f"跨学科知识点导入完成: {len(self.cache['knowledge_points'])} 个")

    # ------------------------------
    # 3. 导入跨学科子知识点（SubKnowledgePoint标签）
    # ------------------------------
    def import_extend_sub_knowledge_points(self, file_path):
        logger.info("导入跨学科子知识点（SubKnowledgePoint标签）")
        if not self.cache["knowledge_points"]:
            raise ValueError("请先导入跨学科知识点（调用import_extend_knowledge_points）")
        
        df = self._load_csv(file_path)
        tx = self.graph.begin()

        for _, row in df.iterrows():
            # 跨学科子知识点ID：EXT_+cross_sub_kp_id
            sub_kp_id = f"EXT_{row['cross_sub_kp_id']}"
            # 关联跨学科知识点：知识点ID=EXT_+CSV的cross_kp_id
            parent_kp_id = f"EXT_{row['cross_kp_id']}"
            if parent_kp_id not in self.cache["knowledge_points"]:
                logger.warning(f"跨学科知识点 {parent_kp_id} 不存在，跳过子知识点 {sub_kp_id}")
                continue

            # 关键修复：从父知识点获取领域名称
            domain_name = self.cache["knowledge_points"][parent_kp_id]["domain"]
            # 创建跨学科子知识点节点
            sub_kp_node = Node(
                "SubKnowledgePoint",
                sub_kp_id=sub_kp_id,
                id=sub_kp_id,
                title=row["title"],
                description=row["description"],
                bloom_level=int(row["bloom_level"]),
                cross_correlation=row["cross_correlation"],
                syllabus_mentions=int(row["syllabus_mentions"]),
                domain=domain_name,  # 继承父知识点的领域名称
                chapter_id=self.cache["knowledge_points"][parent_kp_id]["chapter_id"],  # 继承虚拟章节ID
                class_hours=0
            )
            tx.create(sub_kp_node)
            self.cache["sub_knowledge_points"][sub_kp_id] = sub_kp_node

            # 子知识点→父知识点（CHILD_OF）
            parent_kp_node = self.cache["knowledge_points"][parent_kp_id]
            tx.create(Relationship(sub_kp_node, "CHILD_OF", parent_kp_node))
            tx.create(Relationship(parent_kp_node, "PARENT_OF", sub_kp_node))

            # 处理子知识点依赖
            dependency_id = row["dependency_cross_sub_kp"]
            if dependency_id:
                extend_dependency_id = f"EXT_{dependency_id}"
                if extend_dependency_id in self.cache["sub_knowledge_points"]:
                    tx.create(Relationship(
                        sub_kp_node,
                        "REQUIRES_PREREQUISITE",
                        self.cache["sub_knowledge_points"][extend_dependency_id]
                    ))
                    logger.debug(f"建立依赖关系: {sub_kp_id} → {extend_dependency_id}")

            logger.debug(f"创建跨学科子知识点: {sub_kp_id} - {row['title']}（关联知识点: {parent_kp_id}）")

        tx.commit()
        logger.info(f"跨学科子知识点导入完成: {len(self.cache['sub_knowledge_points'])} 个")

    # ------------------------------
    # 4. 建立跨学科与原学科的关联
    # ------------------------------
    def import_extend_original_links(self, file_path):
        logger.info("建立跨学科与原学科的关联（RELATED_TO关系）")
        if not self.cache["sub_knowledge_points"]:
            raise ValueError("请先导入跨学科子知识点（调用import_extend_sub_knowledge_points）")
        self._preload_original_nodes()
        
        df = self._load_csv(file_path)
        tx = self.graph.begin()
        link_count = 0

        for _, row in df.iterrows():
            # 跨学科子知识点ID：EXT_+CSV的cross_node_id
            cross_sub_kp_id = f"EXT_{row['cross_node_id']}"
            if cross_sub_kp_id not in self.cache["sub_knowledge_points"]:
                logger.warning(f"跨学科子知识点 {cross_sub_kp_id} 不存在，跳过关联")
                continue
            cross_node = self.cache["sub_knowledge_points"][cross_sub_kp_id]

            # 获取原学科节点
            original_node_id = row["original_node_id"]
            original_node_type = row["original_node_type"]
            original_node = None

            if original_node_type == "KnowledgePoint" and original_node_id in self.cache["original_kps"]:
                original_node = self.graph.nodes[self.cache["original_kps"][original_node_id]]
            elif original_node_type == "SubKnowledgePoint" and original_node_id in self.cache["original_sub_kps"]:
                original_node = self.graph.nodes[self.cache["original_sub_kps"][original_node_id]]
            
            if not original_node:
                logger.warning(f"原学科节点 {original_node_id}（{original_node_type}）不存在，跳过关联")
                continue

            # 建立双向关联（RELATED_TO）
            link_weight = row["link_weight"]
            tx.create(Relationship(
                original_node,
                "RELATED_TO",
                cross_node,
                weight=link_weight,
                scenario=row["apply_scenario"],
                link_type="extend_original"
            ))
            tx.create(Relationship(
                cross_node,
                "RELATED_TO",
                original_node,
                weight=link_weight * 0.7,
                scenario=row["apply_scenario"],
                link_type="original_extend"
            ))

            link_count += 1
            logger.debug(f"建立关联: {original_node['title']} ↔ {cross_node['title']}（权重: {link_weight}）")

        tx.commit()
        logger.info(f"跨学科与原学科关联完成: {link_count} 条")

    # ------------------------------
    # 验证导入结果
    # ------------------------------
    def verify_import(self):
        logger.info("\n=== 导入结果验证（统一标签视角）===")
        # 1. 课程统计
        total_courses = self.graph.run("MATCH (n:Course) RETURN count(n) AS cnt").data()[0]["cnt"]
        original_courses = self.graph.run(f"MATCH (n:Course) WHERE n.domain = '{self.original_domain}' RETURN count(n) AS cnt").data()[0]["cnt"]
        extend_courses = total_courses - original_courses
        logger.info(f"课程（Course）: 总计 {total_courses} 个（原学科 {original_courses} 个，跨学科 {extend_courses} 个）")

        # 2. 知识点统计
        total_kps = self.graph.run("MATCH (n:KnowledgePoint) RETURN count(n) AS cnt").data()[0]["cnt"]
        original_kps = self.graph.run(f"MATCH (n:KnowledgePoint) WHERE n.domain = '{self.original_domain}' RETURN count(n) AS cnt").data()[0]["cnt"]
        extend_kps = total_kps - original_kps
        logger.info(f"知识点（KnowledgePoint）: 总计 {total_kps} 个（原学科 {original_kps} 个，跨学科 {extend_kps} 个）")

        # 3. 子知识点统计
        total_sub_kps = self.graph.run("MATCH (n:SubKnowledgePoint) RETURN count(n) AS cnt").data()[0]["cnt"]
        original_sub_kps = self.graph.run(f"MATCH (n:SubKnowledgePoint) WHERE n.domain = '{self.original_domain}' RETURN count(n) AS cnt").data()[0]["cnt"]
        extend_sub_kps = total_sub_kps - original_sub_kps
        logger.info(f"子知识点（SubKnowledgePoint）: 总计 {total_sub_kps} 个（原学科 {original_sub_kps} 个，跨学科 {extend_sub_kps} 个）")

        # 4. 关联统计
        extend_links = self.graph.run("""
            MATCH ()-[r:RELATED_TO {link_type: 'extend_original'}]->()
            RETURN count(r) AS cnt
        """).data()[0]["cnt"]
        logger.info(f"跨学科-原学科关联（RELATED_TO）: {extend_links} 条")
        logger.info("=== 验证完成 ===\n")


def main():
    # 配置参数（根据实际环境修改）
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "123456789"  # 替换为你的Neo4j密码
    EXTEND_DATA_DIR = "./cross_domain_data"  # 跨学科数据目录

    try:
        importer = Neo4jKnowledgeImporter(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            extend_data_dir=EXTEND_DATA_DIR
        )

        # 按顺序导入
        importer.import_extend_courses(os.path.join(EXTEND_DATA_DIR, "cross_domains.csv"))
        importer.import_extend_knowledge_points(os.path.join(EXTEND_DATA_DIR, "cross_knowledge_points.csv"))
        importer.import_extend_sub_knowledge_points(os.path.join(EXTEND_DATA_DIR, "cross_sub_knowledge_points.csv"))
        importer.import_extend_original_links(os.path.join(EXTEND_DATA_DIR, "cross_original_links.csv"))

        # 验证结果
        importer.verify_import()

        logger.info("=== 所有扩展数据导入完成 ===")
        logger.info(f"提示：通过 domain 属性区分节点类型（{importer.original_domain} = 原学科，其他为跨学科）")

    except Exception as e:
        logger.critical(f"导入任务失败: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()