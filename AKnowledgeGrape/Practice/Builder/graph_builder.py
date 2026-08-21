import pandas as pd
from py2neo import Graph, Node, Relationship, Transaction
from typing import Dict
import logging
import warnings
import os

# 禁用 Py2neo 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EduKGBuilder")


class EduKGBuilder:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._validate_connection()

        # 初始化缓存
        self.chapter_cache: Dict[str, Node] = {}
        self.kp_cache: Dict[str, Node] = {}
        self.sub_kp_cache: Dict[str, Node] = {}

    def _validate_connection(self):
        """验证 Neo4j 连接"""
        try:
            self.graph.run("RETURN 1")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.critical(f"Neo4j连接失败: {str(e)}")
            raise

    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        import chardet
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        logger.info(f"文件 {file_path} 编码检测结果: {encoding} (可信度: {result['confidence']})")
        return encoding or 'utf-8'

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """数据加载与预处理"""
        # 使用绝对路径，兼容不同操作系统
        base_path = os.path.join("D:\\", "LunWen (2)", "LunWen", "AKnowledgeGrape", "D-Data")
        required_files = {
            'chapters': (os.path.join(base_path, 'chapters.csv'), 
                        ['chapter_id', 'title', 'order', 'class_hours']),
            'knowledge_points': (os.path.join(base_path, 'knowledge_points.csv'), 
                                ['kp_id', 'chapter_id', 'description', 'bloom_level', 'class_hours', 'syllabus_mentions']),
            'sub_knowledge_points': (os.path.join(base_path, 'sub_knowledge_points.csv'),  
                                    ['sub_kp_id', 'kp_id', 'title', 'description', 'bloom_level', 'dependency_sub_kp', 'class_hours', 'syllabus_mentions', 'chapter_id']),
            'labs': (os.path.join(base_path, 'labs.csv'), 
                    ['lab_id', 'lab_name', 'related_kp'])
        }

        data = {}
        for name, (path, cols) in required_files.items():
            try:
                # 检查文件是否存在
                if not os.path.exists(path):
                    raise FileNotFoundError(f"文件不存在: {path}")
                
                # 尝试多种编码读取
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        logger.info(f"使用 {encoding} 编码成功读取 {name}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                # 如果都失败，尝试自动检测编码
                if df is None:
                    try:
                        detected_encoding = self._detect_encoding(path)
                        df = pd.read_csv(path, encoding=detected_encoding)
                        logger.info(f"使用检测到的编码 {detected_encoding} 成功读取 {name}")
                    except Exception as e:
                        raise ValueError(f"无法读取文件 {path}，所有编码尝试失败") from e

                # 验证必要列
                missing_cols = set(cols) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"{name} CSV 缺少必要列: {missing_cols}")
                
                # 数据预处理
                if name in ['knowledge_points', 'sub_knowledge_points']:
                    # 安全处理 bloom_level
                    df['bloom_level'] = df['bloom_level'].astype(str).str.strip()
                    df['bloom_level'] = pd.to_numeric(df['bloom_level'], errors='coerce').fillna(1).astype(int)
                
                # 清理字符串数据
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'None': ''})
                
                # 处理数值列
                numeric_cols = ['class_hours', 'order', 'syllabus_mentions']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                data[name] = df
                logger.info(f"成功加载 {name}: {len(df)} 条记录")
                
            except Exception as e:
                logger.error(f"加载 {name} 失败: {str(e)}")
                raise
        return data

    def build(self):
        tx = None
        try:
            self._clear_existing_data()
            data = self.load_data()
            tx = self.graph.begin()

            self._build_chapters(tx, data['chapters'])
            self._build_course(tx)
            self._build_knowledge_points(tx, data['knowledge_points'])
            self._build_sub_knowledge_points(tx, data['sub_knowledge_points'])
            self._build_labs(tx, data['labs'])

            # 增强章节内知识点与子知识点关联
            self._enhance_chapter_associations(tx)
            # 构建用于学习路径搜索的 NEXT 关系
            self._build_path_relationships(tx)

            tx.commit()
            logger.info("全量构建完成")
        except Exception as e:
            if tx:
                tx.rollback()
            logger.critical(f"构建流程异常: {str(e)}")
            raise

    def _clear_existing_data(self):
        """安全清空现有数据"""
        try:
            # 分批删除以避免内存问题
            batch_size = 1000
            while True:
                result = self.graph.run(f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) as cnt")
                deleted = result.data()[0]['cnt']
                if deleted == 0:
                    break
                logger.info(f"已删除 {deleted} 个节点")
            logger.info("已清理历史数据")
        except Exception as e:
            logger.error(f"清理失败: {str(e)}")
            raise

    def _build_chapters(self, tx: Transaction, chapters: pd.DataFrame):
        """构建章节节点"""
        try:
            for _, row in chapters.iterrows():
                chapter_id = str(row['chapter_id']).strip()
                if not chapter_id:
                    logger.warning("发现空 chapter_id，跳过")
                    continue
                
                chapter = Node("Chapter",
                               id=chapter_id,
                               chapter_id=chapter_id,
                               title=str(row['title']).strip(),
                               order=int(row['order']),
                               class_hours=float(row['class_hours']))
                tx.create(chapter)
                self.chapter_cache[chapter_id] = chapter
            
            logger.info(f"章节构建完成: {len(self.chapter_cache)} 个")
        except Exception as e:
            logger.error(f"章节构建失败: {str(e)}")
            raise

    def _build_course(self, tx: Transaction):
        """构建课程节点，并与所有章节建立关联"""
        try:
            course_node = Node("Course",
                               id="COURSE001",
                               course_id="COURSE001",
                               name="大数据技术",
                               credit=3,
                               semester="2024春季")
            tx.create(course_node)
            
            for chapter in self.chapter_cache.values():
                tx.create(Relationship(course_node, "HAS_CHAPTER", chapter))
            
            logger.info("课程节点关联完成")
        except Exception as e:
            logger.error(f"课程构建失败: {str(e)}")
            raise

    def _build_knowledge_points(self, tx: Transaction, kps: pd.DataFrame):
        """构建知识点节点，并建立章节与知识点的关联"""
        try:
            for _, row in kps.iterrows():
                kp_id = str(row['kp_id']).strip()
                chapter_id = str(row['chapter_id']).strip()
                
                if not kp_id or not chapter_id:
                    logger.warning(f"无效知识点数据 (kp_id={kp_id}, chapter_id={chapter_id})，跳过")
                    continue
                
                kp = Node("KnowledgePoint",
                          id=kp_id,
                          kp_id=kp_id,
                          chapter_id=chapter_id,
                          title=str(row['description']).strip(),
                          bloom_level=int(row['bloom_level']),
                          class_hours=float(row['class_hours']),
                          syllabus_mentions=int(row['syllabus_mentions']))
                tx.create(kp)
                self.kp_cache[kp_id] = kp

                chapter = self.chapter_cache.get(chapter_id)
                if chapter:
                    tx.create(Relationship(chapter, "HAS_KNOWLEDGE", kp))
                else:
                    logger.warning(f"章节不存在: chapter_id={chapter_id}，知识点 {kp_id} 未关联章节")
            
            logger.info(f"知识点构建完成: {len(self.kp_cache)} 个")
        except Exception as e:
            logger.error(f"知识点构建失败: {str(e)}")
            raise

    def _build_sub_knowledge_points(self, tx: Transaction, sub_kps: pd.DataFrame):
        """构建子知识点节点，并建立父子关系与依赖关系"""
        try:
            # 创建所有子知识点节点
            for _, row in sub_kps.iterrows():
                sub_kp_id = str(row['sub_kp_id']).strip()
                kp_id = str(row['kp_id']).strip()
                chapter_id = str(row['chapter_id']).strip()
                
                if not sub_kp_id or not kp_id or not chapter_id:
                    logger.warning(f"子知识点 {sub_kp_id} 缺少必要字段，跳过")
                    continue
                
                sub_kp = Node("SubKnowledgePoint",
                              id=sub_kp_id,
                              sub_kp_id=sub_kp_id,
                              chapter_id=chapter_id,
                              title=str(row['title']).strip(),
                              description=str(row['description']).strip(),
                              bloom_level=int(row['bloom_level']),
                              class_hours=float(row['class_hours']),
                              syllabus_mentions=int(row['syllabus_mentions']))
                tx.create(sub_kp)
                self.sub_kp_cache[sub_kp_id] = sub_kp

                # 关联到章节
                chapter = self.chapter_cache.get(chapter_id)
                if chapter:
                    tx.create(Relationship(sub_kp, "BELONGS_TO_CHAPTER", chapter))
                else:
                    logger.warning(f"章节不存在: chapter_id={chapter_id}，子知识点 {sub_kp_id} 未关联章节")

            # 建立父子及依赖关系
            for _, row in sub_kps.iterrows():
                sub_kp_id = str(row['sub_kp_id']).strip()
                kp_id = str(row['kp_id']).strip()
                
                if sub_kp_id not in self.sub_kp_cache:
                    continue
                    
                parent_kp = self.kp_cache.get(kp_id)
                if parent_kp:
                    sub_kp = self.sub_kp_cache[sub_kp_id]
                    tx.create(Relationship(sub_kp, "CHILD_OF", parent_kp))
                    tx.create(Relationship(parent_kp, "PARENT_OF", sub_kp))
                    
                    # 处理依赖关系
                    dependency_str = str(row['dependency_sub_kp']).strip()
                    if dependency_str and dependency_str != 'nan':
                        dependency_ids = [d.strip() for d in dependency_str.split(';') if d.strip()]
                        for dep_id in dependency_ids:
                            if dep_id in self.sub_kp_cache and dep_id != sub_kp_id:
                                tx.create(Relationship(sub_kp, "REQUIRES_PREREQUISITE", self.sub_kp_cache[dep_id]))
                            else:
                                logger.warning(f"依赖的子知识点不存在或自引用: {dep_id} (当前子知识点: {sub_kp_id})")
                    else:
                        tx.create(Relationship(sub_kp, "ASSOCIATED_WITH", parent_kp))
                else:
                    logger.warning(f"父知识点不存在: kp_id={kp_id}，子知识点 {sub_kp_id} 未关联父节点")
            
            logger.info(f"子知识点构建完成: {len(self.sub_kp_cache)} 个")
        except Exception as e:
            logger.error(f"子知识点构建失败: {str(e)}")
            raise

    def _build_labs(self, tx: Transaction, labs: pd.DataFrame):
        """构建实验环节，并建立与知识点的关联"""
        try:
            for _, row in labs.iterrows():
                lab_id = str(row['lab_id']).strip()
                lab_name = str(row['lab_name']).strip()
                
                if not lab_id:
                    logger.warning("发现空 lab_id，跳过")
                    continue
                
                lab = Node("Lab",
                           id=lab_id,
                           lab_id=lab_id,
                           title=lab_name)
                tx.create(lab)
                
                # 处理关联知识点
                related_kps_str = str(row.get('related_kp', '')).strip()
                if related_kps_str and related_kps_str != 'nan':
                    related_kps = [kp.strip() for kp in related_kps_str.split(';') if kp.strip()]
                    for kp_id in related_kps:
                        kp = self.kp_cache.get(kp_id)
                        if kp:
                            tx.create(Relationship(lab, "REQUIRES_KNOWLEDGE", kp))
                        else:
                            logger.warning(f"实验 {lab_id} 关联不存在的知识点: {kp_id}")
                    logger.debug(f"实验 {lab_id} 关联了 {len(related_kps)} 个知识点")
                else:
                    logger.warning(f"实验 {lab_id} 没有关联任何知识点")
            
            logger.info(f"实验环节构建完成: {len(labs)} 个")
        except Exception as e:
            logger.error(f"实验构建失败: {str(e)}")
            raise

    def _enhance_chapter_associations(self, tx: Transaction):
        """
        对同一章节内的知识点和子知识点建立全连接关联，
        以增强节点间的传播和 PageRank 计算效果。
        """
        try:
            for chapter_id, chapter_node in self.chapter_cache.items():
                # 章节内知识点全连接
                chapter_kps = [kp for kp in self.kp_cache.values() if kp.get("chapter_id") == chapter_id]
                if len(chapter_kps) >= 2:
                    for i in range(len(chapter_kps)):
                        for j in range(i + 1, len(chapter_kps)):
                            tx.create(Relationship(chapter_kps[i], "RELATED_TO", chapter_kps[j]))
                
                # 章节内子知识点全连接
                chapter_sub_kps = [sub for sub in self.sub_kp_cache.values() if sub.get("chapter_id") == chapter_id]
                if len(chapter_sub_kps) >= 2:
                    for i in range(len(chapter_sub_kps)):
                        for j in range(i + 1, len(chapter_sub_kps)):
                            tx.create(Relationship(chapter_sub_kps[i], "RELATED_TO", chapter_sub_kps[j]))
            
            logger.info("章节内知识点关联增强完成")
        except Exception as e:
            logger.error(f"章节关联增强失败: {str(e)}")
            raise

    def _build_path_relationships(self, tx: Transaction):
        """
        构建用于学习路径搜索的关系：
         1. 创建 NEXT_CHAPTER 关系：严格按章节的 order 顺序建立。
         2. 创建 NEXT_KNOWLEDGE 关系：在同一章节内，根据知识点的 bloom_level 建立后继关系。
         3. 创建 NEXT_SUB_KNOWLEDGE 关系：在同一知识点下，根据子知识点的 bloom_level 建立后继关系。
        """
        try:
            # 1. NEXT_CHAPTER 关系：按章节顺序建立
            chapter_query = """
            MATCH (c:Chapter)
            WITH c ORDER BY toInteger(c.order) ASC
            WITH collect(c) AS chapters
            UNWIND range(0, size(chapters)-2) AS i
            MATCH (c1) WHERE id(c1) = id(chapters[i])
            MATCH (c2) WHERE id(c2) = id(chapters[i+1])
            MERGE (c1)-[:NEXT_CHAPTER {weight:1.0}]->(c2)
            """
            tx.run(chapter_query)

            # 2. NEXT_KNOWLEDGE 关系：在同一章节内建立，按照 bloom_level 升序
            kp_query = """
            MATCH (ch:Chapter)-[:HAS_KNOWLEDGE]->(kp1:KnowledgePoint),
                  (ch)-[:HAS_KNOWLEDGE]->(kp2:KnowledgePoint)
            WHERE kp1 <> kp2 AND toInteger(kp1.bloom_level) < toInteger(kp2.bloom_level)
            MERGE (kp1)-[:NEXT_KNOWLEDGE {weight: 1.0}]->(kp2)
            """
            tx.run(kp_query)

            # 3. NEXT_SUB_KNOWLEDGE 关系：在同一知识点下，按照 bloom_level 升序
            sub_kp_query = """
            MATCH (kp:KnowledgePoint)<-[:CHILD_OF]-(skp1:SubKnowledgePoint),
                  (kp)<-[:CHILD_OF]-(skp2:SubKnowledgePoint)
            WHERE skp1 <> skp2 AND toInteger(skp1.bloom_level) < toInteger(skp2.bloom_level)
            MERGE (skp1)-[:NEXT_SUB_KNOWLEDGE {weight: 1.0}]->(skp2)
            """
            tx.run(sub_kp_query)

            logger.info("路径关系构建完成")
        except Exception as e:
            logger.error(f"路径关系构建失败: {str(e)}")
            raise

if __name__ == "__main__":
    # 添加异常处理
    try:
        builder = EduKGBuilder(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="123456789"
        )
        builder.build()
        logger.info("知识图谱构建成功完成！")
    except Exception as e:
        logger.critical(f"程序执行失败: {str(e)}", exc_info=True)
        raise