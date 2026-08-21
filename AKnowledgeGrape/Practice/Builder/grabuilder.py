import pandas as pd
from py2neo import Graph, Node, Relationship, Transaction
from typing import Dict, List
import logging
import warnings
import os
import csv
from datetime import datetime

# 禁用 Py2neo 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CrossDomainEduKGBuilder")


class CrossDomainDataGenerator:
    """跨学科数据生成器：自动生成4个跨学科CSV文件"""
    def __init__(self, output_dir: str = "./cross_domain_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"跨学科数据将生成至：{os.path.abspath(output_dir)}")

    def generate_cross_domains(self) -> str:
        """生成跨学科领域数据（cross_domains.csv）"""
        data = [
            {
                "cross_domain_id": "CD001",
                "domain_name": "数学基础",
                "description": "提供大数据分析（如统计分析、机器学习）所需的数学工具，包括线性代数、概率论、微积分等",
                "relevance": 0.9
            },
            {
                "cross_domain_id": "CD002",
                "domain_name": "计算机基础",
                "description": "支撑大数据技术的底层技术，包括数据结构、数据库、计算机网络、操作系统等",
                "relevance": 0.95
            },
            {
                "cross_domain_id": "CD003",
                "domain_name": "工程实践",
                "description": "大数据项目落地所需的工程能力，包括云计算、容器化、DevOps、数据可视化等",
                "relevance": 0.85
            }
        ]
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, "cross_domains.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"生成跨学科领域数据：{path}（{len(df)}条）")
        return path

    def generate_cross_knowledge_points(self) -> str:
        """生成跨学科知识点数据（cross_knowledge_points.csv）"""
        data = [
            {
                "cross_kp_id": "CKP001",
                "cross_domain_id": "CD001",
                "title": "线性代数基础",
                "description": "涵盖矩阵运算、向量空间、特征值分解，支撑原学科“机器学习”“数据降维”等知识点的数学原理",
                "bloom_level": 3,
                "related_original_kps": "KP005;KP008",
                "cross_correlation": 0.85,
                "class_hours": 8.0
            },
            {
                "cross_kp_id": "CKP002",
                "cross_domain_id": "CD001",
                "title": "概率论与数理统计",
                "description": "包括随机变量、概率分布、假设检验，为原学科“统计分析”“异常检测”提供理论支撑",
                "bloom_level": 3,
                "related_original_kps": "KP003;KP006",
                "cross_correlation": 0.80,
                "class_hours": 6.0
            },
            {
                "cross_kp_id": "CKP003",
                "cross_domain_id": "CD002",
                "title": "数据结构",
                "description": "包括数组、链表、树、图等结构，支撑原学科“大数据存储”“算法优化”知识点",
                "bloom_level": 4,
                "related_original_kps": "KP002;KP010",
                "cross_correlation": 0.90,
                "class_hours": 10.0
            },
            {
                "cross_kp_id": "CKP004",
                "cross_domain_id": "CD002",
                "title": "计算机网络",
                "description": "涵盖TCP/IP协议、分布式通信、网络安全，支撑原学科“分布式计算”“数据传输”知识点",
                "bloom_level": 4,
                "related_original_kps": "KP007;KP009",
                "cross_correlation": 0.88,
                "class_hours": 8.0
            },
            {
                "cross_kp_id": "CKP005",
                "cross_domain_id": "CD003",
                "title": "云计算基础",
                "description": "包括云服务器、对象存储、弹性计算，支撑原学科“大数据部署”“集群管理”知识点",
                "bloom_level": 5,
                "related_original_kps": "KP012;KP015",
                "cross_correlation": 0.82,
                "class_hours": 12.0
            }
        ]
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, "cross_knowledge_points.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"生成跨学科知识点数据：{path}（{len(df)}条）")
        return path

    def generate_cross_sub_knowledge_points(self) -> str:
        """生成跨学科子知识点数据（cross_sub_knowledge_points.csv）"""
        data = [
            {
                "cross_sub_kp_id": "CSKP001",
                "cross_kp_id": "CKP001",
                "title": "矩阵乘法与逆矩阵",
                "description": "矩阵运算的核心操作，支撑原学科“SKP021（机器学习-特征矩阵）”“SKP023（数据降维-PCA）”的计算过程",
                "bloom_level": 4,
                "dependency_cross_sub_kp": "CSKP002",
                "related_original_sub_kps": "SKP021;SKP023",
                "cross_correlation": 0.92,
                "syllabus_mentions": 15
            },
            {
                "cross_sub_kp_id": "CSKP002",
                "cross_kp_id": "CKP001",
                "title": "向量空间与内积",
                "description": "向量运算的基础，用于原学科“SKP018（推荐系统-用户向量）”的相似度计算",
                "bloom_level": 3,
                "dependency_cross_sub_kp": "",
                "related_original_sub_kps": "SKP018",
                "cross_correlation": 0.87,
                "syllabus_mentions": 12
            },
            {
                "cross_sub_kp_id": "CSKP003",
                "cross_kp_id": "CKP003",
                "title": "哈希表与红黑树",
                "description": "高效的数据存储结构，支撑原学科“SKP005（大数据存储-哈希分区）”“SKP007（索引优化-B树索引）”知识点",
                "bloom_level": 5,
                "dependency_cross_sub_kp": "",
                "related_original_sub_kps": "SKP005;SKP007",
                "cross_correlation": 0.91,
                "syllabus_mentions": 18
            },
            {
                "cross_sub_kp_id": "CSKP004",
                "cross_kp_id": "CKP004",
                "title": "TCP/IP协议栈",
                "description": "分布式系统通信的基础，支撑原学科“SKP032（分布式计算-数据同步）”“SKP035（流处理-实时传输）”知识点",
                "bloom_level": 5,
                "dependency_cross_sub_kp": "CSKP005",
                "related_original_sub_kps": "SKP032;SKP035",
                "cross_correlation": 0.89,
                "syllabus_mentions": 16
            },
            {
                "cross_sub_kp_id": "CSKP005",
                "cross_kp_id": "CKP004",
                "title": "HTTP与WebSocket",
                "description": "应用层协议，支撑原学科“SKP040（API开发-数据接口）”知识点",
                "bloom_level": 4,
                "dependency_cross_sub_kp": "",
                "related_original_sub_kps": "SKP040",
                "cross_correlation": 0.83,
                "syllabus_mentions": 10
            },
            {
                "cross_sub_kp_id": "CSKP006",
                "cross_kp_id": "CKP005",
                "title": "Docker容器化",
                "description": "轻量级虚拟化技术，支撑原学科“SKP045（大数据部署-Docker集群）”“SKP047（DevOps-自动化部署）”知识点",
                "bloom_level": 5,
                "dependency_cross_sub_kp": "",
                "related_original_sub_kps": "SKP045;SKP047",
                "cross_correlation": 0.86,
                "syllabus_mentions": 20
            }
        ]
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, "cross_sub_knowledge_points.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"生成跨学科子知识点数据：{path}（{len(df)}条）")
        return path

    def generate_cross_original_links(self) -> str:
        """生成跨学科-原学科关联数据（cross_original_links.csv）"""
        data = [
            {
                "link_id": "CL001",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP001",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP021",
                "link_weight": 1.2,
                "apply_scenario": "预习（机器学习-特征矩阵运算前需掌握）"
            },
            {
                "link_id": "CL002",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP001",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP023",
                "link_weight": 1.2,
                "apply_scenario": "预习（数据降维-PCA前需掌握）"
            },
            {
                "link_id": "CL003",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP002",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP018",
                "link_weight": 1.1,
                "apply_scenario": "预习（推荐系统-用户向量相似度计算前需掌握）"
            },
            {
                "link_id": "CL004",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP003",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP005",
                "link_weight": 1.3,
                "apply_scenario": "复习（大数据存储-哈希分区优化时拓展）"
            },
            {
                "link_id": "CL005",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP003",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP007",
                "link_weight": 1.3,
                "apply_scenario": "复习（索引优化-B树索引原理拓展）"
            },
            {
                "link_id": "CL006",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP004",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP032",
                "link_weight": 1.2,
                "apply_scenario": "预习（分布式计算-数据同步前需掌握）"
            },
            {
                "link_id": "CL007",
                "cross_node_type": "CrossSubKP",
                "cross_node_id": "CSKP006",
                "original_node_type": "SubKnowledgePoint",
                "original_node_id": "SKP045",
                "link_weight": 1.4,
                "apply_scenario": "拓展（大数据部署-Docker集群实践拓展）"
            }
        ]
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, "cross_original_links.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"生成跨学科-原学科关联数据：{path}（{len(df)}条）")
        return path

    def generate_all(self) -> Dict[str, str]:
        """生成所有跨学科数据文件，返回文件路径字典"""
        logger.info("=== 开始生成跨学科数据 ===")
        paths = {
            "cross_domains": self.generate_cross_domains(),
            "cross_knowledge_points": self.generate_cross_knowledge_points(),
            "cross_sub_knowledge_points": self.generate_cross_sub_knowledge_points(),
            "cross_original_links": self.generate_cross_original_links()
        }
        logger.info("=== 跨学科数据生成完成 ===")
        return paths


class EnhancedEduKGBuilder:
    """增强版教育知识图谱构建器：支持跨学科数据导入（已修正实验统计错误）"""
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, cross_data_dir: str):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.cross_data_dir = cross_data_dir  # 跨学科数据目录
        self._validate_connection()

        # 初始化缓存（含跨学科节点缓存）
        self.chapter_cache: Dict[str, Node] = {}
        self.kp_cache: Dict[str, Node] = {}
        self.sub_kp_cache: Dict[str, Node] = {}
        self.cross_domain_cache: Dict[str, Node] = {}
        self.cross_kp_cache: Dict[str, Node] = {}
        self.cross_sub_kp_cache: Dict[str, Node] = {}

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
        encoding = result['encoding'] or 'utf-8'
        logger.debug(f"文件 {os.path.basename(file_path)} 编码检测结果: {encoding}")
        return encoding

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """加载原有数据 + 跨学科数据"""
        # 1. 原有数据路径（需与用户现有数据目录一致）
        original_base_path = os.path.join("D:\\", "LunWen (2)", "LunWen", "AKnowledgeGrape", "D-Data")
        # 2. 跨学科数据路径
        cross_base_path = self.cross_data_dir

        required_files = {
            # 原有数据文件
            'chapters': (os.path.join(original_base_path, 'chapters.csv'), 
                        ['chapter_id', 'title', 'order', 'class_hours']),
            'knowledge_points': (os.path.join(original_base_path, 'knowledge_points.csv'), 
                                ['kp_id', 'chapter_id', 'description', 'bloom_level', 'class_hours', 'syllabus_mentions']),
            'sub_knowledge_points': (os.path.join(original_base_path, 'sub_knowledge_points.csv'),  
                                    ['sub_kp_id', 'kp_id', 'title', 'description', 'bloom_level', 'dependency_sub_kp', 'class_hours', 'syllabus_mentions', 'chapter_id']),
            'labs': (os.path.join(original_base_path, 'labs.csv'), 
                    ['lab_id', 'lab_name', 'related_kp']),
            # 跨学科数据文件
            'cross_domains': (os.path.join(cross_base_path, 'cross_domains.csv'), 
                             ['cross_domain_id', 'domain_name', 'description', 'relevance']),
            'cross_knowledge_points': (os.path.join(cross_base_path, 'cross_knowledge_points.csv'), 
                                      ['cross_kp_id', 'cross_domain_id', 'title', 'description', 'bloom_level', 'related_original_kps', 'cross_correlation', 'class_hours']),
            'cross_sub_knowledge_points': (os.path.join(cross_base_path, 'cross_sub_knowledge_points.csv'), 
                                          ['cross_sub_kp_id', 'cross_kp_id', 'title', 'description', 'bloom_level', 'dependency_cross_sub_kp', 'related_original_sub_kps', 'cross_correlation', 'syllabus_mentions']),
            'cross_original_links': (os.path.join(cross_base_path, 'cross_original_links.csv'), 
                                    ['link_id', 'cross_node_type', 'cross_node_id', 'original_node_type', 'original_node_id', 'link_weight', 'apply_scenario'])
        }

        data = {}
        for name, (path, cols) in required_files.items():
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"文件不存在: {path}")
                
                # 尝试多种编码读取
                encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        logger.info(f"使用 {encoding} 编码读取 {name}（{os.path.basename(path)}）")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    detected_encoding = self._detect_encoding(path)
                    df = pd.read_csv(path, encoding=detected_encoding)
                    logger.info(f"使用自动检测编码 {detected_encoding} 读取 {name}")

                # 验证必要列
                missing_cols = set(cols) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"{name} 缺少必要列: {missing_cols}")
                
                # 数据预处理
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'None': ''})
                
                # 数值列处理
                numeric_cols = ['class_hours', 'order', 'syllabus_mentions', 'bloom_level', 'relevance', 'cross_correlation', 'link_weight']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                data[name] = df
                logger.info(f"成功加载 {name}: {len(df)} 条记录")
                
            except Exception as e:
                logger.error(f"加载 {name} 失败: {str(e)}")
                raise
        return data

    def _build_chapters(self, tx: Transaction, chapters: pd.DataFrame):
        """构建原有章节节点（补充domain属性）"""
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
                           class_hours=float(row['class_hours']),
                           domain="big_data_technology")  # 原学科领域标识
            tx.create(chapter)
            self.chapter_cache[chapter_id] = chapter
        
        logger.info(f"章节构建完成: {len(self.chapter_cache)} 个")

    def _build_course(self, tx: Transaction):
        """构建原有课程节点"""
        course_node = Node("Course",
                           id="COURSE001",
                           course_id="COURSE001",
                           name="大数据技术",
                           credit=3,
                           semester="2024春季",
                           domain="big_data_technology")
        tx.create(course_node)
        
        for chapter in self.chapter_cache.values():
            tx.create(Relationship(course_node, "HAS_CHAPTER", chapter))
        
        logger.info("课程节点关联完成")

    def _build_knowledge_points(self, tx: Transaction, kps: pd.DataFrame):
        """构建原有知识点节点（补充domain属性）"""
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
                      title=str(row['description']).strip(),  # description映射为title
                      bloom_level=int(row['bloom_level']),
                      class_hours=float(row['class_hours']),
                      syllabus_mentions=int(row['syllabus_mentions']),
                      domain="big_data_technology")  # 原学科领域标识
            tx.create(kp)
            self.kp_cache[kp_id] = kp

            chapter = self.chapter_cache.get(chapter_id)
            if chapter:
                tx.create(Relationship(chapter, "HAS_KNOWLEDGE", kp))
            else:
                logger.warning(f"章节不存在: chapter_id={chapter_id}，知识点 {kp_id} 未关联章节")
        
        logger.info(f"知识点构建完成: {len(self.kp_cache)} 个")

    def _build_sub_knowledge_points(self, tx: Transaction, sub_kps: pd.DataFrame):
        """构建原有子知识点节点（补充domain属性）"""
        # 1. 创建子知识点节点
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
                          syllabus_mentions=int(row['syllabus_mentions']),
                          domain="big_data_technology",  # 原学科领域标识
                          cross_correlation=0.5)  # 默认关联强度（原学科节点）
            tx.create(sub_kp)
            self.sub_kp_cache[sub_kp_id] = sub_kp

            # 关联到章节
            chapter = self.chapter_cache.get(chapter_id)
            if chapter:
                tx.create(Relationship(sub_kp, "BELONGS_TO_CHAPTER", chapter))
            else:
                logger.warning(f"章节不存在: chapter_id={chapter_id}，子知识点 {sub_kp_id} 未关联章节")

        # 2. 建立父子及依赖关系
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
                if dependency_str:
                    dependency_ids = [d.strip() for d in dependency_str.split(';') if d.strip()]
                    for dep_id in dependency_ids:
                        if dep_id in self.sub_kp_cache and dep_id != sub_kp_id:
                            tx.create(Relationship(sub_kp, "REQUIRES_PREREQUISITE", self.sub_kp_cache[dep_id]))
                        else:
                            logger.warning(f"依赖的子知识点不存在或自引用: {dep_id} (当前: {sub_kp_id})")
                else:
                    tx.create(Relationship(sub_kp, "ASSOCIATED_WITH", parent_kp))
            else:
                logger.warning(f"父知识点不存在: kp_id={kp_id}，子知识点 {sub_kp_id} 未关联父节点")
        
        logger.info(f"子知识点构建完成: {len(self.sub_kp_cache)} 个")

    def _build_labs(self, tx: Transaction, labs: pd.DataFrame):
        """构建原有实验节点"""
        for _, row in labs.iterrows():
            lab_id = str(row['lab_id']).strip()
            lab_name = str(row['lab_name']).strip()
            
            if not lab_id:
                logger.warning("发现空 lab_id，跳过")
                continue
            
            lab = Node("Lab",
                       id=lab_id,
                       lab_id=lab_id,
                       title=lab_name,
                       domain="big_data_technology")  # 原学科领域标识
            tx.create(lab)
            
            # 关联知识点
            related_kps_str = str(row.get('related_kp', '')).strip()
            if related_kps_str:
                related_kps = [kp.strip() for kp in related_kps_str.split(';') if kp.strip()]
                for kp_id in related_kps:
                    kp = self.kp_cache.get(kp_id)
                    if kp:
                        tx.create(Relationship(lab, "REQUIRES_KNOWLEDGE", kp))
                    else:
                        logger.warning(f"实验 {lab_id} 关联不存在的知识点: {kp_id}")
        
        logger.info(f"实验环节构建完成: {len(labs)} 个")

    def _build_cross_domains(self, tx: Transaction, cross_domains: pd.DataFrame):
        """构建跨学科领域节点"""
        for _, row in cross_domains.iterrows():
            cross_domain_id = str(row['cross_domain_id']).strip()
            if not cross_domain_id:
                logger.warning("空跨学科领域ID，跳过")
                continue
            # 跨学科领域节点（domain属性为领域名称，用于路径搜索识别）
            cross_domain = Node("CrossDomain",
                                id=cross_domain_id,
                                cross_domain_id=cross_domain_id,
                                domain_name=str(row['domain_name']).strip(),
                                description=str(row['description']).strip(),
                                relevance=float(row['relevance']),
                                domain=str(row['domain_name']).strip())
            tx.create(cross_domain)
            self.cross_domain_cache[cross_domain_id] = cross_domain
        logger.info(f"跨学科领域构建完成: {len(self.cross_domain_cache)} 个")

    def _build_cross_knowledge_points(self, tx: Transaction, cross_kps: pd.DataFrame):
        """构建跨学科知识点节点"""
        for _, row in cross_kps.iterrows():
            cross_kp_id = str(row['cross_kp_id']).strip()
            cross_domain_id = str(row['cross_domain_id']).strip()
            
            if not cross_kp_id or cross_domain_id not in self.cross_domain_cache:
                logger.warning(f"无效跨学科知识点数据（ID={cross_kp_id}），跳过")
                continue
            
            # 跨学科知识点节点
            cross_kp = Node("CrossKnowledgePoint",
                            id=cross_kp_id,
                            cross_kp_id=cross_kp_id,
                            title=str(row['title']).strip(),
                            description=str(row['description']).strip(),
                            bloom_level=int(row['bloom_level']),
                            cross_correlation=float(row['cross_correlation']),
                            class_hours=float(row['class_hours']),
                            domain=self.cross_domain_cache[cross_domain_id]['domain_name'])  # 继承领域名称
            tx.create(cross_kp)
            self.cross_kp_cache[cross_kp_id] = cross_kp
            
            # 关联到跨学科领域
            tx.create(Relationship(self.cross_domain_cache[cross_domain_id], "HAS_CROSS_KP", cross_kp))
        logger.info(f"跨学科知识点构建完成: {len(self.cross_kp_cache)} 个")

    def _build_cross_sub_knowledge_points(self, tx: Transaction, cross_sub_kps: pd.DataFrame):
        """构建跨学科子知识点节点"""
        # 1. 创建跨学科子知识点节点
        for _, row in cross_sub_kps.iterrows():
            cross_sub_kp_id = str(row['cross_sub_kp_id']).strip()
            cross_kp_id = str(row['cross_kp_id']).strip()
            
            if not cross_sub_kp_id or cross_kp_id not in self.cross_kp_cache:
                logger.warning(f"无效跨学科子知识点数据（ID={cross_sub_kp_id}），跳过")
                continue
            
            # 跨学科子知识点节点
            cross_sub_kp = Node("CrossSubKnowledgePoint",
                                id=cross_sub_kp_id,
                                cross_sub_kp_id=cross_sub_kp_id,
                                title=str(row['title']).strip(),
                                description=str(row['description']).strip(),
                                bloom_level=int(row['bloom_level']),
                                cross_correlation=float(row['cross_correlation']),
                                syllabus_mentions=int(row['syllabus_mentions']),
                                domain=self.cross_kp_cache[cross_kp_id]['domain'])  # 继承领域名称
            tx.create(cross_sub_kp)
            self.cross_sub_kp_cache[cross_sub_kp_id] = cross_sub_kp
            
            # 关联到跨学科知识点
            tx.create(Relationship(self.cross_kp_cache[cross_kp_id], "HAS_CROSS_SUB_KP", cross_sub_kp))

        # 2. 建立跨学科子知识点依赖关系
        for _, row in cross_sub_kps.iterrows():
            cross_sub_kp_id = str(row['cross_sub_kp_id']).strip()
            if cross_sub_kp_id not in self.cross_sub_kp_cache:
                continue
            
            dependency_str = str(row['dependency_cross_sub_kp']).strip()
            if dependency_str:
                dependency_ids = [d.strip() for d in dependency_str.split(';') if d.strip()]
                for dep_id in dependency_ids:
                    if dep_id in self.cross_sub_kp_cache and dep_id != cross_sub_kp_id:
                        tx.create(Relationship(
                            self.cross_sub_kp_cache[cross_sub_kp_id],
                            "REQUIRES_CROSS_PREREQUISITE",
                            self.cross_sub_kp_cache[dep_id]
                        ))
        logger.info(f"跨学科子知识点构建完成: {len(self.cross_sub_kp_cache)} 个")

    def _build_cross_original_links(self, tx: Transaction, cross_links: pd.DataFrame):
        """构建跨学科-原学科关联关系（支撑路径搜索的跨域边）"""
        for _, row in cross_links.iterrows():
            cross_node_type = str(row['cross_node_type']).strip()
            cross_node_id = str(row['cross_node_id']).strip()
            original_node_type = str(row['original_node_type']).strip()
            original_node_id = str(row['original_node_id']).strip()
            link_weight = float(row['link_weight'])
            
            # 1. 获取跨学科节点
            cross_node = None
            if cross_node_type == "CrossKP":
                cross_node = self.cross_kp_cache.get(cross_node_id)
            elif cross_node_type == "CrossSubKP":
                cross_node = self.cross_sub_kp_cache.get(cross_node_id)
            else:
                logger.warning(f"未知跨学科节点类型: {cross_node_type}（跳过）")
                continue
            
            # 2. 获取原学科节点
            original_node = None
            if original_node_type == "KnowledgePoint":
                original_node = self.kp_cache.get(original_node_id)
            elif original_node_type == "SubKnowledgePoint":
                original_node = self.sub_kp_cache.get(original_node_id)
            else:
                logger.warning(f"未知原学科节点类型: {original_node_type}（跳过）")
                continue
            
            # 3. 构建双向关联关系（支撑“跨域探索”与“返回原学科”）
            if cross_node and original_node:
                # 原学科→跨学科（探索边）
                tx.create(Relationship(original_node, "RELATED_CROSS_DOMAIN", cross_node, 
                                     weight=link_weight, 
                                     scenario=str(row['apply_scenario']).strip()))
                # 跨学科→原学科（返回边，权重降低以鼓励返回）
                tx.create(Relationship(cross_node, "RELATED_ORIGINAL_DOMAIN", original_node, 
                                     weight=link_weight * 0.7,  # 返回边权重降低30%
                                     scenario=str(row['apply_scenario']).strip()))
                logger.debug(f"构建关联: {original_node['title']} ↔ {cross_node['title']}（权重: {link_weight}）")
        
        logger.info(f"跨学科-原学科关联关系构建完成: {len(cross_links)} 条")

    def _enhance_chapter_associations(self, tx: Transaction):
        """原有章节内关联增强（保留）"""
        for chapter_id, chapter_node in self.chapter_cache.items():
            # 章节内知识点全连接
            chapter_kps = [kp for kp in self.kp_cache.values() if kp.get("chapter_id") == chapter_id]
            if len(chapter_kps) >= 2:
                for i in range(len(chapter_kps)):
                    for j in range(i + 1, len(chapter_kps)):
                        tx.create(Relationship(chapter_kps[i], "RELATED_TO", chapter_kps[j], weight=1.0))
            
            # 章节内子知识点全连接
            chapter_sub_kps = [sub for sub in self.sub_kp_cache.values() if sub.get("chapter_id") == chapter_id]
            if len(chapter_sub_kps) >= 2:
                for i in range(len(chapter_sub_kps)):
                    for j in range(i + 1, len(chapter_sub_kps)):
                        tx.create(Relationship(chapter_sub_kps[i], "RELATED_TO", chapter_sub_kps[j], weight=1.0))
        
        logger.info("章节内知识点关联增强完成")

    def _build_path_relationships(self, tx: Transaction):
        """原有路径关系构建（保留）"""
        # 1. 章节顺序关系
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

        # 2. 章节内知识点顺序关系
        kp_query = """
        MATCH (ch:Chapter)-[:HAS_KNOWLEDGE]->(kp1:KnowledgePoint),
              (ch)-[:HAS_KNOWLEDGE]->(kp2:KnowledgePoint)
        WHERE kp1 <> kp2 AND toInteger(kp1.bloom_level) < toInteger(kp2.bloom_level)
        MERGE (kp1)-[:NEXT_KNOWLEDGE {weight: 1.0}]->(kp2)
        """
        tx.run(kp_query)

        # 3. 知识点内子知识点顺序关系
        sub_kp_query = """
        MATCH (kp:KnowledgePoint)<-[:CHILD_OF]-(skp1:SubKnowledgePoint),
              (kp)<-[:CHILD_OF]-(skp2:SubKnowledgePoint)
        WHERE skp1 <> skp2 AND toInteger(skp1.bloom_level) < toInteger(skp2.bloom_level)
        MERGE (skp1)-[:NEXT_SUB_KNOWLEDGE {weight: 1.0}]->(skp2)
        """
        tx.run(sub_kp_query)

        logger.info("原有路径关系构建完成")

    def _validate_cross_data(self, data: Dict[str, pd.DataFrame]):
        """验证跨学科数据与原数据的关联性"""
        logger.info("=== 开始跨学科数据关联性验证 ===")
        # 1. 验证跨学科知识点关联的原学科知识点是否存在
        cross_kps = data['cross_knowledge_points']
        original_kp_ids = set(self.kp_cache.keys())
        missing_kps = set()
        for _, row in cross_kps.iterrows():
            related_kps = str(row['related_original_kps']).strip().split(';')
            for kp_id in related_kps:
                if kp_id and kp_id not in original_kp_ids:
                    missing_kps.add(kp_id)
        if missing_kps:
            logger.warning(f"跨学科知识点关联的原学科知识点不存在: {missing_kps}（建议检查原数据kp_id）")
        else:
            logger.info("跨学科知识点与原学科知识点关联验证通过")

        # 2. 验证跨学科子知识点关联的原学科子知识点是否存在
        cross_sub_kps = data['cross_sub_knowledge_points']
        original_sub_kp_ids = set(self.sub_kp_cache.keys())
        missing_sub_kps = set()
        for _, row in cross_sub_kps.iterrows():
            related_sub_kps = str(row['related_original_sub_kps']).strip().split(';')
            for sub_kp_id in related_sub_kps:
                if sub_kp_id and sub_kp_id not in original_sub_kp_ids:
                    missing_sub_kps.add(sub_kp_id)
        if missing_sub_kps:
            logger.warning(f"跨学科子知识点关联的原学科子知识点不存在: {missing_sub_kps}（建议检查原数据sub_kp_id）")
        else:
            logger.info("跨学科子知识点与原学科子知识点关联验证通过")
        logger.info("=== 跨学科数据关联性验证完成 ===")

    def _print_build_statistics(self):
        """打印构建统计信息（已修正实验数量统计错误）"""
        logger.info("=== 知识图谱构建统计 ===")
        # 原有节点统计
        logger.info(f"原学科节点:")
        logger.info(f"  - 课程: 1 个（大数据技术）")
        logger.info(f"  - 章节: {len(self.chapter_cache)} 个")
        logger.info(f"  - 知识点: {len(self.kp_cache)} 个")
        logger.info(f"  - 子知识点: {len(self.sub_kp_cache)} 个")
        # 修正：直接获取实验数量（整数），不调用len()
        lab_count = self.graph.run("MATCH (l:Lab) RETURN count(l)").data()[0]['count(l)']
        logger.info(f"  - 实验: {lab_count} 个")
        # 跨学科节点统计
        logger.info(f"跨学科节点:")
        logger.info(f"  - 跨学科领域: {len(self.cross_domain_cache)} 个")
        logger.info(f"  - 跨学科知识点: {len(self.cross_kp_cache)} 个")
        logger.info(f"  - 跨学科子知识点: {len(self.cross_sub_kp_cache)} 个")
        # 关系统计
        rel_counts = self.graph.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """).data()
        logger.info(f"关系统计:")
        for rel in rel_counts:
            logger.info(f"  - {rel['rel_type']}: {rel['count']} 条")
        logger.info("=== 统计完成 ===")

    def build(self):
        """全量构建（原有数据 + 跨学科数据）"""
        tx = None
        try:
            # 1. 加载所有数据（原有 + 跨学科）
            data = self.load_data()
            
            # 2. 开始事务
            tx = self.graph.begin()
            
            # 3. 构建原有节点（补充domain属性）
            logger.info("=== 开始构建原有知识图谱节点 ===")
            self._build_chapters(tx, data['chapters'])
            self._build_course(tx)
            self._build_knowledge_points(tx, data['knowledge_points'])
            self._build_sub_knowledge_points(tx, data['sub_knowledge_points'])
            self._build_labs(tx, data['labs'])
            
            # 4. 验证跨学科数据关联性
            self._validate_cross_data(data)
            
            # 5. 构建跨学科节点与关联
            logger.info("=== 开始构建跨学科节点与关联 ===")
            self._build_cross_domains(tx, data['cross_domains'])
            self._build_cross_knowledge_points(tx, data['cross_knowledge_points'])
            self._build_cross_sub_knowledge_points(tx, data['cross_sub_knowledge_points'])
            self._build_cross_original_links(tx, data['cross_original_links'])
            
            # 6. 构建原有增强关系与路径关系
            logger.info("=== 开始构建增强关系与路径关系 ===")
            self._enhance_chapter_associations(tx)
            self._build_path_relationships(tx)
            
            # 7. 提交事务
            tx.commit()
            logger.info("=== 全量知识图谱（含跨学科）构建完成 ===")
            
            # 8. 输出构建统计（已修正错误）
            self._print_build_statistics()
            
        except Exception as e:
            if tx:
                tx.rollback()
                logger.error("事务回滚")
            logger.critical(f"构建流程异常: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    """执行流程：1. 生成跨学科数据 → 2. 导入现有图谱"""
    try:
        # 1. 配置参数（需根据实际环境修改）
        NEO4J_URI = "bolt://localhost:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = "123456789"
        CROSS_DATA_OUTPUT_DIR = "./cross_domain_data"  # 跨学科数据生成目录
        ORIGINAL_DATA_DIR = os.path.join("D:\\", "LunWen (2)", "LunWen", "AKnowledgeGrape", "D-Data")  # 原有数据目录

        # 2. 生成跨学科数据
        data_generator = CrossDomainDataGenerator(output_dir=CROSS_DATA_OUTPUT_DIR)
        cross_data_paths = data_generator.generate_all()

        # 3. 导入跨学科数据到现有图谱
        builder = EnhancedEduKGBuilder(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            cross_data_dir=CROSS_DATA_OUTPUT_DIR
        )
        builder.build()

        logger.info("=== 跨学科数据生成与导入任务全部完成 ===")
        logger.info(f"跨学科数据文件位置: {os.path.abspath(CROSS_DATA_OUTPUT_DIR)}")
        logger.info("可直接运行前文路径搜索代码，使用扩展模式（expand_mode=1）探索跨学科路径")

    except Exception as e:
        logger.critical(f"任务执行失败: {str(e)}", exc_info=True)
        raise