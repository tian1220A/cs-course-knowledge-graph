from py2neo import Graph
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging

# 中文字体注册
pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedSyllabusGenerator")


class EnhancedSyllabusGenerator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.graph.run("RETURN 1")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.critical(f"Neo4j连接失败: {e}")
            raise

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """配置自定义样式"""
        # 标题样式
        self.styles.add(ParagraphStyle(
            name='ChapterTitle',
            fontName='SimHei',
            fontSize=16,
            leading=22,
            alignment=1,
            spaceAfter=20
        ))
        # 表格头样式
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            fontName='SimHei',
            fontSize=12,
            leading=16,
            textColor=colors.white,
            alignment=1
        ))
        # 正文样式
        self.styles.add(ParagraphStyle(
            name='ChineseBody',
            fontName='SimHei',
            fontSize=11,
            leading=14,
            spaceAfter=6
        ))
        # 强调样式
        self.styles.add(ParagraphStyle(
            name='Highlight',
            fontName='SimHei',
            fontSize=11,
            textColor=colors.red,
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name='HighlightRow',
            fontName='SimHei',
            fontSize=11,
            textColor=colors.black,
            backColor=colors.HexColor('#FFF3CD'),  # 浅黄色背景
            leading=14
        ))

    def fetch_chapter_data(self):
        query = """
        MATCH (c:Chapter)
        WITH c ORDER BY c.order
        OPTIONAL MATCH (c)-[:HAS_KNOWLEDGE]->(k:KnowledgePoint)
        WITH c, COLLECT(k) AS kps
        OPTIONAL MATCH (sub_kp:SubKnowledgePoint)-[:BELONGS_TO_CHAPTER]->(c) 
        WITH c, kps, COLLECT(sub_kp) AS sub_kps
        WITH c, kps + sub_kps AS knowledge_points
        RETURN {
            chapter_id: c.chapter_id,
            title: c.title,
            order: c.order,
            class_hours: c.class_hours,
            knowledge: [kp IN knowledge_points | {
                id: coalesce(kp.kp_id, kp.sub_kp_id),
                type: CASE 
                         WHEN 'KnowledgePoint' IN labels(kp) THEN 'KnowledgePoint'
                         WHEN 'SubKnowledgePoint' IN labels(kp) THEN 'SubKnowledgePoint'
                         ELSE 'Unknown'
                     END,
                title: CASE
                         WHEN 'KnowledgePoint' IN labels(kp) THEN 
                             CASE WHEN kp.title IS NOT NULL AND trim(kp.title) <> '' THEN kp.title ELSE '未命名知识点' END
                         WHEN 'SubKnowledgePoint' IN labels(kp) THEN 
                             CASE WHEN kp.title IS NOT NULL AND trim(kp.title) <> '' THEN kp.title ELSE '未命名子知识点' END
                         ELSE '未知类型'
                       END,
                edu_pr: coalesce(kp.edu_pagerank, 0.0),  
                class_hours: coalesce(kp.class_hours, 0),
                syllabus_mentions: coalesce(kp.syllabus_mentions, 0),
                bloom_level: coalesce(kp.bloom_level, 1)
            }]
        } AS chapter_data
        """
        # 执行查询并记录原始数据
        raw_data = self.graph.run(query).data()
        logger.debug("原始章节数据: %s", raw_data)  # 添加调试日志

        # 检查每个知识点的标题
        for chapter in raw_data:
            logger.debug("章节[%s]包含知识点数量: %d",
                         chapter['chapter_data']['title'],
                         len(chapter['chapter_data']['knowledge']))
            knowledge = chapter['chapter_data']['knowledge']
            for kp in knowledge:
                if not kp['title'] or kp['title'] == '未命名知识点':
                    logger.warning("检测到空标题节点: ID=%s, 类型=%s", kp['id'], kp['type'])

        return sorted(raw_data, key=lambda x: x['chapter_data']['order'])

    def _generate_chapter_table(self, chapter):
        columns = [
            ("排名", 35),
            ("知识点ID", 70),
            ("类型", 45),
            ("标题", 200),
            ("教育权重", 55),
            ("课时", 40),
            ("大纲提及", 55),
            ("布鲁姆等级", 50)
        ]

        sorted_knowledge = sorted(
            chapter['knowledge'],
            key=lambda x: x['edu_pr'],
            reverse=True
        )

        table_data = []
        table_data.append([Paragraph(col[0], self.styles['TableHeader']) for col in columns])
        for idx, item in enumerate(sorted_knowledge, 1):
            row_style = self.styles['HighlightRow'] if idx <= 10 else self.styles['ChineseBody']

            # 深度处理标题字段
            title = item.get('title', '')
            if isinstance(title, str):
                # 清理所有空白字符并标准化
                clean_title = title.strip().replace('\n', ' ').replace('\t', ' ').strip()
                if not clean_title:
                    title_text = f"（无标题-{item['id']}）"
                    logger.warning("检测到无效标题: ID=%s, 类型=%s", item['id'], item['type'])
                else:
                    title_text = clean_title
            else:
                title_text = "（标题格式错误）"
                logger.error("标题字段类型错误: ID=%s, 类型=%s", item['id'], item['type'])

            title_para = Paragraph(title_text, style=row_style)
            row = [
                str(idx),
                item['id'],
                "知识点" if item['type'] == 'KnowledgePoint' else "子知识点",
                title_para,
                f"{item['edu_pr']:.8f}",
                str(item['class_hours']),
                str(item['syllabus_mentions']),
                f"L{item['bloom_level']}"
            ]
            table_data.append(row)

        table = Table(
            table_data,
            colWidths=[col[1] for col in columns],
            repeatRows=1,
            splitByRow=1,
            spaceBefore=12,
            spaceAfter=12
        )
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, 10), colors.HexColor('#FFF3CD')),
            ('TEXTCOLOR', (0, 1), (-1, 10), colors.HexColor('#856404')),
            ('LEADING', (0, 1), (-1, -1), 12),
            ('WORDWRAP', (3, 1), (3, -1), 'CJK')
        ]))
        return table

    def _generate_teaching_advice(self, chapter_data):
        """生成教学建议"""
        elements = []
        top10 = sorted(chapter_data['knowledge'],
                       key=lambda x: x['edu_pr'],
                       reverse=True)[:10]

        # 教学建议标题
        elements.append(Paragraph("<b>本章教学优化建议</b>", self.styles['Highlight']))
        elements.append(Spacer(1, 10))

        # 核心知识点分层建议
        high_impact = [k for k in top10 if k['edu_pr'] >= 0.15]
        if high_impact:
            elements.append(Paragraph(
                f"▪️ 检测到{len(high_impact)}个高影响力知识点（权重≥0.15）：",
                self.styles['ChineseBody']
            ))
            elements.append(Paragraph(
                "建议教学策略：<br/>"
                "1. 采用BOPPPS模组化教学设计<br/>"
                "2. 配套开发3-5个微课视频（8-15分钟/个）<br/>"
                "3. 设计分层测试题库（基础:综合:创新=3:5:2）<br/>"
                "4. 安排至少2次小组协作任务",
                self.styles['ChineseBody']
            ))
            elements.append(Spacer(1, 8))

        # 布鲁姆认知层级分析
        bloom_levels = {f'L{i}': 0 for i in range(1, 7)}
        for k in top10:
            level = f"L{k['bloom_level']}"
            bloom_levels[level] += 1

        elements.append(Paragraph(
            f"▪️ 认知层级分布：{', '.join([f'{k}({v})' for k, v in bloom_levels.items() if v > 0])}",
            self.styles['ChineseBody']
        ))

        if bloom_levels['L4'] + bloom_levels['L5'] + bloom_levels['L6'] >= 4:
            elements.append(Paragraph(
                "检测到高阶认知知识点占比≥40%，建议：<br/>"
                "1. 采用案例教学法（Case-Based Learning）<br/>"
                "2. 设计跨章节综合实践项目<br/>"
                "3. 开展课堂辩论/模拟答辩活动",
                self.styles['ChineseBody']
            ))
        else:
            elements.append(Paragraph(
                "建议加强高阶认知训练：<br/>"
                "1. 在作业中增加分析类题目（不少于30%）<br/>"
                "2. 设计1-2个翻转课堂单元",
                self.styles['ChineseBody']
            ))
        elements.append(Spacer(1, 8))

        # 课时分配分析
        total_hours = sum(k['class_hours'] for k in top10)
        avg_hours = total_hours / len(top10) if top10 else 0
        imbalance = [k for k in top10 if k['class_hours'] > 1.8 * avg_hours]

        if imbalance:
            elements.append(Paragraph(
                f"▪️ 检测到{len(imbalance)}个知识点课时分配超出均值80%：",
                self.styles['ChineseBody']
            ))
            elements.append(Paragraph(
                "优化建议：<br/>"
                "1. 拆分复杂知识点为2-3个子单元<br/>"
                "2. 开发配套自主学习材料<br/>"
                "3. 调整实验课与理论课比例至1:1.5",
                self.styles['ChineseBody']
            ))
        else:
            elements.append(Paragraph(
                "▪️ 课时分配相对均衡（波动<80%）",
                self.styles['ChineseBody']
            ))

        elements.append(Spacer(1, 15))
        return elements

    def generate_pdf(self, filename: str = "syllabus_analysis.pdf"):
        """生成教学大纲"""
        # 获取数据
        chapters = self.fetch_chapter_data()

        # 创建文档
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        # ==================== 封面页 ====================
        elements.append(Paragraph("课程核心知识点分析报告", self.styles['ChapterTitle']))
        elements.append(Spacer(1, 40))

        cover_content = [
            Paragraph("——基于教育权重动态标注", self.styles['ChineseBody']),
            Spacer(1, 20),
            Paragraph("<b>课程基本信息</b>", self.styles['ChineseBody']),
            Paragraph(f"课程名称：大数据技术", self.styles['ChineseBody']),
            Paragraph(f"学分：3 | 学期：2024春季", self.styles['ChineseBody']),
            Spacer(1, 15),
            Paragraph("<b>分析说明</b>", self.styles['ChineseBody']),
            Paragraph("本报告通过教育PageRank算法，综合以下指标动态标注核心知识点：", self.styles['ChineseBody']),
            Paragraph("• 课时分配权重", self.styles['ChineseBody']),
            Paragraph("• 大纲提及频次", self.styles['ChineseBody']),
            Paragraph("• 布鲁姆认知层级", self.styles['ChineseBody']),
            Spacer(1, 15),
            Paragraph("<b>教学应用特征</b>", self.styles['ChineseBody']),
            Paragraph("• 多维度指标融合：课时/大纲/认知层级", self.styles['ChineseBody']),
            Paragraph("• 动态权重计算：教育PageRank算法", self.styles['ChineseBody']),
            Paragraph("• 智能建议生成：教学策略推荐", self.styles['ChineseBody']),
            Spacer(1, 15),
            Paragraph("<i>生成时间：2025年6月</i>", self.styles['ChineseBody']),
            Paragraph("<i>数据来源：Neo4j知识图谱</i>", self.styles['ChineseBody']),
            Paragraph("<i>注：本报告数据来自知识图谱实体关系分析，支持教学决策优化</i>",
                      self.styles['ChineseBody'])
        ]
        elements.extend(cover_content)
        elements.append(PageBreak())

        # ==================== 章节内容 ====================
        for chapter in chapters:
            # 章节标题
            title = f"第{chapter['chapter_data']['order']}章 {chapter['chapter_data']['title']}"
            elements.append(Paragraph(title, self.styles['ChapterTitle']))

            # 章节概览
            overview = [
                f"▪ 总课时：{chapter['chapter_data']['class_hours']}小时",
                f"▪ 知识单元：{len(chapter['chapter_data']['knowledge'])}个",
                f"▪ 最高权重：{max(k['edu_pr'] for k in chapter['chapter_data']['knowledge']) if chapter['chapter_data']['knowledge'] else 0:.4f}"
            ]
            for text in overview:
                elements.append(Paragraph(text, self.styles['ChineseBody']))
            elements.append(Spacer(1, 12))

            # 全量知识点表格
            if chapter['chapter_data']['knowledge']:
                table = self._generate_chapter_table(chapter['chapter_data'])
                elements.append(table)
                elements.append(Spacer(1, 15))

                # 生成教学建议
                elements += self._generate_teaching_advice(chapter['chapter_data'])
            else:
                elements.append(Paragraph("本章暂无有效知识点数据", self.styles['ChineseBody']))

            elements.append(PageBreak())

        # 生成PDF
        doc.build(elements)
        logger.info(f"教学大纲分析报告已生成：{filename}")


if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "123456789"

    try:
        generator = EnhancedSyllabusGenerator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        generator.generate_pdf()
    except Exception as e:
        logger.error(f"生成失败: {e}")