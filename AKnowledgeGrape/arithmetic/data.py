import requests
import pandas as pd
from lxml import etree
import time

# 存储课程实体与关系数据
course_entities = []
knowledge_relations = []

class MOOCCourseCrawler:
    def __init__(self, course_url):
        self.course_url = course_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        }

    def crawl_course_info(self):
        """爬取课程基本信息（对应MOOC平台“课程介绍”模块）"""
        response = requests.get(self.course_url, headers=self.headers)
        response.encoding = "utf-8"
        html = etree.HTML(response.text)

        # 1. 提取课程核心信息（课程ID、名称、院校、教师、学分等）
        course_id = self.course_url.split("/")[-1].split("?")[0]
        course_name = html.xpath("//h1[@class='f-ib f-24 f-bold']/text()")
        course_name = course_name[0].strip() if course_name else "未获取到课程名称"

        school = html.xpath("//div[@class='t2']/a/text()")
        school = school[0].strip() if school else "未知院校"

        teachers = html.xpath("//div[@class='teacher']/a/text()")
        teacher_str = ",".join(teachers) if teachers else "未知教师"

        credit = html.xpath("//div[@class='info_term']/span[3]/text()")
        credit = credit[0].split("：")[-1].strip() if credit else "无学分"

        description = html.xpath("//div[@class='course-description f-16']/text()")
        description = description[0].strip() if description else "暂无课程介绍"

        # 封装为“课程实体”（对应MOOC平台的课程卡片信息）
        course_entity = {
            "实体类型": "课程",
            "课程ID": course_id,
            "课程名称": course_name,
            "开设院校": school,
            "授课教师": teacher_str,
            "学分要求": credit,
            "课程简介": description
        }
        course_entities.append(course_entity)
        print(f"已提取课程基本信息：《{course_name}》")

    def crawl_chapter_knowledge(self):
        """爬取章节与知识点结构（对应MOOC平台“课程大纲”模块）"""
        response = requests.get(self.course_url, headers=self.headers)
        response.encoding = "utf-8"
        html = etree.HTML(response.text)

        # 2. 提取章节列表（对应MOOC平台的“第X周/第X章”）
        chapters = html.xpath("//div[@class='j-chapter']")  # 适配新页面结构
        for chapter_idx, chapter in enumerate(chapters):
            chapter_title = chapter.xpath(".//h3/text()")
            chapter_title = chapter_title[0].strip() if chapter_title else f"章节{chapter_idx+1}"

            # 封装为“章节实体”
            chapter_entity = {
                "实体类型": "章节",
                "章节ID": f"{course_id}_chap{chapter_idx}",
                "章节名称": chapter_title,
                "所属课程ID": course_id,
                "章节顺序": chapter_idx + 1
            }
            course_entities.append(chapter_entity)
            print(f"  ├─ 章节 {chapter_idx+1}：{chapter_title}")

            # 3. 提取知识点（对应MOOC平台的“课时”）
            lessons = chapter.xpath(".//div[@class='u-lesson']")  # 适配新页面结构
            for lesson_idx, lesson in enumerate(lessons):
                kp_title = lesson.xpath(".//span[@class='txt']/text()")
                kp_title = kp_title[0].strip() if kp_title else f"知识点{lesson_idx+1}"

                # 封装为“知识点实体”
                kp_entity = {
                    "实体类型": "知识点",
                    "知识点ID": f"{course_id}_chap{chapter_idx}_kp{lesson_idx}",
                    "知识点名称": kp_title,
                    "所属章节ID": chapter_entity["章节ID"],
                    "知识点顺序": lesson_idx + 1
                }
                course_entities.append(kp_entity)
                print(f"  │  ├─ 知识点 {lesson_idx+1}：{kp_title}")

                # 4. 提取子知识点（对应MOOC平台的“子课时/扩展内容”）
                sub_kps = lesson.xpath(".//div[@class='unit']")
                for sub_idx, sub_kp in enumerate(sub_kps):
                    sub_kp_title = sub_kp.xpath(".//span[@class='txt']/text()")
                    sub_kp_title = sub_kp_title[0].strip() if sub_kp_title else f"子知识点{sub_idx+1}"

                    # 封装为“子知识点实体”
                    sub_kp_entity = {
                        "实体类型": "子知识点",
                        "子知识点ID": f"{kp_entity['知识点ID']}_sub{sub_idx}",
                        "子知识点名称": sub_kp_title,
                        "所属知识点ID": kp_entity["知识点ID"],
                        "子知识点顺序": sub_idx + 1
                    }
                    course_entities.append(sub_kp_entity)
                    print(f"  │  │  └─ 子知识点 {sub_idx+1}：{sub_kp_title}")

                    # 记录“层级关系”（知识点→子知识点）
                    knowledge_relations.append({
                        "关系类型": "包含",
                        "来源ID": kp_entity["知识点ID"],
                        "目标ID": sub_kp_entity["子知识点ID"],
                        "关系权重": 1.0
                    })
                    knowledge_relations.append({
                        "关系类型": "被包含",
                        "来源ID": sub_kp_entity["子知识点ID"],
                        "目标ID": kp_entity["知识点ID"],
                        "关系权重": 1.0
                    })

            # 记录“章节顺序关系”（章节N→章节N+1）
            if chapter_idx < len(chapters) - 1:
                next_chapter_id = f"{course_id}_chap{chapter_idx + 1}"
                knowledge_relations.append({
                    "关系类型": "后续章节",
                    "来源ID": chapter_entity["章节ID"],
                    "目标ID": next_chapter_id,
                    "关系权重": 1.0
                })

    def crawl_prerequisite(self):
        """爬取先修课程关系（对应MOOC平台“课程须知”模块）"""
        response = requests.get(self.course_url, headers=self.headers)
        response.encoding = "utf-8"
        html = etree.HTML(response.text)

        # 5. 提取先修课程
        prereq_text = html.xpath("//div[contains(text(), '先修课程')]/following-sibling::div/text()")
        if prereq_text:
            for prereq in prereq_text[0].split("、"):
                prereq = prereq.strip()
                if prereq:
                    knowledge_relations.append({
                        "关系类型": "先修要求",
                        "来源ID": course_id,
                        "目标ID": prereq,
                        "关系权重": 2.0
                    })
            print(f"已提取先修课程关系：{prereq_text[0]}")

    def save_to_csv(self):
        """保存数据为MOOC风格的CSV（便于后续分析）"""
        if course_entities:
            entity_df = pd.DataFrame(course_entities)
            entity_df.to_csv("MOOC课程实体.csv", index=False, encoding="utf-8-sig")
            print(f"\n已保存 {len(course_entities)} 条课程实体数据到“MOOC课程实体.csv”")
        
        if knowledge_relations:
            relation_df = pd.DataFrame(knowledge_relations)
            relation_df.to_csv("MOOC知识关系.csv", index=False, encoding="utf-8-sig")
            print(f"已保存 {len(knowledge_relations)} 条知识关系数据到“MOOC知识关系.csv”")


if __name__ == "__main__":
    # 目标课程URL（替换为你想爬取的MOOC课程链接）
    target_course_url = "https://www.icourse163.org/learn/PKU-1002534001?tid=1475372482#/learn/announce"
    crawler = MOOCCourseCrawler(target_course_url)
    
    print("===== 开始爬取MOOC课程数据 =====")
    crawler.crawl_course_info()
    crawler.crawl_chapter_knowledge()
    crawler.crawl_prerequisite()
    crawler.save_to_csv()
    print("===== 爬取完成 =====")