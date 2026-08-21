import pandas as pd
import os
import numpy as np

# 配置路径
input_dir = r'D:\Pycharm\LunWen\AKnowledgeGrape\O-Data'
output_dir = r'D:\Pycharm\LunWen\AKnowledgeGrape\D-Data'

# 需要转换的文件列表
excel_files = [
    'chapters.xlsx',
    'knowledge_points.xlsx',
    'labs.xlsx',
    'sub_knowledge_points.xlsx'
]

# 布鲁姆层级映射
bloom_mapping = {
    "记忆": 1,
    "理解": 2,
    "应用": 3,
    "分析": 4,
    "评价": 5,
    "创造": 6
}


def get_class_hours_from_chapters(chapter_id, chapters_df):
    """从chapters_df中获取class_hours"""
    try:
        chapter_id = str(chapter_id)
        chapters_df['chapter_id'] = chapters_df['chapter_id'].astype(str)
        return chapters_df.loc[chapters_df['chapter_id'] == chapter_id, 'class_hours'].values[0]
    except Exception as e:
        print(f"获取class_hours失败: {e}")
        return None


def get_class_hours_from_knowledge_points(kp_id, kp_df):
    """从knowledge_points_df中获取class_hours"""
    try:
        kp_id = str(kp_id)
        kp_df['kp_id'] = kp_df['kp_id'].astype(str)
        return kp_df.loc[kp_df['kp_id'] == kp_id, 'class_hours'].values[0]
    except Exception as e:
        print(f"获取class_hours失败: {e}")
        return None


def excel_to_csv_converter():
    """Excel转CSV转换器，保留原始数据结构并进行特定处理"""
    os.makedirs(output_dir, exist_ok=True)

    chapters_df = None
    knowledge_points_df = None

    for file in excel_files:
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace('.xlsx', '.csv'))

            print(f'正在处理: {os.path.basename(input_path)}')
            df = pd.read_excel(input_path, engine='openpyxl')

            if file == 'chapters.xlsx':
                if 'chapter_id' not in df.columns:
                    df['chapter_id'] = df.index + 1
                chapters_df = df.copy()
                print("chapters_df:")
                print(chapters_df.head())

            elif file == 'knowledge_points.xlsx':
                if 'bloom' in df.columns:
                    df = df.rename(columns={'bloom': 'bloom_level'})
                if 'bloom_level' in df.columns:
                    df['bloom_level'] = df['bloom_level'].map(bloom_mapping)
                if 'chapter_id' in df.columns and chapters_df is not None:
                    df['class_hours'] = df['chapter_id'].apply(
                        lambda x: get_class_hours_from_chapters(x, chapters_df))
                if 'bloom_level' in df.columns:
                    df['syllabus_mentions'] = df['bloom_level'].apply(
                        lambda x: np.random.randint(1, 11 - x) if x <= 10 else 1)
                knowledge_points_df = df.copy()
                print("knowledge_points_df:")
                print(knowledge_points_df.head())


            elif file == 'sub_knowledge_points.xlsx':
                if 'difficulty_level' in df.columns:
                    df = df.rename(columns={'difficulty_level': 'bloom_level'})

                if 'kp_id' in df.columns and knowledge_points_df is not None:
                    df['kp_id'] = df['kp_id'].astype(str)
                    knowledge_points_df['kp_id'] = knowledge_points_df['kp_id'].astype(str)
                    kp_to_class_hours = knowledge_points_df.set_index('kp_id')['class_hours'].to_dict()
                    df['class_hours'] = df['kp_id'].map(kp_to_class_hours)

                if 'kp_id' in df.columns and knowledge_points_df is not None:
                    df['kp_id'] = df['kp_id'].astype(str)
                    knowledge_points_df['kp_id'] = knowledge_points_df['kp_id'].astype(str)
                    kp_to_chapter = knowledge_points_df.set_index('kp_id')['chapter_id'].to_dict()
                    df['chapter_id'] = df['kp_id'].map(kp_to_chapter)
                    df['chapter_id'] = df['chapter_id'].astype('object')
                if 'bloom_level' in df.columns:
                    df['syllabus_mentions'] = df['bloom_level'].apply(
                        lambda x: np.random.randint(1, 6 - x // 2) if x <= 10 else 1)
                print("sub_knowledge_points_df:")
                print(df.head())

            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f'成功生成: {os.path.basename(output_path)}\n')

        except Exception as e:
            print(f'处理 {file} 时发生错误: {str(e)}\n')


if __name__ == '__main__':
    excel_to_csv_converter()
    print('转换完成！检查输出目录：' + output_dir)