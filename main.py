import os
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unittest
import cProfile
import pstats
import io

class CosineSimilarity(object):
    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_text(content):
        # 删除html标签并解码html实体
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        content = html.unescape(content)
        return content.strip()

    def main(self):
        # 从内容中提取文本
        text_x = self.extract_text(self.s1)
        text_y = self.extract_text(self.s2)

        # 检查文本是否为空
        if not text_x or not text_y:
            return 0.0

        # 将普通文本转化成向量表述
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text_x, text_y])

        # 计算余弦相似度
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return sim_matrix[0][0]


def process_files(file_pairs, outputfile):
    # 确保字典存在
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)

    with open(outputfile, 'a+', encoding='utf-8') as answer:
        for file1, file2 in file_pairs:
            if not os.path.isfile(file1) or not os.path.isfile(file2):
                print(f"Error: One or both input files '{file1}' and '{file2}' do not exist.")
                continue

            try:
                with open(file1, 'r', encoding='utf-8') as f1, \
                        open(file2, 'r', encoding='utf-8') as f2:

                    content1 = f1.read()
                    content2 = f2.read()
                    similarity = CosineSimilarity(content1, content2)
                    similarity_score = similarity.main()
                    result = f'{file1} 和 {file2} 相似度: {similarity_score * 100:.2f}%\n'
                    answer.writelines(result)
                    print(result)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except IOError as e:
                print(f"File I/O error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with files '{file1}' and '{file2}': {e}")


class TestCosineSimilarity(unittest.TestCase):
    def test_empty_file(self):
        sim = CosineSimilarity("", "")
        self.assertEqual(sim.main(), 0.0)

    def test_no_common_text(self):
        sim = CosineSimilarity("这是一个测试", "完全不同的内容")
        self.assertEqual(sim.main(), 0.0)

    def test_identical_files(self):
        sim = CosineSimilarity("内容完全相同", "内容完全相同")
        self.assertEqual(sim.main(), 1.0)


if __name__ == '__main__':

    # 创建性能分析器对象
    profiler = cProfile.Profile()
    profiler.enable()  # 启动性能分析

    # 执行主要程序逻辑
    file_pairs = [
        (os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig.txt'),
         os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig_0.8_add.txt')),
        (os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig.txt'),
         os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig_0.8_del.txt')),
        (os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig.txt'),
         os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig_0.8_dis_1.txt')),
        (os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig.txt'),
         os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig_0.8_dis_10.txt')),
        (os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig.txt'),
         os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'orig_0.8_dis_15.txt')),
    ]
    outputfile = os.path.join('D://', 'pythonProject1', 'Software_Engineer', 'output.txt')
    process_files(file_pairs, outputfile)

    profiler.disable()  # 停止性能分析

    # 生成性能分析报告
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('time')
    stats.print_stats()

    # 将分析结果写入文件
    with open('profile_report.txt', 'w') as f:
        f.write(stream.getvalue())



