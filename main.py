import os
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unittest

class CosineSimilarity(object):
    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_text(content):
        # Remove HTML tags and decode HTML entities
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        content = html.unescape(content)
        return content.strip()  # Strip leading/trailing whitespace

    def main(self):
        # Extract text from contents
        text_x = self.extract_text(self.s1)
        text_y = self.extract_text(self.s2)

        # Check if either document is empty
        if not text_x or not text_y:
            return 0.0

        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text_x, text_y])

        # Compute Cosine Similarity
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return sim_matrix[0][0]


def process_files(file_pairs, outputfile):
    # Ensure the directory exists
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
    # Use os.path.join to create file paths
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


    def process_files(file_pairs, outputfile):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)

        with open(outputfile, 'a+', encoding='utf-8') as answer:
            for file1, file2 in file_pairs:
                print(f"Checking files: {file1}, {file2}")  # Debug print to check paths
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


    # Run unit tests
    unittest.main()
