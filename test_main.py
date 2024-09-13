import unittest
from unittest.mock import patch, mock_open
import io
import os
from main import CosineSimilarity, process_files

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

    def test_partial_similarity(self):
        sim = CosineSimilarity("这是一个测试", "这是另一个测试")
        result = sim.main()
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)
        print(f"Partial similarity test result: {result * 100:.2f}%")

class TestProcessFiles(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data="测试内容")
    @patch('os.path.isfile', return_value=True)
    @patch('os.makedirs')
    def test_process_files(self, mock_makedirs, mock_isfile, mock_open):
        file_pairs = [('file1.txt', 'file2.txt')]
        outputfile = 'output.txt'

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            process_files(file_pairs, outputfile)
            self.assertIn("file1.txt 和 file2.txt 相似度:", mock_stdout.getvalue())
            print(mock_stdout.getvalue())

    @patch('builtins.open', new_callable=mock_open, read_data="测试内容")
    @patch('os.path.isfile', side_effect=[True, False])
    @patch('os.makedirs')
    def test_process_files_missing_file(self, mock_makedirs, mock_isfile, mock_open):
        file_pairs = [('file1.txt', 'file2.txt')]
        outputfile = 'output.txt'

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            process_files(file_pairs, outputfile)
            self.assertIn("Error: One or both input files 'file1.txt' and 'file2.txt' do not exist.", mock_stdout.getvalue())
            print(mock_stdout.getvalue())

if __name__ == '__main__':
    unittest.main()
