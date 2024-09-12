import unittest
import os
import tempfile
from unittest.mock import patch, mock_open
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


class TestProcessFiles(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.isfile", return_value=True)
    def test_process_files(self, mock_isfile, mock_open):
        file_pairs = [("file1.txt", "file2.txt")]
        outputfile = "output.txt"

        # 模拟文件内容
        mock_open().read.side_effect = [
            "This is the content of file1.",
            "This is the content of file2."
        ]

        # 不再调用 os.makedirs
        with patch("os.makedirs") as mock_makedirs:
            process_files(file_pairs, outputfile)
            mock_makedirs.assert_not_called()

        # 验证文件是否正确打开并写入内容
        mock_open.assert_called_with(outputfile, 'a+', encoding='utf-8')
        handle = mock_open()
        handle.writelines.assert_called()

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.isfile", return_value=False)
    def test_process_files_missing_file(self, mock_isfile, mock_open, mock_makedirs):
        file_pairs = [("file1.txt", "file2.txt")]
        outputfile = "output.txt"

        with self.assertLogs(level='ERROR') as log:
            process_files(file_pairs, outputfile)
            # 如果函数中没有日志记录，可以直接检查print输出
            self.assertTrue(any("Error: One or both input files" in message for message in log.output))


if __name__ == '__main__':
    unittest.main()
