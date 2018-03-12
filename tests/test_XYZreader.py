import unittest
from cheml.initialization import XYZreader


class TestXYZreader(unittest.TestCase):
    def test_string(self):
        reader= XYZreader(path_pattern='[2-3]/*.opt.xyz',
                                           path_root = 'data/',
                                           reader='manual',
                                           skip_lines=[0, 0])
        molecules = reader.read()
        self.assertEqual(len(molecules) , 2)
        self.assertEqual(len(molecules[1]['mol']), 38)

    def test_list(self):
        reader = XYZreader(path_pattern = ['[2-3]/*.opt.xyz', '[1-2][1-2]/*.opt.xyz'],
                                           path_root = 'data/',
                                           reader='manual',
                                           skip_lines = [0, 0])
        molecules = reader.read()
        self.assertEqual(len(molecules) , 4)
        self.assertEqual(len(molecules[1]['mol']), 38)
        self.assertEqual(len(molecules[4]['mol']), 41)


if __name__== '__main__':
    unittest.main()



