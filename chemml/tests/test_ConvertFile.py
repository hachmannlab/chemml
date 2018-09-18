import unittest
from cheml.initialization import XYZreader
from cheml.initialization import ConvertFile


class TestConvertFile(unittest.TestCase):
    def test_xyz_cml(self):
        reader= XYZreader(path_pattern='[2-3]/*.opt.xyz',
                                           path_root = 'data/',
                                           reader='manual',
                                           skip_lines=[0, 0])
        molecules = reader.read()
        model = ConvertFile(molecules, 'xyz', 'cml')
        # commented out since this command generates some files
        # s = model.convert()
        # self.assertEqual(len(s) , 2)
        # self.assertEqual(s[1]['file'][-3:], 'cml')

if __name__== '__main__':
    unittest.main()



