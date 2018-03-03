import unittest
import numpy as np
from cheml.initialization import XYZreader


class TestXYZreader(unittest.TestCase):
    def test_string(self):
        reader= XYZreader(path_pattern='[1-2]/*.opt.xyz',
                                           path_root = 'benchmarks/RI_project/PI_R2_xyz',
                                           reader='manual',
                                           skip_lines=[0, 0])
        molecules = reader.read()
        self.assertEqual(len(molecules) , 2)
        self.assertEqual(len(molecules[1]['mol']), 0)
        print molecules

    def test_list(self):
        reader = XYZreader(path_pattern = ['[1-2]/*.opt.xyz', '[11-12]/*.opt.xyz'],
                                           path_root = 'benchmarks/RI_project/PI_R2_xyz',
                                           reader='manual',
                                           skip_lines = [0, 0])
        molecules = reader.read()
        self.assertEqual(len(molecules) , 4)
        self.assertEqual(len(molecules[1]['mol']), 0)


if __name__== '__main__':
    unittest.main()



from qml.representations import *
from cheml.initialization import XYZreader
from cheml.chem import Coulomb_Matrix
import numpy as np
import pandas as pd

m = XYZreader(path_pattern='2/*.opt.xyz',path_root = 'benchmarks/RI_project/PI_R2_xyz',reader='manual',skip_lines=[0, 0])
ms = m.read()
n = ms[1]['mol'][:,0]
c=ms[1]['mol'][:,1:]
mq = generate_coulomb_matrix(n, c, len(c), sorting='unsorted')
cml = Coulomb_Matrix('U')
mc = cml.represent(ms)
mc = mc.values.reshape((741,))

n = [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,
        6.,  6.,  6.,  6.,  6.,  6.,  6.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]
a = [ 'C',  'C',  'C',  'C',  'C',  'C',  'C',  'C',  'C',  'C',  'C',  'C',  'C',
        'C',  'C',  'C',  'C',  'C',  'C',  'C',  'H',  'H',  'H',  'H',  'H',  'H',
        'H',  'H',  'H',  'H',  'H',  'H',  'H',  'H',  'H',  'H',  'H',  'H']
c = [[-1.21686,  0.41556, -2.14005],
       [-0.03633,  0.14887, -2.83743],
       [ 1.1869 ,  0.10322, -2.15699],
       [ 1.23353,  0.3213 , -0.76963],
       [ 0.03873,  0.5883 , -0.07534],
       [-1.17814,  0.63391, -0.76049],
       [ 2.55433,  0.29419, -0.03289],
       [ 3.24563, -1.04668, -0.13228],
       [ 4.51597, -1.14995, -0.72401],
       [ 5.15945, -2.38987, -0.81095],
       [ 4.53781, -3.54757, -0.31393],
       [ 3.26781, -3.44413,  0.27801],
       [ 2.63028, -2.2023 ,  0.37608],
       [ 5.20479, -4.89524, -0.46487],
       [ 6.16466, -5.22891,  0.65193],
       [ 7.52531, -5.45861,  0.37936],
       [ 8.40054, -5.81729,  1.41123],
       [ 7.92602, -5.95113,  2.71866],
       [ 6.57638, -5.72316,  2.99886],
       [ 5.69785, -5.36205,  1.97215],
       [-2.16142,  0.45443, -2.66803],
       [-0.06698, -0.01812, -3.90628],
       [ 2.0959 , -0.09908, -2.70944],
       [ 0.05246,  0.76498,  0.99251],
       [-2.09556,  0.843  , -0.22181],
       [ 2.40516,  0.52508,  1.0439 ],
       [ 3.19559,  1.09681, -0.45712],
       [ 5.0064 , -0.27006, -1.12325],
       [ 6.13268, -2.45282, -1.28132],
       [ 2.77316, -4.32634,  0.66446],
       [ 1.65413, -2.13983,  0.84071],
       [ 5.70426, -4.93161, -1.45843],
       [ 4.43152, -5.69284, -0.49008],
       [ 7.9055 , -5.37051, -0.63068],
       [ 9.44583, -5.99912,  1.19626],
       [ 8.60303, -6.237  ,  3.51418],
       [ 6.2094 , -5.83387,  4.0113 ],
       [ 4.65206, -5.20256,  2.20156]]

from molml.features import CoulombMatrix
feat = CoulombMatrix()
m=(a,c)
feat.fit([m])
feat.transform([m])
a = ['N', 'H', 'H']
c = [[1.464, 0.707, 1.056],
 [0.878, 1.218, 0.498],
 [2.319, 1.126, 0.952]]

c = np.array([[1.464, 0.707, 1.056],
                        [0.878, 1.218, 0.498],
                        [2.319, 1.126, 0.952]])

# Oxygen, Hydrogen, Hydrogen
n = np.array([8, 1, 1])

nu = n.reshape((3,1))
nu = np.append(nu,c,1)
cml = Coulomb_Matrix('E')
mc = cml.represent(pd.DataFrame(nu))

