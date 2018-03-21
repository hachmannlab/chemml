import unittest
import numpy as np
from cheml.chem import CoulombMatrix

# Oxygen, Hydrogen, Hydrogen
n = np.array([8, 1, 1])
c = np.array([[1.464, 0.707, 1.056],
             [0.878, 1.218, 0.498],
             [2.319, 1.126, 0.952]])
nc = n.reshape((3,1))
nc = np.append(nc,c,1)

class TestCoulombMatrix(unittest.TestCase):
    def test_UM(self):
        cm = CoulombMatrix('UM')
        h2o = cm.represent(np.array([nc]))

        self.assertEqual(h2o.shape, (1, cm.max_n_atoms**2))
        a = np.array([[ 73.51669472, 8.3593106 ,8.35237809, 8.3593106,
          0.5, 0.66066557, 8.35237809, 0.66066557, 0.5]])
        self.assertAlmostEqual(a[0][0], h2o.values[0][0])
        self.assertAlmostEqual(a[0][1], h2o.values[0][1])
        self.assertAlmostEqual(a[0][-1], h2o.values[0][-1])

    def test_UT(self):
        cm = CoulombMatrix('UT')
        h2o = cm.represent(np.array([nc]))

        self.assertEqual(h2o.shape, (1, cm.max_n_atoms*(cm.max_n_atoms+1)/2))
        a = np.array([[ 73.51669472, 8.3593106, 0.5, 8.35237809,
          0.66066557, 0.5]])
        self.assertAlmostEqual(a[0][0], h2o.values[0][0])
        self.assertAlmostEqual(a[0][1], h2o.values[0][1])
        self.assertAlmostEqual(a[0][-1], h2o.values[0][-1])

    def test_E(self):
        cm = CoulombMatrix('E')
        h2o = cm.represent(np.array([nc]))

        self.assertEqual(h2o.shape, (1, cm.max_n_atoms))
        a = np.array([[ 75.39770052,  -0.16066482,  -0.72034098]])
        self.assertAlmostEqual(a[0][0], h2o.values[0][0])
        self.assertAlmostEqual(a[0][1], h2o.values[0][1])
        self.assertAlmostEqual(a[0][-1], h2o.values[0][-1])


    def test_SC(self):
        cm = CoulombMatrix('SC')
        h2o = cm.represent(np.array([nc]))

        self.assertEqual(h2o.shape, (1, cm.max_n_atoms*(cm.max_n_atoms+1)/2))
        a = np.array([[ 73.51669472,   8.3593106 ,   0.5       ,   8.35237809,
          0.66066557,   0.5       ]])
        self.assertAlmostEqual(a[0][0], h2o.values[0][0])
        self.assertAlmostEqual(a[0][1], h2o.values[0][1])
        self.assertAlmostEqual(a[0][-1], h2o.values[0][-1])

    def test_RC(self):
        cm = CoulombMatrix('RC')
        h2o = cm.represent(np.array([nc]))

        self.assertEqual(h2o.shape, (1, cm.nPerm*cm.max_n_atoms*(cm.max_n_atoms+1)/2))
        a = np.array([[  0.5,   8.35237809,  73.51669472,   0.66066557,
          8.3593106 ,   0.5       ,  73.51669472,   8.35237809,
          0.5       ,   8.3593106 ,   0.66066557,   0.5       ,
          0.5       ,   8.3593106 ,  73.51669472,   0.66066557,
          8.35237809,   0.5       ]])




if __name__== '__main__':
    unittest.main()

