import unittest
import numpy as np

from cheml.chem import BagofBonds


# Oxygen, Hydrogen, Hydrogen
n = np.array([8, 1, 1])
c = np.array([[1.464, 0.707, 1.056],
             [0.878, 1.218, 0.498],
             [2.319, 1.126, 0.952]])

nc = n.reshape((3,1))
nc = np.append(nc,c,1)

class TestBagofBonds(unittest.TestCase):
    def test_h2o(self):
        bob = BagofBonds(const=1.0)
        h2o = bob.represent(np.array([nc]))

        self.assertEqual(h2o.shape, (1, 6))
        a = np.array([[  0.66066557,   0.5       ,   0.5       ,   8.3593106 ,
          8.35237809,  73.51669472]])
        self.assertAlmostEqual(a[0][0], h2o.values[0][0])
        self.assertAlmostEqual(a[0][1], h2o.values[0][1])
        self.assertAlmostEqual(a[0][-1], h2o.values[0][-1])




if __name__== '__main__':
    unittest.main()

