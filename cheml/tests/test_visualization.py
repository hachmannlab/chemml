import unittest
import os
import shutil, tempfile
import numpy as np
import pandas as pd
from cheml.visualization import scatter2D, hist
from cheml.visualization import decorator
from cheml.visualization import SavePlot

# dummy data
x = pd.DataFrame(np.arange(0.0, 1.0, 0.01))
y = pd.DataFrame(np.sin(2*np.pi*x))



class Testvisualization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_scatter2D(self):
        sc = scatter2D('r', linestyle='--')
        fig = sc.plot(x,y,0,0)
        fig.show()

    def test_hist(self):
        hg = hist(20,'g',{'normed':True})
        fig = hg.plot(y,0)
        fig.show()

    def test_decorator(self):
        hg = hist(20,'g',{'normed':True})
        fig = hg.plot(y,0)
        dec = decorator('histogram',xlabel='sin', ylabel='sin%', xlim=(4,None), ylim=(0,None),
                grid=True, grid_color='g', grid_linestyle=':', grid_linewidth=0.5)
        fig = dec.fit(fig)
        dec.matplotlib_font()
        fig.show()

    def test_SavePlot(self):
        sc = scatter2D('r', linestyle='--')
        fig = sc.plot(x,y,0,0)
        sp = SavePlot('Sin', os.path.join(self.test_dir,'plots'), 'eps', {'facecolor': 'w', 'dpi': 100, 'pad_inches': 0.1, 'bbox_inches': 'tight'})
        sp.save(fig)


if __name__== '__main__':
    unittest.main()

