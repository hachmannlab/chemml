import pytest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from chemml.visualization import scatter2D, hist
from chemml.visualization import decorator
from chemml.visualization import SavePlot


@pytest.fixture()
def dummy_data():
    x = pd.DataFrame(np.arange(0.0, 1.0, 0.01))
    y = pd.DataFrame(np.sin(2 * np.pi * x))
    return (x, y)


@pytest.fixture()
def setup_teardown():
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    # return test directory to save figures
    yield test_dir
    # Remove the directory after the test
    shutil.rmtree(test_dir)


def test_scatter2D(dummy_data):
    sc = scatter2D('r', linestyle='--')
    x, y = dummy_data
    fig = sc.plot(x, y, 0, 0)
    # fig.show()


def test_hist(dummy_data):
    hg = hist(20, 'g', {'density': True})
    x, y = dummy_data
    fig = hg.plot(y, 0)
    # fig.show()


def test_decorator(dummy_data):
    hg = hist(20, 'g', {'density': True})
    x, y = dummy_data
    fig = hg.plot(y, 0)
    dec = decorator(
        'histogram',
        xlabel='sin',
        ylabel='sin%',
        xlim=(4, None),
        ylim=(0, None),
        grid=True,
        grid_color='g',
        grid_linestyle=':',
        grid_linewidth=0.5)
    fig = dec.fit(fig)
    dec.matplotlib_font()
    # fig.show()


def test_SavePlot(dummy_data, setup_teardown):
    sc = scatter2D('r', linestyle='--')
    x, y = dummy_data
    fig = sc.plot(x, y, 0, 0)
    sp = SavePlot('Sin', os.path.join(setup_teardown, 'plots'), 'png', {
        'facecolor': 'w',
        'dpi': 100,
        'pad_inches': 0.1,
        'bbox_inches': 'tight'
    })
    sp.save(fig)
