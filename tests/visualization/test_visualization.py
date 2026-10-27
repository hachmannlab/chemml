import pytest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from chemml.visualization import scatter2D, hist
from chemml.visualization import decorator
from chemml.visualization import SavePlot
from chemml.visualization import ClassificationPlots


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


@pytest.fixture()
def binary_classification_data():
    """Fixture for binary classification test data"""
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.85, 0.6, 0.95, 0.15])
    return y_true, y_pred, y_pred_proba


@pytest.fixture()
def multiclass_classification_data():
    """Fixture for multiclass classification test data"""
    np.random.seed(42)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0])
    # One-vs-rest probability matrix for multiclass
    y_pred_proba = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.4, 0.3],
        [0.9, 0.05, 0.05],
        [0.2, 0.3, 0.5],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1]
    ])
    return y_true, y_pred, y_pred_proba


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


def test_ClassificationPlots_confusion_matrix(binary_classification_data):
    """Test confusion matrix plot generation"""
    y_true, y_pred, y_pred_proba = binary_classification_data
    cp = ClassificationPlots(plot_type='confusion_matrix')
    fig = cp.plot(y_true, y_pred)
    assert fig is not None
    assert hasattr(fig, 'show')  # Check it's a matplotlib figure


def test_ClassificationPlots_roc(binary_classification_data):
    """Test ROC curve plot generation"""
    y_true, y_pred, y_pred_proba = binary_classification_data
    cp = ClassificationPlots(plot_type='roc')
    fig = cp.plot(y_true, y_pred, y_pred_proba)
    assert fig is not None
    assert hasattr(fig, 'show')  # Check it's a matplotlib figure


def test_ClassificationPlots_both(binary_classification_data):
    """Test both confusion matrix and ROC curve plots"""
    y_true, y_pred, y_pred_proba = binary_classification_data
    cp = ClassificationPlots(plot_type='both')
    fig = cp.plot(y_true, y_pred, y_pred_proba)
    assert fig is not None
    assert hasattr(fig, 'show')


def test_ClassificationPlots_both_without_proba(binary_classification_data):
    """Test both plots when ROC probabilities are not provided"""
    y_true, y_pred, _ = binary_classification_data
    cp = ClassificationPlots(plot_type='both')
    fig = cp.plot(y_true, y_pred, y_pred_proba=None)
    assert fig is not None
    assert hasattr(fig, 'show')


def test_ClassificationPlots_roc_missing_proba(binary_classification_data):
    """Test that ROC plot raises error when probabilities are missing"""
    y_true, y_pred, _ = binary_classification_data
    cp = ClassificationPlots(plot_type='roc')
    with pytest.raises(ValueError, match='y_pred_proba is required for ROC curve'):
        cp.plot(y_true, y_pred, y_pred_proba=None)


def test_ClassificationPlots_invalid_plot_type(binary_classification_data):
    """Test that invalid plot_type raises error"""
    y_true, y_pred, _ = binary_classification_data
    cp = ClassificationPlots(plot_type='invalid_type')
    with pytest.raises(ValueError, match="plot_type must be 'confusion_matrix', 'roc', or 'both'"):
        cp.plot(y_true, y_pred)


def test_ClassificationPlots_custom_figsize(binary_classification_data):
    """Test custom figure size"""
    y_true, y_pred, y_pred_proba = binary_classification_data
    custom_figsize = (10, 5)
    cp = ClassificationPlots(plot_type='both', figsize=custom_figsize)
    fig = cp.plot(y_true, y_pred, y_pred_proba)
    assert fig is not None
    # Check if figsize is applied
    assert fig.get_size_inches()[0] == custom_figsize[0]
    assert fig.get_size_inches()[1] == custom_figsize[1]


def test_ClassificationPlots_custom_cmap(binary_classification_data):
    """Test custom colormap"""
    y_true, y_pred, _ = binary_classification_data
    cp = ClassificationPlots(plot_type='confusion_matrix', cmap='Reds')
    fig = cp.plot(y_true, y_pred)
    assert fig is not None


def test_ClassificationPlots_multiclass_roc(multiclass_classification_data):
    """Test ROC curves for multiclass classification"""
    y_true, y_pred, y_pred_proba = multiclass_classification_data
    cp = ClassificationPlots(plot_type='roc')
    fig = cp.plot(y_true, y_pred, y_pred_proba)
    assert fig is not None
    assert hasattr(fig, 'show')


def test_ClassificationPlots_multiclass_both(multiclass_classification_data):
    """Test both plots for multiclass classification"""
    y_true, y_pred, y_pred_proba = multiclass_classification_data
    cp = ClassificationPlots(plot_type='both')
    fig = cp.plot(y_true, y_pred, y_pred_proba)
    assert fig is not None
    assert hasattr(fig, 'show')

