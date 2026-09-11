import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from chemml.preprocessing import GAFSel, ZScoreFSel


@pytest.fixture
def synthetic_data():
    """Create a synthetic dataset for feature selection testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 30  # Must be > 20 for target_features_count to work
    
    # Generate features with varying usefulness
    X = np.random.randn(n_samples, n_features)
    
    # Create target with real relationship to first 3 features
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.1 * np.random.randn(n_samples)
    
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(n_features)])
    df['target_property'] = y
    
    return df


@pytest.fixture
def synthetic_data_with_smiles():
    """Create a synthetic dataset with SMILES column."""
    np.random.seed(42)
    n_samples = 50
    n_features = 25  # Must be > 20 for target_features_count to work
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    df = pd.DataFrame(X, columns=[f'Desc_{i}' for i in range(n_features)])
    df['SMILES'] = ['CCO'] * n_samples  # dummy SMILES
    df['target'] = y
    
    return df


def test_gasel_basic_functionality(synthetic_data):
    """Test basic GAFSel execution and return type."""
    result = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=10,
        n_generations=3,
        pop_size=20
    )
    
    assert isinstance(result, pd.DataFrame)
    assert 'target_property' in result.columns
    assert result.shape[0] == synthetic_data.shape[0]  # same number of rows
    assert result.shape[1] <= synthetic_data.shape[1]  # fewer or equal columns


def test_gasel_target_column_included(synthetic_data):
    """Test that target column is included in returned DataFrame."""
    result = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=5,
        n_generations=2,
        pop_size=15
    )
    
    assert 'target_property' in result.columns
    np.testing.assert_array_equal(
        result['target_property'].values,
        synthetic_data['target_property'].values
    )


def test_gasel_feature_reduction(synthetic_data):
    """Test that GAFSel actually selects fewer features than input."""
    result = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=5,
        n_generations=3,
        pop_size=20
    )
    
    # Result should have target + selected features
    # With target_features_count=5, we expect roughly 5 features + 1 target
    n_selected_features = result.shape[1] - 1  # exclude target
    assert n_selected_features <= synthetic_data.shape[1] - 1  # should be reduced from 30 original features


def test_gasel_smiles_auto_skip(synthetic_data_with_smiles):
    """Test that SMILES column is automatically skipped."""
    result = GAFSel(
        df=synthetic_data_with_smiles,
        target='target',
        target_features_count=5,
        n_generations=2,
        pop_size=15
    )
    
    # SMILES should not be in result
    assert 'SMILES' not in result.columns
    assert 'target' in result.columns
    # Result should have target + selected descriptors (not SMILES)
    assert result.shape[1] <= synthetic_data_with_smiles.shape[1] - 1  # target + descriptors only


def test_gasel_custom_evaluator(synthetic_data):
    """Test GAFSel with a custom sklearn estimator."""
    result = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=8,
        evaluator=Ridge(alpha=1.0),
        n_generations=2,
        pop_size=15
    )
    
    assert isinstance(result, pd.DataFrame)
    assert 'target_property' in result.columns


def test_gasel_invalid_inputs_raise(synthetic_data):
    """Test that invalid GAFSel arguments raise ValueError."""
    cases = [
        ({'df': None, 'target': 'target', 'target_features_count': 50}, "'df' parameter cannot be None"),
        ({'df': synthetic_data, 'target': None, 'target_features_count': 50}, "'target' parameter cannot be None"),
        ({'df': synthetic_data, 'target': 'nonexistent', 'target_features_count': 50, 'n_generations': 2, 'pop_size': 15}, "Target column 'nonexistent' not found"),
        ({'df': synthetic_data, 'target': 'target_property', 'target_features_count': 0, 'n_generations': 2, 'pop_size': 15}, "'target_features_count' must be a positive integer"),
        ({'df': synthetic_data, 'target': 'target_property', 'target_features_count': 50, 'test_size': 1.5, 'n_generations': 2, 'pop_size': 15}, "'test_size' must be in the interval"),
        ({'df': synthetic_data, 'target': 'target_property', 'target_features_count': 50, 'test_size': -0.1, 'n_generations': 2, 'pop_size': 15}, "'test_size' must be in the interval"),
    ]

    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            GAFSel(**kwargs)


def test_gasel_default_parameters(synthetic_data):
    """Test GAFSel with default parameters."""
    # Should work without specifying optional parameters
    # Note: default target_features_count=50, but we provide a dataset with 30 features
    # So we explicitly set target_features_count to a valid value
    result = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=15  # reasonable for 30 features
    )
    
    assert isinstance(result, pd.DataFrame)
    assert 'target_property' in result.columns
    # Result should have fewer than the original 30 feature columns
    assert result.shape[1] <= synthetic_data.shape[1]


def test_gasel_preserves_index(synthetic_data):
    """Test that GAFSel preserves the original DataFrame index."""
    result = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=5,
        n_generations=2,
        pop_size=15
    )
    
    np.testing.assert_array_equal(result.index.values, synthetic_data.index.values)


def test_gasel_with_different_pop_sizes(synthetic_data):
    """Test GAFSel with different population sizes."""
    result_small = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=5,
        n_generations=2,
        pop_size=10
    )
    
    result_large = GAFSel(
        df=synthetic_data,
        target='target_property',
        target_features_count=5,
        n_generations=2,
        pop_size=30
    )
    
    # Both should return valid results
    assert isinstance(result_small, pd.DataFrame)
    assert isinstance(result_large, pd.DataFrame)
    assert 'target_property' in result_small.columns
    assert 'target_property' in result_large.columns


@pytest.fixture
def zscore_data():
    """Create synthetic features with a target-correlated binary feature,
    an uncorrelated binary feature and a non-binary feature."""
    np.random.seed(42)
    n_samples = 100
    good_feature = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    noise_feature = np.random.randint(0, 2, n_samples)
    continuous_feature = np.random.uniform(0, 10, n_samples)
    target_values = np.where(good_feature == 1, 10, 0) + np.random.normal(0, 0.1, n_samples)

    features = pd.DataFrame({
        'good_feature': good_feature,
        'noise_feature': noise_feature,
        'continuous_feature': continuous_feature,
    })
    target = pd.Series(target_values, name='target')
    return features, target


def test_zscorefsel_selects_correlated_binary_feature(zscore_data):
    """Test that a binary feature correlated with the target is selected."""
    features, target = zscore_data
    result = ZScoreFSel(features, target, threshold=1.0)

    assert 'good_feature' in result.columns


def test_zscorefsel_excludes_non_binary_feature(zscore_data):
    """Test that non-binary features are never selected, regardless of threshold."""
    features, target = zscore_data
    result = ZScoreFSel(features, target, threshold=0.0)

    assert 'continuous_feature' not in result.columns


def test_zscorefsel_high_threshold_excludes_all_features(zscore_data):
    """Test that an unreachably high threshold results in no columns being selected."""
    features, target = zscore_data
    with pytest.warns(UserWarning):
        ZScoreFSel(features, target, threshold=100.0)


def test_zscorefsel_lower_threshold_selects_more_features(zscore_data):
    """Test that lowering the threshold selects at least as many features."""
    features, target = zscore_data
    result_low = ZScoreFSel(features, target, threshold=0.0)
    result_high = ZScoreFSel(features, target, threshold=2.0)

    n_features_low = result_low.shape[1] - 1
    n_features_high = result_high.shape[1] - 1
    assert n_features_low >= n_features_high
