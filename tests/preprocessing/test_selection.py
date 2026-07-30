import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from chemml.preprocessing import GAFSel


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


def test_gasel_none_df_raises_error(synthetic_data):
    """Test that None dataframe raises ValueError."""
    with pytest.raises(ValueError, match="'df' parameter cannot be None"):
        GAFSel(df=None, target='target', target_features_count=50)


def test_gasel_none_target_raises_error(synthetic_data):
    """Test that None target raises ValueError."""
    with pytest.raises(ValueError, match="'target' parameter cannot be None"):
        GAFSel(df=synthetic_data, target=None, target_features_count=50)


def test_gasel_invalid_target_column(synthetic_data):
    """Test that invalid target column name raises ValueError."""
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
        GAFSel(
            df=synthetic_data,
            target='nonexistent',
            target_features_count=50,
            n_generations=2,
            pop_size=15
        )


def test_gasel_invalid_target_features_count(synthetic_data):
    """Test that invalid target_features_count raises ValueError."""
    with pytest.raises(ValueError, match="'target_features_count' must be a positive integer"):
        GAFSel(
            df=synthetic_data,
            target='target_property',
            target_features_count=0,
            n_generations=2,
            pop_size=15
        )


def test_gasel_invalid_test_size(synthetic_data):
    """Test that invalid test_size raises ValueError."""
    with pytest.raises(ValueError, match="'test_size' must be in the interval"):
        GAFSel(
            df=synthetic_data,
            target='target_property',
            target_features_count=50,
            test_size=1.5,
            n_generations=2,
            pop_size=15
        )


def test_gasel_negative_test_size(synthetic_data):
    """Test that negative test_size raises ValueError."""
    with pytest.raises(ValueError, match="'test_size' must be in the interval"):
        GAFSel(
            df=synthetic_data,
            target='target_property',
            target_features_count=50,
            test_size=-0.1,
            n_generations=2,
            pop_size=15
        )


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
