"""
The feature_selection module provides functions for selecting the most informative 
features based on their direct contribution to model performance, using techniques 
like genetic algorithms. This complements feature_cleaning by focusing on modeling 
usefulness rather than statistical properties.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from chemml.optimization import GeneticAlgorithm


def GAFSel(df=None, target=None, target_features_count=50, evaluator=None, 
           n_generations=50, pop_size=100, test_size=0.7, crossover_ratio = 0.6, random_state=42):
    """
    Genetic Algorithm-based Feature Selection wrapper for standardized feature selection workflows.
    
    Selects the most modeling-useful features from a dataset using a genetic algorithm that 
    directly optimizes for a given estimator's predictive performance (Mean Absolute Error).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing both feature columns and the target column.
        A 'SMILES' column, if present, is automatically excluded from the feature set.
        
    target : str
        The name of the column containing the target variable to predict.
        
    target_features_count : int, optional (default=50)
        Target number of features to select. The genetic algorithm will bias selection 
        toward solutions with approximately this many active features.
        
    evaluator : sklearn-style estimator, optional (default=None)
        A scikit-learn compatible regression estimator with fit() and predict() methods.
        If None, defaults to LinearRegression().
        
    n_generations : int, optional (default=50)
        Number of generations to evolve the GA population for.
        
    pop_size : int, optional (default=100)
        Size of the GA population in each generation.
    
    crossover_ratio : float, optional (default=0.6)
        Fraction of the population to use for crossover in each generation. Must be in (0, 1).
        
    test_size : float, optional (default=0.7)
        Fraction of data to use for testing in each fitness evaluation (train/test split).
        Must be in the interval (0, 1).
        
    random_state : int, optional (default=42)
        Random seed for reproducibility in train/test splits.
        
    Returns
    -------
    pandas.DataFrame
        Dataframe containing only the selected features plus the target column.
        The number of selected features will typically be close to target_features_count,
        depending on GA convergence and the data.
        
    Raises
    ------
    ValueError
        If df or target is None, target is not in df.columns, target_features_count 
        is not a positive integer, or test_size is not in (0, 1).
        
    Notes
    -----
    The genetic algorithm uses:
    - Crossover type: Uniform
    - Algorithm variant: 3 (best members from init/crossover/mutation pools)
    - Fitness: minimization (MAE)
    - Crossover size: 60% of population
    - Mutation size: 40% of population
    
    Each individual in the GA population is a binary string where each bit represents 
    whether a feature is included (1) or excluded (0). The objective function trains 
    the estimator on a subset of features and returns the MAE on held-out test data.

    Note that due to its inherently random nature, GA won't hit the exact target_features_count, but will start with a bias toward that number of features. 
    The final selection may vary slightly depending on the data and random seed.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from chemml.preprocessing import GAFSel
    >>> df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 
    ...                     'target': [10, 20, 30]})
    >>> selected = GAFSel(df=df, target='target', target_features_count=1, 
    ...                    n_generations=5, pop_size=20)
    >>> print(selected.shape)  # (3, 2) — target + 1 selected feature
    """
    # Input validation
    if df is None:
        raise ValueError("'df' parameter cannot be None.")
    if target is None:
        raise ValueError("'target' parameter cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe columns.")
    if not isinstance(target_features_count, int) or target_features_count <= 0:
        raise ValueError("'target_features_count' must be a positive integer.")
    if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
        raise ValueError("'test_size' must be in the interval (0, 1).")
    
    # Separate features and target
    y = df[target].values.flatten()
    
    # Drop target and SMILES (if present) from features
    features = df.drop(columns=[target])
    if 'SMILES' in features.columns:
        features = features.drop(columns=['SMILES'])
    
    n_features = features.shape[1]
    if n_features == 0:
        raise ValueError("No features available after removing target and SMILES columns.")
    
    # Default evaluator
    if evaluator is None:
        evaluator = LinearRegression()
    
    # Define binary GA search space: each feature is either selected (1) or not (0)
    space = tuple([{i: {'choice': [0, 1]}} for i in range(n_features)])
    
    # Define the objective function: train on selected features and return MAE
    def obj(individual):
        """
        Objective function for GA: evaluates fitness of a feature selection.
        
        Parameters
        ----------
        individual : list or tuple
            Binary string (list/tuple of 0s and 1s) indicating selected features.
            
        Returns
        -------
        tuple
            MAE value wrapped in a tuple (GA expects tuple return).
        """
        # Select columns based on individual (binary mask)
        selected_mask = list(map(bool, individual))
        selected_features = features.iloc[:, selected_mask]
        
        # If no features selected, return high penalty
        if selected_features.shape[1] == 0:
            return (1e6,)  # Large penalty for no features
        
        X = selected_features.values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Standardize features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train estimator
        try:
            est = evaluator.__class__(**evaluator.get_params())  # Fresh copy of estimator
        except:
            est = evaluator  # Fallback if get_params() fails
        
        est.fit(X_train_scaled, y_train_scaled)
        
        # Predict on test set
        y_pred_scaled = est.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Compute MAE
        mae = mean_absolute_error(y_test, y_pred)
        
        return (mae,)
    
    # Compute crossover and mutation sizes based on population
    crossover_size = int(crossover_ratio * pop_size)
    mutation_size = pop_size - crossover_size
    
    # Instantiate and run genetic algorithm
    ga = GeneticAlgorithm(
        evaluate=obj,
        space=space,
        fitness=("min",),
        pop_size=pop_size,
        crossover_size=crossover_size,
        mutation_size=mutation_size,
        crossover_type="Uniform",
        algorithm=3,
        target_features_count=target_features_count
    )
    
    fitness_df, final_best_features = ga.search(n_generations=n_generations)
    
    # Convert final best features dict to boolean mask
    col_mask = [bool(final_best_features.get(i, 0)) for i in range(n_features)]
    
    # Select features based on mask
    selected_features_final = features.iloc[:, col_mask]
    
    # Append target column
    result_df = selected_features_final.copy()
    result_df[target] = y
    
    return result_df
