"""
Data preprocessing module for machine learning analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_model.domain_profiles import domain_profiles


def load_dataset(filepath, sample_size=None, random_state=42):
    """
    Load and prepare dataset for ML training.
    
    Parameters:
    -----------
    filepath : str
        Path to dataset file (CSV or Parquet)
    
    sample_size : int, optional
        If specified, sample this many rows randomly from the dataset
    
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    # Load data based on file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    return df


def split_features_target(df, target_column, domain=None):
    """
    Split dataset into features and target variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    
    target_column : str
        Column name to use as target variable
    
    domain : str, optional
        If specified, filter to only this domain's data
    
    Returns:
    --------
    tuple
        X (features) and y (target) arrays
    """
    # Filter by domain if specified
    if domain is not None:
        if 'domain' in df.columns:
            df = df[df['domain'] == domain]
        else:
            # Try target column pattern matching
            domain_target = f"expertise_{domain}"
            if domain_target in df.columns:
                target_column = domain_target
    
    # Identify feature columns
    feature_columns = ['harm', 'complexity', 'error_rate']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract features and target
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y


def create_dataset_splits(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Create train/validation/test splits.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
    
    y : pandas.Series
        Target data
    
    test_size : float
        Proportion of data to use for test set
    
    val_size : float
        Proportion of training data to use for validation
    
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing train, validation, and test splits
    """
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Then split training set into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def preprocess_data(data_splits, scaler_type='standard'):
    """
    Preprocess data with scaling.
    
    Parameters:
    -----------
    data_splits : dict
        Dictionary containing train, validation, and test splits
    
    scaler_type : str
        Type of scaler to use ('standard' or 'minmax')
    
    Returns:
    --------
    tuple
        Preprocessed data splits and scaler object
    """
    # Choose scaler based on type
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    # Fit scaler on training data only
    scaler.fit(data_splits['X_train'])
    
    # Transform all datasets
    processed_splits = {}
    for key, data in data_splits.items():
        if key.startswith('X_'):
            processed_splits[key] = pd.DataFrame(
                scaler.transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            processed_splits[key] = data
    
    return processed_splits, scaler


def create_polynomial_features(data_splits, degree=2, include_bias=False):
    """
    Create polynomial features for improved model fit.
    
    Parameters:
    -----------
    data_splits : dict
        Dictionary containing train, validation, and test splits
    
    degree : int
        Polynomial degree
    
    include_bias : bool
        Whether to include a bias column
    
    Returns:
    --------
    dict
        Data splits with polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    # Create polynomial feature transformer
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    
    # Fit on training data
    poly.fit(data_splits['X_train'])
    
    # Transform all datasets
    poly_splits = {}
    for key, data in data_splits.items():
        if key.startswith('X_'):
            # Get feature names
            if hasattr(poly, 'get_feature_names_out'):
                feature_names = poly.get_feature_names_out(data.columns)
            else:
                feature_names = poly.get_feature_names(data.columns)
            
            # Transform data
            poly_splits[key] = pd.DataFrame(
                poly.transform(data),
                columns=feature_names,
                index=data.index
            )
        else:
            poly_splits[key] = data
    
    return poly_splits


def create_domain_specific_features(data_splits, domain):
    """
    Create domain-specific features based on domain profiles.
    
    Parameters:
    -----------
    data_splits : dict
        Dictionary containing train, validation, and test splits
    
    domain : str
        Domain to use for feature engineering
    
    Returns:
    --------
    dict
        Data splits with additional domain-specific features
    """
    if domain not in domain_profiles:
        raise ValueError(f"Unknown domain: {domain}")
    
    profile = domain_profiles[domain]
    
    # Create new splits dictionary
    enhanced_splits = {}
    
    # Process each split
    for key, data in data_splits.items():
        if key.startswith('X_'):
            # Create a copy of the original data
            enhanced_data = data.copy()
            
            # Get beta and alpha parameters for this domain
            betas = profile['beta']
            alphas = profile['alpha']
            
            # Add domain-specific features
            enhanced_data['harm_weighted'] = data['harm'] ** alphas[0] * betas[0]
            enhanced_data['complexity_weighted'] = data['complexity'] ** alphas[1] * betas[1]
            enhanced_data['error_weighted'] = data['error_rate'] ** alphas[2] * betas[2]
            
            # Add interaction term
            enhanced_data['interaction'] = (
                data['harm'] * data['complexity'] * data['error_rate']
            ) ** alphas[3] * betas[3]
            
            enhanced_splits[key] = enhanced_data
        else:
            enhanced_splits[key] = data
    
    return enhanced_splits


if __name__ == "__main__":
    # Example usage
    print("Testing data preprocessing...")
    
    # Generate a small sample dataset
    import sys
    sys.path.append('..')
    from data_generation.batch_generator import generate_uniform_dataset
    
    sample_df = generate_uniform_dataset(n_samples=1000)
    print(f"Generated sample dataset with {len(sample_df)} rows")
    
    # Split features and target
    X, y = split_features_target(sample_df, 'expertise_medical')
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Create train/val/test splits
    splits = create_dataset_splits(X, y)
    print(f"Train size: {len(splits['X_train'])}, Val size: {len(splits['X_val'])}, Test size: {len(splits['X_test'])}")
    
    # Preprocess data
    processed_splits, scaler = preprocess_data(splits)
    print(f"Processed train set shape: {processed_splits['X_train'].shape}")
    
    # Create polynomial features
    poly_splits = create_polynomial_features(processed_splits)
    print(f"Polynomial train set shape: {poly_splits['X_train'].shape}")
    
    # Create domain-specific features
    enhanced_splits = create_domain_specific_features(splits, 'medical')
    print(f"Enhanced train set shape: {enhanced_splits['X_train'].shape}")