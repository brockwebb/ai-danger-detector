"""
Model training module for machine learning analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_linear_model(X_train, y_train, model_type='linear', alpha=1.0, **kwargs):
    """
    Train a linear model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    
    y_train : pandas.Series
        Training targets
    
    model_type : str
        Type of linear model ('linear', 'ridge', 'lasso', or 'elasticnet')
    
    alpha : float
        Regularization strength (for ridge, lasso, elasticnet)
    
    **kwargs : dict
        Additional arguments to pass to the model constructor
    
    Returns:
    --------
    model
        Trained model
    """
    # Select model type
    if model_type == 'linear':
        model = LinearRegression(**kwargs)
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha, **kwargs)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha, **kwargs)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=alpha, **kwargs)
    else:
        raise ValueError(f"Unsupported linear model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def train_tree_model(X_train, y_train, model_type='random_forest', **kwargs):
    """
    Train a tree-based model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    
    y_train : pandas.Series
        Training targets
    
    model_type : str
        Type of tree model ('random_forest' or 'gradient_boosting')
    
    **kwargs : dict
        Additional arguments to pass to the model constructor
    
    Returns:
    --------
    model
        Trained model
    """
    # Set default values if not provided
    if 'n_estimators' not in kwargs:
        kwargs['n_estimators'] = 100
    
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 42
    
    # Select model type
    if model_type == 'random_forest':
        model = RandomForestRegressor(**kwargs)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(**kwargs)
    else:
        raise ValueError(f"Unsupported tree model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def train_advanced_model(X_train, y_train, model_type='svm', **kwargs):
    """
    Train an advanced model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    
    y_train : pandas.Series
        Training targets
    
    model_type : str
        Type of advanced model ('svm' or 'mlp')
    
    **kwargs : dict
        Additional arguments to pass to the model constructor
    
    Returns:
    --------
    model
        Trained model
    """
    # Select model type
    if model_type == 'svm':
        if 'kernel' not in kwargs:
            kwargs['kernel'] = 'rbf'
        
        model = SVR(**kwargs)
    elif model_type == 'mlp':
        if 'hidden_layer_sizes' not in kwargs:
            kwargs['hidden_layer_sizes'] = (100, 50)
        
        if 'max_iter' not in kwargs:
            kwargs['max_iter'] = 1000
        
        model = MLPRegressor(**kwargs)
    else:
        raise ValueError(f"Unsupported advanced model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_val, y_val, X_test=None, y_test=None):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : model
        Trained model
    
    X_val : pandas.DataFrame
        Validation features
    
    y_val : pandas.Series
        Validation targets
    
    X_test : pandas.DataFrame, optional
        Test features
    
    y_test : pandas.Series, optional
        Test targets
    
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Make predictions on validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Create results dictionary
    results = {
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'val_predictions': y_val_pred
    }
    
    # Add test metrics if test data provided
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.update({
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_predictions': y_test_pred
        })
    
    return results


def plot_predictions(y_true, y_pred, title=None, figsize=(10, 6), save_path=None):
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    
    y_pred : array-like
        Predicted target values
    
    title : str, optional
        Plot title
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Predicted vs Actual Values')
    
    # Add R² annotation
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(y_true, y_pred, title=None, figsize=(10, 6), save_path=None):
    """
    Plot residuals analysis.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    
    y_pred : array-like
        Predicted target values
    
    title : str, optional
        Plot title
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Value')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def train_and_evaluate_multiple_models(X_train, y_train, X_val, y_val, models_config=None):
    """
    Train and evaluate multiple models for comparison.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    
    y_train : pandas.Series
        Training targets
    
    X_val : pandas.DataFrame
        Validation features
    
    y_val : pandas.Series
        Validation targets
    
    models_config : list, optional
        List of model configurations to try
    
    Returns:
    --------
    dict
        Dictionary of trained models and their performance metrics
    """
    # Default models to try if none specified
    if models_config is None:
        models_config = [
            {'name': 'Linear Regression', 'type': 'linear', 'func': train_linear_model, 'params': {}},
            {'name': 'Ridge Regression', 'type': 'ridge', 'func': train_linear_model, 'params': {'alpha': 1.0}},
            {'name': 'Random Forest', 'type': 'random_forest', 'func': train_tree_model, 'params': {'n_estimators': 100}},
            {'name': 'Gradient Boosting', 'type': 'gradient_boosting', 'func': train_tree_model, 'params': {'n_estimators': 100}}
        ]
    
    # Results storage
    results = {}
    
    # Train and evaluate each model
    for model_config in models_config:
        print(f"Training {model_config['name']}...")
        start_time = time.time()
        
        # Train model
        model = model_config['func'](
            X_train, y_train, 
            model_type=model_config['type'], 
            **model_config['params']
        )
        
        # Evaluate model
        eval_results = evaluate_model(model, X_val, y_val)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store results
        results[model_config['name']] = {
            'model': model,
            'config': model_config,
            'metrics': eval_results,
            'training_time': training_time
        }
        
        print(f"  Validation RMSE: {eval_results['val_rmse']:.4f}")
        print(f"  Validation R²: {eval_results['val_r2']:.4f}")
        print(f"  Training Time: {training_time:.2f} seconds")
    
    return results


def compare_models(model_results, metric='val_rmse', figsize=(10, 6), save_path=None):
    """
    Compare performance of multiple models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary of model results from train_and_evaluate_multiple_models
    
    metric : str
        Metric to use for comparison
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Extract metric values for each model
    models = []
    values = []
    
    for model_name, result in model_results.items():
        models.append(model_name)
        values.append(result['metrics'][metric])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Model': models,
        'Value': values
    })
    
    # Sort by metric value (ascending for error metrics, descending for R²)
    ascending = True
    if metric.endswith('r2'):
        ascending = False
    
    df = df.sort_values('Value', ascending=ascending)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    ax.barh(df['Model'], df['Value'])
    
    # Format metric name for display
    metric_display = metric.replace('val_', 'Validation ').replace('test_', 'Test ')
    metric_display = metric_display.upper() if metric_display.endswith('rmse') or metric_display.endswith('mae') or metric_display.endswith('mse') else metric_display
    
    # Set labels and title
    ax.set_xlabel(metric_display)
    ax.set_title(f'Model Comparison: {metric_display}')
    
    # Add values on bars
    for i, v in enumerate(df['Value']):
        ax.text(v * 1.01, i, f'{v:.4f}', va='center')
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_model(model, filepath, metadata=None):
    """
    Save trained model to file.
    
    Parameters:
    -----------
    model : model
        Trained model to save
    
    filepath : str
        Path to save model
    
    metadata : dict, optional
        Additional metadata to save with model
    
    Returns:
    --------
    str
        Path to saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare model data for saving
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    # Add timestamp
    model_data['metadata']['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filepath}")
    
    return filepath


def load_model(filepath):
    """
    Load model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
    
    Returns:
    --------
    tuple
        (model, metadata)
    """
    # Load model
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model and metadata
    model = model_data['model']
    metadata = model_data['metadata']
    
    print(f"Model loaded from {filepath}")
    print(f"Model saved at: {metadata.get('saved_at', 'unknown')}")
    
    return model, metadata


if __name__ == "__main__":
    # Example usage
    print("Testing model training and evaluation...")
    
    # Generate a small sample dataset
    import sys
    sys.path.append('..')
    from data_generation.batch_generator import generate_uniform_dataset
    from machine_learning.preprocessing import split_features_target, create_dataset_splits, preprocess_data
    
    # Generate data
    sample_df = generate_uniform_dataset(n_samples=1000)
    
    # Split features and target
    X, y = split_features_target(sample_df, 'expertise_medical')
    
    # Create train/val/test splits
    splits = create_dataset_splits(X, y)
    
    # Preprocess data
    processed_splits, _ = preprocess_data(splits)
    
    # Train a simple model
    model = train_linear_model(processed_splits['X_train'], processed_splits['y_train'])
    results = evaluate_model(model, processed_splits['X_val'], processed_splits['y_val'])
    
    print("Linear Regression Results:")
    print(f"Validation RMSE: {results['val_rmse']:.4f}")
    print(f"Validation R²: {results['val_r2']:.4f}")
    
    # Train multiple models for comparison
    model_results = train_and_evaluate_multiple_models(
        processed_splits['X_train'], processed_splits['y_train'],
        processed_splits['X_val'], processed_splits['y_val']
    )
    
    print("\nModel Comparison:")
    for model_name, result in model_results.items():
        print(f"{model_name}: RMSE = {result['metrics']['val_rmse']:.4f}, R² = {result['metrics']['val_r2']:.4f}")