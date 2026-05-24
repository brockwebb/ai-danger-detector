"""
Model interpretability module for machine learning analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_feature_importance(model, feature_names=None):
    """
    Extract feature importance from a trained model.
    
    Parameters:
    -----------
    model : model
        Trained model
    
    feature_names : list, optional
        List of feature names
    
    Returns:
    --------
    pandas.DataFrame
        Feature importance scores
    """
    # Check if model type supports feature importance
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1 and importance.shape[0] == 1:
            importance = importance[0]
    else:
        raise ValueError("Model does not support direct feature importance extraction")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df, top_n=None, figsize=(10, 6), save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        Feature importance DataFrame
    
    top_n : int, optional
        Number of top features to plot
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Limit to top N features if specified
    if top_n is not None and top_n < len(importance_df):
        plot_df = importance_df.head(top_n).copy()
    else:
        plot_df = importance_df.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart (reversed order for better viewing)
    plot_df = plot_df.iloc[::-1]  # Reverse order
    ax.barh(plot_df['Feature'], plot_df['Importance'])
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def partial_dependence_plot(model, X, features, feature_names=None, figsize=(12, 8), 
                          n_points=50, save_path=None):
    """
    Create partial dependence plots for specified features.
    
    Parameters:
    -----------
    model : model
        Trained model
    
    X : pandas.DataFrame
        Feature data
    
    features : list
        List of feature indices or names to plot
    
    feature_names : list, optional
        List of feature names
    
    figsize : tuple
        Figure size
    
    n_points : int
        Number of points to sample for each feature
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create feature names if not provided
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Convert X to numpy if it's a DataFrame
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    
    # Convert feature names to indices if needed
    feature_indices = []
    for feature in features:
        if isinstance(feature, str) and isinstance(X, pd.DataFrame):
            feature_indices.append(list(X.columns).index(feature))
        else:
            feature_indices.append(feature)
    
    # Create figure with subplots
    n_features = len(feature_indices)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Create PDP for each feature
    for i, feature_idx in enumerate(feature_indices):
        # Get subplot axis
        ax = axes[i // n_cols, i % n_cols]
        
        # Get feature name
        feature_name = feature_names[feature_idx]
        
        # Get feature values
        feature_values = X_values[:, feature_idx]
        
        # Create grid for feature values
        grid = np.linspace(np.min(feature_values), np.max(feature_values), n_points)
        
        # Create PDP
        pdp_values = []
        
        for value in grid:
            # Create copies with the feature set to this value
            X_modified = X_values.copy()
            X_modified[:, feature_idx] = value
            
            # Predict with the model
            predictions = model.predict(X_modified)
            
            # Take average prediction
            pdp_values.append(np.mean(predictions))
        
        # Plot PDP
        ax.plot(grid, pdp_values, 'b-', linewidth=2)
        
        # Add actual data distribution as histogram
        ax_twin = ax.twinx()
        ax_twin.hist(feature_values, bins=20, alpha=0.3, color='gray')
        ax_twin.set_ylabel('Frequency')
        ax_twin.set_yticks([])  # Hide y-ticks for cleaner look
        
        # Set labels
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'Partial Dependence: {feature_name}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove empty subplots
    for i in range(n_features, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols, i % n_cols])
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_interaction_effects(model, X, feature_pairs, feature_names=None, 
                           figsize=(12, 10), n_points=20, save_path=None):
    """
    Plot interaction effects between pairs of features.
    
    Parameters:
    -----------
    model : model
        Trained model
    
    X : pandas.DataFrame
        Feature data
    
    feature_pairs : list of tuples
        List of feature pairs (indices or names) to plot
    
    feature_names : list, optional
        List of feature names
    
    figsize : tuple
        Figure size
    
    n_points : int
        Number of points to sample for each feature
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create feature names if not provided
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Convert X to numpy if it's a DataFrame
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    
    # Convert feature names to indices if needed
    feature_pairs_indices = []
    for feature1, feature2 in feature_pairs:
        idx1 = feature1
        idx2 = feature2
        
        if isinstance(feature1, str) and isinstance(X, pd.DataFrame):
            idx1 = list(X.columns).index(feature1)
        
        if isinstance(feature2, str) and isinstance(X, pd.DataFrame):
            idx2 = list(X.columns).index(feature2)
        
        feature_pairs_indices.append((idx1, idx2))
    
    # Create figure with subplots
    n_pairs = len(feature_pairs_indices)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Plot interaction effects for each pair
    for i, (idx1, idx2) in enumerate(feature_pairs_indices):
        # Get subplot axis
        ax = axes[i // n_cols, i % n_cols]
        
        # Get feature names
        name1 = feature_names[idx1]
        name2 = feature_names[idx2]
        
        # Get feature values
        values1 = X_values[:, idx1]
        values2 = X_values[:, idx2]
        
        # Create grids for feature values
        grid1 = np.linspace(np.min(values1), np.max(values1), n_points)
        grid2 = np.linspace(np.min(values2), np.max(values2), n_points)
        
        # Create meshgrid
        G1, G2 = np.meshgrid(grid1, grid2)
        
        # Create prediction matrix
        Z = np.zeros_like(G1)
        
        # Base predictions by fixing other features to their means
        X_base = np.mean(X_values, axis=0).reshape(1, -1).repeat(n_points * n_points, axis=0)
        
        # Flatten grids
        G1_flat = G1.flatten()
        G2_flat = G2.flatten()
        
        # Update features
        X_modified = X_base.copy()
        X_modified[:, idx1] = G1_flat
        X_modified[:, idx2] = G2_flat
        
        # Make predictions
        predictions = model.predict(X_modified)
        
        # Reshape predictions to grid
        Z = predictions.reshape(n_points, n_points)
        
        # Create contour plot
        contour = ax.contourf(G1, G2, Z, levels=20, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Predicted Value')
        
        # Set labels and title
        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_title(f'Interaction: {name1} × {name2}')
        
        # Add scatter plot of actual data points (sample to avoid clutter)
        if len(X_values) > 500:
            idx = np.random.choice(len(X_values), 500, replace=False)
            ax.scatter(values1[idx], values2[idx], alpha=0.3, s=10, color='white', edgecolor='black')
        else:
            ax.scatter(values1, values2, alpha=0.3, s=10, color='white', edgecolor='black')
    
    # Remove empty subplots
    for i in range(n_pairs, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols, i % n_cols])
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def try_shap_analysis(model, X, feature_names=None, sample_size=None, 
                    plot_type='summary', figsize=(10, 8), save_path=None):
    """
    Attempt to run SHAP analysis on the model.
    
    Parameters:
    -----------
    model : model
        Trained model
    
    X : pandas.DataFrame
        Feature data
    
    feature_names : list, optional
        List of feature names
    
    sample_size : int, optional
        Number of samples to use for SHAP analysis
    
    plot_type : str
        Type of SHAP plot ('summary', 'bar', 'beeswarm', or 'waterfall')
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    tuple
        (figure, shap_values)
    """
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP package not installed. Install with: pip install shap")
        return None, None
    
    # Create feature names if not provided
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Sample data if needed
    if sample_size is not None and sample_size < len(X):
        if isinstance(X, pd.DataFrame):
            X_sample = X.sample(sample_size, random_state=42)
        else:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
    else:
        X_sample = X
    
    # Create figure
    plt.figure(figsize=figsize)
    
    try:
        # Try to create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # For classifiers
            explainer = shap.Explainer(model, X_sample)
        else:
            # For regressors
            explainer = shap.Explainer(model, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        # Create plot based on type
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X_sample, plot_type='bar', feature_names=feature_names, show=False)
        elif plot_type == 'beeswarm':
            shap.plots.beeswarm(shap_values, show=False)
        elif plot_type == 'waterfall':
            if isinstance(X_sample, pd.DataFrame):
                example_idx = 0
                shap.plots.waterfall(shap_values[example_idx], show=False)
            else:
                warnings.warn("Waterfall plot requires DataFrame input")
                return None, None
        else:
            warnings.warn(f"Unknown plot type: {plot_type}")
            return None, None
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf(), shap_values
    
    except Exception as e:
        warnings.warn(f"SHAP analysis failed: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Example usage
    print("Testing model interpretability methods...")
    
    # Generate a small sample dataset
    import sys
    sys.path.append('..')
    from data_generation.batch_generator import generate_uniform_dataset
    from machine_learning.preprocessing import split_features_target, create_dataset_splits
    from machine_learning.model_training import train_tree_model
    
    # Generate data
    sample_df = generate_uniform_dataset(n_samples=1000)
    
    # Split features and target
    X, y = split_features_target(sample_df, 'expertise_medical')
    
    # Create train/val/test splits
    splits = create_dataset_splits(X, y)
    
    # Train a tree model for interpretability
    model = train_tree_model(splits['X_train'], splits['y_train'])
    
    # Extract feature importance
    importance_df = extract_feature_importance(model, feature_names=splits['X_train'].columns)
    print("\nFeature Importance:")
    print(importance_df)
    
    # Create partial dependence plot
    print("\nCreating partial dependence plots...")
    pdp_fig = partial_dependence_plot(model, splits['X_val'], features=[0, 1, 2], 
                                  feature_names=splits['X_val'].columns)
    
    # Create interaction effects plot
    print("\nCreating interaction effects plot...")
    interaction_fig = plot_interaction_effects(model, splits['X_val'], 
                                          feature_pairs=[(0, 1), (0, 2)], 
                                          feature_names=splits['X_val'].columns)
    
    # Try SHAP analysis
    print("\nAttempting SHAP analysis...")
    try:
        shap_fig, shap_values = try_shap_analysis(model, splits['X_val'], 
                                               feature_names=splits['X_val'].columns)
        if shap_fig is not None:
            print("SHAP analysis successful!")
        else:
            print("SHAP analysis not available.")
    except:
        print("SHAP analysis failed or not available.")