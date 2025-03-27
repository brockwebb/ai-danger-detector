"""
Machine learning module for expertise-harm modeling.
"""

from .preprocessing import (
    load_dataset,
    split_features_target,
    create_dataset_splits,
    preprocess_data,
    create_polynomial_features,
    create_domain_specific_features
)

from .model_training import (
    train_linear_model,
    train_tree_model,
    train_advanced_model,
    evaluate_model,
    plot_predictions,
    plot_residuals,
    train_and_evaluate_multiple_models,
    compare_models,
    save_model,
    load_model
)

from .interpretability import (
    extract_feature_importance,
    plot_feature_importance,
    partial_dependence_plot,
    plot_interaction_effects,
    try_shap_analysis
)

__all__ = [
    # Preprocessing
    'load_dataset',
    'split_features_target',
    'create_dataset_splits',
    'preprocess_data',
    'create_polynomial_features',
    'create_domain_specific_features',
    
    # Model training
    'train_linear_model',
    'train_tree_model',
    'train_advanced_model',
    'evaluate_model',
    'plot_predictions',
    'plot_residuals',
    'train_and_evaluate_multiple_models',
    'compare_models',
    'save_model',
    'load_model',
    
    # Interpretability
    'extract_feature_importance',
    'plot_feature_importance',
    'partial_dependence_plot',
    'plot_interaction_effects',
    'try_shap_analysis'
]