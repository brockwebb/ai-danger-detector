"""
Parameter sensitivity analysis for the expertise-harm model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_model.model_definition import expertise_required
from core_model.domain_profiles import domain_profiles


def parameter_sweep(parameter_name, values, fixed_params=None):
    """
    Perform a sweep over a range of parameter values.
    
    Parameters:
    -----------
    parameter_name : str
        Name of parameter to sweep ('harm', 'complexity', 'error_rate', 'E_min', 'beta_i', 'alpha_i')
    
    values : list or numpy.ndarray
        Parameter values to test
    
    fixed_params : dict, optional
        Fixed parameters for the model
        
    Returns:
    --------
    pandas.DataFrame
        Results of parameter sweep
    """
    # Set default fixed parameters
    if fixed_params is None:
        fixed_params = {
            'harm': 5.0,
            'complexity': 5.0,
            'error_rate': 0.15,
            'E_min': 1.0,
            'beta': [0.2, 0.3, 0.2, 0.1],
            'alpha': [1.5, 1.2, 0.8, 1.1]
        }
    
    results = []
    
    for value in values:
        # Create a copy of fixed parameters
        params = fixed_params.copy()
        
        # Update parameter being swept
        if parameter_name in ['harm', 'complexity', 'error_rate', 'E_min']:
            params[parameter_name] = value
        elif parameter_name.startswith('beta_'):
            idx = int(parameter_name.split('_')[1])
            beta_copy = params['beta'].copy()
            beta_copy[idx] = value
            params['beta'] = beta_copy
        elif parameter_name.startswith('alpha_'):
            idx = int(parameter_name.split('_')[1])
            alpha_copy = params['alpha'].copy()
            alpha_copy[idx] = value
            params['alpha'] = alpha_copy
        else:
            raise ValueError(f"Unknown parameter name: {parameter_name}")
        
        # Calculate expertise with updated parameters
        expertise = expertise_required(
            params['harm'],
            params['complexity'],
            params['error_rate'],
            E_min=params['E_min'],
            beta=params['beta'],
            alpha=params['alpha']
        )
        
        # Store result
        results.append({
            'parameter_value': value,
            'expertise': expertise
        })
    
    return pd.DataFrame(results)


def sensitivity_analysis(parameter_ranges=None, n_points=20, fixed_params=None, plot=True):
    """
    Perform sensitivity analysis on model parameters.
    
    Parameters:
    -----------
    parameter_ranges : dict, optional
        Dictionary mapping parameter names to (min, max) tuples
    
    n_points : int
        Number of points to sample in each range
    
    fixed_params : dict, optional
        Fixed parameters for the model
    
    plot : bool
        Whether to generate plot
    
    Returns:
    --------
    dict
        Dictionary mapping parameter names to DataFrames with sweep results
    """
    # Set default parameter ranges
    if parameter_ranges is None:
        parameter_ranges = {
            'harm': (1, 10),
            'complexity': (1, 10),
            'error_rate': (0.01, 0.5),
            'E_min': (0, 5),
            'beta_0': (0.05, 0.5),  # Harm coefficient
            'beta_1': (0.05, 0.5),  # Complexity coefficient
            'beta_2': (0.05, 0.5),  # Error rate coefficient
            'beta_3': (0.01, 0.3),  # Interaction coefficient
            'alpha_0': (0.5, 2.0),  # Harm exponent
            'alpha_1': (0.5, 2.0),  # Complexity exponent
            'alpha_2': (0.5, 2.0),  # Error rate exponent
            'alpha_3': (0.5, 2.0)   # Interaction exponent
        }
    
    results = {}
    
    # Perform parameter sweep for each parameter
    for param_name, (min_val, max_val) in parameter_ranges.items():
        param_values = np.linspace(min_val, max_val, n_points)
        df = parameter_sweep(param_name, param_values, fixed_params)
        results[param_name] = df
    
    # Plot results if requested
    if plot:
        plot_sensitivity_analysis(results)
    
    return results


def plot_sensitivity_analysis(results, figsize=(16, 12)):
    """
    Generate sensitivity analysis plot.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping parameter names to DataFrames with sweep results
    
    figsize : tuple
        Figure size (width, height)
    """
    # Organize parameters into categories
    param_categories = {
        'Primary': ['harm', 'complexity', 'error_rate'],
        'Minimum Expertise': ['E_min'],
        'Beta Coefficients': ['beta_0', 'beta_1', 'beta_2', 'beta_3'],
        'Alpha Exponents': ['alpha_0', 'alpha_1', 'alpha_2', 'alpha_3']
    }
    
    # Create subplots
    n_categories = len(param_categories)
    fig, axes = plt.subplots(n_categories, 1, figsize=figsize)
    
    # Plot each category
    for ax, (category, params) in zip(axes, param_categories.items()):
        for param in params:
            if param in results:
                df = results[param]
                ax.plot(df['parameter_value'], df['expertise'], label=param)
        
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Expertise Required')
        ax.set_title(f'{category} Parameters')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_sensitivity_gradients(results):
    """
    Calculate sensitivity gradients for each parameter.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping parameter names to DataFrames with sweep results
    
    Returns:
    --------
    pandas.DataFrame
        Gradients and sensitivity metrics for each parameter
    """
    gradients = []
    
    for param_name, df in results.items():
        # Calculate gradients
        df['gradient'] = df['expertise'].diff() / df['parameter_value'].diff()
        
        # Calculate sensitivity metrics
        mean_gradient = df['gradient'].mean()
        max_gradient = df['gradient'].abs().max()
        range_impact = df['expertise'].max() - df['expertise'].min()
        
        # Store results
        gradients.append({
            'parameter': param_name,
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'range_impact': range_impact,
            'normalized_impact': range_impact / (df['parameter_value'].max() - df['parameter_value'].min())
        })
    
    return pd.DataFrame(gradients).sort_values('normalized_impact', ascending=False)


if __name__ == "__main__":
    print("Performing parameter sensitivity analysis...")
    
    # Perform sensitivity analysis with default settings
    results = sensitivity_analysis()
    
    # Calculate sensitivity gradients
    gradients_df = calculate_sensitivity_gradients(results)
    print("\nParameter sensitivity ranking:")
    print(gradients_df)
    
    # Test with domain-specific parameters
    print("\nTesting domain-specific parameter sensitivity:")
    for domain, profile in domain_profiles.items():
        print(f"\nAnalyzing {domain} domain...")
        fixed_params = {
            'harm': 5.0,
            'complexity': 5.0,
            'error_rate': 0.15,
            'E_min': profile['E_min'],
            'beta': profile['beta'],
            'alpha': profile['alpha']
        }
        
        domain_results = sensitivity_analysis(
            parameter_ranges={'harm': (1, 10), 'complexity': (1, 10), 'error_rate': (0.01, 0.5)},
            fixed_params=fixed_params,
            plot=False
        )
        
        domain_gradients = calculate_sensitivity_gradients(domain_results)
        print(domain_gradients)