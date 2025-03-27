"""
Batch generation of expertise requirement data for various scenarios.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# Add the parent directory to sys.path to import core_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.calculator import calculate_expertise_for_scenario
from core_model.domain_profiles import domain_profiles

def generate_uniform_dataset(n_samples=1000, random_seed=42):
    """
    Generate a dataset with uniform random sampling across all parameters.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with columns for all parameters and expertise requirements
    """
    np.random.seed(random_seed)
    
    # Generate random parameter values
    harm_values = np.random.uniform(1, 10, n_samples)
    complexity_values = np.random.uniform(1, 10, n_samples)
    error_rates = np.random.uniform(0.01, 0.5, n_samples)
    
    # Prepare dataframe
    df = pd.DataFrame({
        'harm': harm_values,
        'complexity': complexity_values,
        'error_rate': error_rates
    })
    
    # Calculate expertise requirements for each domain
    for domain in domain_profiles.keys():
        df[f'expertise_{domain}'] = df.apply(
            lambda row: calculate_expertise_for_scenario(
                row['harm'], 
                row['complexity'], 
                row['error_rate'], 
                domain
            ),
            axis=1
        )
    
    # Add default expertise calculation
    df['expertise_default'] = df.apply(
        lambda row: calculate_expertise_for_scenario(
            row['harm'], 
            row['complexity'], 
            row['error_rate']
        ),
        axis=1
    )
    
    return df


def generate_domain_specific_dataset(domain, n_samples=1000, random_seed=42):
    """
    Generate a dataset with parameter distributions tailored to a specific domain.
    
    Parameters:
    -----------
    domain : str
        Domain name to use for parameter skewing
    
    n_samples : int
        Number of samples to generate
    
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with parameters skewed according to domain characteristics
    """
    np.random.seed(random_seed)
    
    # Domain-specific parameter distributions
    domain_distributions = {
        "medical": {
            "harm": {"mean": 7.0, "std": 1.5, "min": 1, "max": 10},
            "complexity": {"mean": 6.0, "std": 2.0, "min": 1, "max": 10},
            "error_rate": {"mean": 0.15, "std": 0.08, "min": 0.01, "max": 0.5}
        },
        "legal": {
            "harm": {"mean": 6.0, "std": 1.8, "min": 1, "max": 10},
            "complexity": {"mean": 7.0, "std": 1.5, "min": 1, "max": 10},
            "error_rate": {"mean": 0.18, "std": 0.1, "min": 0.01, "max": 0.5}
        },
        "creative": {
            "harm": {"mean": 3.0, "std": 1.5, "min": 1, "max": 10},
            "complexity": {"mean": 4.0, "std": 2.0, "min": 1, "max": 10},
            "error_rate": {"mean": 0.25, "std": 0.15, "min": 0.01, "max": 0.5}
        },
        "statistical": {
            "harm": {"mean": 5.0, "std": 1.8, "min": 1, "max": 10},
            "complexity": {"mean": 6.5, "std": 1.8, "min": 1, "max": 10},
            "error_rate": {"mean": 0.12, "std": 0.08, "min": 0.01, "max": 0.5}
        },
        "financial": {
            "harm": {"mean": 6.5, "std": 1.5, "min": 1, "max": 10},
            "complexity": {"mean": 6.5, "std": 1.5, "min": 1, "max": 10},
            "error_rate": {"mean": 0.12, "std": 0.07, "min": 0.01, "max": 0.5}
        },
        "educational": {
            "harm": {"mean": 4.5, "std": 1.8, "min": 1, "max": 10},
            "complexity": {"mean": 5.5, "std": 2.0, "min": 1, "max": 10},
            "error_rate": {"mean": 0.2, "std": 0.1, "min": 0.01, "max": 0.5}
        }
    }
    
    if domain not in domain_distributions:
        raise ValueError(f"No distribution defined for domain: {domain}")
    
    dist = domain_distributions[domain]
    
    # Generate random parameter values from truncated normal distributions
    def truncated_normal(mean, std, min_val, max_val, size):
        values = np.random.normal(mean, std, size)
        return np.clip(values, min_val, max_val)
    
    harm_values = truncated_normal(
        dist["harm"]["mean"], 
        dist["harm"]["std"], 
        dist["harm"]["min"], 
        dist["harm"]["max"], 
        n_samples
    )
    
    complexity_values = truncated_normal(
        dist["complexity"]["mean"], 
        dist["complexity"]["std"], 
        dist["complexity"]["min"], 
        dist["complexity"]["max"], 
        n_samples
    )
    
    error_rates = truncated_normal(
        dist["error_rate"]["mean"], 
        dist["error_rate"]["std"], 
        dist["error_rate"]["min"], 
        dist["error_rate"]["max"], 
        n_samples
    )
    
    # Prepare dataframe
    df = pd.DataFrame({
        'harm': harm_values,
        'complexity': complexity_values,
        'error_rate': error_rates,
        'domain': domain
    })
    
    # Calculate expertise requirements
    df['expertise'] = df.apply(
        lambda row: calculate_expertise_for_scenario(
            row['harm'], 
            row['complexity'], 
            row['error_rate'], 
            domain
        ),
        axis=1
    )
    
    return df


def generate_grid_dataset(n_points=10, domain=None):
    """
    Generate a systematic grid of parameter values for visualization.
    
    Parameters:
    -----------
    n_points : int
        Number of points along each dimension
    
    domain : str, optional
        Domain to calculate expertise values for
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with systematic grid sampling of parameter space
    """
    # Create grid points
    harm_values = np.linspace(1, 10, n_points)
    complexity_values = np.linspace(1, 10, n_points)
    error_rates = np.linspace(0.05, 0.5, n_points)
    
    # Create meshgrid for all combinations
    grid = []
    for h in harm_values:
        for c in complexity_values:
            for e in error_rates:
                grid.append((h, c, e))
    
    # Convert to dataframe
    df = pd.DataFrame(grid, columns=['harm', 'complexity', 'error_rate'])
    
    # Calculate expertise for specified domain or all domains
    if domain is not None:
        df['expertise'] = df.apply(
            lambda row: calculate_expertise_for_scenario(
                row['harm'], 
                row['complexity'], 
                row['error_rate'], 
                domain
            ),
            axis=1
        )
    else:
        for domain in domain_profiles.keys():
            df[f'expertise_{domain}'] = df.apply(
                lambda row: calculate_expertise_for_scenario(
                    row['harm'], 
                    row['complexity'], 
                    row['error_rate'], 
                    domain
                ),
                axis=1
            )
    
    return df


def save_dataset(df, filename, **kwargs):
    """Save dataset to file with appropriate format."""
    if filename.endswith('.csv'):
        df.to_csv(filename, index=False, **kwargs)
    elif filename.endswith('.parquet'):
        df.to_parquet(filename, index=False, **kwargs)
    elif filename.endswith('.pkl'):
        df.to_pickle(filename, **kwargs)
    else:
        df.to_csv(filename + '.csv', index=False, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Generating example datasets...")
    
    # Generate small uniform dataset
    df_uniform = generate_uniform_dataset(n_samples=100)
    print(f"Uniform dataset shape: {df_uniform.shape}")
    print(df_uniform.head())
    
    # Generate domain-specific dataset
    df_medical = generate_domain_specific_dataset(domain="medical", n_samples=100)
    print(f"\nMedical domain dataset shape: {df_medical.shape}")
    print(df_medical.head())
    
    # Generate small grid dataset
    df_grid = generate_grid_dataset(n_points=5, domain="legal")
    print(f"\nGrid dataset shape: {df_grid.shape}")
    print(df_grid.head())