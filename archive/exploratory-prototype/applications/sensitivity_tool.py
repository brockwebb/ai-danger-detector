"""
Sensitivity analyzer tool for the expertise-harm model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_model.model_definition import expertise_required
from core_model.domain_profiles import domain_profiles


class SensitivityAnalyzer:
    """Tool for analyzing parameter sensitivity in the expertise-harm model."""
    
    def __init__(self, domain=None, fixed_params=None):
        """
        Initialize the sensitivity analyzer.
        
        Parameters:
        -----------
        domain : str, optional
            Domain to use for analysis
        
        fixed_params : dict, optional
            Fixed parameters for the model
        """
        self.domain = domain
        
        # Set default fixed parameters
        if fixed_params is None:
            self.fixed_params = {
                'harm': 5.0,
                'complexity': 5.0,
                'error_rate': 0.15,
                'E_min': 1.0,
                'beta': [0.2, 0.3, 0.2, 0.1],
                'alpha': [1.5, 1.2, 0.8, 1.1]
            }
        else:
            self.fixed_params = fixed_params
        
        # Update fixed parameters with domain profile if specified
        if domain is not None and domain in domain_profiles:
            profile = domain_profiles[domain]
            self.fixed_params['E_min'] = profile['E_min']
            self.fixed_params['beta'] = profile['beta']
            self.fixed_params['alpha'] = profile['alpha']
    
    def calculate_expertise(self, **params):
        """
        Calculate expertise with the given parameters.
        
        Parameters:
        -----------
        **params : dict
            Parameters to override fixed parameters
        
        Returns:
        --------
        float
            Calculated expertise
        """
        # Start with fixed parameters
        p = self.fixed_params.copy()
        
        # Update with provided parameters
        for key, value in params.items():
            if key in p:
                p[key] = value
            elif key.startswith('beta_'):
                idx = int(key.split('_')[1])
                beta_copy = p['beta'].copy()
                beta_copy[idx] = value
                p['beta'] = beta_copy
            elif key.startswith('alpha_'):
                idx = int(key.split('_')[1])
                alpha_copy = p['alpha'].copy()
                alpha_copy[idx] = value
                p['alpha'] = alpha_copy
        
        # Calculate expertise
        expertise = expertise_required(
            p['harm'],
            p['complexity'],
            p['error_rate'],
            E_min=p['E_min'],
            beta=p['beta'],
            alpha=p['alpha']
        )
        
        return expertise
    
    def analyze_parameter_range(self, parameter, values):
        """
        Analyze how expertise changes across a range of parameter values.
        
        Parameters:
        -----------
        parameter : str
            Parameter to analyze
        
        values : array-like
            Values to test
        
        Returns:
        --------
        pandas.DataFrame
            Results of the analysis
        """
        results = []
        
        for value in values:
            # Calculate expertise with this parameter value
            expertise = self.calculate_expertise(**{parameter: value})
            
            # Store result
            results.append({
                'parameter': parameter,
                'value': value,
                'expertise': expertise
            })
        
        return pd.DataFrame(results)
    
    def analyze_multiple_parameters(self, parameter_ranges):
        """
        Analyze multiple parameters at once.
        
        Parameters:
        -----------
        parameter_ranges : dict
            Dictionary mapping parameter names to lists of values
        
        Returns:
        --------
        dict
            Dictionary mapping parameter names to analysis results
        """
        results = {}
        
        for parameter, values in parameter_ranges.items():
            results[parameter] = self.analyze_parameter_range(parameter, values)
        
        return results
    
    def plot_sensitivity_curves(self, results, figsize=(12, 8), save_path=None):
        """
        Plot sensitivity curves for analyzed parameters.
        
        Parameters:
        -----------
        results : dict
            Dictionary mapping parameter names to analysis results
        
        figsize : tuple
            Figure size
        
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Group parameters by type
        param_groups = {
            'Primary': ['harm', 'complexity', 'error_rate'],
            'Minimum Expertise': ['E_min'],
            'Beta Coefficients': [f'beta_{i}' for i in range(4)],
            'Alpha Exponents': [f'alpha_{i}' for i in range(4)]
        }
        
        # Count how many groups have parameters in results
        active_groups = sum(1 for _, params in param_groups.items() 
                           if any(p in results for p in params))
        
        # Create figure with subplots
        fig, axes = plt.subplots(active_groups, 1, figsize=figsize, sharex=False)
        
        # Ensure axes is iterable
        if active_groups == 1:
            axes = [axes]
        
        # Counter for active groups
        i = 0
        
        # Plot each group
        for group_name, params in param_groups.items():
            # Check if any parameters in this group were analyzed
            group_params = [p for p in params if p in results]
            
            if not group_params:
                continue
            
            ax = axes[i]
            i += 1
            
            # Plot each parameter in this group
            for param in group_params:
                df = results[param]
                ax.plot(df['value'], df['expertise'], 
                        label=param, linewidth=2, marker='o')
            
            # Add expert threshold line
            ax.axhline(y=7, color='r', linestyle='--', alpha=0.5, 
                      label='Expert Level Threshold')
            
            # Set labels and title
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Expertise Required')
            ax.set_title(f'{group_name} Parameters')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_threshold_crossings(self, parameter, values, thresholds=None):
        """
        Analyze where parameter values cross expertise thresholds.
        
        Parameters:
        -----------
        parameter : str
            Parameter to analyze
        
        values : array-like
            Values to test
        
        thresholds : list, optional
            Expertise thresholds to check
        
        Returns:
        --------
        pandas.DataFrame
            Threshold crossing points
        """
        # Default thresholds if none provided
        if thresholds is None:
            thresholds = [3, 5, 7, 9]  # Basic, familiar, educated, expert
        
        # Analyze parameter across values
        df = self.analyze_parameter_range(parameter, values)
        
        # Find crossings
        crossings = []
        
        for threshold in thresholds:
            # Find where expertise crosses the threshold
            for i in range(1, len(df)):
                prev_val = df.iloc[i-1]['expertise']
                curr_val = df.iloc[i]['expertise']
                
                if (prev_val < threshold and curr_val >= threshold) or \
                   (prev_val >= threshold and curr_val < threshold):
                    # Interpolate to find exact crossing point
                    prev_param = df.iloc[i-1]['value']
                    curr_param = df.iloc[i]['value']
                    
                    # Linear interpolation
                    t = (threshold - prev_val) / (curr_val - prev_val)
                    crossing_point = prev_param + t * (curr_param - prev_param)
                    
                    direction = 'up' if curr_val > prev_val else 'down'
                    
                    crossings.append({
                        'parameter': parameter,
                        'threshold': threshold,
                        'crossing_point': crossing_point,
                        'direction': direction
                    })
        
        return pd.DataFrame(crossings)
    
    def plot_parameter_heatmap(self, param1, values1, param2, values2, figsize=(10, 8), save_path=None):
        """
        Create a heatmap showing expertise as a function of two parameters.
        
        Parameters:
        -----------
        param1 : str
            First parameter to vary
        
        values1 : array-like
            Values for first parameter
        
        param2 : str
            Second parameter to vary
        
        values2 : array-like
            Values for second parameter
        
        figsize : tuple
            Figure size
        
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Create grid of values
        V1, V2 = np.meshgrid(values1, values2)
        
        # Calculate expertise for each combination
        Z = np.zeros_like(V1)
        
        for i in range(len(values1)):
            for j in range(len(values2)):
                Z[j, i] = self.calculate_expertise(**{
                    param1: values1[i],
                    param2: values2[j]
                })
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.pcolormesh(V1, V2, Z, cmap='viridis', shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Expertise Required (1-10)')
        
        # Set labels and title
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        
        domain_str = f" for {domain_profiles[self.domain]['name']}" if self.domain else ""
        ax.set_title(f'Expertise Required{domain_str}: {param1} vs {param2}')
        
        # Add expert threshold contour
        contour = ax.contour(V1, V2, Z, levels=[3, 5, 7, 9], 
                           colors=['blue', 'green', 'orange', 'red'], 
                           linestyles='dashed')
        
        # Add contour labels
        fmt = {
            3: 'Basic',
            5: 'Familiar',
            7: 'Educated',
            9: 'Expert'
        }
        
        ax.clabel(contour, inline=True, fontsize=10, 
                fmt=lambda x: fmt.get(x, f'{x:.1f}'))
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("Testing Sensitivity Analyzer...")
    
    # Create analyzer for medical domain
    analyzer = SensitivityAnalyzer(domain='medical')
    
    # Analyze how expertise changes with harm
    harm_results = analyzer.analyze_parameter_range('harm', np.linspace(1, 10, 50))
    print("\nExpertise vs. Harm:")
    print(harm_results.head())
    
    # Analyze multiple parameters
    param_ranges = {
        'harm': np.linspace(1, 10, 20),
        'complexity': np.linspace(1, 10, 20),
        'error_rate': np.linspace(0.01, 0.5, 20)
    }
    
    results = analyzer.analyze_multiple_parameters(param_ranges)
    
    # Plot sensitivity curves
    analyzer.plot_sensitivity_curves(results)
    
    # Find threshold crossings
    crossings = analyzer.analyze_threshold_crossings('harm', np.linspace(1, 10, 100))
    print("\nThreshold Crossings:")
    print(crossings)
    
    # Create parameter heatmap
    analyzer.plot_parameter_heatmap('harm', np.linspace(1, 10, 20), 
                                 'complexity', np.linspace(1, 10, 20))
    
    plt.show()