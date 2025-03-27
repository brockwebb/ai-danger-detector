"""
Basic plotting functions for expertise-harm model visualization.
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
from data_generation.calculator import calculate_expertise_for_scenario


def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Set consistent colors for domains
    domain_colors = {
        'medical': '#1f77b4',
        'legal': '#ff7f0e',
        'creative': '#2ca02c',
        'statistical': '#d62728',
        'financial': '#9467bd',
        'educational': '#8c564b',
        'default': '#7f7f7f'
    }
    
    # Set up matplotlib parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    return domain_colors


def plot_expertise_vs_harm(domains=None, fixed_complexity=5, fixed_error=0.15, 
                         figsize=(10, 6), save_path=None):
    """
    Plot expertise required vs. harm potential for different domains.
    
    Parameters:
    -----------
    domains : list, optional
        List of domains to include, defaults to all domains
    
    fixed_complexity : float
        Fixed complexity value to use (1-10)
    
    fixed_error : float
        Fixed error rate to use (0-1)
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    domain_colors = setup_plotting_style()
    
    # Default to all domains if none specified
    if domains is None:
        domains = list(domain_profiles.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate harm values
    harm_values = np.linspace(1, 10, 100)
    
    # Plot each domain
    for domain in domains:
        expertise_values = [
            calculate_expertise_for_scenario(h, fixed_complexity, fixed_error, domain)
            for h in harm_values
        ]
        
        ax.plot(harm_values, expertise_values, 
                label=domain_profiles[domain]['name'],
                color=domain_colors.get(domain, None),
                linewidth=2.5)
    
    # Add reference line
    ax.axhline(y=7, color='r', linestyle='--', alpha=0.5, 
              label='Expert Level Threshold')
    
    # Set labels and title
    ax.set_xlabel('Harm Potential (1-10)')
    ax.set_ylabel('Expertise Required (1-10)')
    title = f'Expertise Required vs. Harm Potential\n(Complexity={fixed_complexity}, Error Rate={fixed_error*100:.0f}%)'
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 10.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Add annotations for key thresholds
    ax.text(9.5, 3, 'Low Risk Zone', fontsize=10, 
            verticalalignment='center', horizontalalignment='right')
    ax.text(9.5, 8.5, 'High Risk Zone', fontsize=10, 
            verticalalignment='center', horizontalalignment='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_expertise_vs_complexity(domains=None, fixed_harm=7, fixed_error=0.15, 
                               figsize=(10, 6), save_path=None):
    """
    Plot expertise required vs. complexity for different domains.
    
    Parameters:
    -----------
    domains : list, optional
        List of domains to include, defaults to all domains
    
    fixed_harm : float
        Fixed harm value to use (1-10)
    
    fixed_error : float
        Fixed error rate to use (0-1)
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    domain_colors = setup_plotting_style()
    
    # Default to all domains if none specified
    if domains is None:
        domains = list(domain_profiles.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate complexity values
    complexity_values = np.linspace(1, 10, 100)
    
    # Plot each domain
    for domain in domains:
        expertise_values = [
            calculate_expertise_for_scenario(fixed_harm, c, fixed_error, domain)
            for c in complexity_values
        ]
        
        ax.plot(complexity_values, expertise_values, 
                label=domain_profiles[domain]['name'],
                color=domain_colors.get(domain, None),
                linewidth=2.5)
    
    # Add reference line
    ax.axhline(y=7, color='r', linestyle='--', alpha=0.5, 
              label='Expert Level Threshold')
    
    # Set labels and title
    ax.set_xlabel('Complexity (1-10)')
    ax.set_ylabel('Expertise Required (1-10)')
    title = f'Expertise Required vs. Complexity\n(Harm={fixed_harm}, Error Rate={fixed_error*100:.0f}%)'
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 10.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_expertise_vs_error_rate(domains=None, fixed_harm=7, fixed_complexity=5, 
                               figsize=(10, 6), save_path=None):
    """
    Plot expertise required vs. error rate for different domains.
    
    Parameters:
    -----------
    domains : list, optional
        List of domains to include, defaults to all domains
    
    fixed_harm : float
        Fixed harm value to use (1-10)
    
    fixed_complexity : float
        Fixed complexity value to use (1-10)
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    domain_colors = setup_plotting_style()
    
    # Default to all domains if none specified
    if domains is None:
        domains = list(domain_profiles.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate error rate values
    error_values = np.linspace(0.01, 0.5, 100)
    
    # Plot each domain
    for domain in domains:
        expertise_values = [
            calculate_expertise_for_scenario(fixed_harm, fixed_complexity, e, domain)
            for e in error_values
        ]
        
        ax.plot(error_values, expertise_values, 
                label=domain_profiles[domain]['name'],
                color=domain_colors.get(domain, None),
                linewidth=2.5)
    
    # Add reference line
    ax.axhline(y=7, color='r', linestyle='--', alpha=0.5, 
              label='Expert Level Threshold')
    
    # Set labels and title
    ax.set_xlabel('Error Rate (0-1)')
    ax.set_ylabel('Expertise Required (1-10)')
    title = f'Expertise Required vs. Error Rate\n(Harm={fixed_harm}, Complexity={fixed_complexity})'
    ax.set_title(title)
    
    # Format x-axis as percentage
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.55, 0.05))
    ax.set_xticklabels([f'{x*100:.0f}%' for x in ax.get_xticks()])
    
    # Set y-axis limits
    ax.set_ylim(0, 10.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_3d_surface(domain, figsize=(12, 10), save_path=None):
    """
    Create a 3D surface plot showing expertise as a function of harm and complexity.
    
    Parameters:
    -----------
    domain : str
        Domain to plot
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Set fixed error rate
    fixed_error = 0.15
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate grid values
    harm_values = np.linspace(1, 10, 20)
    complexity_values = np.linspace(1, 10, 20)
    H, C = np.meshgrid(harm_values, complexity_values)
    
    # Calculate expertise for each point
    Z = np.zeros_like(H)
    for i in range(len(harm_values)):
        for j in range(len(complexity_values)):
            Z[j, i] = calculate_expertise_for_scenario(
                H[j, i], C[j, i], fixed_error, domain
            )
    
    # Create surface plot
    surf = ax.plot_surface(H, C, Z, cmap='viridis', alpha=0.8,
                         linewidth=0, antialiased=True)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Expertise Required (1-10)')
    
    # Set labels and title
    ax.set_xlabel('Harm Potential (1-10)')
    ax.set_ylabel('Complexity (1-10)')
    ax.set_zlabel('Expertise Required (1-10)')
    domain_name = domain_profiles[domain]['name'] if domain in domain_profiles else domain
    title = f'Expertise Required for {domain_name}\n(Error Rate={fixed_error*100:.0f}%)'
    ax.set_title(title)
    
    # Set z-axis limits
    ax.set_zlim(0, 10.5)
    
    # Add contour plot projection on the bottom
    ax.contourf(H, C, Z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
    
    # Add annotation for expert threshold
    # Create a plane at z=7 to represent expert threshold
    xx, yy = np.meshgrid(np.linspace(1, 10, 2), np.linspace(1, 10, 2))
    zz = np.ones_like(xx) * 7
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.2)
    ax.text(5, 5, 7.5, 'Expert Level Threshold', color='r')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_heatmap(domain, parameter1='harm', parameter2='complexity', 
                         fixed_params=None, figsize=(10, 8), save_path=None):
    """
    Create a heatmap showing expertise required as a function of two parameters.
    
    Parameters:
    -----------
    domain : str
        Domain to plot
    
    parameter1 : str
        First parameter to vary ('harm', 'complexity', or 'error_rate')
    
    parameter2 : str
        Second parameter to vary ('harm', 'complexity', or 'error_rate')
    
    fixed_params : dict, optional
        Fixed values for the non-varied parameter
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Set default fixed parameters
    if fixed_params is None:
        fixed_params = {
            'harm': 5.0,
            'complexity': 5.0,
            'error_rate': 0.15
        }
    
    # Validate parameters
    if parameter1 == parameter2:
        raise ValueError("parameter1 and parameter2 must be different")
    
    valid_params = ['harm', 'complexity', 'error_rate']
    if parameter1 not in valid_params or parameter2 not in valid_params:
        raise ValueError(f"Parameters must be one of: {valid_params}")
    
    # Determine the fixed parameter
    fixed_param = [p for p in valid_params if p != parameter1 and p != parameter2][0]
    fixed_value = fixed_params[fixed_param]
    
    # Create parameter ranges
    if parameter1 == 'error_rate':
        param1_values = np.linspace(0.01, 0.5, 20)
    else:
        param1_values = np.linspace(1, 10, 20)
        
    if parameter2 == 'error_rate':
        param2_values = np.linspace(0.01, 0.5, 20)
    else:
        param2_values = np.linspace(1, 10, 20)
    
    # Create meshgrid
    P1, P2 = np.meshgrid(param1_values, param2_values)
    
    # Calculate expertise for each point
    Z = np.zeros_like(P1)
    for i in range(len(param1_values)):
        for j in range(len(param2_values)):
            # Set parameters for this calculation
            params = {
                parameter1: P1[j, i],
                parameter2: P2[j, i],
                fixed_param: fixed_value
            }
            
            Z[j, i] = calculate_expertise_for_scenario(
                params['harm'], 
                params['complexity'], 
                params['error_rate'], 
                domain
            )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.pcolormesh(P1, P2, Z, cmap='viridis', shading='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Expertise Required (1-10)')
    
    # Set labels and title
    # Format labels based on parameter type
    param1_label = parameter1.replace('_', ' ').title()
    if parameter1 == 'error_rate':
        param1_label += ' (0-1)'
    else:
        param1_label += ' (1-10)'
        
    param2_label = parameter2.replace('_', ' ').title()
    if parameter2 == 'error_rate':
        param2_label += ' (0-1)'
    else:
        param2_label += ' (1-10)'
    
    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    
    domain_name = domain_profiles[domain]['name'] if domain in domain_profiles else domain
    fixed_param_label = fixed_param.replace('_', ' ').title()
    
    if fixed_param == 'error_rate':
        fixed_display = f"{fixed_value*100:.0f}%"
    else:
        fixed_display = f"{fixed_value:.1f}"
    
    title = f'Expertise Required for {domain_name}\n({fixed_param_label}={fixed_display})'
    ax.set_title(title)
    
    # Add expert threshold contour
    contour = ax.contour(P1, P2, Z, levels=[7], colors='r', linestyles='dashed')
    plt.clabel(contour, inline=True, fontsize=10, fmt='Expert Level')
    
    # Format axes if showing error rate
    if parameter1 == 'error_rate':
        ax.set_xlim(0.01, 0.5)
        ax.set_xticks(np.linspace(0.05, 0.5, 10))
        ax.set_xticklabels([f'{x*100:.0f}%' for x in ax.get_xticks()])
    
    if parameter2 == 'error_rate':
        ax.set_ylim(0.01, 0.5)
        ax.set_yticks(np.linspace(0.05, 0.5, 10))
        ax.set_yticklabels([f'{y*100:.0f}%' for y in ax.get_yticks()])
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_domain_radar_chart(domains=None, figsize=(10, 8), save_path=None):
    """
    Create a radar chart comparing domain profile parameters.
    
    Parameters:
    -----------
    domains : list, optional
        List of domains to include, defaults to all domains
    
    figsize : tuple
        Figure size
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    domain_colors = setup_plotting_style()
    
    # Default to all domains if none specified
    if domains is None:
        domains = list(domain_profiles.keys())
    
    # Categories for radar chart
    categories = [
        'Minimum\nExpertise',
        'Harm\nSensitivity',
        'Complexity\nSensitivity',
        'Error\nSensitivity',
        'Interaction\nEffect'
    ]
    
    # Number of categories
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each domain
    for domain in domains:
        profile = domain_profiles[domain]
        
        # Extract normalized values from profile
        values = [
            profile['E_min'] / 5.0,  # Normalize by maximum E_min
            profile['beta'][0] * 2.5,  # Harm sensitivity (beta_0)
            profile['beta'][1] * 2.5,  # Complexity sensitivity (beta_1)
            profile['beta'][2] * 5.0,  # Error sensitivity (beta_2)
            profile['beta'][3] * 10.0  # Interaction effect (beta_3)
        ]
        
        # Close the polygon
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, label=profile['name'], 
               color=domain_colors.get(domain, None))
        ax.fill(angles, values, alpha=0.1, color=domain_colors.get(domain, None))
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Remove radial labels and set y-axis limits
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    ax.set_title('Domain Parameter Comparison', size=15)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Generating example plots...")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Generate basic plots
    plot_expertise_vs_harm(save_path='plots/expertise_vs_harm.png')
    plot_expertise_vs_complexity(save_path='plots/expertise_vs_complexity.png')
    plot_expertise_vs_error_rate(save_path='plots/expertise_vs_error_rate.png')
    
    # Generate specific domain plots
    plot_3d_surface('medical', save_path='plots/medical_3d_surface.png')
    plot_parameter_heatmap('legal', save_path='plots/legal_heatmap.png')
    
    # Generate comparative plots
    plot_domain_radar_chart(save_path='plots/domain_radar.png')
    
    print("Example plots saved to 'plots' directory.")