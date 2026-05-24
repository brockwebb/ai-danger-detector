"""
Visualization functions for comparing domains and scenarios.
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
from visualization.plotting import setup_plotting_style


def plot_domain_comparison(scenario_type='harm', fixed_values=None, domains=None, 
                         figsize=(12, 6), save_path=None):
    """
    Create a comparison plot of expertise requirements across domains.
    
    Parameters:
    -----------
    scenario_type : str
        Type of scenario to compare ('harm', 'complexity', or 'error_rate')
    
    fixed_values : dict, optional
        Fixed values for non-varied parameters
    
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
    
    # Set default fixed values
    if fixed_values is None:
        fixed_values = {
            'harm': 7.0,
            'complexity': 5.0,
            'error_rate': 0.15
        }
    
    # Default to all domains if none specified
    if domains is None:
        domains = list(domain_profiles.keys())
    
    # Validate scenario type
    valid_types = ['harm', 'complexity', 'error_rate']
    if scenario_type not in valid_types:
        raise ValueError(f"scenario_type must be one of: {valid_types}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate parameter values
    if scenario_type == 'error_rate':
        param_values = np.linspace(0.01, 0.5, 100)
        param_label = 'Error Rate'
        x_display = lambda x: f"{x*100:.0f}%"
    else:
        param_values = np.linspace(1, 10, 100)
        param_label = scenario_type.title()
        x_display = lambda x: f"{x:.1f}"
    
    # Plot each domain
    for domain in domains:
        # Calculate expertise for each parameter value
        expertise_values = []
        
        for val in param_values:
            # Set parameters based on scenario type
            params = fixed_values.copy()
            params[scenario_type] = val
            
            # Calculate expertise
            expertise = calculate_expertise_for_scenario(
                params['harm'], 
                params['complexity'], 
                params['error_rate'], 
                domain
            )
            
            expertise_values.append(expertise)
        
        # Plot curve
        ax.plot(param_values, expertise_values, 
                label=domain_profiles[domain]['name'],
                color=domain_colors.get(domain, None),
                linewidth=2.5)
    
    # Add reference line
    ax.axhline(y=7, color='r', linestyle='--', alpha=0.5, 
              label='Expert Level Threshold')
    
    # Set labels and title
    if scenario_type == 'error_rate':
        ax.set_xlabel(f'{param_label} (0-1)')
        ax.set_xlim(0, 0.5)
        ax.set_xticks(np.arange(0, 0.55, 0.05))
        ax.set_xticklabels([f'{x*100:.0f}%' for x in ax.get_xticks()])
    else:
        ax.set_xlabel(f'{param_label} (1-10)')
    
    ax.set_ylabel('Expertise Required (1-10)')
    
    # Create title with fixed parameters
    fixed_params_display = []
    for param, value in fixed_values.items():
        if param != scenario_type:
            if param == 'error_rate':
                fixed_params_display.append(f"{param.replace('_', ' ').title()}={value*100:.0f}%")
            else:
                fixed_params_display.append(f"{param.title()}={value:.1f}")
    
    fixed_params_str = ', '.join(fixed_params_display)
    title = f'Domain Comparison: Expertise vs. {param_label}\n({fixed_params_str})'
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


def plot_scenario_comparison(scenario_specs, domain=None, figsize=(12, 6), save_path=None):
    """
    Compare expertise requirements for different scenarios.
    
    Parameters:
    -----------
    scenario_specs : list of dict
        List of scenario specifications with 'name', 'harm', 'complexity', and 'error_rate'
    
    domain : str, optional
        Domain to use, if None will compare across domains
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if domain is None:
        # Compare scenarios across domains
        domains = list(domain_profiles.keys())
        
        # Create DataFrame to store results
        results = []
        
        for scenario in scenario_specs:
            for domain_name in domains:
                # Calculate expertise
                expertise = calculate_expertise_for_scenario(
                    scenario['harm'],
                    scenario['complexity'],
                    scenario['error_rate'],
                    domain_name
                )
                
                # Store result
                results.append({
                    'scenario': scenario['name'],
                    'domain': domain_profiles[domain_name]['name'],
                    'expertise': expertise,
                    'harm': scenario['harm'],
                    'complexity': scenario['complexity'],
                    'error_rate': scenario['error_rate']
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create grouped bar chart
        sns.barplot(x='scenario', y='expertise', hue='domain', data=df, ax=ax)
        
        title = 'Expertise Required by Scenario Across Domains'
    else:
        # Compare scenarios for a single domain
        results = []
        
        for scenario in scenario_specs:
            # Calculate expertise
            expertise = calculate_expertise_for_scenario(
                scenario['harm'],
                scenario['complexity'],
                scenario['error_rate'],
                domain
            )
            
            # Store result
            results.append({
                'scenario': scenario['name'],
                'expertise': expertise,
                'harm': scenario['harm'],
                'complexity': scenario['complexity'],
                'error_rate': scenario['error_rate']
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create bar chart
        sns.barplot(x='scenario', y='expertise', data=df, ax=ax, color=domain_colors.get(domain, None))
        
        domain_name = domain_profiles[domain]['name'] if domain in domain_profiles else domain
        title = f'Expertise Required by Scenario for {domain_name}'
    
    # Add reference line
    ax.axhline(y=7, color='r', linestyle='--', alpha=0.5, label='Expert Level Threshold')
    
    # Set labels and title
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Expertise Required (1-10)')
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 10.5)
    
    # Rotate x-tick labels if needed
    if len(scenario_specs) > 3:
        plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend if comparing across domains
    if domain is None:
        ax.legend(title='Domain')
    

    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_expertise_thresholds(domains=None, thresholds=None, figsize=(12, 8), save_path=None):
    """
    Plot expertise threshold contours across harm and complexity.
    
    Parameters:
    -----------
    domains : list, optional
        List of domains to include, defaults to all domains
    
    thresholds : list, optional
        List of expertise thresholds to plot
    
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
    
    # Default thresholds if none specified
    if thresholds is None:
        thresholds = [3, 5, 7, 9]  # Basic, familiar, educated, expert
    
    # Fixed error rate
    fixed_error = 0.15
    
    # Create figure with subplots for each domain
    n_domains = len(domains)
    n_cols = min(3, n_domains)
    n_rows = (n_domains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Generate parameter values
    harm_values = np.linspace(1, 10, 50)
    complexity_values = np.linspace(1, 10, 50)
    H, C = np.meshgrid(harm_values, complexity_values)
    
    # Plot each domain
    for i, domain in enumerate(domains):
        # Get subplot axis
        ax = axes[i // n_cols, i % n_cols]
        
        # Calculate expertise for each point
        Z = np.zeros_like(H)
        for j in range(len(harm_values)):
            for k in range(len(complexity_values)):
                Z[k, j] = calculate_expertise_for_scenario(
                    H[k, j], C[k, j], fixed_error, domain
                )
        
        # Create filled contour plot
        contour = ax.contourf(H, C, Z, levels=20, cmap='viridis', alpha=0.7)
        
        # Add threshold contours
        threshold_contours = ax.contour(H, C, Z, levels=thresholds, 
                                     colors=['blue', 'green', 'orange', 'red'],
                                     linewidths=2)
        
        # Add contour labels
        threshold_labels = {
            3: 'Basic',
            5: 'Familiar',
            7: 'Educated',
            9: 'Expert'
        }
        
        # Add custom labels
        for level, collection in zip(threshold_contours.levels, threshold_contours.collections):
            if level in threshold_labels:
                # Find a point on the contour for the label
                paths = collection.get_paths()
                if paths:
                    path = paths[0]
                    vertices = path.vertices
                    mid_point = vertices[len(vertices) // 2]
                    ax.annotate(threshold_labels[level], xy=(mid_point[0], mid_point[1]),
                              fontsize=9, fontweight='bold', color='white',
                              bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))
        
        # Set labels and title
        ax.set_xlabel('Harm Potential (1-10)')
        ax.set_ylabel('Complexity (1-10)')
        domain_name = domain_profiles[domain]['name'] if domain in domain_profiles else domain
        ax.set_title(f'{domain_name}\n(Error Rate={fixed_error*100:.0f}%)')
    
    # Remove empty subplots
    for i in range(n_domains, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols, i % n_cols])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Expertise Required (1-10)')
    
    # Add overall title
    fig.suptitle('Expertise Threshold Contours by Domain', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_interactive_scenario_explorer(domain, save_path=None):
    """
    Create an interactive scenario explorer using ipywidgets (for Jupyter notebooks).
    
    Parameters:
    -----------
    domain : str
        Domain to explore
    
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    function
        Interactive function for Jupyter notebook
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("This function requires ipywidgets and IPython. Please install with: pip install ipywidgets")
        return None
    
    # Create sliders for parameters
    harm_slider = widgets.FloatSlider(
        value=5.0,
        min=1.0,
        max=10.0,
        step=0.1,
        description='Harm:',
        continuous_update=False
    )
    
    complexity_slider = widgets.FloatSlider(
        value=5.0,
        min=1.0,
        max=10.0,
        step=0.1,
        description='Complexity:',
        continuous_update=False
    )
    
    error_slider = widgets.FloatSlider(
        value=0.15,
        min=0.01,
        max=0.5,
        step=0.01,
        description='Error Rate:',
        continuous_update=False,
        readout_format='.0%'
    )
    
    # Create output widget for the plot
    output = widgets.Output()
    
    # Create function to update the plot
    def update_plot(harm, complexity, error_rate):
        with output:
            output.clear_output(wait=True)
            
            # Calculate expertise
            expertise = calculate_expertise_for_scenario(harm, complexity, error_rate, domain)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Add a gauge visualization
            ax.axvspan(0, 3, alpha=0.2, color='green', label='Basic (1-3)')
            ax.axvspan(3, 7, alpha=0.2, color='yellow', label='Intermediate (3-7)')
            ax.axvspan(7, 10, alpha=0.2, color='red', label='Expert (7-10)')
            
            ax.barh(0, expertise, height=0.5, color='blue')
            ax.axvline(x=expertise, color='blue', linestyle='--', linewidth=2)
            
            # Add annotation
            ax.text(expertise, 0.25, f'{expertise:.2f}', 
                    fontsize=12, fontweight='bold', 
                    horizontalalignment='center',
                    verticalalignment='bottom')
            
            # Set labels and title
            ax.set_xlim(0, 10)
            ax.set_yticks([])
            ax.set_xlabel('Expertise Required (1-10)')
            
            domain_name = domain_profiles[domain]['name'] if domain in domain_profiles else domain
            ax.set_title(f'Required Expertise for {domain_name}\nHarm={harm:.1f}, Complexity={complexity:.1f}, Error Rate={error_rate:.0%}')
            
            # Add legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            plt.tight_layout()
            plt.show()
    
    # Create interactive function
    interactive_plot = widgets.interactive(
        update_plot,
        harm=harm_slider,
        complexity=complexity_slider,
        error_rate=error_slider
    )
    
    # Create layout
    layout = widgets.VBox([
        widgets.HTML(f"<h3>Expertise Explorer for {domain_profiles[domain]['name']}</h3>"),
        widgets.HBox([
            widgets.VBox([harm_slider, complexity_slider, error_slider]),
            output
        ])
    ])
    
    # Update the plot with initial values
    update_plot(harm_slider.value, complexity_slider.value, error_slider.value)
    
    return layout


if __name__ == "__main__":
    # Example usage
    print("Generating domain comparison plots...")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Define common scenarios
    scenarios = [
        {'name': 'Low Risk', 'harm': 2, 'complexity': 3, 'error_rate': 0.05},
        {'name': 'Medium Risk', 'harm': 5, 'complexity': 5, 'error_rate': 0.15},
        {'name': 'High Risk', 'harm': 8, 'complexity': 7, 'error_rate': 0.25},
        {'name': 'High Harm', 'harm': 9, 'complexity': 4, 'error_rate': 0.15},
        {'name': 'High Complexity', 'harm': 4, 'complexity': 9, 'error_rate': 0.15},
        {'name': 'High Error', 'harm': 5, 'complexity': 5, 'error_rate': 0.40}
    ]
    
    # Generate comparison plots
    plot_domain_comparison(scenario_type='harm', save_path='plots/domain_comparison_harm.png')
    plot_domain_comparison(scenario_type='complexity', save_path='plots/domain_comparison_complexity.png')
    plot_domain_comparison(scenario_type='error_rate', save_path='plots/domain_comparison_error.png')
    
    # Generate scenario comparison
    plot_scenario_comparison(scenarios, save_path='plots/scenario_comparison.png')
    
    # Generate threshold plots
    plot_expertise_thresholds(domains=['medical', 'legal', 'creative'], 
                           save_path='plots/expertise_thresholds.png')
    
    print("Domain comparison plots saved to 'plots' directory.")