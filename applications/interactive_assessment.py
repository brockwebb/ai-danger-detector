"""
Interactive risk assessment tool for the expertise-harm model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_model.model_definition import expertise_required
from core_model.domain_profiles import domain_profiles
from data_generation.calculator import calculate_expertise_for_scenario


class RiskAssessmentTool:
    """Interactive tool for assessing expertise requirements for AI applications."""
    
    def __init__(self):
        """Initialize the risk assessment tool."""
        self.domains = list(domain_profiles.keys())
        self.expertise_levels = {
            (1, 3): ('Basic', 'General user with basic knowledge'),
            (3, 5): ('Familiar', 'Domain-familiar user with some training'),
            (5, 7): ('Educated', 'Domain-educated user with formal education'),
            (7, 10): ('Expert', 'Domain expert with specialized training')
        }
    
    def assess_scenario(self, harm, complexity, error_rate, domain=None):
        """
        Assess expertise requirements for a specific scenario.
        
        Parameters:
        -----------
        harm : float
            Harm potential (1-10)
        
        complexity : float
            Complexity (1-10)
        
        error_rate : float
            Error rate (0-1)
        
        domain : str, optional
            Domain to use for assessment
        
        Returns:
        --------
        dict
            Assessment results
        """
        # Calculate expertise requirement
        if domain is not None:
            expertise = calculate_expertise_for_scenario(harm, complexity, error_rate, domain)
            domain_name = domain_profiles[domain]['name']
        else:
            expertise = expertise_required(harm, complexity, error_rate)
            domain_name = 'Default'
        
        # Determine expertise level
        level_name = None
        level_description = None
        
        for (min_val, max_val), (name, description) in self.expertise_levels.items():
            if min_val <= expertise <= max_val:
                level_name = name
                level_description = description
                break
        
        # Create assessment result
        result = {
            'domain': domain_name,
            'harm': harm,
            'complexity': complexity,
            'error_rate': error_rate,
            'expertise_required': expertise,
            'level_name': level_name,
            'level_description': level_description,
            'is_expert_level': expertise >= 7.0,
            'is_basic_level': expertise <= 3.0
        }
        
        return result
    
def compare_domains(self, harm, complexity, error_rate, domains=None):
        """
        Compare expertise requirements across domains.
        
        Parameters:
        -----------
        harm : float
            Harm potential (1-10)
        
        complexity : float
            Complexity (1-10)
        
        error_rate : float
            Error rate (0-1)
        
        domains : list, optional
            List of domains to compare
        
        Returns:
        --------
        pandas.DataFrame
            Comparison results
        """
        # Default to all domains if none specified
        if domains is None:
            domains = self.domains
        
        # Assess each domain
        results = []
        
        for domain in domains:
            result = self.assess_scenario(harm, complexity, error_rate, domain)
            results.append(result)
        
        # Add default assessment
        default_result = self.assess_scenario(harm, complexity, error_rate)
        default_result['domain'] = 'Default Parameters'
        results.append(default_result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns
        columns = ['domain', 'expertise_required', 'level_name', 'level_description',
                  'harm', 'complexity', 'error_rate', 'is_expert_level', 'is_basic_level']
        df = df[columns]
        
        return df
    
    def plot_assessment_gauge(self, result, figsize=(10, 6), save_path=None):
        """
        Create a gauge visualization of the assessment result.
        
        Parameters:
        -----------
        result : dict
            Assessment result from assess_scenario
        
        figsize : tuple
            Figure size
        
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Add a gauge visualization
        ax.axvspan(0, 3, alpha=0.2, color='green', label='Basic (1-3)')
        ax.axvspan(3, 7, alpha=0.2, color='yellow', label='Intermediate (3-7)')
        ax.axvspan(7, 10, alpha=0.2, color='red', label='Expert (7-10)')
        
        ax.barh(0, result['expertise_required'], height=0.5, color='blue')
        ax.axvline(x=result['expertise_required'], color='blue', linestyle='--', linewidth=2)
        
        # Add annotation
        ax.text(result['expertise_required'], 0.25, f"{result['expertise_required']:.2f}", 
               fontsize=12, fontweight='bold', 
               horizontalalignment='center',
               verticalalignment='bottom')
        
        # Set labels and title
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.set_xlabel('Expertise Required (1-10)')
        
        ax.set_title(f"Required Expertise for {result['domain']}\n"
                    f"Harm={result['harm']:.1f}, Complexity={result['complexity']:.1f}, "
                    f"Error Rate={result['error_rate']:.1%}")
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_domain_comparison(self, comparison_df, figsize=(10, 6), save_path=None):
        """
        Create a bar chart comparing expertise requirements across domains.
        
        Parameters:
        -----------
        comparison_df : pandas.DataFrame
            Comparison results from compare_domains
        
        figsize : tuple
            Figure size
        
        save_path : str, optional
            Path to save figure
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by expertise required
        df = comparison_df.sort_values('expertise_required')
        
        # Create bar chart
        bars = ax.barh(df['domain'], df['expertise_required'])
        
        # Color bars based on expertise level
        for i, bar in enumerate(bars):
            level = df.iloc[i]['level_name']
            if level == 'Expert':
                bar.set_color('red')
            elif level == 'Educated':
                bar.set_color('orange')
            elif level == 'Familiar':
                bar.set_color('yellow')
            else:
                bar.set_color('green')
        
        # Add expertise values on bars
        for i, v in enumerate(df['expertise_required']):
            ax.text(v + 0.1, i, f"{v:.2f}", va='center')
        
        # Set labels and title
        ax.set_xlabel('Expertise Required (1-10)')
        ax.set_title(f"Domain Comparison: Required Expertise\n"
                    f"Harm={df['harm'].iloc[0]:.1f}, Complexity={df['complexity'].iloc[0]:.1f}, "
                    f"Error Rate={df['error_rate'].iloc[0]:.1%}")
        
        # Add reference lines
        ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5)
        
        # Add expertise level labels
        ax.text(1.5, -0.5, 'Basic', ha='center', color='green')
        ax.text(5.0, -0.5, 'Intermediate', ha='center', color='orange')
        ax.text(8.5, -0.5, 'Expert', ha='center', color='red')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_recommendations(self, result):
        """
        Generate recommendations based on assessment result.
        
        Parameters:
        -----------
        result : dict
            Assessment result from assess_scenario
        
        Returns:
        --------
        dict
            Recommendations
        """
        # Initialize recommendations
        recommendations = {
            'summary': None,
            'expertise_requirement': None,
            'risk_mitigation': [],
            'training_needs': [],
            'supervision': None,
            'alternative_approaches': []
        }
        
        # Basic summary
        expertise = result['expertise_required']
        level = result['level_name']
        domain = result['domain']
        
        # Generate summary
        if expertise < 3:
            recommendations['summary'] = (
                f"This {domain} application has low expertise requirements. "
                f"It is suitable for general users with basic knowledge."
            )
        elif expertise < 5:
            recommendations['summary'] = (
                f"This {domain} application has moderate expertise requirements. "
                f"Users should be familiar with the domain."
            )
        elif expertise < 7:
            recommendations['summary'] = (
                f"This {domain} application has substantial expertise requirements. "
                f"Users should have formal education or training in the domain."
            )
        else:
            recommendations['summary'] = (
                f"This {domain} application has high expertise requirements. "
                f"Only domain experts with specialized training should use this application."
            )
        
        # Expertise requirement
        recommendations['expertise_requirement'] = (
            f"Required expertise level: {level} ({expertise:.2f}/10)"
        )
        
        # Risk mitigation strategies
        if result['harm'] > 7:
            recommendations['risk_mitigation'].append(
                "Implement additional safety mechanisms to reduce potential harm."
            )
        
        if result['complexity'] > 7:
            recommendations['risk_mitigation'].append(
                "Consider simplifying the application or providing better guidance on complex aspects."
            )
        
        if result['error_rate'] > 0.2:
            recommendations['risk_mitigation'].append(
                "Use a more accurate AI model to reduce error rates, or implement stronger verification processes."
            )
        
        # Training needs
        if expertise >= 5:
            recommendations['training_needs'].append(
                f"Users should have specific training in {domain} concepts and terminology."
            )
        
        if expertise >= 7:
            recommendations['training_needs'].append(
                "Users should have advanced training in identifying and correcting AI errors."
            )
        
        # Supervision
        if expertise < 3:
            recommendations['supervision'] = "Minimal supervision required."
        elif expertise < 7:
            recommendations['supervision'] = "Periodic review by domain experts recommended."
        else:
            recommendations['supervision'] = "Continuous expert supervision required."
        
        # Alternative approaches
        if expertise >= 7:
            recommendations['alternative_approaches'].append(
                "Consider a human-AI collaborative approach with clearer division of responsibilities."
            )
            recommendations['alternative_approaches'].append(
                "Implement a tiered system where AI makes initial recommendations, but experts make final decisions."
            )
        
        if result['error_rate'] > 0.3:
            recommendations['alternative_approaches'].append(
                "Consider using AI for augmentation rather than automation in this domain."
            )
        
        return recommendations
    
    def create_interactive_assessment(self):
        """
        Create an interactive assessment tool using ipywidgets (for Jupyter notebooks).
        
        Returns:
        --------
        function
            Interactive function for Jupyter notebook
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
        except ImportError:
            print("This function requires ipywidgets and IPython. Please install with: pip install ipywidgets")
            return None
        
        # Create input widgets
        domain_dropdown = widgets.Dropdown(
            options=[('Default Parameters', None)] + [(domain_profiles[d]['name'], d) for d in self.domains],
            description='Domain:',
            value=None
        )
        
        harm_slider = widgets.FloatSlider(
            value=5.0,
            min=1.0,
            max=10.0,
            step=0.1,
            description='Harm:',
            continuous_update=False,
            readout_format='.1f'
        )
        
        complexity_slider = widgets.FloatSlider(
            value=5.0,
            min=1.0,
            max=10.0,
            step=0.1,
            description='Complexity:',
            continuous_update=False,
            readout_format='.1f'
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
        
        # Create output widgets
        output = widgets.Output()
        
        # Create function to update the assessment
        def update_assessment(domain, harm, complexity, error_rate):
            with output:
                output.clear_output(wait=True)
                
                # Perform assessment
                result = self.assess_scenario(harm, complexity, error_rate, domain)
                
                # Generate recommendations
                recommendations = self.generate_recommendations(result)
                
                # Display assessment
                display(HTML(f"<h3>Expertise Assessment</h3>"))
                display(HTML(f"<p><b>Domain:</b> {result['domain']}</p>"))
                display(HTML(f"<p><b>Required Expertise:</b> {result['expertise_required']:.2f}/10 ({result['level_name']})</p>"))
                display(HTML(f"<p><b>Level Description:</b> {result['level_description']}</p>"))
                
                # Plot gauge
                fig = self.plot_assessment_gauge(result)
                plt.tight_layout()
                plt.show()
                
                # Display recommendations
                display(HTML(f"<h3>Recommendations</h3>"))
                display(HTML(f"<p><b>Summary:</b> {recommendations['summary']}</p>"))
                
                if recommendations['risk_mitigation']:
                    display(HTML(f"<p><b>Risk Mitigation:</b></p>"))
                    display(HTML("<ul>" + "".join([f"<li>{item}</li>" for item in recommendations['risk_mitigation']]) + "</ul>"))
                
                if recommendations['training_needs']:
                    display(HTML(f"<p><b>Training Needs:</b></p>"))
                    display(HTML("<ul>" + "".join([f"<li>{item}</li>" for item in recommendations['training_needs']]) + "</ul>"))
                
                display(HTML(f"<p><b>Supervision:</b> {recommendations['supervision']}</p>"))
                
                if recommendations['alternative_approaches']:
                    display(HTML(f"<p><b>Alternative Approaches:</b></p>"))
                    display(HTML("<ul>" + "".join([f"<li>{item}</li>" for item in recommendations['alternative_approaches']]) + "</ul>"))
        
        # Create interactive function
        interactive_assessment = widgets.interactive(
            update_assessment,
            domain=domain_dropdown,
            harm=harm_slider,
            complexity=complexity_slider,
            error_rate=error_slider
        )
        
        # Create layout
        layout = widgets.VBox([
            widgets.HTML("<h2>AI Risk Assessment Tool</h2>"),
            widgets.HBox([
                widgets.VBox([domain_dropdown, harm_slider, complexity_slider, error_slider]),
                output
            ])
        ])
        
        # Initialize with default values
        update_assessment(domain_dropdown.value, harm_slider.value, 
                         complexity_slider.value, error_slider.value)
        
        return layout


if __name__ == "__main__":
    # Example usage
    print("Testing Risk Assessment Tool...")
    
    # Create assessment tool
    tool = RiskAssessmentTool()
    
    # Assess a scenario
    result = tool.assess_scenario(harm=8, complexity=7, error_rate=0.15, domain='medical')
    print("\nAssessment Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Compare domains
    comparison = tool.compare_domains(harm=5, complexity=5, error_rate=0.15)
    print("\nDomain Comparison:")
    print(comparison)
    
    # Generate recommendations
    recommendations = tool.generate_recommendations(result)
    print("\nRecommendations:")
    for key, value in recommendations.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    # Create visualization
    tool.plot_assessment_gauge(result)
    tool.plot_domain_comparison(comparison)
    
    plt.show()