"""
Validation of domain profiles against expected behaviors and patterns.
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
from data_generation.calculator import calculate_expertise_for_scenario


def validate_domain_against_expectations(domain, expectations=None):
    """
    Validate a domain profile against expected behaviors.
    
    Parameters:
    -----------
    domain : str
        Domain name to validate
    
    expectations : dict, optional
        Dictionary of expected behaviors
        
    Returns:
    --------
    dict
        Validation results
    """
    if domain not in domain_profiles:
        raise ValueError(f"Unknown domain: {domain}")
    
    profile = domain_profiles[domain]
    
    # Set default expectations if none provided
    if expectations is None:
        expectations = {
            'min_expertise': None,  # Minimum expertise level expected
            'max_expertise': None,  # Maximum expertise level expected
            'high_harm_threshold': 7,  # Harm level considered "high"
            'high_complexity_threshold': 7,  # Complexity level considered "high"
            'high_error_threshold': 0.3,  # Error rate considered "high"
            'expected_expertise_in_low_risk': None,  # Expected expertise for low risk scenario
            'expected_expertise_in_high_risk': None,  # Expected expertise for high risk scenario
            'primary_sensitivity': None  # 'harm', 'complexity', or 'error'
        }
    
    # Generate test cases
    test_cases = [
        {'name': 'low_risk', 'harm': 2, 'complexity': 2, 'error_rate': 0.05},
        {'name': 'medium_risk', 'harm': 5, 'complexity': 5, 'error_rate': 0.15},
        {'name': 'high_risk', 'harm': 9, 'complexity': 8, 'error_rate': 0.3},
        {'name': 'high_harm_only', 'harm': 9, 'complexity': 3, 'error_rate': 0.05},
        {'name': 'high_complexity_only', 'harm': 3, 'complexity': 9, 'error_rate': 0.05},
        {'name': 'high_error_only', 'harm': 3, 'complexity': 3, 'error_rate': 0.4}
    ]
    
    # Calculate expertise for each test case
    results = []
    for case in test_cases:
        expertise = calculate_expertise_for_scenario(
            case['harm'], case['complexity'], case['error_rate'], domain
        )
        results.append({**case, 'expertise': expertise})
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate parameter sensitivities
    harm_range = np.linspace(1, 10, 20)
    complexity_range = np.linspace(1, 10, 20)
    error_range = np.linspace(0.05, 0.5, 20)
    
    # Fixed values for sensitivity tests
    fixed_values = {'harm': 5, 'complexity': 5, 'error_rate': 0.15}
    
    harm_sensitivity = [
        calculate_expertise_for_scenario(h, fixed_values['complexity'], fixed_values['error_rate'], domain)
        for h in harm_range
    ]
    
    complexity_sensitivity = [
        calculate_expertise_for_scenario(fixed_values['harm'], c, fixed_values['error_rate'], domain)
        for c in complexity_range
    ]
    
    error_sensitivity = [
        calculate_expertise_for_scenario(fixed_values['harm'], fixed_values['complexity'], e, domain)
        for e in error_range
    ]
    
    # Calculate gradients
    harm_gradient = np.mean(np.diff(harm_sensitivity) / np.diff(harm_range))
    complexity_gradient = np.mean(np.diff(complexity_sensitivity) / np.diff(complexity_range))
    error_gradient = np.mean(np.diff(error_sensitivity) / np.diff(error_range))
    
    # Determine primary sensitivity
    gradients = {
        'harm': harm_gradient,
        'complexity': complexity_gradient,
        'error': error_gradient
    }
    primary_sensitivity = max(gradients, key=gradients.get)
    
    # Validation checks
    validation_results = {
        'domain': domain,
        'profile': profile,
        'test_results': results_df,
        'min_expertise': results_df['expertise'].min(),
        'max_expertise': results_df['expertise'].max(),
        'parameter_sensitivities': {
            'harm': harm_gradient,
            'complexity': complexity_gradient,
            'error': error_gradient
        },
        'primary_sensitivity': primary_sensitivity,
        'validation': {
            'min_expertise_check': True,
            'max_expertise_check': True,
            'primary_sensitivity_check': True,
            'low_risk_expertise_check': True,
            'high_risk_expertise_check': True
        },
        'messages': []
    }
    
    # Check minimum expertise
    if expectations['min_expertise'] is not None:
        validation_results['validation']['min_expertise_check'] = (
            validation_results['min_expertise'] >= expectations['min_expertise']
        )
        if not validation_results['validation']['min_expertise_check']:
            validation_results['messages'].append(
                f"Minimum expertise {validation_results['min_expertise']:.2f} is below expected {expectations['min_expertise']:.2f}"
            )
    
    # Check maximum expertise
    if expectations['max_expertise'] is not None:
        validation_results['validation']['max_expertise_check'] = (
            validation_results['max_expertise'] <= expectations['max_expertise']
        )
        if not validation_results['validation']['max_expertise_check']:
            validation_results['messages'].append(
                f"Maximum expertise {validation_results['max_expertise']:.2f} is above expected {expectations['max_expertise']:.2f}"
            )
    
    # Check primary sensitivity
    if expectations['primary_sensitivity'] is not None:
        validation_results['validation']['primary_sensitivity_check'] = (
            validation_results['primary_sensitivity'] == expectations['primary_sensitivity']
        )
        if not validation_results['validation']['primary_sensitivity_check']:
            validation_results['messages'].append(
                f"Primary sensitivity is {validation_results['primary_sensitivity']} but expected {expectations['primary_sensitivity']}"
            )
    
    # Check expertise in low risk scenario
    low_risk_expertise = results_df.loc[results_df['name'] == 'low_risk', 'expertise'].values[0]
    if expectations['expected_expertise_in_low_risk'] is not None:
        expected = expectations['expected_expertise_in_low_risk']
        validation_results['validation']['low_risk_expertise_check'] = (
            abs(low_risk_expertise - expected) <= 1.0
        )
        if not validation_results['validation']['low_risk_expertise_check']:
            validation_results['messages'].append(
                f"Low risk expertise {low_risk_expertise:.2f} differs significantly from expected {expected:.2f}"
            )
    
    # Check expertise in high risk scenario
    high_risk_expertise = results_df.loc[results_df['name'] == 'high_risk', 'expertise'].values[0]
    if expectations['expected_expertise_in_high_risk'] is not None:
        expected = expectations['expected_expertise_in_high_risk']
        validation_results['validation']['high_risk_expertise_check'] = (
            abs(high_risk_expertise - expected) <= 1.0
        )
        if not validation_results['validation']['high_risk_expertise_check']:
            validation_results['messages'].append(
                f"High risk expertise {high_risk_expertise:.2f} differs significantly from expected {expected:.2f}"
            )
    
    # Overall validation check
    validation_results['passed'] = all(validation_results['validation'].values())
    
    return validation_results


def validate_all_domains(domain_expectations=None):
    """
    Validate all domain profiles and return results.
    
    Parameters:
    -----------
    domain_expectations : dict, optional
        Dictionary mapping domain names to expectation dictionaries
    
    Returns:
    --------
    dict
        Validation results for all domains
    """
    # Set default expectations for domains
    if domain_expectations is None:
        domain_expectations = {
            'medical': {
                'min_expertise': 2.0,
                'max_expertise': 10.0,
                'expected_expertise_in_low_risk': 3.0,
                'expected_expertise_in_high_risk': 9.0,
                'primary_sensitivity': 'harm'
            },
            'legal': {
                'min_expertise': 2.0,
                'max_expertise': 10.0,
                'expected_expertise_in_low_risk': 3.0,
                'expected_expertise_in_high_risk': 9.0,
                'primary_sensitivity': 'complexity'
            },
            'creative': {
                'min_expertise': 1.0,
                'max_expertise': 8.0,
                'expected_expertise_in_low_risk': 1.5,
                'expected_expertise_in_high_risk': 6.0,
                'primary_sensitivity': None
            }
        }
    
    # Validate each domain
    results = {}
    for domain in domain_profiles.keys():
        expectations = domain_expectations.get(domain, None)
        results[domain] = validate_domain_against_expectations(domain, expectations)
    
    return results


def plot_domain_validation_results(validation_results, figsize=(15, 10)):
    """
    Plot validation results for visual comparison.
    
    Parameters:
    -----------
    validation_results : dict
        Validation results from validate_all_domains
    
    figsize : tuple
        Figure size
    """
    # Extract test results for each domain
    test_results = {}
    for domain, result in validation_results.items():
        test_results[domain] = result['test_results']
    
    # Combine into a single DataFrame
    combined_df = pd.concat([df.assign(domain=domain) for domain, df in test_results.items()])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot test case results
    sns.barplot(x='name', y='expertise', hue='domain', data=combined_df, ax=ax1)
    ax1.set_title('Expertise Required by Test Case and Domain')
    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('Expertise Required (1-10)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot parameter sensitivities
    sensitivity_data = []
    for domain, result in validation_results.items():
        for param, sensitivity in result['parameter_sensitivities'].items():
            sensitivity_data.append({
                'domain': domain,
                'parameter': param,
                'sensitivity': sensitivity
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    sns.barplot(x='parameter', y='sensitivity', hue='domain', data=sensitivity_df, ax=ax2)
    ax2.set_title('Parameter Sensitivities by Domain')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Sensitivity (Average Gradient)')
    
    plt.tight_layout()
    plt.savefig('domain_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":
    print("Validating domain profiles...")
    
    # Validate all domains
    validation_results = validate_all_domains()
    
    # Display results
    for domain, result in validation_results.items():
        print(f"\n{domain.upper()} DOMAIN:")
        print(f"Passed: {result['passed']}")
        if not result['passed']:
            print("Validation messages:")
            for msg in result['messages']:
                print(f"- {msg}")
        
        print("\nTest case results:")
        print(result['test_results'][['name', 'expertise']])
        
        print("\nParameter sensitivities:")
        for param, sensitivity in result['parameter_sensitivities'].items():
            print(f"- {param}: {sensitivity:.4f}")
        
        print(f"Primary sensitivity: {result['primary_sensitivity']}")
    
    # Plot validation results
    plot_domain_validation_results(validation_results)