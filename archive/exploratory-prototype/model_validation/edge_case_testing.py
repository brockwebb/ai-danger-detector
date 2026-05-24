"""
Edge case testing for the expertise-harm model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_model.model_definition import expertise_required
from core_model.domain_profiles import domain_profiles
from data_generation.calculator import calculate_expertise_for_scenario


def test_extreme_values():
    """
    Test model behavior with extreme parameter values.
    
    Returns:
    --------
    dict
        Test results for extreme values
    """
    # Define extreme test cases
    extreme_tests = [
        {'name': 'min_all', 'harm': 1, 'complexity': 1, 'error_rate': 0.01},
        {'name': 'max_all', 'harm': 10, 'complexity': 10, 'error_rate': 0.5},
        {'name': 'max_harm_only', 'harm': 10, 'complexity': 1, 'error_rate': 0.01},
        {'name': 'max_complexity_only', 'harm': 1, 'complexity': 10, 'error_rate': 0.01},
        {'name': 'max_error_only', 'harm': 1, 'complexity': 1, 'error_rate': 0.5},
        {'name': 'zero_error', 'harm': 5, 'complexity': 5, 'error_rate': 0},
        {'name': 'very_high_error', 'harm': 5, 'complexity': 5, 'error_rate': 0.9}
    ]
    
    # Test each case for all domains
    results = []
    
    for test_case in extreme_tests:
        # Test default model
        default_expertise = expertise_required(
            test_case['harm'], test_case['complexity'], test_case['error_rate']
        )
        
        results.append({
            **test_case,
            'domain': 'default',
            'expertise': default_expertise
        })
        
        # Test each domain
        for domain in domain_profiles.keys():
            expertise = calculate_expertise_for_scenario(
                test_case['harm'], test_case['complexity'], test_case['error_rate'], domain
            )
            
            results.append({
                **test_case,
                'domain': domain,
                'expertise': expertise
            })
    
    return pd.DataFrame(results)


def test_parameter_edge_cases():
    """
    Test edge cases for model parameters (beta and alpha).
    
    Returns:
    --------
    dict
        Test results for parameter edge cases
    """
    # Define baseline parameters
    baseline = {
        'harm': 5.0,
        'complexity': 5.0,
        'error_rate': 0.15,
        'E_min': 1.0,
        'beta': [0.2, 0.3, 0.2, 0.1],
        'alpha': [1.5, 1.2, 0.8, 1.1]
    }
    
    # Define parameter edge cases
    parameter_tests = [
        {'name': 'baseline', 'changes': {}},
        {'name': 'zero_beta', 'changes': {'beta': [0, 0, 0, 0]}},
        {'name': 'zero_alpha', 'changes': {'alpha': [0, 0, 0, 0]}},
        {'name': 'high_beta', 'changes': {'beta': [1.0, 1.0, 1.0, 1.0]}},
        {'name': 'high_alpha', 'changes': {'alpha': [3.0, 3.0, 3.0, 3.0]}},
        {'name': 'zero_E_min', 'changes': {'E_min': 0}},
        {'name': 'high_E_min', 'changes': {'E_min': 5.0}},
        {'name': 'negative_beta', 'changes': {'beta': [-0.2, -0.3, -0.2, -0.1]}},
        {'name': 'negative_alpha', 'changes': {'alpha': [-1.5, -1.2, -0.8, -1.1]}},
        {'name': 'uneven_beta', 'changes': {'beta': [0.01, 0.5, 0.01, 0.01]}},
        {'name': 'uneven_alpha', 'changes': {'alpha': [0.5, 3.0, 0.5, 0.5]}}
    ]
    
    results = []
    
    for test in parameter_tests:
        # Create parameter set for this test
        params = baseline.copy()
        for key, value in test['changes'].items():
            params[key] = value
        
        try:
            # Calculate expertise with modified parameters
            expertise = expertise_required(
                params['harm'],
                params['complexity'],
                params['error_rate'],
                E_min=params['E_min'],
                beta=params['beta'],
                alpha=params['alpha']
            )
            
            error = None
        except Exception as e:
            expertise = None
            error = str(e)
        
        results.append({
            'name': test['name'],
            'expertise': expertise,
            'error': error,
            'valid': error is None
        })
    
    return pd.DataFrame(results)


def test_discontinuities():
    """
    Test for discontinuities in the model.
    
    Returns:
    --------
    dict
        Test results for discontinuity tests
    """
    results = {}
    
    for domain in list(domain_profiles.keys()) + ['default']:
        # Test for discontinuities in harm
        harm_values = np.linspace(1, 10, 1000)
        harm_expertise = []
        
        for h in harm_values:
            if domain == 'default':
                e = expertise_required(h, 5.0, 0.15)
            else:
                e = calculate_expertise_for_scenario(h, 5.0, 0.15, domain)
            harm_expertise.append(e)
        
        harm_diff = np.diff(harm_expertise)
        max_harm_jump = np.max(np.abs(harm_diff))
        
        # Test for discontinuities in complexity
        complexity_values = np.linspace(1, 10, 1000)
        complexity_expertise = []
        
        for c in complexity_values:
            if domain == 'default':
                e = expertise_required(5.0, c, 0.15)
            else:
                e = calculate_expertise_for_scenario(5.0, c, 0.15, domain)
            complexity_expertise.append(e)
        
        complexity_diff = np.diff(complexity_expertise)
        max_complexity_jump = np.max(np.abs(complexity_diff))
        
        # Test for discontinuities in error rate
        error_values = np.linspace(0.01, 0.5, 1000)
        error_expertise = []
        
        for e in error_values:
            if domain == 'default':
                exp = expertise_required(5.0, 5.0, e)
            else:
                exp = calculate_expertise_for_scenario(5.0, 5.0, e, domain)
            error_expertise.append(exp)
        
        error_diff = np.diff(error_expertise)
        max_error_jump = np.max(np.abs(error_diff))
        
        # Store results
        results[domain] = {
            'max_harm_jump': max_harm_jump,
            'max_complexity_jump': max_complexity_jump,
            'max_error_jump': max_error_jump,
            'has_discontinuity': (max_harm_jump > 0.01 or 
                                max_complexity_jump > 0.01 or 
                                max_error_jump > 0.01)
        }
    
    return results


def test_consistency():
    """
    Test for internal consistency in the model.
    
    Returns:
    --------
    dict
        Test results for consistency tests
    """
    results = {}
    
    # Test proportionality
    # If harm doubles, how does expertise change?
    proportionality_tests = []
    
    base_cases = [
        {'harm': 2, 'complexity': 3, 'error_rate': 0.1},
        {'harm': 3, 'complexity': 5, 'error_rate': 0.2},
        {'harm': 4, 'complexity': 4, 'error_rate': 0.15}
    ]
    
    for case in base_cases:
        # Original case
        original_expertise = {}
        for domain in list(domain_profiles.keys()) + ['default']:
            if domain == 'default':
                original_expertise[domain] = expertise_required(
                    case['harm'], case['complexity'], case['error_rate']
                )
            else:
                original_expertise[domain] = calculate_expertise_for_scenario(
                    case['harm'], case['complexity'], case['error_rate'], domain
                )
        
        # Double harm
        double_harm_expertise = {}
        for domain in list(domain_profiles.keys()) + ['default']:
            if domain == 'default':
                double_harm_expertise[domain] = expertise_required(
                    case['harm'] * 2, case['complexity'], case['error_rate']
                )
            else:
                double_harm_expertise[domain] = calculate_expertise_for_scenario(
                    case['harm'] * 2, case['complexity'], case['error_rate'], domain
                )
        
        # Double complexity
        double_complexity_expertise = {}
        for domain in list(domain_profiles.keys()) + ['default']:
            if domain == 'default':
                double_complexity_expertise[domain] = expertise_required(
                    case['harm'], case['complexity'] * 2, case['error_rate']
                )
            else:
                double_complexity_expertise[domain] = calculate_expertise_for_scenario(
                    case['harm'], case['complexity'] * 2, case['error_rate'], domain
                )
        
        # Double error rate
        double_error_expertise = {}
        for domain in list(domain_profiles.keys()) + ['default']:
            if domain == 'default':
                double_error_expertise[domain] = expertise_required(
                    case['harm'], case['complexity'], min(case['error_rate'] * 2, 0.5)
                )
            else:
                double_error_expertise[domain] = calculate_expertise_for_scenario(
                    case['harm'], case['complexity'], min(case['error_rate'] * 2, 0.5), domain
                )
        
        # Calculate ratios
        for domain in list(domain_profiles.keys()) + ['default']:
            proportionality_tests.append({
                'base_case': case,
                'domain': domain,
                'original_expertise': original_expertise[domain],
                'double_harm_expertise': double_harm_expertise[domain],
                'double_complexity_expertise': double_complexity_expertise[domain],
                'double_error_expertise': double_error_expertise[domain],
                'harm_ratio': double_harm_expertise[domain] / original_expertise[domain],
                'complexity_ratio': double_complexity_expertise[domain] / original_expertise[domain],
                'error_ratio': double_error_expertise[domain] / original_expertise[domain]
            })
    
    # Convert to DataFrame
    proportionality_df = pd.DataFrame(proportionality_tests)
    
    # Test monotonicity
    monotonicity_tests = {}
    
    for domain in list(domain_profiles.keys()) + ['default']:
        # Test monotonicity for harm
        harm_values = np.linspace(1, 10, 100)
        harm_expertise = []
        
        for h in harm_values:
            if domain == 'default':
                e = expertise_required(h, 5.0, 0.15)
            else:
                e = calculate_expertise_for_scenario(h, 5.0, 0.15, domain)
            harm_expertise.append(e)
        
        harm_monotonic = all(i <= j for i, j in zip(harm_expertise[:-1], harm_expertise[1:]))
        
        # Test monotonicity for complexity
        complexity_values = np.linspace(1, 10, 100)
        complexity_expertise = []
        
        for c in complexity_values:
            if domain == 'default':
                e = expertise_required(5.0, c, 0.15)
            else:
                e = calculate_expertise_for_scenario(5.0, c, 0.15, domain)
            complexity_expertise.append(e)
        
        complexity_monotonic = all(i <= j for i, j in zip(complexity_expertise[:-1], complexity_expertise[1:]))
        
        # Test monotonicity for error rate
        error_values = np.linspace(0.01, 0.5, 100)
        error_expertise = []
        
        for e in error_values:
            if domain == 'default':
                exp = expertise_required(5.0, 5.0, e)
            else:
                exp = calculate_expertise_for_scenario(5.0, 5.0, e, domain)
            error_expertise.append(exp)
        
        error_monotonic = all(i <= j for i, j in zip(error_expertise[:-1], error_expertise[1:]))
        
        monotonicity_tests[domain] = {
            'harm_monotonic': harm_monotonic,
            'complexity_monotonic': complexity_monotonic,
            'error_monotonic': error_monotonic,
            'all_monotonic': harm_monotonic and complexity_monotonic and error_monotonic
        }
    
    return {
        'proportionality': proportionality_df,
        'monotonicity': monotonicity_tests
    }


def plot_edge_case_results(extreme_results, figsize=(15, 10)):
    """
    Plot results from edge case testing.
    
    Parameters:
    -----------
    extreme_results : pandas.DataFrame
        Results from test_extreme_values function
    
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group by test case and domain
    pivoted = extreme_results.pivot(index='name', columns='domain', values='expertise')
    
    # Create bar chart
    ax = pivoted.plot(kind='bar', figsize=figsize)
    ax.set_ylabel('Expertise Required (1-10)')
    ax.set_title('Model Behavior in Extreme Cases')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('edge_case_testing.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Testing model behavior in edge cases...")
    
    # Test extreme values
    print("\nTesting extreme parameter values...")
    extreme_results = test_extreme_values()
    print(extreme_results)
    
    # Test parameter edge cases
    print("\nTesting parameter edge cases...")
    param_results = test_parameter_edge_cases()
    print(param_results)
    
    # Test for discontinuities
    print("\nTesting for discontinuities...")
    discontinuity_results = test_discontinuities()
    for domain, result in discontinuity_results.items():
        print(f"\n{domain}:")
        print(f"  Max harm jump: {result['max_harm_jump']:.6f}")
        print(f"  Max complexity jump: {result['max_complexity_jump']:.6f}")
        print(f"  Max error jump: {result['max_error_jump']:.6f}")
        print(f"  Has discontinuity: {result['has_discontinuity']}")
    
    # Test for consistency
    print("\nTesting for consistency...")
    consistency_results = test_consistency()
   
    print("\nProportionality test results (expertise ratios when doubling parameters):")
    proportionality_df = consistency_results['proportionality']
    # Group and summarize
    summary = proportionality_df.groupby('domain')[['harm_ratio', 'complexity_ratio', 'error_ratio']].mean()
    print(summary)
   
    print("\nMonotonicity test results:")
    monotonicity = consistency_results['monotonicity']
    for domain, result in monotonicity.items():
        print(f"\n{domain}:")
        print(f"  Harm monotonic: {result['harm_monotonic']}")
        print(f"  Complexity monotonic: {result['complexity_monotonic']}")
        print(f"  Error monotonic: {result['error_monotonic']}")
        print(f"  All monotonic: {result['all_monotonic']}")
   
    # Plot results
    plot_edge_case_results(extreme_results)