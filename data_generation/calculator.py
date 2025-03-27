"""
Single-point calculator for expertise requirements.
"""

import sys
import os

# Add the parent directory to sys.path to import core_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_model.model_definition import expertise_required
from core_model.domain_profiles import domain_profiles

def calculate_expertise_for_scenario(harm, complexity, error_rate, domain=None, custom_params=None):
    """
    Calculate expertise required for a specific scenario.
    
    Parameters:
    -----------
    harm : float
        Harm potential (1-10)
    
    complexity : float
        Domain complexity (1-10)
    
    error_rate : float
        AI model error rate (0-1)
    
    domain : str, optional
        Domain name to use predefined parameters from domain_profiles
    
    custom_params : dict, optional
        Custom parameters to use instead of domain profiles
        Format: {'E_min': float, 'beta': list, 'alpha': list}
    
    Returns:
    --------
    float
        Required expertise level (1-10)
    """
    if custom_params is not None:
        # Use custom parameters
        return expertise_required(
            harm, 
            complexity, 
            error_rate,
            E_min=custom_params.get('E_min', 1.0),
            beta=custom_params.get('beta', [0.2, 0.3, 0.2, 0.1]),
            alpha=custom_params.get('alpha', [1.5, 1.2, 0.8, 1.1])
        )
    elif domain is not None and domain in domain_profiles:
        # Use domain-specific profile
        profile = domain_profiles[domain]
        return expertise_required(
            harm, 
            complexity, 
            error_rate,
            E_min=profile['E_min'],
            beta=profile['beta'],
            alpha=profile['alpha']
        )
    else:
        # Use default parameters
        return expertise_required(harm, complexity, error_rate)


def get_available_domains():
    """
    Returns list of available domain profiles.
    """
    return list(domain_profiles.keys())


def get_domain_info(domain):
    """
    Returns detailed information about a specific domain profile.
    """
    if domain in domain_profiles:
        return domain_profiles[domain]
    else:
        return None


if __name__ == "__main__":
    # Example usage
    print("Testing expertise calculator:")
    
    # Test with default parameters
    print("\nDefault parameters:")
    print(f"Low stakes (h=2, c=3, e=0.1): {calculate_expertise_for_scenario(2, 3, 0.1):.2f}")
    print(f"Medium stakes (h=5, c=5, e=0.2): {calculate_expertise_for_scenario(5, 5, 0.2):.2f}")
    print(f"High stakes (h=9, c=8, e=0.3): {calculate_expertise_for_scenario(9, 8, 0.3):.2f}")
    
    # Test with domain profiles
    print("\nDomain-specific parameters:")
    for domain in get_available_domains():
        print(f"\n{domain_profiles[domain]['name']}:")
        print(f"Low stakes (h=2, c=3, e=0.1): {calculate_expertise_for_scenario(2, 3, 0.1, domain):.2f}")
        print(f"Medium stakes (h=5, c=5, e=0.2): {calculate_expertise_for_scenario(5, 5, 0.2, domain):.2f}")
        print(f"High stakes (h=9, c=8, e=0.3): {calculate_expertise_for_scenario(9, 8, 0.3, domain):.2f}")