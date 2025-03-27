"""
Data generation module for expertise-harm modeling.
"""

from .calculator import calculate_expertise_for_scenario, get_available_domains, get_domain_info
from .batch_generator import (
    generate_uniform_dataset, 
    generate_domain_specific_dataset, 
    generate_grid_dataset,
    save_dataset
)

__all__ = [
    'calculate_expertise_for_scenario',
    'get_available_domains',
    'get_domain_info',
    'generate_uniform_dataset',
    'generate_domain_specific_dataset',
    'generate_grid_dataset',
    'save_dataset'
]