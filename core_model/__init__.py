"""
Core model module for expertise-harm relationship modeling.
"""

from .model_definition import expertise_required
from .domain_profiles import domain_profiles

__all__ = ['expertise_required', 'domain_profiles']