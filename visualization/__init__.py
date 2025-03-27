"""
Visualization module for expertise-harm modeling.
"""

from .plotting import (
    plot_expertise_vs_harm,
    plot_expertise_vs_complexity,
    plot_expertise_vs_error_rate,
    plot_3d_surface,
    plot_parameter_heatmap,
    plot_domain_radar_chart
)

from .domain_comparisons import (
    plot_domain_comparison,
    plot_scenario_comparison,
    plot_expertise_thresholds,
    plot_interactive_scenario_explorer
)

__all__ = [
    'plot_expertise_vs_harm',
    'plot_expertise_vs_complexity',
    'plot_expertise_vs_error_rate',
    'plot_3d_surface',
    'plot_parameter_heatmap',
    'plot_domain_radar_chart',
    'plot_domain_comparison',
    'plot_scenario_comparison',
    'plot_expertise_thresholds',
    'plot_interactive_scenario_explorer'
]