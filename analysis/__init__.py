"""
Post-analysis tools for BCG candidate classification models.

This module provides feature importance analysis, interpretability tools,
and visualization capabilities for understanding model behavior.
"""

__version__ = "1.0.0"
__author__ = "BCG Analysis Framework"

from .feature_importance import (
    FeatureImportanceAnalyzer,
    SHAPAnalyzer
)

from .importance_plots import (
    ImportancePlotter,
    create_shap_summary_plot,
    create_feature_ranking_plot,
    create_individual_explanation_plot
)

__all__ = [
    'FeatureImportanceAnalyzer',
    'SHAPAnalyzer',
    'ImportancePlotter',
    'create_shap_summary_plot',
    'create_feature_ranking_plot', 
    'create_individual_explanation_plot'
]