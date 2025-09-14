"""
Main analysis runner for BCG candidate classification feature importance analysis.

This script provides a comprehensive interface for running feature importance
analysis on trained BCG classification models, including:
- Global feature importance analysis (SHAP, permutation, gradient-based)
- Individual sample explanations
- Feature group analysis
- Comprehensive visualization reports

Usage:
    python analysis/run_analysis.py --config analysis_config.yaml
    or
    python analysis/run_analysis.py --model_path path/to/model.pth --data_path path/to/data
"""

import os
# Fix NUMEXPR warning for HPC systems - set high enough for cluster nodes
os.environ['NUMEXPR_MAX_THREADS'] = '128'

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle
from typing import Dict, List, Optional
import warnings
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

# Import analysis modules
from .feature_importance import (
    FeatureImportanceAnalyzer, 
    create_default_feature_groups,
    FeatureGroupAnalyzer
)
from .importance_plots import ImportancePlotter, create_comprehensive_report
from .individual_analysis import IndividualSampleAnalyzer, analyze_prediction_boundary

# Import model and data utilities (adjust imports based on your project structure)
import sys
sys.path.append('..')
from ml_models.candidate_classifier import BCGCandidateClassifier
from ml_models.uq_classifier import BCGProbabilisticClassifier
from data.candidate_dataset_bcgs import BCGCandidateDataset
from .feature_utils import create_bcg_feature_names


class BCGAnalysisRunner:
    """
    Main runner for comprehensive BCG classification analysis.
    """
    
    def __init__(self, config_path=None, **kwargs):
        """
        Initialize analysis runner.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Direct configuration parameters
        """
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs
        
        self.model = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        # Force CPU for analysis to avoid CUDA memory issues and multiprocessing problems
        self.device = torch.device('cpu')
        
        # Initialize analyzers
        self.importance_analyzer = None
        self.individual_analyzer = None
        self.plotter = None
    
    @staticmethod
    def load_config(config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup(self):
        """Setup models, data, and analyzers."""
        print("Setting up analysis environment...")
        
        # Load model
        self.load_model()
        
        # Load data
        self.load_data()
        
        # Setup feature names
        self.setup_feature_names()
        
        # Initialize analyzers
        self.importance_analyzer = FeatureImportanceAnalyzer(
            model=self.model,
            feature_names=self.feature_names,
            device=self.device,
            probabilistic=self.config.get('probabilistic_model', False)
        )
        
        self.individual_analyzer = IndividualSampleAnalyzer(
            model=self.model,
            feature_names=self.feature_names,
            device=self.device,
            probabilistic=self.config.get('probabilistic_model', False)
        )
        
        self.plotter = ImportancePlotter()
        
        print("Setup complete!")
    
    def load_model(self):
        """Load trained model."""
        model_path = self.config['model_path']
        model_type = self.config.get('model_type', 'deterministic')
        
        print(f"Loading model from: {model_path}")
        
        if model_type == 'probabilistic':
            # Load probabilistic model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get actual feature dimension from saved model weights (smart detection)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Read actual feature dimension from first layer
            if 'network.0.weight' in state_dict:
                feature_dim = state_dict['network.0.weight'].shape[1]
                print(f"Detected feature dimension from model: {feature_dim}")
            else:
                raise RuntimeError(
                    f"Cannot determine feature dimension from model at {model_path}. "
                    f"Expected 'network.0.weight' in state dict but found keys: {list(state_dict.keys())}. "
                    f"Model may be corrupted or incompatible."
                )
            
            hidden_dims = checkpoint.get('hidden_dims', checkpoint.get('hidden_sizes', self.config.get('hidden_sizes', [128, 64, 32])))
            
            self.model = BCGProbabilisticClassifier(
                feature_dim=feature_dim,
                hidden_dims=hidden_dims
            )
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        else:
            # Load deterministic model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get actual feature dimension from saved model weights (smart detection)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Read actual feature dimension from first layer
            if 'network.0.weight' in state_dict:
                feature_dim = state_dict['network.0.weight'].shape[1]
                print(f"Detected feature dimension from model: {feature_dim}")
            else:
                raise RuntimeError(
                    f"Cannot determine feature dimension from model at {model_path}. "
                    f"Expected 'network.0.weight' in state dict but found keys: {list(state_dict.keys())}. "
                    f"Model may be corrupted or incompatible."
                )
            
            hidden_dims = checkpoint.get('hidden_dims', checkpoint.get('hidden_sizes', self.config.get('hidden_sizes', [128, 64, 32])))
            
            self.model = BCGCandidateClassifier(
                feature_dim=feature_dim,
                hidden_dims=hidden_dims
            )
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully: {type(self.model).__name__}")
    
    def load_data(self):
        """Load test data for analysis."""
        data_path = self.config['data_path']
        
        print(f"Loading data from: {data_path}")
        
        if data_path.endswith('.npz'):
            # Load numpy data
            data = np.load(data_path)
            self.X_test = data['X']
            self.y_test = data['y'] if 'y' in data else None
            
        elif data_path.endswith('.pkl'):
            # Load pickle data
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                self.X_test = data['X']
                self.y_test = data.get('y', None)
            else:
                self.X_test = data
                self.y_test = None
                
        elif data_path.endswith('.csv'):
            # CSV files from testing don't contain the model features, only evaluation results
            # For feature importance analysis, we need to extract features from the original data
            raise ValueError(
                f"CSV evaluation results cannot be used directly for feature importance analysis. "
                f"CSV files contain prediction results, not the original model features. "
                f"Feature importance analysis requires the original high-dimensional feature vectors "
                f"that were input to the model, not the evaluation metrics. "
                f"Please provide a .npz file with the original features and labels, "
                f"or modify the test script to save features during evaluation."
            )
                
        else:
            raise ValueError(
                f"Unsupported data format: {data_path}. "
                f"Expected .npz or .pkl file containing original model features. "
                f"CSV evaluation files cannot be used for feature importance analysis."
            )
        
        print(f"Data loaded: {self.X_test.shape} samples")
        if self.y_test is not None:
            print(f"Labels available: {len(self.y_test)} labels")
    
    def setup_feature_names(self):
        """Setup feature names."""
        if 'feature_names' in self.config:
            self.feature_names = self.config['feature_names']
        else:
            # Create default feature names based on feature configuration
            feature_config = self.config.get('features', {})
            self.feature_names = create_bcg_feature_names(
                use_color_features=feature_config.get('use_color', True),
                use_auxiliary_features=feature_config.get('use_auxiliary', True),
                color_pca_components=feature_config.get('color_pca_components', 8)
            )
        
        print(f"Feature names setup: {len(self.feature_names)} features")
    
    def run_global_analysis(self):
        """Run global feature importance analysis."""
        print("Running global feature importance analysis...")
        
        # Configuration
        methods = self.config.get('analysis_methods', ['permutation', 'shap', 'gradient'])
        n_samples = self.config.get('analysis_samples', min(1000, len(self.X_test)))
        
        # Sample data for analysis (to speed up computation)
        if n_samples < len(self.X_test):
            indices = np.random.choice(len(self.X_test), n_samples, replace=False)
            X_analysis = self.X_test[indices]
            y_analysis = self.y_test[indices] if self.y_test is not None else None
        else:
            X_analysis = self.X_test
            y_analysis = self.y_test
        
        # Run analysis
        results = self.importance_analyzer.analyze_feature_importance(
            X_analysis, y_analysis, 
            methods=methods,
            n_repeats=self.config.get('permutation_repeats', 10)
        )
        
        return results
    
    def run_individual_analysis(self, sample_indices=None):
        """Run individual sample analysis."""
        print("Running individual sample analysis...")
        
        if sample_indices is None:
            # Select diverse samples
            n_samples = min(5, len(self.X_test))
            sample_indices = np.linspace(0, len(self.X_test)-1, n_samples, dtype=int)
        
        individual_results = []
        
        for idx in sample_indices:
            print(f"Analyzing sample {idx}...")
            
            sample = self.X_test[idx]
            explanation = self.individual_analyzer.explain_sample(
                sample, 
                background_data=self.X_test[:100]  # Use first 100 samples as background
            )
            
            explanation['sample_index'] = int(idx)
            if self.y_test is not None:
                explanation['true_label'] = int(self.y_test[idx])
            
            individual_results.append(explanation)
        
        return individual_results
    
    def run_group_analysis(self, importance_results):
        """Run feature group analysis."""
        print("Running feature group analysis...")
        
        # Setup feature groups
        if 'feature_groups' in self.config:
            feature_groups = self.config['feature_groups']
        else:
            feature_groups = create_default_feature_groups()
        
        group_analyzer = FeatureGroupAnalyzer(feature_groups)
        
        group_results = {}
        for method, results in importance_results.items():
            if method == 'shap':
                importance = results['mean_abs_shap']
            else:
                importance = results['importance']
            
            group_scores = group_analyzer.compute_group_importance(
                importance, self.feature_names
            )
            group_results[method] = group_scores
        
        return group_results
    
    def generate_reports(self, importance_results, individual_results, group_results):
        """Generate comprehensive analysis reports."""
        output_dir = Path(self.config.get('output_dir', 'analysis_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating reports in: {output_dir}")
        
        # 1. Save raw results
        results_dir = output_dir / 'raw_results'
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'global_importance.pkl', 'wb') as f:
            pickle.dump(importance_results, f)
        
        with open(results_dir / 'individual_analysis.pkl', 'wb') as f:
            pickle.dump(individual_results, f)
        
        with open(results_dir / 'group_analysis.pkl', 'wb') as f:
            pickle.dump(group_results, f)
        
        # 2. Generate CSV reports
        csv_dir = output_dir / 'csv_reports'
        csv_dir.mkdir(exist_ok=True)
        
        for method in importance_results:
            ranking_df = self.importance_analyzer.get_feature_ranking(
                importance_results, method
            )
            ranking_df.to_csv(csv_dir / f'{method}_feature_ranking.csv', index=False)
        
        # 3. Generate comprehensive visual report
        plots_dir = output_dir / 'plots'
        plot_paths = create_comprehensive_report(
            importance_results, 
            self.X_test[:100],  # Sample for plotting
            self.feature_names,
            plots_dir,
            sample_indices=[r['sample_index'] for r in individual_results[:3]]
        )
        
        # 4. Generate individual sample plots
        individual_plots_dir = output_dir / 'individual_plots'
        individual_plots_dir.mkdir(exist_ok=True)
        
        for result in individual_results:
            idx = result['sample_index']
            fig = self.individual_analyzer.plot_sample_explanation(
                result,
                save_path=individual_plots_dir / f'sample_{idx}_explanation.png'
            )
            plt.close(fig)
        
        # 5. Generate summary report
        self.generate_summary_report(
            importance_results, individual_results, group_results, output_dir
        )
        
        print(f"Reports generated successfully in: {output_dir}")
        return output_dir
    
    def generate_summary_report(self, importance_results, individual_results, 
                              group_results, output_dir):
        """Generate text summary report."""
        
        report_lines = []
        report_lines.append("=== BCG Classification Feature Importance Analysis Report ===\n")
        
        # Model info
        report_lines.append(f"Model Type: {type(self.model).__name__}")
        report_lines.append(f"Number of Features: {len(self.feature_names)}")
        report_lines.append(f"Test Samples: {len(self.X_test)}")
        report_lines.append(f"Analysis Methods: {list(importance_results.keys())}")
        report_lines.append("\n")
        
        # Global importance summary
        report_lines.append("=== Global Feature Importance Summary ===\n")
        
        for method in importance_results:
            report_lines.append(f"--- {method.upper()} Results ---")
            
            ranking_df = self.importance_analyzer.get_feature_ranking(
                importance_results, method
            )
            
            # Top 10 features
            report_lines.append("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(ranking_df.head(10).iterrows()):
                report_lines.append(f"  {i+1}. {row['feature_name']}: {row['importance']:.4f}")
            
            report_lines.append("")
        
        # Feature group summary
        report_lines.append("=== Feature Group Analysis ===\n")
        
        for method in group_results:
            report_lines.append(f"--- {method.upper()} Group Results ---")
            
            groups = group_results[method]
            sorted_groups = sorted(groups.items(), 
                                 key=lambda x: x[1]['mean_importance'], 
                                 reverse=True)
            
            for group_name, group_info in sorted_groups:
                report_lines.append(
                    f"  {group_name}: {group_info['mean_importance']:.4f} "
                    f"(avg), {group_info['sum_importance']:.4f} (total), "
                    f"{group_info['feature_count']} features"
                )
            
            report_lines.append("")
        
        # Individual analysis summary
        report_lines.append("=== Individual Sample Analysis Summary ===\n")
        
        for result in individual_results:
            idx = result['sample_index']
            pred_info = result['prediction']
            
            report_lines.append(f"Sample {idx}:")
            report_lines.append(f"  Predicted Class: {pred_info['predicted_class']}")
            report_lines.append(f"  Confidence: {pred_info['prediction_confidence']:.4f}")
            
            if 'true_label' in result:
                correct = result['true_label'] == pred_info['predicted_class']
                report_lines.append(f"  True Label: {result['true_label']} ({'Correct' if correct else 'Incorrect'})")
            
            if 'is_uncertain' in pred_info:
                report_lines.append(f"  Uncertain: {'Yes' if pred_info['is_uncertain'] else 'No'}")
            
            report_lines.append("")
        
        # Key insights
        report_lines.append("=== Key Insights ===\n")
        
        # Most consistent important features across methods
        if len(importance_results) > 1:
            all_rankings = {}
            for method in importance_results:
                ranking_df = self.importance_analyzer.get_feature_ranking(
                    importance_results, method
                )
                for i, (_, row) in enumerate(ranking_df.iterrows()):
                    if row['feature_name'] not in all_rankings:
                        all_rankings[row['feature_name']] = []
                    all_rankings[row['feature_name']].append(i + 1)  # rank (1-indexed)
            
            # Features with consistently high rankings
            avg_rankings = {feat: np.mean(ranks) for feat, ranks in all_rankings.items()}
            consistent_features = sorted(avg_rankings.items(), key=lambda x: x[1])[:10]
            
            report_lines.append("Most Consistently Important Features (across methods):")
            for i, (feat, avg_rank) in enumerate(consistent_features):
                report_lines.append(f"  {i+1}. {feat} (avg rank: {avg_rank:.1f})")
            report_lines.append("")
        
        # Most important feature group
        if group_results:
            method = list(group_results.keys())[0]  # Use first method
            groups = group_results[method]
            top_group = max(groups.items(), key=lambda x: x[1]['mean_importance'])
            
            report_lines.append(f"Most Important Feature Group: {top_group[0]}")
            report_lines.append(f"  Average Importance: {top_group[1]['mean_importance']:.4f}")
            report_lines.append(f"  Number of Features: {top_group[1]['feature_count']}")
            report_lines.append("")
        
        # Save report
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report_lines))
    
    def run_complete_analysis(self):
        """Run complete feature importance analysis pipeline."""
        print("Starting complete BCG feature importance analysis...")
        
        # Setup
        self.setup()
        
        # Run analyses
        importance_results = self.run_global_analysis()
        individual_results = self.run_individual_analysis()
        group_results = self.run_group_analysis(importance_results)
        
        # Generate reports
        output_dir = self.generate_reports(
            importance_results, individual_results, group_results
        )
        
        print("Analysis complete!")
        return {
            'global_importance': importance_results,
            'individual_analysis': individual_results,
            'group_analysis': group_results,
            'output_directory': output_dir
        }


def create_default_config():
    """Create default analysis configuration."""
    return {
        'model_path': 'path/to/your/model.pth',
        'data_path': 'path/to/your/test_data.npz',
        'model_type': 'deterministic',  # or 'probabilistic'
        'probabilistic_model': False,
        'input_size': 58,
        'hidden_sizes': [128, 64, 32],
        'output_dir': 'analysis_results',
        'analysis_methods': ['permutation', 'shap', 'gradient'],
        'analysis_samples': 1000,
        'permutation_repeats': 10,
        'features': {
            'use_color': True,
            'use_auxiliary': True,
            'color_pca_components': 8
        }
    }


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Run BCG classification feature importance analysis'
    )
    
    parser.add_argument('--config', type=str, 
                       help='Path to configuration YAML file')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Output directory for results')
    parser.add_argument('--methods', nargs='+', 
                       choices=['permutation', 'shap', 'gradient'],
                       default=['permutation', 'shap'],
                       help='Analysis methods to use')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to use for analysis')
    parser.add_argument('--create_config', action='store_true',
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        with open('analysis_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Default configuration saved to: analysis_config.yaml")
        return
    
    # Setup configuration
    if args.config:
        runner = BCGAnalysisRunner(config_path=args.config)
    else:
        config = {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'output_dir': args.output_dir,
            'analysis_methods': args.methods,
            'analysis_samples': args.samples
        }
        
        # Add default values
        default_config = create_default_config()
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        runner = BCGAnalysisRunner(**config)
    
    # Run analysis
    try:
        results = runner.run_complete_analysis()
        print(f"Analysis completed successfully!")
        print(f"Results saved to: {results['output_directory']}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()