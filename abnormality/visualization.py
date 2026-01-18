# visualization.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.figure_factory as ff

class VisualizationEngine:
    """Create interactive visualizations for anomaly detection"""
    
    def __init__(self, config):
        self.config = config
        self.color_scheme = {
            'normal': '#2E86AB',
            'anomaly': '#A23B72',
            'warning': '#F18F01',
            'success': '#73AB84'
        }
    
    def create_anomaly_distribution(self, results_df: pd.DataFrame) -> go.Figure:
        """Create distribution of anomaly scores"""
        fig = go.Figure()
        
        if len(results_df) > 0:
            # Histogram of anomaly scores
            fig.add_trace(go.Histogram(
                x=results_df['anomaly_score'],
                nbinsx=50,
                name='Anomaly Scores',
                marker_color=self.color_scheme['normal'],
                opacity=0.7
            ))
            
            # Add threshold line
            fig.add_vline(
                x=self.config.ANOMALY_THRESHOLD,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {self.config.ANOMALY_THRESHOLD}",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title='Distribution of Anomaly Scores',
            xaxis_title='Anomaly Score',
            yaxis_title='Count',
            template=self.config.PLOTLY_TEMPLATE,
            showlegend=True
        )
        
        return fig
    
    def create_feature_comparison(self, normal_data: pd.DataFrame, 
                                anomaly_data: pd.DataFrame, 
                                feature: str) -> go.Figure:
        """Compare feature distributions between normal and anomalous samples"""
        fig = go.Figure()
        
        if len(normal_data) > 0:
            fig.add_trace(go.Box(
                y=normal_data[feature],
                name='Normal',
                marker_color=self.color_scheme['normal'],
                boxpoints='outliers'
            ))
        
        if len(anomaly_data) > 0:
            fig.add_trace(go.Box(
                y=anomaly_data[feature],
                name='Anomaly',
                marker_color=self.color_scheme['anomaly'],
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=f'Distribution Comparison: {feature}',
            yaxis_title=feature,
            template=self.config.PLOTLY_TEMPLATE,
            showlegend=True
        )
        
        return fig
    
    def create_time_series_plot(self, results_df: pd.DataFrame) -> go.Figure:
        """Create time series plot of anomalies"""
        if 'date' not in results_df.columns or len(results_df) == 0:
            return go.Figure()
        
        # Convert date column
        results_df['date_dt'] = pd.to_datetime(results_df['date'], errors='coerce')
        results_df = results_df.dropna(subset=['date_dt'])
        
        # Group by date
        daily_counts = results_df.groupby(results_df['date_dt'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'anomaly_count']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['anomaly_count'],
            mode='lines+markers',
            name='Daily Anomalies',
            line=dict(color=self.color_scheme['anomaly'], width=2),
            marker=dict(size=8)
        ))
        
        # Add rolling average
        if len(daily_counts) > 7:
            daily_counts['rolling_avg'] = daily_counts['anomaly_count'].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['rolling_avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color=self.color_scheme['normal'], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Anomaly Detection Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Anomalies',
            template=self.config.PLOTLY_TEMPLATE,
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, 
                                 features: List[str]) -> go.Figure:
        """Create correlation heatmap for features"""
        # Select only numeric columns
        numeric_data = data[features].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            colorscale='RdBu',
            showscale=True
        )
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            template=self.config.PLOTLY_TEMPLATE
        )
        
        return fig
    
    def create_parallel_coordinates(self, data: pd.DataFrame, 
                                  features: List[str], 
                                  color_by: str = 'is_anomaly') -> go.Figure:
        """Create parallel coordinates plot"""
        if len(data) == 0 or len(features) < 2:
            return go.Figure()
        
        # Select only features that exist in data
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            return go.Figure()
        
        fig = px.parallel_coordinates(
            data,
            dimensions=available_features[:5],  # Limit to 5 features for clarity
            color=color_by if color_by in data.columns else None,
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Parallel Coordinates Plot'
        )
        
        fig.update_layout(
            template=self.config.PLOTLY_TEMPLATE
        )
        
        return fig
    
    def create_3d_scatter(self, data: pd.DataFrame, 
                         features: List[str]) -> go.Figure:
        """Create 3D scatter plot"""
        if len(data) == 0 or len(features) < 3:
            return go.Figure()
        
        # Select only features that exist in data
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 3:
            return go.Figure()
        
        x_feat, y_feat, z_feat = available_features[:3]
        
        fig = px.scatter_3d(
            data,
            x=x_feat,
            y=y_feat,
            z=z_feat,
            color='is_anomaly' if 'is_anomaly' in data.columns else None,
            title=f'3D Scatter: {x_feat} vs {y_feat} vs {z_feat}',
            opacity=0.7
        )
        
        fig.update_layout(
            template=self.config.PLOTLY_TEMPLATE,
            scene=dict(
                xaxis_title=x_feat,
                yaxis_title=y_feat,
                zaxis_title=z_feat
            )
        )
        
        return fig
    
    def create_model_performance_dashboard(self, model_metrics: Dict[str, Dict]) -> go.Figure:
        """Create dashboard showing model performance"""
        if not model_metrics:
            return go.Figure()
        
        models = list(model_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig = make_subplots(
            rows=1, 
            cols=len(metrics),
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics]
        )
        
        for idx, metric in enumerate(metrics, 1):
            values = [model_metrics[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=self.color_scheme['normal']
                ),
                row=1, col=idx
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            template=self.config.PLOTLY_TEMPLATE,
            showlegend=False,
            height=400
        )
        
        return fig
    
    # visualization.py - Add these methods
    def create_failure_pattern_visualization(self, patterns_data: Dict[str, Any]) -> go.Figure:
        """Create visualization of failure patterns"""
        if not patterns_data.get('by_failure_type'):
            return go.Figure()
        
        failure_types = list(patterns_data['by_failure_type'].keys())
        pattern_counts = [patterns_data['by_failure_type'][ft] for ft in failure_types]
        
        colors = {
            'WW': '#A23B72',  # Red/purple for complete rejection
            'GW': '#F18F01',  # Orange for partial failure
            'WG': '#73AB84',  # Green for other partial failure
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=failure_types,
                y=pattern_counts,
                marker_color=[colors.get(ft, '#2E86AB') for ft in failure_types],
                text=pattern_counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Learned Failure Patterns by Type',
            xaxis_title='Failure Type',
            yaxis_title='Number of Patterns',
            template=self.config.PLOTLY_TEMPLATE
        )
        
        return fig
    
    def create_pattern_parameter_ranges(self, pattern: Dict[str, Any]) -> go.Figure:
        """Visualize parameter ranges for a failure pattern"""
        if 'parameter_ranges' not in pattern:
            return go.Figure()
        
        param_ranges = pattern['parameter_ranges']
        parameters = list(param_ranges.keys())
        
        # Extract min, max, mean
        mins = [param_ranges[p].get('min', 0) for p in parameters]
        maxs = [param_ranges[p].get('max', 0) for p in parameters]
        means = [param_ranges[p].get('mean', 0) for p in parameters]
        
        fig = go.Figure()
        
        # Add range bars
        for i, param in enumerate(parameters):
            fig.add_trace(go.Scatter(
                x=[param, param],
                y=[mins[i], maxs[i]],
                mode='lines',
                line=dict(color='gray', width=10),
                showlegend=False,
                name=f'Range: {mins[i]:.2f} to {maxs[i]:.2f}'
            ))
        
        # Add mean points
        fig.add_trace(go.Scatter(
            x=parameters,
            y=means,
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='diamond'
            ),
            name='Mean Value'
        ))
        
        fig.update_layout(
            title=f"Parameter Ranges for {pattern.get('failure_type', 'Unknown')} Pattern",
            xaxis_title='Parameter',
            yaxis_title='Value Range',
            template=self.config.PLOTLY_TEMPLATE,
            showlegend=True
        )
        
        return fig