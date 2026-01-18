# failure_patterns.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
import joblib
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FailurePattern:
    """Represents a learned failure pattern for an item code"""
    item_code: str
    failure_type: str  # WW, GW, WG
    pattern_id: str
    features: Dict[str, Any]
    cluster_center: np.ndarray
    cluster_radius: float
    samples_count: int
    first_seen: str
    last_seen: str
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    common_operators: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'item_code': self.item_code,
            'failure_type': self.failure_type,
            'pattern_id': self.pattern_id,
            'features': self.features,
            'cluster_center': self.cluster_center.tolist() if hasattr(self.cluster_center, 'tolist') else list(self.cluster_center),
            'cluster_radius': self.cluster_radius,
            'samples_count': self.samples_count,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'description': self.description,
            'severity': self.severity,
            'common_operators': self.common_operators,
            'parameter_ranges': self.parameter_ranges
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        data['cluster_center'] = np.array(data['cluster_center'])
        return cls(**data)
    
    def matches(self, features: np.ndarray, threshold: float = 0.8) -> Tuple[bool, float]:
        """Check if features match this failure pattern"""
        if len(features) != len(self.cluster_center):
            return False, 0.0
        
        # Calculate similarity (1 - normalized distance)
        distance = euclidean(features, self.cluster_center)
        max_distance = self.cluster_radius * 2  # Allow some margin
        similarity = 1.0 - min(distance / max_distance, 1.0) if max_distance > 0 else 1.0
        
        return similarity >= threshold, similarity
    
    def get_failure_description(self) -> str:
        """Generate a human-readable description of the failure pattern"""
        desc_parts = []
        
        if self.description:
            desc_parts.append(self.description)
        else:
            desc_parts.append(f"{self.failure_type} failure pattern")
        
        # Add parameter information
        abnormal_params = []
        for param, ranges in self.parameter_ranges.items():
            if 'min' in ranges and 'max' in ranges:
                abnormal_params.append(f"{param}: {ranges['min']:.3f}-{ranges['max']:.3f}")
        
        if abnormal_params:
            desc_parts.append(f"Affected parameters: {', '.join(abnormal_params[:3])}")
        
        if self.common_operators:
            desc_parts.append(f"Common operators: {', '.join(self.common_operators[:3])}")
        
        desc_parts.append(f"Severity: {self.severity}")
        desc_parts.append(f"Based on {self.samples_count} samples")
        
        return ". ".join(desc_parts)

class FailurePatternManager:
    """Manages learning and matching of failure patterns"""
    
    def __init__(self, config):
        self.config = config
        self.patterns: Dict[str, List[FailurePattern]] = {}  # item_code -> list of patterns
        self.load_patterns()
    
    def learn_failure_patterns(self, historical_data: pd.DataFrame):
        """Learn failure patterns from historical data"""
        logger.info("Learning failure patterns from historical data...")
        
        # Filter failure items
        failure_data = historical_data[
            historical_data['result'].isin(self.config.FAILURE_RESULTS)
        ].copy()
        
        if len(failure_data) == 0:
            logger.info("No failure data found for pattern learning")
            return {}
        
        patterns_learned = {}
        
        # Group by item code and failure type
        for item_code in failure_data['item code'].unique():
            item_failures = failure_data[failure_data['item code'] == item_code]
            
            for failure_type in self.config.FAILURE_RESULTS:
                type_failures = item_failures[item_failures['result'] == failure_type]
                
                if len(type_failures) >= self.config.FAILURE_PATTERN_MIN_SAMPLES:
                    patterns = self._learn_patterns_for_type(
                        item_code, failure_type, type_failures
                    )
                    
                    if patterns:
                        if item_code not in patterns_learned:
                            patterns_learned[item_code] = []
                        patterns_learned[item_code].extend(patterns)
                        
                        # Add to global patterns
                        if item_code not in self.patterns:
                            self.patterns[item_code] = []
                        self.patterns[item_code].extend(patterns)
        
        # Save learned patterns
        self.save_patterns()
        
        logger.info(f"Learned failure patterns for {len(patterns_learned)} items")
        return patterns_learned
    
    def _learn_patterns_for_type(self, item_code: str, failure_type: str, 
                                failures_df: pd.DataFrame) -> List[FailurePattern]:
        """Learn patterns for specific item code and failure type"""
        try:
            # Extract numeric features
            numeric_features = []
            feature_names = []
            
            for col in self.config.NUMERIC_FEATURES:
                if col in failures_df.columns:
                    # Extract numeric values
                    col_series = failures_df[col].apply(self._extract_numeric)
                    if col_series.notna().sum() > 0:
                        numeric_features.append(col_series.values)
                        feature_names.append(col)
            
            if not numeric_features:
                return []
            
            # Create feature matrix
            X = np.column_stack(numeric_features)
            
            # Remove rows with all NaN
            valid_mask = ~np.isnan(X).all(axis=1)
            X = X[valid_mask]
            
            if len(X) < self.config.FAILURE_PATTERN_MIN_SAMPLES:
                return []
            
            # Handle missing values with column means
            for i in range(X.shape[1]):
                col_mean = np.nanmean(X[:, i])
                if np.isnan(col_mean):
                    col_mean = 0
                X[np.isnan(X[:, i]), i] = col_mean
            
            # Cluster similar failures
            patterns = []
            
            # Use DBSCAN to find clusters
            eps = 0.5  # Maximum distance between samples
            min_samples = max(2, len(X) // 3)
            
            try:
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                
                # Create pattern for each cluster
                unique_labels = set(clustering.labels_)
                
                for label in unique_labels:
                    if label == -1:  # Noise points
                        continue
                    
                    cluster_mask = clustering.labels_ == label
                    cluster_samples = X[cluster_mask]
                    
                    if len(cluster_samples) >= self.config.FAILURE_PATTERN_MIN_SAMPLES:
                        # Calculate cluster center
                        cluster_center = np.mean(cluster_samples, axis=0)
                        
                        # Calculate cluster radius (max distance from center)
                        distances = [euclidean(x, cluster_center) for x in cluster_samples]
                        cluster_radius = max(distances) if distances else 0.1
                        
                        # Get original rows for this cluster
                        cluster_indices = np.where(cluster_mask)[0]
                        original_rows = failures_df.iloc[cluster_indices]
                        
                        # Extract parameter ranges
                        parameter_ranges = {}
                        for i, feature in enumerate(feature_names):
                            values = cluster_samples[:, i]
                            if len(values) > 0:
                                parameter_ranges[feature] = {
                                    'min': float(np.min(values)),
                                    'max': float(np.max(values)),
                                    'mean': float(np.mean(values)),
                                    'std': float(np.std(values))
                                }
                        
                        # Extract common operators
                        if 'operator' in original_rows.columns:
                            operators = original_rows['operator'].dropna().unique()
                            common_operators = operators.tolist()[:5]
                        else:
                            common_operators = []
                        
                        # Create pattern
                        pattern = FailurePattern(
                            item_code=item_code,
                            failure_type=failure_type,
                            pattern_id=f"{item_code}_{failure_type}_{label}_{datetime.now().strftime('%Y%m%d')}",
                            features={name: feature_names[i] for i, name in enumerate(feature_names)},
                            cluster_center=cluster_center,
                            cluster_radius=cluster_radius,
                            samples_count=len(cluster_samples),
                            first_seen=original_rows['date'].min() if 'date' in original_rows.columns else datetime.now().strftime('%Y-%m-%d'),
                            last_seen=original_rows['date'].max() if 'date' in original_rows.columns else datetime.now().strftime('%Y-%m-%d'),
                            description=f"Cluster {label} of {failure_type} failures for {item_code}",
                            severity=self._determine_severity(failure_type, parameter_ranges),
                            common_operators=common_operators,
                            parameter_ranges=parameter_ranges
                        )
                        
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning(f"Clustering failed for {item_code}-{failure_type}: {e}")
                # Create a single pattern from all samples
                if len(X) >= self.config.FAILURE_PATTERN_MIN_SAMPLES:
                    pattern = self._create_single_pattern(
                        item_code, failure_type, failures_df, X, feature_names
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error learning patterns for {item_code}-{failure_type}: {e}")
            return []
    
    def _create_single_pattern(self, item_code: str, failure_type: str,
                             failures_df: pd.DataFrame, X: np.ndarray,
                             feature_names: List[str]) -> FailurePattern:
        """Create a single pattern when clustering fails"""
        cluster_center = np.mean(X, axis=0)
        
        # Calculate distances
        distances = [euclidean(x, cluster_center) for x in X]
        cluster_radius = max(distances) if distances else 0.1
        
        # Extract parameter ranges
        parameter_ranges = {}
        for i, feature in enumerate(feature_names):
            values = X[:, i]
            parameter_ranges[feature] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Extract common operators
        if 'operator' in failures_df.columns:
            operators = failures_df['operator'].dropna().unique()
            common_operators = operators.tolist()[:5]
        else:
            common_operators = []
        
        return FailurePattern(
            item_code=item_code,
            failure_type=failure_type,
            pattern_id=f"{item_code}_{failure_type}_single_{datetime.now().strftime('%Y%m%d')}",
            features={name: feature_names[i] for i, name in enumerate(feature_names)},
            cluster_center=cluster_center,
            cluster_radius=cluster_radius,
            samples_count=len(X),
            first_seen=failures_df['date'].min() if 'date' in failures_df.columns else datetime.now().strftime('%Y-%m-%d'),
            last_seen=failures_df['date'].max() if 'date' in failures_df.columns else datetime.now().strftime('%Y-%m-%d'),
            description=f"Single pattern of {failure_type} failures for {item_code}",
            severity=self._determine_severity(failure_type, parameter_ranges),
            common_operators=common_operators,
            parameter_ranges=parameter_ranges
        )
    
    def _determine_severity(self, failure_type: str, 
                           parameter_ranges: Dict[str, Dict[str, float]]) -> str:
        """Determine severity based on failure type and parameter deviations"""
        if failure_type == 'WW':  # Completely rejected
            return 'critical'
        elif failure_type in ['GW', 'WG']:  # Partially rejected
            # Check for extreme parameter values
            for param, ranges in parameter_ranges.items():
                mean_val = ranges.get('mean', 0)
                std_val = ranges.get('std', 0)
                if std_val > mean_val * 0.5:  # High variability
                    return 'high'
            return 'medium'
        return 'medium'
    
    def _extract_numeric(self, value: Any) -> Optional[float]:
        """Extract numeric value from mixed data"""
        if pd.isna(value):
            return np.nan
        
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            value_str = str(value).strip().lower()
            
            if value_str in ['na', 'nd', '', 'nan', 'none', 'null']:
                return np.nan
            
            # Extract numbers from text like "(leakage 1.15)"
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', value_str)
            if numbers:
                return float(numbers[0])
            
            return np.nan
        except:
            return np.nan
    
    def match_failure_patterns(self, item_code: str, features: np.ndarray,
                              feature_names: List[str]) -> List[Dict[str, Any]]:
        """Match features against known failure patterns"""
        if item_code not in self.patterns:
            return []
        
        matches = []
        for pattern in self.patterns[item_code]:
            # Align features with pattern features
            aligned_features = []
            for pattern_feat in pattern.features.values():
                if pattern_feat in feature_names:
                    idx = feature_names.index(pattern_feat)
                    aligned_features.append(features[idx])
                else:
                    aligned_features.append(0)  # Missing feature
            
            aligned_array = np.array(aligned_features)
            
            # Check match
            matches_pattern, similarity = pattern.matches(
                aligned_array, 
                self.config.FAILURE_SIMILARITY_THRESHOLD
            )
            
            if matches_pattern:
                matches.append({
                    'pattern': pattern,
                    'similarity': similarity,
                    'failure_type': pattern.failure_type,
                    'description': pattern.get_failure_description(),
                    'severity': pattern.severity,
                    'confidence': min(similarity * 100, 100),
                    'parameter_ranges': pattern.parameter_ranges,
                    'common_operators': pattern.common_operators,
                    'samples_based_on': pattern.samples_count
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def save_patterns(self):
        """Save failure patterns to disk"""
        save_data = {}
        for item_code, patterns in self.patterns.items():
            save_data[item_code] = [pattern.to_dict() for pattern in patterns]
        
        save_path = f"{self.config.FAILURE_PATTERNS_DIR}/failure_patterns.json"
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Also save as pickle for faster loading
        pickle_path = f"{self.config.FAILURE_PATTERNS_DIR}/failure_patterns.pkl"
        joblib.dump(self.patterns, pickle_path)
        
        logger.info(f"Saved {sum(len(p) for p in self.patterns.values())} failure patterns")
    
    def load_patterns(self):
        """Load failure patterns from disk"""
        pickle_path = f"{self.config.FAILURE_PATTERNS_DIR}/failure_patterns.pkl"
        json_path = f"{self.config.FAILURE_PATTERNS_DIR}/failure_patterns.json"
        
        try:
            self.patterns = joblib.load(pickle_path)
            logger.info(f"Loaded {sum(len(p) for p in self.patterns.values())} failure patterns")
        except:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                self.patterns = {}
                for item_code, patterns_data in data.items():
                    self.patterns[item_code] = [
                        FailurePattern.from_dict(pattern_data) 
                        for pattern_data in patterns_data
                    ]
                logger.info(f"Loaded {sum(len(p) for p in self.patterns.values())} failure patterns from JSON")
            except:
                self.patterns = {}
                logger.info("No existing failure patterns found")
    
    def get_patterns_for_item(self, item_code: str) -> List[Dict[str, Any]]:
        """Get all patterns for a specific item code"""
        if item_code in self.patterns:
            return [pattern.to_dict() for pattern in self.patterns[item_code]]
        return []
    
    def add_feedback_pattern(self, feedback_data: Dict[str, Any], 
                           processed_features: np.ndarray,
                           feature_names: List[str]):
        """Add a new pattern based on user feedback"""
        item_code = feedback_data.get('item_code')
        failure_type = feedback_data.get('feedback_type', 'unknown')
        
        if item_code not in self.patterns:
            self.patterns[item_code] = []
        
        # Create new pattern from feedback
        pattern = FailurePattern(
            item_code=item_code,
            failure_type=failure_type.upper() if failure_type != 'unknown' else 'CUSTOM',
            pattern_id=f"{item_code}_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            features={name: name for name in feature_names},
            cluster_center=processed_features,
            cluster_radius=0.1,  # Small radius for single sample
            samples_count=1,
            first_seen=datetime.now().strftime('%Y-%m-%d'),
            last_seen=datetime.now().strftime('%Y-%m-%d'),
            description=feedback_data.get('business_justification', 'User-confirmed failure'),
            severity=feedback_data.get('severity', 'medium'),
            common_operators=[feedback_data.get('operator', 'Unknown')] if feedback_data.get('operator') else [],
            parameter_ranges=self._create_parameter_ranges_from_features(
                processed_features, feature_names
            )
        )
        
        self.patterns[item_code].append(pattern)
        self.save_patterns()
        
        logger.info(f"Added feedback-based pattern for {item_code}")