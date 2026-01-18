# detector.py (Updated)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json

from data_preprocessor import DataPreprocessor
from model_manager import ModelManager
from failure_patterns import FailurePatternManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Main anomaly detection orchestrator with failure pattern learning"""
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.model_manager = ModelManager(config)
        self.failure_manager = FailurePatternManager(config)
        self.db_manager = db_manager
        
        # Track trained item codes
        self.trained_items = set()
    
    def prepare_training_data(self, historical_data: pd.DataFrame, 
                            include_failures: bool = False,
                            min_samples: int = 5) -> Dict[str, pd.DataFrame]:
        """Prepare training data for each item code"""
        logger.info(f"Preparing training data (include_failures={include_failures})...")
        
        if include_failures:
            # Use all data for pattern learning
            training_data = historical_data.copy()
        else:
            # Use only GG items for normal model training
            training_data = historical_data[historical_data['result'].isin(self.config.GOOD_RESULTS)].copy()
        
        if len(training_data) == 0:
            logger.warning("No training data found!")
            return {}
        
        # Group by item code
        item_groups = {}
        item_codes = training_data['item code'].unique()
        
        logger.info(f"Found {len(item_codes)} unique item codes")
        
        for item_code in item_codes:
            item_data = training_data[training_data['item code'] == item_code].copy()
            
            if include_failures:
                # For failure pattern learning, we need samples of each failure type
                has_failures = any(
                    item_data['result'].isin(self.config.FAILURE_RESULTS).sum() >= min_samples
                )
                if has_failures:
                    item_groups[item_code] = item_data
                    logger.info(f"Item {item_code}: {len(item_data)} samples for pattern learning")
            else:
                # For normal model training, need enough GG samples
                gg_count = len(item_data[item_data['result'].isin(self.config.GOOD_RESULTS)])
                if gg_count >= min_samples:
                    item_groups[item_code] = item_data
                    logger.info(f"Item {item_code}: {gg_count} GG samples")
        
        logger.info(f"Prepared training data for {len(item_groups)} items")
        return item_groups
    
    def train_all_models(self, historical_data: pd.DataFrame, 
                        selected_models: List[str] = None,
                        min_samples: int = 5,
                        learn_failure_patterns: bool = True):
        """Train models and learn failure patterns"""
        logger.info("Starting comprehensive training...")
        
        training_results = {}
        
        # Step 1: Learn failure patterns from all historical data
        if learn_failure_patterns:
            logger.info("Step 1: Learning failure patterns...")
            failure_patterns = self.failure_manager.learn_failure_patterns(historical_data)
            
            for item_code, patterns in failure_patterns.items():
                if item_code not in training_results:
                    training_results[item_code] = {}
                training_results[item_code]['failure_patterns'] = {
                    'count': len(patterns),
                    'types': list(set(p.failure_type for p in patterns))
                }
        
        # Step 2: Train normal models from GG items
        logger.info("Step 2: Training normal models from GG items...")
        normal_models_result = self._train_normal_models(
            historical_data, selected_models, min_samples
        )
        
        # Merge results
        for item_code, result in normal_models_result.items():
            if item_code not in training_results:
                training_results[item_code] = {}
            training_results[item_code].update(result)
        
        return training_results
    
    def _train_normal_models(self, historical_data: pd.DataFrame,
                           selected_models: List[str],
                           min_samples: int) -> Dict[str, Dict]:
        """Train normal behavior models from GG items"""
        # Prepare GG training data
        item_groups = self.prepare_training_data(
            historical_data, include_failures=False, min_samples=min_samples
        )
        
        if not item_groups:
            logger.warning("No items with sufficient GG samples found for training.")
            return {}
        
        training_results = {}
        
        for item_code, item_data in item_groups.items():
            try:
                logger.info(f"Training normal models for {item_code}...")
                
                # Preprocess data
                preprocessed_data = self.preprocessor.preprocess_dataframe(
                    item_data, is_training=True, item_code=item_code
                )
                
                # Extract features
                feature_columns = [col for col in preprocessed_data.columns 
                                 if col not in self.config.IDENTIFIER_FEATURES + ['result']]
                
                if not feature_columns:
                    logger.warning(f"No features available for {item_code}")
                    continue
                
                X_train = preprocessed_data[feature_columns].values
                
                # Train models
                trained_models = self.model_manager.train_models_for_item(
                    item_code, X_train, selected_models
                )
                
                # Save preprocessors
                self.preprocessor.save_preprocessors(item_code)
                
                training_results[item_code] = {
                    'status': 'success',
                    'num_samples': len(item_data),
                    'num_features': X_train.shape[1],
                    'trained_models': list(trained_models.keys()),
                    'model_type': 'normal_behavior'
                }
                
                self.trained_items.add(item_code)
                
                logger.info(f"Successfully trained {len(trained_models)} normal models for {item_code}")
                
            except Exception as e:
                training_results[item_code] = {
                    'status': 'failed',
                    'error': str(e),
                    'model_type': 'normal_behavior'
                }
                logger.error(f"Error training normal models for {item_code}: {e}")
        
        return training_results
    
    def detect_anomalies(self, new_data: pd.DataFrame, 
                        item_code: str = None,
                        check_failure_patterns: bool = True) -> pd.DataFrame:
        """Detect anomalies with failure pattern matching"""
        logger.info(f"Detecting anomalies in new data ({len(new_data)} records)...")
        
        if item_code:
            # Detect for specific item code
            item_data = new_data[new_data['item code'] == item_code].copy()
            return self._detect_for_item(item_code, item_data, check_failure_patterns)
        else:
            # Detect for all item codes
            all_results = []
            
            for current_item_code in new_data['item code'].unique():
                try:
                    item_data = new_data[new_data['item code'] == current_item_code].copy()
                    
                    if len(item_data) > 0:
                        item_results = self._detect_for_item(
                            current_item_code, item_data, check_failure_patterns
                        )
                        all_results.append(item_results)
                        
                except Exception as e:
                    logger.error(f"Error detecting anomalies for {current_item_code}: {e}")
                    continue
            
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            else:
                return pd.DataFrame()
    
    def _detect_for_item(self, item_code: str, item_data: pd.DataFrame,
                        check_failure_patterns: bool) -> pd.DataFrame:
        """Detect anomalies for a specific item code"""
        # Check if models exist
        if item_code not in self.trained_items:
            self.model_manager.load_models_for_item(item_code)
            if item_code in self.model_manager.models:
                self.trained_items.add(item_code)
            else:
                logger.warning(f"No trained models found for {item_code}")
                # We can still check failure patterns even without normal models
                pass
        
        # Load preprocessors
        self.preprocessor.load_preprocessors(item_code)
        
        # Preprocess data
        preprocessed_data = self.preprocessor.preprocess_dataframe(
            item_data, is_training=False, item_code=item_code
        )
        
        # Extract features
        feature_columns = [col for col in preprocessed_data.columns 
                         if col not in self.config.IDENTIFIER_FEATURES + ['result']]
        
        results = []
        
        for idx, row in item_data.iterrows():
            if idx < len(preprocessed_data):
                processed_row = preprocessed_data.iloc[idx]
                X = processed_row[feature_columns].values.reshape(1, -1)
                
                # Get normal anomaly predictions
                is_normal_anomaly = False
                anomaly_score = 0.0
                model_predictions = {}
                model_probabilities = {}
                
                if item_code in self.model_manager.models:
                    predictions = self.model_manager.ensemble_predict(item_code, X)
                    is_normal_anomaly = bool(predictions['ensemble_predictions'][0])
                    anomaly_score = float(predictions['ensemble_probabilities'][0])
                    model_predictions = {
                        model: int(preds[0]) if len(preds) > 0 else 0
                        for model, preds in predictions.get('individual_predictions', {}).items()
                    }
                    model_probabilities = {
                        model: float(probs[0]) if len(probs) > 0 else 0.0
                        for model, probs in predictions.get('individual_probabilities', {}).items()
                    }
                
                # Check failure patterns
                failure_matches = []
                if check_failure_patterns:
                    failure_matches = self.failure_manager.match_failure_patterns(
                        item_code, X.flatten(), feature_columns
                    )
                
                # Determine if this is an anomaly
                is_anomaly = is_normal_anomaly or len(failure_matches) > 0
                
                if is_anomaly:
                    # Find abnormal fields
                    abnormal_fields = self._identify_abnormal_fields(
                        item_code, row, processed_row if idx < len(preprocessed_data) else None
                    )
                    
                    # Determine anomaly type
                    if failure_matches:
                        # Matches known failure pattern
                        primary_match = failure_matches[0]
                        anomaly_type = f"KNOWN_FAILURE_{primary_match['failure_type']}"
                        failure_info = {
                            'matched_patterns': [
                                {
                                    'failure_type': match['failure_type'],
                                    'similarity': match['similarity'],
                                    'description': match['description'],
                                    'severity': match['severity'],
                                    'confidence': match['confidence']
                                }
                                for match in failure_matches[:3]  # Top 3 matches
                            ]
                        }
                        confidence = primary_match['confidence']
                    else:
                        # Unknown anomaly (normal model detected)
                        anomaly_type = "UNKNOWN_ANOMALY"
                        failure_info = {}
                        confidence = min(anomaly_score * 100, 100)
                    
                    result = {
                        'bar_code': row.get('bar code', ''),
                        'item_code': item_code,
                        'date': row.get('date', ''),
                        'time': row.get('time', ''),
                        'operator': row.get('operator', ''),
                        'description': row.get('description', ''),
                        'result': row.get('result', 'Unknown'),
                        'is_anomaly': int(is_anomaly),
                        'anomaly_score': anomaly_score,
                        'confidence_level': confidence,
                        'anomaly_type': anomaly_type,
                        'failure_matches': json.dumps(failure_info),
                        'abnormal_fields': abnormal_fields,
                        'model_predictions': model_predictions,
                        'model_probabilities': model_probabilities
                    }
                    
                    # Save to database if available
                    if self.db_manager:
                        self.db_manager.save_detection_result({
                            'bar_code': result['bar_code'],
                            'item_code': result['item_code'],
                            'operator': result['operator'],
                            'anomaly_score': result['anomaly_score'],
                            'confidence_level': result['confidence_level'],
                            'anomaly_type': result['anomaly_type'],
                            'abnormal_fields': result['abnormal_fields']
                        })
                    
                    results.append(result)
        
        results_df = pd.DataFrame(results)
        logger.info(f"Detected {len(results_df)} anomalies for {item_code}")
        
        return results_df
    
    def _identify_abnormal_fields(self, item_code: str, original_row: pd.Series, 
                                processed_row: pd.Series = None) -> List[Dict[str, Any]]:
        """Identify which specific fields are abnormal"""
        abnormal_fields = []
        
        for feature in self.config.NUMERIC_FEATURES:
            if feature in original_row:
                value = original_row[feature]
                
                # Check if value is significantly different from normal
                if pd.isna(value) or str(value).strip() in ['', 'NA', 'ND']:
                    abnormal_fields.append({
                        'field': feature,
                        'value': str(value),
                        'reason': 'Missing or invalid value',
                        'severity': 'medium'
                    })
                elif processed_row is not None and feature in processed_row:
                    processed_value = processed_row[feature]
                    if abs(processed_value) > 3:  # More than 3 standard deviations
                        abnormal_fields.append({
                            'field': feature,
                            'value': str(value),
                            'reason': 'Value significantly different from normal',
                            'severity': 'high'
                        })
        
        return abnormal_fields
    
    def update_with_feedback(self, feedback_data: Dict[str, Any]):
        """Update models based on user feedback"""
        item_code = feedback_data.get('item_code')
        
        if feedback_data.get('feedback_type') in ['reject', 'accept']:
            # Add to failure patterns if rejected (confirmed anomaly)
            if feedback_data.get('feedback_type') == 'reject':
                # Extract features from the anomaly
                # In a full implementation, you would store and process the features
                pass
        
        # Save feedback to database
        if self.db_manager:
            self.db_manager.save_feedback(feedback_data)
        
        return True
    
    def get_failure_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of learned failure patterns"""
        summary = {
            'total_items': len(self.failure_manager.patterns),
            'total_patterns': sum(len(p) for p in self.failure_manager.patterns.values()),
            'by_failure_type': {},
            'items_with_patterns': list(self.failure_manager.patterns.keys())
        }
        
        # Count by failure type
        for item_code, patterns in self.failure_manager.patterns.items():
            for pattern in patterns:
                failure_type = pattern.failure_type
                if failure_type not in summary['by_failure_type']:
                    summary['by_failure_type'][failure_type] = 0
                summary['by_failure_type'][failure_type] += 1
        
        return summary