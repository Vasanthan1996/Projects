# data_preprocessor.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import re
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handle mixed data types and preprocessing for manufacturing data"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.numeric_pattern = re.compile(r'[-+]?\d*\.?\d+')
        
    def extract_numeric_from_text(self, value: Any) -> Optional[float]:
        """Extract numeric values from text annotations"""
        if pd.isna(value):
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).lower()
        
        # Handle special cases
        if value_str in ['na', 'nd', '', 'nan', 'none']:
            return None
        
        # Extract first numeric value from text
        match = self.numeric_pattern.search(value_str)
        if match:
            return float(match.group())
        
        return None
    
    def preprocess_column(self, series: pd.Series) -> Tuple[pd.Series, str]:
        """Preprocess a single column, determining if it's numeric or categorical"""
        # Extract numeric values
        numeric_series = series.apply(self.extract_numeric_from_text)
        
        # Check if we have enough numeric values
        numeric_count = numeric_series.notna().sum()
        total_count = len(series)
        
        if numeric_count / total_count > 0.7:  # If >70% numeric, treat as numeric
            return numeric_series, 'numeric'
        else:
            return series.astype(str), 'categorical'
    
    def preprocess_dataframe(self, df: pd.DataFrame, is_training: bool = True, 
                           item_code: str = None) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        logger.info(f"Preprocessing dataframe with {len(df)} rows")
        
        processed_data = {}
        
        # Process each feature column
        for col in self.config.NUMERIC_FEATURES:
            if col in df.columns:
                series, col_type = self.preprocess_column(df[col])
                
                if col_type == 'numeric':
                    # Handle missing values
                    if is_training:
                        imputer = SimpleImputer(strategy='median')
                        processed_series = imputer.fit_transform(series.values.reshape(-1, 1)).flatten()
                        self.imputers[col] = imputer
                    else:
                        imputer = self.imputers.get(col, SimpleImputer(strategy='median'))
                        processed_series = imputer.transform(series.values.reshape(-1, 1)).flatten()
                    
                    # Scale numeric features
                    if is_training:
                        scaler = StandardScaler()
                        scaled_values = scaler.fit_transform(processed_series.reshape(-1, 1)).flatten()
                        self.scalers[f"{item_code}_{col}"] = scaler
                    else:
                        scaler_key = f"{item_code}_{col}"
                        scaler = self.scalers.get(scaler_key)
                        if scaler:
                            scaled_values = scaler.transform(processed_series.reshape(-1, 1)).flatten()
                        else:
                            scaled_values = processed_series
                    
                    processed_data[col] = scaled_values
                else:
                    # Handle categorical numeric columns
                    processed_data[col] = series.fillna('missing')
        
        # Process categorical features
        for col in self.config.CATEGORICAL_FEATURES:
            if col in df.columns:
                if is_training:
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(df[col].fillna('missing').astype(str))
                    self.encoders[col] = encoder
                else:
                    encoder = self.encoders.get(col)
                    if encoder:
                        try:
                            encoded = encoder.transform(df[col].fillna('missing').astype(str))
                        except:
                            encoded = np.zeros(len(df))
                    else:
                        encoded = np.zeros(len(df))
                
                processed_data[col] = encoded
        
        # Add identifier columns
        for col in self.config.IDENTIFIER_FEATURES:
            if col in df.columns:
                processed_data[col] = df[col]
        
        # Add result if present
        if 'result' in df.columns:
            processed_data['result'] = df['result']
        
        processed_df = pd.DataFrame(processed_data)
        logger.info(f"Preprocessing complete. Processed shape: {processed_df.shape}")
        
        return processed_df
    
    def save_preprocessors(self, item_code: str):
        """Save preprocessors for a specific item code"""
        preprocessor_data = {
            'scalers': {k: v for k, v in self.scalers.items() if item_code in k},
            'encoders': self.encoders,
            'imputers': self.imputers
        }
        
        save_path = f"{self.config.MODEL_DIR}/{item_code}_preprocessors.pkl"
        joblib.dump(preprocessor_data, save_path)
        logger.info(f"Saved preprocessors to {save_path}")
    
    def load_preprocessors(self, item_code: str):
        """Load preprocessors for a specific item code"""
        load_path = f"{self.config.MODEL_DIR}/{item_code}_preprocessors.pkl"
        try:
            preprocessor_data = joblib.load(load_path)
            self.scalers.update(preprocessor_data['scalers'])
            self.encoders.update(preprocessor_data['encoders'])
            self.imputers.update(preprocessor_data['imputers'])
            logger.info(f"Loaded preprocessors from {load_path}")
        except FileNotFoundError:
            logger.warning(f"No preprocessors found for {item_code}")