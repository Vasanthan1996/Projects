# utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import os

def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{log_dir}/anomaly_detection_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Validate dataframe structure"""
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'issues': []
    }
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_result['is_valid'] = False
        validation_result['missing_columns'] = missing_columns
        validation_result['issues'].append(f"Missing columns: {', '.join(missing_columns)}")
    
    # Check for empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        validation_result['empty_columns'] = empty_columns
        validation_result['issues'].append(f"Empty columns: {', '.join(empty_columns)}")
    
    # Check for duplicate bar codes
    if 'bar code' in df.columns:
        duplicates = df[df.duplicated('bar code', keep=False)]
        if len(duplicates) > 0:
            validation_result['issues'].append(f"Found {len(duplicates)} duplicate bar codes")
    
    return validation_result

def create_sample_data():
    """Create sample data for testing"""
    sample_data = {
        'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'time': ['08:00', '09:00', '10:00', '11:00'],
        'operator': ['Operator1', 'Operator2', 'Operator1', 'Operator2'],
        'bar code': ['BC001', 'BC002', 'BC003', 'BC004'],
        'item code': ['A16384J', 'A16384J', '003243852B', '003243852B'],
        'description': ['Part A', 'Part A', 'Part B', 'Part B'],
        'result': ['GG', 'GG', 'WG', 'GW'],
        'stroke 1': [1.08, 1.09, 0.95, 1.02],
        'Q L feed': [0.047, 0.048, 0.163, 'NA'],
        'Q R feed': [0.009, 0.010, 0.113, '(leakage 1.15)'],
        'cyl 1': [395, 396, 397, 398],
        'cyl 2': [np.nan, np.nan, np.nan, 'ND'],
        'stroke 2': [1.07, 1.08, 0.91, 1.05],
        'stroke 3': [1.1, 1.11, 1.01, 1.08]
    }
    
    return pd.DataFrame(sample_data)

def export_results(results_df: pd.DataFrame, format: str = 'csv') -> str:
    """Export results in specified format"""
    if format == 'csv':
        return results_df.to_csv(index=False)
    elif format == 'json':
        return results_df.to_json(orient='records', indent=2)
    elif format == 'excel':
        # This would return a bytes object in production
        return results_df.to_excel(index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    import tensorflow as tf
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total // (1024**3),
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'timestamp': datetime.now().isoformat()
    }
    
    return system_info