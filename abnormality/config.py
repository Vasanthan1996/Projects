# config.py
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json

@dataclass
class Config:
    """Configuration settings for the anomaly detection system"""
    
    # Data settings
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        'stroke 1', 'Q L feed', 'Q R feed', 'cyl 1', 'cyl 2', 'stroke 2', 'stroke 3'
    ])
    
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        'description', 'operator'
    ])
    
    IDENTIFIER_FEATURES: List[str] = field(default_factory=lambda: [
        'bar code', 'item code', 'date', 'time'
    ])
    
    # Result categories
    GOOD_RESULTS: List[str] = field(default_factory=lambda: ['GG'])
    FAILURE_RESULTS: List[str] = field(default_factory=lambda: ['WW', 'GW', 'WG'])
    ALL_RESULTS: List[str] = field(default_factory=lambda: ['GG', 'WW', 'GW', 'WG'])
    
    # Failure pattern learning
    FAILURE_PATTERN_MIN_SAMPLES: int = 3
    FAILURE_SIMILARITY_THRESHOLD: float = 0.8
    
    # Model settings
    ML_MODELS: List[str] = field(default_factory=lambda: [
        'isolation_forest',
        'local_outlier_factor',
        'one_class_svm',
        'elliptic_envelope'
    ])
    
    DL_MODELS: List[str] = field(default_factory=lambda: [
        'autoencoder',
        'lstm_autoencoder'
    ])
    
    # Training settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    ANOMALY_THRESHOLD: float = 0.85
    
    # Database settings
    DATABASE_PATH: str = "anomaly_detection.db"
    
    # File paths
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"
    UPLOAD_DIR: str = "uploads"
    FAILURE_PATTERNS_DIR: str = "failure_patterns"
    
    # Visualization settings
    PLOTLY_TEMPLATE: str = "plotly_white"
    
    def __post_init__(self):
        """Create necessary directories"""
        for directory in [self.MODEL_DIR, self.LOG_DIR, self.UPLOAD_DIR, self.FAILURE_PATTERNS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)