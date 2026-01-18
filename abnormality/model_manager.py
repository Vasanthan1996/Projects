# model_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

# ML models
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# DL models
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Ensemble
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for individual model"""
    name: str
    model_type: str  # 'ml' or 'dl'
    hyperparameters: Dict[str, Any]
    is_active: bool = True

class BaseModel(ABC):
    """Abstract base class for all anomaly detection models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.training_history = None
        self.metrics = {}
        
    @abstractmethod
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities"""
        pass
    
    def save(self, filepath: str):
        """Save model to disk"""
        if self.model is not None:
            if hasattr(self.model, 'save'):
                self.model.save(filepath)
            else:
                joblib.dump(self.model, filepath)
            logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        try:
            # Try loading as TensorFlow model
            self.model = load_model(filepath)
        except:
            # Try loading as sklearn model
            self.model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        self.metrics = {
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions)
        }
        
        return self.metrics

class IsolationForestModel(BaseModel):
    """Isolation Forest anomaly detection"""
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        self.model = IsolationForest(
            **self.config.hyperparameters,
            random_state=42,
            contamination='auto'
        )
        self.model.fit(X_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X)
        # Convert to probabilities (0 for anomaly, 1 for normal)
        probs = (predictions + 1) / 2
        return probs

class AutoencoderModel(BaseModel):
    """Deep Learning Autoencoder for anomaly detection"""
    
    def build_model(self, input_dim: int):
        """Build autoencoder architecture"""
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='mse')
        
        return autoencoder
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        input_dim = X_train.shape[1]
        self.model = self.build_model(input_dim)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f"models/{self.config.name}_best.h5",
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        self.training_history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
    
    def predict(self, X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Use quantile as threshold
        if len(mse) > 0:
            threshold_value = np.quantile(mse, threshold)
            return (mse > threshold_value).astype(int)
        return np.zeros(len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        # Normalize MSE to [0, 1] range
        if mse.max() > 0:
            return mse / mse.max()
        return mse

class LSTMAutoencoderModel(BaseModel):
    """LSTM Autoencoder for sequence data"""
    
    def build_model(self, input_shape: tuple):
        """Build LSTM autoencoder"""
        # Encoder
        inputs = Input(shape=input_shape)
        encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
        encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
        
        # Repeat vector for decoder
        repeated = tf.keras.layers.RepeatVector(input_shape[0])(encoded)
        
        # Decoder
        decoded = LSTM(16, activation='relu', return_sequences=True)(repeated)
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = tf.keras.layers.TimeDistributed(Dense(input_shape[1]))(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        # Reshape for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        input_shape = (1, X_train.shape[1])
        self.model = self.build_model(input_shape)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        self.training_history = self.model.fit(
            X_train_reshaped, X_train_reshaped,
            validation_data=(
                X_val.reshape(X_val.shape[0], 1, X_val.shape[1]), 
                X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            ) if X_val is not None else None,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
    
    def predict(self, X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        reconstructions = self.model.predict(X_reshaped)
        mse = np.mean(np.power(X_reshaped - reconstructions, 2), axis=(1, 2))
        
        if len(mse) > 0:
            threshold_value = np.quantile(mse, threshold)
            return (mse > threshold_value).astype(int)
        return np.zeros(len(X))

class ModelManager:
    """Manage multiple models for each item code"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_factories = {
            'isolation_forest': IsolationForestModel,
            'autoencoder': AutoencoderModel,
            'lstm_autoencoder': LSTMAutoencoderModel
        }
    
    def create_model(self, model_name: str, model_type: str) -> BaseModel:
        """Create a model instance based on name and type"""
        hyperparameters = self.get_default_hyperparameters(model_name)
        
        model_config = ModelConfig(
            name=model_name,
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        
        if model_name in self.model_factories:
            return self.model_factories[model_name](model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_default_hyperparameters(self, model_name: str) -> Dict:
        """Get default hyperparameters for each model"""
        hyperparameters = {
            'isolation_forest': {
                'n_estimators': 100,
                'max_samples': 'auto',
                'contamination': 0.1
            },
            'autoencoder': {
                'encoding_dim': 16,
                'epochs': 100,
                'batch_size': 32
            },
            'lstm_autoencoder': {
                'lstm_units': [32, 16],
                'epochs': 50,
                'batch_size': 32
            }
        }
        return hyperparameters.get(model_name, {})
    
    def train_models_for_item(self, item_code: str, X_train: np.ndarray, 
                            selected_models: List[str] = None):
        """Train multiple models for a specific item code"""
        logger.info(f"Training models for item code: {item_code}")
        
        if selected_models is None:
            selected_models = self.config.ML_MODELS[:2]  # Default to first 2 models
        
        # Split data for training
        X_train_split, X_val_split = train_test_split(
            X_train, test_size=0.2, random_state=42
        )
        
        trained_models = {}
        
        for model_name in selected_models:
            try:
                logger.info(f"Training {model_name} for {item_code}")
                
                # Create and train model
                if model_name in self.config.ML_MODELS:
                    model = self.create_model(model_name, 'ml')
                else:
                    model = self.create_model(model_name, 'dl')
                
                model.train(X_train_split, X_val_split)
                
                # Evaluate on validation set
                # For unsupervised, we don't have labels, so we'll use reconstruction error
                
                trained_models[model_name] = model
                
                # Save the model
                model.save(f"{self.config.MODEL_DIR}/{item_code}_{model_name}.pkl")
                
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.models[item_code] = trained_models
        return trained_models
    
    def load_models_for_item(self, item_code: str) -> Dict[str, BaseModel]:
        """Load trained models for a specific item code"""
        trained_models = {}
        
        for model_name in self.config.ML_MODELS + self.config.DL_MODELS:
            try:
                filepath = f"{self.config.MODEL_DIR}/{item_code}_{model_name}.pkl"
                model = self.create_model(model_name, 'ml' if model_name in self.config.ML_MODELS else 'dl')
                model.load(filepath)
                trained_models[model_name] = model
                logger.info(f"Loaded {model_name} for {item_code}")
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                continue
        
        self.models[item_code] = trained_models
        return trained_models
    
    def ensemble_predict(self, item_code: str, X: np.ndarray) -> Dict[str, Any]:
        """Make ensemble predictions using all trained models"""
        if item_code not in self.models:
            self.load_models_for_item(item_code)
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.get(item_code, {}).items():
            try:
                pred = model.predict(X)
                proba = model.predict_proba(X)
                
                predictions[model_name] = pred
                probabilities[model_name] = proba
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
                continue
        
        # Combine predictions (voting)
        if predictions:
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = np.mean(all_preds, axis=0) > 0.5
            ensemble_proba = np.mean(list(probabilities.values()), axis=0)
        else:
            ensemble_pred = np.zeros(len(X))
            ensemble_proba = np.zeros(len(X))
        
        return {
            'ensemble_predictions': ensemble_pred.astype(int),
            'ensemble_probabilities': ensemble_proba,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }