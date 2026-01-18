# database.py
import sqlite3
from sqlite3 import Error
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage SQLite database for the anomaly detection system"""
    
    def __init__(self, db_path: str = "anomaly_detection.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            
            # Create feedback table
            feedback_table = """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                bar_code TEXT NOT NULL,
                item_code TEXT NOT NULL,
                operator TEXT,
                date TEXT,
                time TEXT,
                anomaly_fields TEXT,
                original_value TEXT,
                feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'discussion')),
                business_justification TEXT,
                reviewed_by TEXT,
                model_predictions TEXT,
                confidence_score REAL
            )
            """
            
            # Create model updates table
            model_updates_table = """
            CREATE TABLE IF NOT EXISTS model_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                item_code TEXT NOT NULL,
                model_name TEXT NOT NULL,
                update_type TEXT,
                before_parameters TEXT,
                after_parameters TEXT,
                updated_by TEXT
            )
            """
            
            # Create audit log table
            audit_log_table = """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user TEXT,
                action TEXT,
                details TEXT,
                ip_address TEXT
            )
            """
            
            # Create detection history table
            detection_history_table = """
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_date DATE DEFAULT CURRENT_DATE,
                bar_code TEXT,
                item_code TEXT,
                operator TEXT,
                anomaly_score REAL,
                confidence_level REAL,
                abnormal_fields TEXT,
                detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed INTEGER DEFAULT 0
            )
            """
            
            cursor = self.connection.cursor()
            cursor.execute(feedback_table)
            cursor.execute(model_updates_table)
            cursor.execute(audit_log_table)
            cursor.execute(detection_history_table)
            
            self.connection.commit()
            logger.info("Database initialized successfully")
            
        except Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> int:
        """Save user feedback to database"""
        try:
            cursor = self.connection.cursor()
            
            insert_query = """
            INSERT INTO feedback (
                bar_code, item_code, operator, date, time,
                anomaly_fields, original_value, feedback_type,
                business_justification, reviewed_by, model_predictions,
                confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_query, (
                feedback_data.get('bar_code'),
                feedback_data.get('item_code'),
                feedback_data.get('operator'),
                feedback_data.get('date'),
                feedback_data.get('time'),
                json.dumps(feedback_data.get('anomaly_fields', [])),
                json.dumps(feedback_data.get('original_value', {})),
                feedback_data.get('feedback_type'),
                feedback_data.get('business_justification'),
                feedback_data.get('reviewed_by'),
                json.dumps(feedback_data.get('model_predictions', {})),
                feedback_data.get('confidence_score', 0.0)
            ))
            
            self.connection.commit()
            feedback_id = cursor.lastrowid
            
            # Log the feedback action
            self.log_action(
                user=feedback_data.get('reviewed_by'),
                action=f"feedback_{feedback_data.get('feedback_type')}",
                details=f"Feedback for {feedback_data.get('bar_code')}"
            )
            
            logger.info(f"Saved feedback with ID: {feedback_id}")
            return feedback_id
            
        except Error as e:
            logger.error(f"Error saving feedback: {e}")
            return -1
    
    def get_feedback(self, item_code: str = None, start_date: str = None, 
                    end_date: str = None) -> pd.DataFrame:
        """Retrieve feedback with optional filters"""
        try:
            query = "SELECT * FROM feedback WHERE 1=1"
            params = []
            
            if item_code:
                query += " AND item_code = ?"
                params.append(item_code)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, self.connection, params=params)
            return df
            
        except Error as e:
            logger.error(f"Error retrieving feedback: {e}")
            return pd.DataFrame()
    
    def log_action(self, user: str, action: str, details: str, ip_address: str = None):
        """Log user actions for audit trail"""
        try:
            cursor = self.connection.cursor()
            
            insert_query = """
            INSERT INTO audit_log (user, action, details, ip_address)
            VALUES (?, ?, ?, ?)
            """
            
            cursor.execute(insert_query, (user, action, details, ip_address))
            self.connection.commit()
            
        except Error as e:
            logger.error(f"Error logging action: {e}")
    
    def save_detection_result(self, detection_data: Dict[str, Any]):
        """Save detection results to history"""
        try:
            cursor = self.connection.cursor()
            
            insert_query = """
            INSERT INTO detection_history (
                bar_code, item_code, operator, anomaly_score,
                confidence_level, abnormal_fields
            ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_query, (
                detection_data.get('bar_code'),
                detection_data.get('item_code'),
                detection_data.get('operator'),
                detection_data.get('anomaly_score'),
                detection_data.get('confidence_level'),
                json.dumps(detection_data.get('abnormal_fields', []))
            ))
            
            self.connection.commit()
            
        except Error as e:
            logger.error(f"Error saving detection result: {e}")
    
    def export_to_csv(self, table_name: str, filepath: str):
        """Export table to CSV file"""
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, self.connection)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {table_name} to {filepath}")
        except Error as e:
            logger.error(f"Error exporting {table_name}: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")