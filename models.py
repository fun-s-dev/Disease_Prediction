"""
Database Models for MediGuard AI
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    """Prediction model"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_id = db.Column(db.String(100), nullable=False)
    patient_name = db.Column(db.String(200))
    patient_age = db.Column(db.Integer)
    patient_sex = db.Column(db.String(10))
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    raw_features = db.Column(db.Text, nullable=False)  # JSON string
    probabilities = db.Column(db.Text, nullable=False)  # JSON string
    block_hash = db.Column(db.String(64), nullable=False)
    data_quality_issues = db.Column(db.Text)  # JSON string
    data_quality_warnings = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction}>'
    
    def get_raw_features(self):
        """Parse raw features JSON"""
        import json
        return json.loads(self.raw_features)
    
    def get_probabilities(self):
        """Parse probabilities JSON"""
        import json
        return json.loads(self.probabilities)
    
    def get_data_quality_issues(self):
        """Parse data quality issues JSON"""
        import json
        if self.data_quality_issues:
            return json.loads(self.data_quality_issues)
        return []
    
    def get_data_quality_warnings(self):
        """Parse data quality warnings JSON"""
        import json
        if self.data_quality_warnings:
            return json.loads(self.data_quality_warnings)
        return []

