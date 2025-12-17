"""
Model Training Module
Trains classification and regression models with MLflow tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Class to train and evaluate ML models"""
    
    def __init__(self, experiment_name="real_estate_investment"):
        """Initialize model trainer with MLflow experiment"""
        mlflow.set_experiment(experiment_name)
        self.classification_model = None
        self.regression_model = None
        self.feature_names = None
        
    def prepare_data(self, df, target_col, feature_cols, test_size=0.2):
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature column names
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nPreparing data for '{target_col}'...")
        
        # Select features and target
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Decide whether to stratify (only when class distribution supports it)
        stratify_param = None
        if target_col == 'Good_Investment':
            try:
                vc = y.value_counts()
                if vc.min() >= 2:
                    stratify_param = y
                else:
                    print("  Warning: class distribution too small for stratified split; proceeding without stratify")
            except Exception:
                stratify_param = None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        print(f"  Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_classification_model(self, X_train, X_test, y_train, y_test, model_type='rf'):
        """
        Train classification model with MLflow tracking
        
        Args:
            model_type: 'lr' (Logistic), 'rf' (Random Forest), 'xgb' (XGBoost)
        """
        print(f"\nTraining Classification Model: {model_type.upper()}")
        
        with mlflow.start_run(run_name=f"Classification_{model_type}"):
            
            # Choose model
            if model_type == 'lr':
                model = LogisticRegression(random_state=42, max_iter=1000)
                params = {'max_iter': 1000}
            elif model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                }
            elif model_type == 'xgb':
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            
            # Train model
            print("  Training...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log model
            mlflow.sklearn.log_model(model, "classification_model")
            
            # Print results
            print(f"\n  {model_type.upper()} Classification Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n  Confusion Matrix:")
            print(f"  {cm}")
            
            # Store best model
            if self.classification_model is None or roc_auc > getattr(self, 'best_class_auc', 0):
                self.classification_model = model
                self.best_class_auc = roc_auc
            
            return model, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'confusion_matrix': cm
            }
    
    def train_regression_model(self, X_train, X_test, y_train, y_test, model_type='rf'):
        """
        Train regression model with MLflow tracking
        
        Args:
            model_type: 'lr' (Linear), 'rf' (Random Forest), 'xgb' (XGBoost)
        """
        print(f"\nTraining Regression Model: {model_type.upper()}")
        
        with mlflow.start_run(run_name=f"Regression_{model_type}"):
            
            # Choose model
            if model_type == 'lr':
                model = LinearRegression()
                params = {}
            elif model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                }
            elif model_type == 'xgb':
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            
            # Train model
            print("  Training...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "regression_model")
            
            # Print results
            print(f"\n  {model_type.upper()} Regression Results:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R2:   {r2:.4f}")
            
            # Store best model
            if self.regression_model is None or r2 > getattr(self, 'best_reg_r2', 0):
                self.regression_model = model
                self.best_reg_r2 = r2
            
            return model, {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
    
    def save_models(self, class_path='../models/classification_model.pkl',
                   reg_path='../models/regression_model.pkl'):
        """Save trained models"""
        print("\nSaving Models...")
        
        if self.classification_model:
            joblib.dump(self.classification_model, class_path)
            print(f"  Classification model: {class_path}")
        
        if self.regression_model:
            joblib.dump(self.regression_model, reg_path)
            print(f"  Regression model: {reg_path}")
