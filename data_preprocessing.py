"""
Data Preprocessing Module
Handles missing values, duplicates, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

class DataPreprocessor:
    """Class to preprocess real estate data"""
    
    def __init__(self):
        """Initialize encoders and scalers"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        - Numerical: Fill with median
        - Categorical: Fill with mode
        """
        print("\n🔍 Handling Missing Values...")
        
        # Get numerical and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  ✓ Filled {missing_count} missing values in '{col}' with median")
        
        # Handle categorical missing values
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  ✓ Filled {missing_count} missing values in '{col}' with mode")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("\n🔍 Removing Duplicates...")
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_shape - df.shape[0]
        print(f"  ✓ Removed {removed} duplicate rows")
        return df
    
    def encode_categorical(self, df, columns):
        """
        Encode categorical variables using Label Encoding
        Saves encoders for later use
        """
        print("\n🔍 Encoding Categorical Variables...")
        
        for col in columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    # Create new encoder
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    print(f"  ✓ Encoded '{col}' ({len(self.label_encoders[col].classes_)} unique values)")
                else:
                    # Use existing encoder
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def handle_outliers(self, df, columns, method='iqr'):
        """
        Handle outliers using IQR method
        """
        print("\n🔍 Handling Outliers...")
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  ✓ Capped {outliers} outliers in '{col}'")
        
        return df
    
    def save_encoders(self, filepath='models/encoders.pkl'):
        """Save label encoders for later use"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"\n💾 Saved encoders to '{filepath}'")
    
    def load_encoders(self, filepath='models/encoders.pkl'):
        """Load saved label encoders"""
        with open(filepath, 'rb') as f:
            self.label_encoders = pickle.load(f)
        print(f"\n📂 Loaded encoders from '{filepath}'")
    
    def preprocess(self, df, categorical_cols, outlier_cols=None):
        """
        Complete preprocessing pipeline
        """
        print("\n" + "="*60)
        print("🚀 STARTING DATA PREPROCESSING")
        print("="*60)
        
        # Make a copy
        df = df.copy()
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3: Encode categorical variables
        df = self.encode_categorical(df, categorical_cols)
        
        # Step 4: Handle outliers (optional)
        if outlier_cols:
            df = self.handle_outliers(df, outlier_cols)
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETED")
        print("="*60)
        
        return df
