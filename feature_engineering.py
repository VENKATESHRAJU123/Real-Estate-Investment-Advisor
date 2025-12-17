"""
Feature Engineering Module
Creates new features and target variables
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Class to engineer features for real estate prediction"""
    
    def __init__(self, growth_rate=0.08, current_year=2025):
        """
        Initialize feature engineer
        
        Args:
            growth_rate: Annual price growth rate (default 8%)
            current_year: Current year for age calculation
        """
        self.growth_rate = growth_rate
        self.current_year = current_year
    
    def create_price_per_sqft(self, df):
        """Calculate price per square foot"""
        print("  Creating Price_per_SqFt feature")
        
        if 'Price_per_SqFt' not in df.columns:
            df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
        
        return df
    
    def create_age_of_property(self, df):
        """Calculate age of property"""
        print("  Creating Age_of_Property feature")
        
        if 'Age_of_Property' not in df.columns:
            df['Age_of_Property'] = self.current_year - df['Year_Built']
        
        return df
    
    def create_school_density_score(self, df):
        """Create school density score"""
        print("  Creating School_Density_Score feature")
        # Ensure Nearby_Schools is numeric (handles strings like '3' or lists)
        ns = pd.to_numeric(df['Nearby_Schools'], errors='coerce').fillna(0)
        df['School_Density_Score'] = ns / (df['Age_of_Property'] + 1)
        return df
    
    def create_infrastructure_score(self, df):
        """
        Create infrastructure score
        Combines schools, hospitals, and transport
        """
        print("  Creating Infrastructure_Score feature")
        # Coerce the relevant columns to numeric to avoid multiplying sequences (strings/lists)
        ns = pd.to_numeric(df['Nearby_Schools'], errors='coerce').fillna(0)
        nh = pd.to_numeric(df['Nearby_Hospitals'], errors='coerce').fillna(0)
        pt = pd.to_numeric(df['Public_Transport_Accessibility'], errors='coerce').fillna(0)

        df['Infrastructure_Score'] = ns * 0.3 + nh * 0.3 + pt * 0.4
        
        return df
    
    def create_amenities_count(self, df):
        """Count number of amenities"""
        print("  Creating Amenities_Count feature")
        
        # If Amenities is a comma-separated string
        if df['Amenities'].dtype == 'object':
            df['Amenities_Count'] = df['Amenities'].str.split(',').str.len()
        else:
            # If already a number, keep as is
            df['Amenities_Count'] = df['Amenities']
        
        # Fill NaN with 0
        df['Amenities_Count'] = df['Amenities_Count'].fillna(0)
        
        return df
    
    def create_floor_ratio(self, df):
        """Create ratio of floor number to total floors"""
        print("  Creating Floor_Ratio feature")
        
        df['Floor_Ratio'] = df['Floor_No'] / (df['Total_Floors'] + 1)
        return df
    
    def create_future_price_5y(self, df):
        """
        CREATE REGRESSION TARGET
        Predict price after 5 years using compound growth
        Formula: Future = Current × (1 + r)^t
        """
        print("\nCreating REGRESSION Target: Future_Price_5Y")
        
        years = 5
        df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1 + self.growth_rate) ** years)
        
        print(f"  Applied {self.growth_rate*100}% annual growth for {years} years")
        print(f"  Average future price: INR{df['Future_Price_5Y'].mean():.2f} Lakhs")
        
        return df
    
    def create_good_investment_label(self, df):
        """
        CREATE CLASSIFICATION TARGET
        Determine if property is a good investment
        
        Criteria (score-based approach):
        1. Price <= city median (1 point)
        2. Price per sqft <= city median (1 point)
        3. BHK >= 2 (1 point)
        4. Age < 10 years (1 point)
        5. Available or Under Construction (1 point)
        
        Good Investment if score >= 3
        """
        print("\nCreating CLASSIFICATION Target: Good_Investment")
        
        # Calculate city-wise medians
        city_median_price = df.groupby('City')['Price_in_Lakhs'].transform('median')
        city_median_price_per_sqft = df.groupby('City')['Price_per_SqFt'].transform('median')
        
        # Initialize score
        score = 0
        
        # Criterion 1: Price below city median
        criterion1 = (df['Price_in_Lakhs'] <= city_median_price).astype(int)
        score += criterion1
        print(f"  Criterion 1 (Price <= median): {criterion1.sum()} properties")
        
        # Criterion 2: Price per sqft below median
        criterion2 = (df['Price_per_SqFt'] <= city_median_price_per_sqft).astype(int)
        score += criterion2
        print(f"  Criterion 2 (Price/sqft <= median): {criterion2.sum()} properties")
        
        # Criterion 3: BHK >= 2
        criterion3 = (df['BHK'] >= 2).astype(int)
        score += criterion3
        print(f"  Criterion 3 (BHK >= 2): {criterion3.sum()} properties")
        
        # Criterion 4: Property age < 10 years
        criterion4 = (df['Age_of_Property'] < 10).astype(int)
        score += criterion4
        print(f"  Criterion 4 (Age < 10 years): {criterion4.sum()} properties")
        
        # Criterion 5: Availability status
        if 'Availability_Status' in df.columns:
            criterion5 = df['Availability_Status'].isin(['Available', 'Under Construction']).astype(int)
            score += criterion5
            print(f"  Criterion 5 (Available/Under Construction): {criterion5.sum()} properties")
        
        # Good investment if score >= 3 out of 5
        df['Good_Investment'] = (score >= 3).astype(int)
        
        good_count = df['Good_Investment'].sum()
        total_count = len(df)
        print(f"\n  Good Investments: {good_count} / {total_count} ({good_count/total_count*100:.1f}%)")
        
        return df
    
    def engineer_all_features(self, df):
        """
        Run complete feature engineering pipeline
        """
        print("\n" + "="*60)
        print("STARTING FEATURE ENGINEERING")
        print("="*60)
        
        # Create derived features
        df = self.create_price_per_sqft(df)
        df = self.create_age_of_property(df)
        df = self.create_school_density_score(df)
        df = self.create_infrastructure_score(df)
        df = self.create_amenities_count(df)
        df = self.create_floor_ratio(df)
        
        # Create target variables
        df = self.create_future_price_5y(df)
        df = self.create_good_investment_label(df)
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETED")
        print("="*60)
        
        return df
