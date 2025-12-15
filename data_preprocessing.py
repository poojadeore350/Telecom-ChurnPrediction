"""
Data Preprocessing Module for Telecom Customer Churn Prediction

This module contains functions for data cleaning, preprocessing, and feature engineering
specifically designed for telecom customer churn analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ChurnDataPreprocessor:
    """
    A comprehensive data preprocessor for telecom churn prediction.
    
    This class handles data cleaning, feature engineering, and preprocessing
    tasks specific to telecom customer data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = 'Churn'
        
    def load_and_combine_data(self, train_path: str, test_path: str) -> pd.DataFrame:
        """
        Load and combine training and test datasets.
        
        Args:
            train_path: Path to training data CSV file
            test_path: Path to test data CSV file
            
        Returns:
            Combined DataFrame with source indicator
        """
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Add source indicator
        train_data['data_source'] = 'train'
        test_data['data_source'] = 'test'
        
        # Combine datasets
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Combined data shape: {combined_data.shape}")
        
        return combined_data
    
    def basic_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate basic information about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing basic dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        # Handle missing values (if any)
        missing_counts = df_clean.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values found:")
            print(missing_counts[missing_counts > 0])
            
            # For numerical columns, fill with median
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"Data cleaning completed:")
        print(f"- Removed {removed_duplicates} duplicate rows")
        print(f"- Final dataset shape: {df_clean.shape}")
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features based on domain knowledge.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Total usage features
        df_engineered['Total_minutes'] = (df_engineered['Total day minutes'] + 
                                        df_engineered['Total eve minutes'] + 
                                        df_engineered['Total night minutes'] + 
                                        df_engineered['Total intl minutes'])
        
        df_engineered['Total_calls'] = (df_engineered['Total day calls'] + 
                                      df_engineered['Total eve calls'] + 
                                      df_engineered['Total night calls'] + 
                                      df_engineered['Total intl calls'])
        
        df_engineered['Total_charge'] = (df_engineered['Total day charge'] + 
                                       df_engineered['Total eve charge'] + 
                                       df_engineered['Total night charge'] + 
                                       df_engineered['Total intl charge'])
        
        # Average call duration
        df_engineered['Avg_call_duration'] = df_engineered['Total_minutes'] / df_engineered['Total_calls']
        df_engineered['Avg_call_duration'] = df_engineered['Avg_call_duration'].replace([np.inf, -np.inf], 0)
        df_engineered['Avg_call_duration'] = df_engineered['Avg_call_duration'].fillna(0)
        
        # Charge per minute ratios
        df_engineered['Day_charge_per_minute'] = df_engineered['Total day charge'] / df_engineered['Total day minutes']
        df_engineered['Eve_charge_per_minute'] = df_engineered['Total eve charge'] / df_engineered['Total eve minutes']
        df_engineered['Night_charge_per_minute'] = df_engineered['Total night charge'] / df_engineered['Total night minutes']
        df_engineered['Intl_charge_per_minute'] = df_engineered['Total intl charge'] / df_engineered['Total intl minutes']
        
        # Replace inf and NaN values with 0
        charge_per_minute_cols = ['Day_charge_per_minute', 'Eve_charge_per_minute', 
                                'Night_charge_per_minute', 'Intl_charge_per_minute']
        for col in charge_per_minute_cols:
            df_engineered[col] = df_engineered[col].replace([np.inf, -np.inf], 0)
            df_engineered[col] = df_engineered[col].fillna(0)
        
        # Usage patterns
        df_engineered['Day_usage_ratio'] = df_engineered['Total day minutes'] / df_engineered['Total_minutes']
        df_engineered['Eve_usage_ratio'] = df_engineered['Total eve minutes'] / df_engineered['Total_minutes']
        df_engineered['Night_usage_ratio'] = df_engineered['Total night minutes'] / df_engineered['Total_minutes']
        df_engineered['Intl_usage_ratio'] = df_engineered['Total intl minutes'] / df_engineered['Total_minutes']
        
        # Replace inf and NaN values with 0
        usage_ratio_cols = ['Day_usage_ratio', 'Eve_usage_ratio', 'Night_usage_ratio', 'Intl_usage_ratio']
        for col in usage_ratio_cols:
            df_engineered[col] = df_engineered[col].replace([np.inf, -np.inf], 0)
            df_engineered[col] = df_engineered[col].fillna(0)
        
        # High usage indicators
        df_engineered['High_day_usage'] = (df_engineered['Total day minutes'] > df_engineered['Total day minutes'].quantile(0.75)).astype(int)
        df_engineered['High_eve_usage'] = (df_engineered['Total eve minutes'] > df_engineered['Total eve minutes'].quantile(0.75)).astype(int)
        df_engineered['High_intl_usage'] = (df_engineered['Total intl minutes'] > df_engineered['Total intl minutes'].quantile(0.75)).astype(int)
        
        # Customer service interaction indicators
        df_engineered['High_service_calls'] = (df_engineered['Customer service calls'] >= 4).astype(int)
        df_engineered['No_service_calls'] = (df_engineered['Customer service calls'] == 0).astype(int)
        
        # Plan combination features
        df_engineered['Has_both_plans'] = ((df_engineered['International plan'] == 'Yes') & 
                                         (df_engineered['Voice mail plan'] == 'Yes')).astype(int)
        df_engineered['Has_no_plans'] = ((df_engineered['International plan'] == 'No') & 
                                       (df_engineered['Voice mail plan'] == 'No')).astype(int)
        
        # Account tenure categories
        df_engineered['Account_tenure_category'] = pd.cut(df_engineered['Account length'], 
                                                        bins=[0, 50, 100, 150, 250], 
                                                        labels=['Short', 'Medium', 'Long', 'Very_Long'])
        
        print(f"Feature engineering completed. New shape: {df_engineered.shape}")
        print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
        
        return df_engineered
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Convert boolean target to binary
        if 'Churn' in df_encoded.columns:
            df_encoded['Churn'] = df_encoded['Churn'].map({True: 1, False: 0})
        
        # Binary categorical features (Yes/No)
        binary_features = ['International plan', 'Voice mail plan']
        for feature in binary_features:
            if feature in df_encoded.columns:
                df_encoded[feature] = df_encoded[feature].map({'Yes': 1, 'No': 0})
        
        # One-hot encode State (high cardinality categorical)
        if 'State' in df_encoded.columns:
            state_dummies = pd.get_dummies(df_encoded['State'], prefix='State')
            df_encoded = pd.concat([df_encoded, state_dummies], axis=1)
            df_encoded.drop('State', axis=1, inplace=True)
        
        # Handle Account_tenure_category if it exists
        if 'Account_tenure_category' in df_encoded.columns:
            tenure_dummies = pd.get_dummies(df_encoded['Account_tenure_category'], prefix='Tenure')
            df_encoded = pd.concat([df_encoded, tenure_dummies], axis=1)
            df_encoded.drop('Account_tenure_category', axis=1, inplace=True)
        
        print(f"Categorical encoding completed. Final shape: {df_encoded.shape}")
        
        return df_encoded
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Columns to exclude from features
        exclude_cols = ['Churn', 'data_source']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['Churn'].copy() if 'Churn' in df.columns else None
        
        self.feature_names = feature_cols
        
        print(f"Features shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
            print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled training and test features
        """
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print(f"Feature scaling completed for {len(numerical_cols)} numerical features")
        
        return X_train_scaled, X_test_scaled
    
    def full_preprocessing_pipeline(self, train_path: str, test_path: str) -> Dict[str, Any]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Dictionary containing processed data and metadata
        """
        print("Starting full preprocessing pipeline...")
        print("=" * 50)
        
        # Step 1: Load and combine data
        print("\n1. Loading and combining data...")
        combined_df = self.load_and_combine_data(train_path, test_path)
        
        # Step 2: Basic data info
        print("\n2. Analyzing basic data information...")
        data_info = self.basic_data_info(combined_df)
        
        # Step 3: Clean data
        print("\n3. Cleaning data...")
        clean_df = self.clean_data(combined_df)
        
        # Step 4: Feature engineering
        print("\n4. Engineering features...")
        engineered_df = self.engineer_features(clean_df)
        
        # Step 5: Encode categorical features
        print("\n5. Encoding categorical features...")
        encoded_df = self.encode_categorical_features(engineered_df)
        
        # Step 6: Separate train and test data
        print("\n6. Separating train and test data...")
        train_df = encoded_df[encoded_df['data_source'] == 'train'].copy()
        test_df = encoded_df[encoded_df['data_source'] == 'test'].copy()
        
        # Step 7: Prepare features and target
        print("\n7. Preparing features and target...")
        X_train, y_train = self.prepare_features_target(train_df)
        X_test, y_test = self.prepare_features_target(test_df)
        
        # Step 8: Scale features
        print("\n8. Scaling features...")
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n" + "=" * 50)
        print("Preprocessing pipeline completed successfully!")
        
        return {
            'X_train': X_train,
            'X_train_scaled': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'data_info': data_info,
            'preprocessor': self
        }