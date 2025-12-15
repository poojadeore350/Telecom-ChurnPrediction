"""
Model Training and Evaluation Module for Telecom Customer Churn Prediction

This module contains advanced machine learning models, hyperparameter optimization,
and comprehensive evaluation metrics for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Any
import joblib
import time


class ChurnModelTrainer:
    """
    Advanced model training class for telecom churn prediction.
    
    This class provides comprehensive model training, hyperparameter optimization,
    and evaluation capabilities with support for imbalanced data handling.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all machine learning models with default parameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'AdaBoost': AdaBoostClassifier(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Neural Network': MLPClassifier(random_state=self.random_state, max_iter=500)
        }
    
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced dataset using various techniques.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Resampling method ('smote', 'smoteenn', 'none')
            
        Returns:
            Resampled feature matrix and target vector
        """
        print(f"Original dataset shape: {X.shape}")
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        print(f"Resampled dataset shape: {X_resampled.shape}")
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_baseline_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            y_train: pd.Series, y_test: pd.Series, 
                            use_resampling: bool = True, resampling_method: str = 'smote') -> Dict[str, Any]:
        """
        Train all baseline models and evaluate their performance.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            use_resampling: Whether to use resampling for imbalanced data
            resampling_method: Type of resampling method
            
        Returns:
            Dictionary containing model results
        """
        print("🚀 Training Baseline Models...")
        print("=" * 60)
        
        # Handle imbalanced data if requested
        if use_resampling:
            X_train_resampled, y_train_resampled = self.handle_imbalanced_data(
                X_train, y_train, resampling_method
            )
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train_resampled, y_train_resampled)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                metrics['training_time'] = time.time() - start_time
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, 
                                          cv=5, scoring='roc_auc')
                metrics['cv_auc_mean'] = cv_scores.mean()
                metrics['cv_auc_std'] = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ {name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                model_names: List[str] = None, 
                                optimization_method: str = 'random') -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for selected models.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_names: List of model names to optimize (None for top 3)
            optimization_method: 'grid' or 'random'
            
        Returns:
            Dictionary containing optimized models
        """
        print("🔧 Hyperparameter Optimization...")
        print("=" * 60)
        
        # Select models to optimize
        if model_names is None:
            # Select top 3 models based on AUC score
            if not self.results:
                print("❌ No baseline results found. Run train_baseline_models first.")
                return {}
            
            sorted_models = sorted(self.results.items(), 
                                 key=lambda x: x[1]['metrics']['auc'], reverse=True)
            model_names = [name for name, _ in sorted_models[:3]]
        
        # Define hyperparameter grids
        param_grids = self._get_hyperparameter_grids()
        
        optimized_results = {}
        
        for model_name in model_names:
            if model_name not in param_grids:
                print(f"⚠️ No hyperparameter grid defined for {model_name}")
                continue
            
            print(f"\n🎯 Optimizing {model_name}...")
            
            try:
                # Get base model
                base_model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                # Choose optimization method
                if optimization_method == 'grid':
                    search = GridSearchCV(
                        base_model, param_grid, cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=0
                    )
                else:
                    search = RandomizedSearchCV(
                        base_model, param_grid, cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=0, n_iter=20, random_state=self.random_state
                    )
                
                # Fit the search
                start_time = time.time()
                search.fit(X_train, y_train)
                optimization_time = time.time() - start_time
                
                optimized_results[model_name] = {
                    'best_model': search.best_estimator_,
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'optimization_time': optimization_time
                }
                
                print(f"   ✅ Best AUC: {search.best_score_:.4f}")
                print(f"   ⏱️ Time: {optimization_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Optimization failed: {str(e)}")
                continue
        
        return optimized_results
    
    def evaluate_optimized_models(self, optimized_models: Dict[str, Any], 
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate optimized models on test data.
        
        Args:
            optimized_models: Dictionary of optimized models
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation results
        """
        print("📈 Evaluating Optimized Models...")
        print("=" * 60)
        
        evaluation_results = {}
        
        for model_name, model_info in optimized_models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            model = model_info['best_model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            evaluation_results[model_name] = {
                'model': model,
                'best_params': model_info['best_params'],
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"   📊 AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Define hyperparameter grids for optimization.
        
        Returns:
            Dictionary of hyperparameter grids
        """
        return {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
    
    def plot_model_comparison(self, results: Dict[str, Any]) -> None:
        """
        Create comprehensive model comparison visualizations.
        
        Args:
            results: Dictionary containing model results
        """
        if not results:
            print("No results to plot")
            return
        
        # Extract metrics
        model_names = list(results.keys())
        metrics_data = {
            'Model': model_names,
            'AUC': [results[name]['metrics']['auc'] for name in model_names],
            'F1': [results[name]['metrics']['f1'] for name in model_names],
            'Accuracy': [results[name]['metrics']['accuracy'] for name in model_names],
            'Precision': [results[name]['metrics']['precision'] for name in model_names],
            'Recall': [results[name]['metrics']['recall'] for name in model_names]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. AUC Comparison
        axes[0, 0].barh(metrics_df['Model'], metrics_df['AUC'], color='skyblue')
        axes[0, 0].set_title('Model Comparison - AUC Score', fontweight='bold')
        axes[0, 0].set_xlabel('AUC Score')
        for i, v in enumerate(metrics_df['AUC']):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 2. F1 Score Comparison
        axes[0, 1].barh(metrics_df['Model'], metrics_df['F1'], color='lightcoral')
        axes[0, 1].set_title('Model Comparison - F1 Score', fontweight='bold')
        axes[0, 1].set_xlabel('F1 Score')
        for i, v in enumerate(metrics_df['F1']):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 3. Precision vs Recall
        axes[1, 0].scatter(metrics_df['Recall'], metrics_df['Precision'], 
                          s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        axes[1, 0].set_title('Precision vs Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        for i, model in enumerate(model_names):
            axes[1, 0].annotate(model, (metrics_df['Recall'][i], metrics_df['Precision'][i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Overall Performance Radar Chart (using matplotlib)
        metrics_normalized = metrics_df[['AUC', 'F1', 'Accuracy', 'Precision', 'Recall']].values
        
        # Find best model
        best_model_idx = np.argmax(metrics_df['AUC'])
        best_model_name = metrics_df['Model'][best_model_idx]
        
        axes[1, 1].plot(range(5), metrics_normalized[best_model_idx], 'o-', linewidth=2, label=best_model_name)
        axes[1, 1].fill(range(5), metrics_normalized[best_model_idx], alpha=0.25)
        axes[1, 1].set_xticks(range(5))
        axes[1, 1].set_xticklabels(['AUC', 'F1', 'Accuracy', 'Precision', 'Recall'])
        axes[1, 1].set_title(f'Best Model Performance - {best_model_name}', fontweight='bold')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n🏆 Model Performance Summary:")
        print("-" * 60)
        sorted_models = metrics_df.sort_values('AUC', ascending=False)
        for i, (_, row) in enumerate(sorted_models.iterrows()):
            print(f"{i+1:2d}. {row['Model']:20} | AUC: {row['AUC']:.4f} | F1: {row['F1']:.4f}")
    
    def plot_roc_curves(self, results: Dict[str, Any], y_test: pd.Series) -> None:
        """
        Plot ROC curves for all models.
        
        Args:
            results: Dictionary containing model results
            y_test: True test labels
        """
        plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                auc_score = result['metrics']['auc']
                plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_confusion_matrices(self, results: Dict[str, Any], y_test: pd.Series) -> None:
        """
        Plot confusion matrices for all models.
        
        Args:
            results: Dictionary containing model results
            y_test: True test labels
        """
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i, (name, result) in enumerate(results.items()):
            if i < len(axes):
                cm = confusion_matrix(y_test, result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{name}\nAccuracy: {result["metrics"]["accuracy"]:.3f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def save_best_model(self, results: Dict[str, Any], filepath: str) -> str:
        """
        Save the best performing model.
        
        Args:
            results: Dictionary containing model results
            filepath: Path to save the model
            
        Returns:
            Name of the best model
        """
        if not results:
            print("No results to save")
            return None
        
        # Find best model based on AUC score
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['auc'])
        best_model = results[best_model_name]['model']
        
        # Save model
        joblib.dump(best_model, filepath)
        
        print(f"✅ Best model ({best_model_name}) saved to {filepath}")
        print(f"   AUC Score: {results[best_model_name]['metrics']['auc']:.4f}")
        
        self.best_model = best_model
        return best_model_name
    
    def generate_model_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate a comprehensive model performance report.
        
        Args:
            results: Dictionary containing model results
            
        Returns:
            DataFrame with model performance metrics
        """
        if not results:
            print("No results to generate report")
            return pd.DataFrame()
        
        report_data = []
        
        for name, result in results.items():
            metrics = result['metrics']
            report_data.append({
                'Model': name,
                'AUC': metrics['auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'Training_Time': metrics.get('training_time', 'N/A')
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('AUC', ascending=False).reset_index(drop=True)
        report_df['Rank'] = range(1, len(report_df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Model', 'AUC', 'F1_Score', 'Accuracy', 'Precision', 'Recall', 'Training_Time']
        report_df = report_df[cols]
        
        print("📊 Model Performance Report:")
        print("=" * 80)
        print(report_df.to_string(index=False, float_format='%.4f'))
        
        return report_df