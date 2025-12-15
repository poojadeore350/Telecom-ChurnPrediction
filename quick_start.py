"""
Quick Start Script for Telecom Customer Churn Prediction Project

This script provides a quick way to run the complete analysis pipeline
and generate results without opening Jupyter Notebook.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from data_preprocessing import ChurnDataPreprocessor
from eda_utils import ChurnEDA
from model_training import ChurnModelTrainer
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run the complete analysis pipeline"""
    print("🚀 Starting Telecom Customer Churn Prediction Analysis")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n📊 Step 1: Data Preprocessing and Feature Engineering")
    print("-" * 40)
    
    preprocessor = ChurnDataPreprocessor()
    processed_data = preprocessor.full_preprocessing_pipeline(
        train_path="churn-bigml-80.csv",
        test_path="churn-bigml-20.csv"
    )
    
    X_train_scaled = processed_data['X_train_scaled']
    X_test_scaled = processed_data['X_test_scaled']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    # Step 2: Quick EDA
    print("\n🔍 Step 2: Exploratory Data Analysis")
    print("-" * 40)
    
    # Load combined data for EDA
    combined_data = preprocessor.load_and_combine_data(
        train_path="churn-bigml-80.csv",
        test_path="churn-bigml-20.csv"
    )
    
    eda = ChurnEDA()
    train_data = combined_data[combined_data['data_source'] == 'train'].copy()
    
    # Basic dataset overview
    eda.dataset_overview(train_data)
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Model Training and Evaluation")
    print("-" * 40)
    
    trainer = ChurnModelTrainer(random_state=42)
    
    # Train baseline models
    baseline_results = trainer.train_baseline_models(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        use_resampling=True,
        resampling_method='smote'
    )
    
    # Generate performance report
    print("\n📈 Step 4: Model Performance Report")
    print("-" * 40)
    
    performance_report = trainer.generate_model_report(baseline_results)
    
    # Save best model
    print("\n💾 Step 5: Saving Best Model")
    print("-" * 40)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    best_model_name = trainer.save_best_model(
        results=baseline_results,
        filepath="models/best_churn_model.pkl"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    
    best_metrics = baseline_results[best_model_name]['metrics']
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Performance Metrics:")
    print(f"   • AUC Score: {best_metrics['auc']:.4f}")
    print(f"   • Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   • Precision: {best_metrics['precision']:.4f}")
    print(f"   • Recall: {best_metrics['recall']:.4f}")
    print(f"   • F1-Score: {best_metrics['f1']:.4f}")
    
    print(f"\n📁 Files Generated:")
    print(f"   • Best model saved: models/best_churn_model.pkl")
    print(f"   • Performance report displayed above")
    
    print(f"\n🎯 Business Impact:")
    churn_rate = y_test.mean() * 100
    print(f"   • Current churn rate: {churn_rate:.1f}%")
    print(f"   • Model can identify {best_metrics['recall']*100:.1f}% of churners")
    print(f"   • Precision of {best_metrics['precision']*100:.1f}% reduces false alarms")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. Open Jupyter notebook for detailed analysis")
    print(f"   2. Review visualizations and insights")
    print(f"   3. Implement business recommendations")
    print(f"   4. Deploy model for production use")
    
    print(f"\n🚀 For complete analysis, run:")
    print(f"   jupyter notebook")
    print(f"   # Then open: Telecom_Customer_Churn_Prediction_Professional.ipynb")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print(f"   2. Check that data files exist: churn-bigml-80.csv and churn-bigml-20.csv")
        print(f"   3. Run test script: python test_modules.py")
        print(f"   4. If issues persist, check the documentation in README.md")