"""
Test script to verify all modules work correctly
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test if all custom modules can be imported"""
    try:
        from data_preprocessing import ChurnDataPreprocessor
        print("✅ data_preprocessing module imported successfully")
        
        from eda_utils import ChurnEDA
        print("✅ eda_utils module imported successfully")
        
        from model_training import ChurnModelTrainer
        print("✅ model_training module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of modules"""
    try:
        from data_preprocessing import ChurnDataPreprocessor
        from eda_utils import ChurnEDA
        from model_training import ChurnModelTrainer
        
        # Test preprocessor
        preprocessor = ChurnDataPreprocessor()
        print("✅ ChurnDataPreprocessor initialized successfully")
        
        # Test EDA
        eda = ChurnEDA()
        print("✅ ChurnEDA initialized successfully")
        
        # Test model trainer
        trainer = ChurnModelTrainer()
        print("✅ ChurnModelTrainer initialized successfully")
        print(f"   Available models: {len(trainer.models)}")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_data_files():
    """Test if data files exist"""
    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"
    
    if os.path.exists(train_file):
        print(f"✅ Training data file found: {train_file}")
    else:
        print(f"❌ Training data file not found: {train_file}")
        return False
    
    if os.path.exists(test_file):
        print(f"✅ Test data file found: {test_file}")
    else:
        print(f"❌ Test data file not found: {test_file}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Telecom Churn Prediction Project")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing module imports...")
    imports_ok = test_imports()
    
    # Test basic functionality
    print("\n2. Testing basic functionality...")
    functionality_ok = test_basic_functionality()
    
    # Test data files
    print("\n3. Testing data files...")
    data_ok = test_data_files()
    
    # Summary
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok and data_ok:
        print("🎉 All tests passed! Your project is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print("\n📝 Next steps:")
    print("1. Open Jupyter Notebook")
    print("2. Run: jupyter notebook")
    print("3. Open: Telecom_Customer_Churn_Prediction_Professional.ipynb")
    print("4. Execute all cells to run the complete analysis")