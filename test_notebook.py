"""
Test script to verify notebook imports work correctly
"""

print("🧪 Testing Notebook Imports...")

try:
    # Test core libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Core libraries imported successfully")
    
    # Test plotly
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("✅ Plotly libraries imported successfully")
    
    # Test custom modules
    import sys
    sys.path.append('src')
    
    from data_preprocessing import ChurnDataPreprocessor
    from eda_utils import ChurnEDA
    from model_training import ChurnModelTrainer
    print("✅ Custom modules imported successfully")
    
    # Test basic functionality
    preprocessor = ChurnDataPreprocessor()
    eda = ChurnEDA()
    trainer = ChurnModelTrainer()
    print("✅ All classes initialized successfully")
    
    print("\n🎉 All notebook imports working correctly!")
    print("✅ Your Jupyter notebook is ready to run!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the error above and ensure all dependencies are installed.")