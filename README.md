# Models Directory

This directory contains trained machine learning models for the telecom churn prediction project.

## Model Files

- `best_churn_model.pkl` - The best performing model saved after hyperparameter optimization
- Model metadata and performance metrics are stored alongside the model files

## Model Information

### Best Model Performance
- **Model Type**: [Will be determined after training]
- **AUC Score**: [Will be updated after training]
- **Precision**: [Will be updated after training]
- **Recall**: [Will be updated after training]
- **F1-Score**: [Will be updated after training]

### Usage

```python
import joblib

# Load the best model
model = joblib.load('models/best_churn_model.pkl')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

## Model Versioning

Models are versioned based on:
- Training date
- Performance metrics
- Feature set used
- Hyperparameters

## Deployment Notes

- Models are serialized using joblib for compatibility
- Ensure the same preprocessing pipeline is used for new predictions
- Monitor model performance over time for drift detection