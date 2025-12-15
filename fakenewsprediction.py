# === 1. Import required libraries ===
# Core data manipulation & visualization libraries, NLP, ML models and evaluation metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")


# === 2. Load and prepare data ===
# Download and load spaCy's medium English model
!python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

# Load Fake and True news datasets
df_fake = pd.read_csv('/content/drive/MyDrive/DATASET/Fake.csv')
df_true = pd.read_csv('/content/drive/MyDrive/DATASET/True.csv')

# Assign labels (0 = fake, 1 = true)
df_fake['lable'] = 0
df_true['lable'] = 1

# Combine datasets and shuffle
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
df = df.drop(columns=['subject', 'date'], errors='ignore')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# === 3. Text preprocessing ===
# Merge title and text into a single column
df['full_text'] = df['title'] + " " + df['text']
df = df.drop(columns=['title', 'text'], errors='ignore')

# Function to get sentence/document vector using spaCy
def get_vector(text):
    doc = nlp(text)
    if len(doc) == 0:
        return np.zeros(nlp.vocab.vectors_length)
    return doc.vector

# Apply vectorization
df['vector'] = df['full_text'].apply(get_vector)

# Save embeddings for later use
df.to_csv("embedding.csv", index=False)


# === 4. Train-test split ===
X = df['vector']
y = df['lable']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=42)

# Ensure all vectors are of the same dimension and stack them into a matrix
vecs = [np.array(v, dtype=np.float32) for v in df["vector"].tolist()]
dim = len(vecs[0])
assert all(len(v) == dim for v in vecs), "Inconsistent vector sizes"
X = np.vstack(vecs)


# === 5. LightGBM model training and evaluation ===
import lightgbm as lgb
lgbm = lgb.LGBMClassifier(
    objective="binary",
    class_weight="balanced",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
y_proba = lgbm.predict_proba(X_test)[:, 1]

print("=== LightGBM ===")
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# === 6. MLP model training and evaluation ===
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                    alpha=1e-4, max_iter=50, random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
y_proba = mlp.predict_proba(X_test)[:, 1]

print("=== MLP ===")
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
