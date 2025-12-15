# Fake News Detection (Kaggle Dataset)

This project uses the **Fake and Real News Dataset** from Kaggle by Clément Bisaillon for building a fake news detection model via NLP embeddings and machine learning.

## Dataset Source

Download the dataset from Kaggle:
- [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
The dataset contains two CSV files:
- `True.csv`: real news articles  
- `Fake.csv`: fake news articles  
Typical columns include: `title`, `text`, `subject`, `date`.

## Project Overview

- Load both CSV files and label them (`0 = Fake`, `1 = True`).
- Combine, shuffle, and preprocess the data.
- Generate embeddings from **spaCy** (`en_core_web_md`) using the concatenated `title + text`.
- Train and evaluate two models:
  - **LightGBM**
  - **MLP (Multi-Layer Perceptron)**

## Example LightGBM Performance

ROC-AUC: 0.9978
Accuracy: 98.03%
Confusion matrix:
[[3453 69]
[ 64 3149]]



## How to Run

1. Download the dataset from Kaggle using the link above.
2. Place `True.csv` and `Fake.csv` under a folder named `DATASET/` (or adjust the paths).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
