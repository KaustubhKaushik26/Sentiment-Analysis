# 🎬 Sentiment Analysis on IMDb Reviews using Deep Learning (NN, CNN, LSTM)

This project implements and compares three deep learning models — a simple Neural Network, a Convolutional Neural Network (CNN), and a Recurrent Neural Network (LSTM) — for sentiment analysis on movie reviews from the IMDb dataset.

## 📌 Overview

- **Dataset**: 50,000 IMDb movie reviews (balanced positive and negative).
- **Objective**: Classify reviews as positive (1) or negative (0).
- **Approach**: Compare performance of NN, CNN, and LSTM models using the glove word embeddings and FastText word embeddings and preprocessing pipeline.

## 🗃️ Dataset

- Source: [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Format: CSV (Review text + Sentiment label)

## 🛠️ Tools & Libraries

- Python
- TensorFlow / Keras
- NLTK
- GloVe 100D Word Embeddings
- Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

## 🧹 Preprocessing

- Removed HTML tags, special characters, and stopwords
- Converted all reviews to lowercase
- Tokenized and padded sequences to max length of 100
- Transformed labels: `positive` → 1, `negative` → 0
- Used GloVe embeddings to build a custom embedding matrix of shape `(92,394, 100)`

## 📊 Models and Results

| Model Type | Architecture Summary | Test Accuracy |
|------------|----------------------|---------------|
| Simple NN  | Embedding → Flatten → Dense | ~75.0% |
| CNN        | Embedding → Conv1D → GlobalMaxPooling → Dense | ~85.8% |
| LSTM       | Embedding → LSTM → Dense | **~86.4%** ✅ |

> 📈 LSTM performed best, with strong generalization on unseen IMDb reviews.

## 📦 Output

- Trained models are saved as `.h5` files (e.g., `c1_lstm_model_acc_0.864.h5`)
- Evaluation on live unseen IMDb reviews with predicted sentiment scores (0–10 scale)
- Exported results to `c2_IMDb_Unseen_Predictions.csv`

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
