# Sentiment Analysis using RNN

streamlit :https://sentiment-analysis-imdb-rnn.streamlit.app/
## Overview
This project implements a text classification model for sentiment analysis using Recurrent Neural Networks (RNNs). The dataset used is the IMDb movie reviews dataset, with reviews labeled as either positive or negative. Multiple RNN architectures (Vanilla RNN, LSTM, Bi-directional LSTM, GRU) are explored, leveraging pre-trained GloVe embeddings.

---

## Features
- Pre-trained GloVe embeddings for word representation.
- Extensive text preprocessing pipeline.
- Implementation of multiple RNN architectures:
  - Vanilla RNN
  - LSTM
  - Bi-directional LSTM
  - GRU
- Performance evaluation with metrics like accuracy.

---

## Dataset
The IMDb movie reviews dataset is used for training and testing. Download the dataset using the Kaggle API:
```bash
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
!unzip imdb-dataset-of-50k-movie-reviews.zip
```
---

## Text Preprocessing Pipeline
- Removing special characters
- Removing non-ASCII characters
- Tokenization
- Stopword removal
- Stemming/Lemmatization

---

## Model Architectures
### Vanilla RNN
- A simple RNN with one recurrent layer.

### LSTM
- Long Short-Term Memory architecture for capturing long-term dependencies.

### Bi-Directional LSTM
- Processes sequences in both forward and backward directions.

### GRU
- Gated Recurrent Unit, a simplified version of LSTM.
