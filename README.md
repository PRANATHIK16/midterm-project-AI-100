# IMDB Sentiment Analysis

## Project Description
This project classifies movie reviews as **Positive** or **Negative** using a deep learning model (LSTM).  
The goal is to demonstrate a binary text classification task using the IMDB Movie Review dataset.

**Problem Type:** Binary Classification  
**Dataset:** [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
**Model:** LSTM (Long Short-Term Memory) Neural Network  

---

## Repository Structure
```
IMDB_Sentiment_Analysis/
│
├── README.md
├── requirements.txt
├── code/
│ ├── preprocess.py # Prepare dataset: cleaning, tokenization, padding
│ ├── train_model.py # Build and train the LSTM model
│ └── evaluate_model.py # Evaluate model, generate accuracy/loss plots and confusion matrix
├── results/
│ ├── accuracy_plot.png
│ ├── loss_plot.png
│ └── confusion_matrix.png
└── report.pdf # PDF report with problem, model, results, lessons
```
---
```
## Requirements

- Python 3.9+  
- TensorFlow 2.x  
- numpy  
- matplotlib  

Install requirements with:

```bash
pip install -r requirements.txt
```
