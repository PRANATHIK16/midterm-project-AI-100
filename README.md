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
---
## How to Run

**Download and preprocess dataset**  
```bash
python code/preprocess.py
```
**Train the LSTM model**

```
python code/train_model.py
```
**Evaluate the model and generate plots**
```
python code/evaluate_model.py
```

---

## How It Works

**Data Processing:**  
The IMDB dataset is cleaned and preprocessed to remove noise, tokenize text, and pad sequences to a uniform length.

**Model Training:**  
An LSTM (Long Short-Term Memory) neural network is trained on the processed data to learn patterns associated with positive and negative sentiments.

**Prediction:**  
The trained model predicts the sentiment of new, unseen movie reviews.

**Results Visualization:**  
Training and validation accuracy/loss plots, as well as a confusion matrix, are generated to evaluate model performance.

---
## Technologies Used

**Python:** Core programming language for data processing and model implementation.  

**TensorFlow / Keras:** Framework for building and training the LSTM deep learning model.  

**Matplotlib:** Library for plotting accuracy/loss curves and confusion matrices.  

**Jupyter Notebook (Optional):** Environment for exploratory analysis and testing model ideas.  

---



