# code/preprocess.py

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Parameters
num_words = 10000  # Top 10,000 words
max_len = 200      # Maximum sequence length

# Create a directory to save preprocessed data
os.makedirs('../data', exist_ok=True)

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Save preprocessed data as numpy files
np.save('../data/X_train.npy', X_train)
np.save('../data/y_train.npy', y_train)
np.save('../data/X_test.npy', X_test)
np.save('../data/y_test.npy', y_test)

print("Preprocessing complete. Data saved in '../data/' folder.")
