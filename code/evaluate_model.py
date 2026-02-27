# code/evaluate_model.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Load preprocessed test data
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

# Load trained model
model = load_model('../results/lstm_model.h5')
history = np.load('../results/history.npy', allow_pickle=True).item()

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")

# Plot accuracy
plt.figure()
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../results/accuracy_plot.png')

# Plot loss
plt.figure()
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../results/loss_plot.png')

# Confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('../results/confusion_matrix.png')

print("Evaluation complete. Plots saved in '../results/' folder.")

import matplotlib.pyplot as plt

# accuracy plot
plt.figure()
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('../results/accuracy_plot.png')  # saves plot in results
plt.close()

# loss plot
plt.figure()
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../results/loss_plot.png')
plt.close()
