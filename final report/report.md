# IMDB Sentiment Analysis Final Report

## Authors: Pranathi Kovvuru and Amal Parekh 

## 1.Problem Definition & Dataset Curation

The task we worked on was binary sentiment analysis on movie reviews. The goal of this problem is to predict whether a given movie review expresses a 
positive or negative opinion. In other words, the model reads a piece of text and determines the sentiment behind it. 

Sentiment analysis matters because it allows opinions to be analyzed at scale. Companies, streaming platforms, and social media platforms collect thousands (or even millions) of user reviews and comments. Manually reading and interpreting that much text is unrealistic. A reliable sentiment classifier can automatically summarize public opinion, detect trends, and support decision-making. For example, studios can quickly understand how audiences react to a new movie release.

For this project, we used the IMDB movie review dataset, which is a well-known benchmark dataset for sentiment classification tasks. Each review in the dataset is labeled as either positive or negative, making it ideal for supervised learning. One helpful feature of this dataset is that it already comes split into training and test sets, which simplified our workflow and ensured consistent evaluation.

Before training the model, we performed basic preprocessing. Since neural networks cannot directly process raw text, the reviews were first converted into numerical sequences where each word is represented by an integer index. After tokenization, we padded the sequences so that all reviews had the same length. This step is necessary because neural networks require fixed-size inputs. Overall, preprocessing helped us transform unstructured text into a structured format suitable for deep learning.


## 2. Deep Learning Model

For this task, we used an LSTM-based neural network for text classification. LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) specifically designed to handle sequential data, which makes them a strong choice for text-based tasks.

The model consisted of three main components. First, an embedding layer was used to convert each word index into a dense vector representation. Instead of treating words as isolated symbols, embeddings allow the model to learn meaningful relationships between words in a continuous vector space.

Next, the core of the model was an LSTM layer, which processes the sequence of word embeddings. Unlike simple feedforward networks, LSTMs maintain memory over previous words in the sequence. This is important because meaning in language often depends on word order. For example, “not good” has a very different sentiment than “good,” and LSTMs are capable of learning such dependencies.

Finally, the model included a dense output layer with a sigmoid activation function. Since this is a binary classification problem, sigmoid outputs a probability between 0 and 1, representing how likely the review is positive.

The model was trained for 5 epochs using the training dataset, with a validation split to monitor performance during training. Accuracy and loss were used as evaluation metrics. We chose LSTM because it is specifically designed to capture sequence information and long-term dependencies in text, making it a natural fit for sentiment analysis.



## 3. Results 
After training the model, the final performance showed a training accuracy of approximately 96% and a test accuracy of around 85–86%. These results indicate that the model learned strong patterns from the training data and performed reasonably well on unseen data.

Looking at the training graphs, accuracy steadily increased over the epochs, which suggests that the model was learning meaningful patterns in the data. At the same time, the loss decreased over epochs, indicating that prediction errors were becoming smaller. This is generally a good sign during training.

However, there was a noticeable gap between training and test accuracy. Since the training accuracy is significantly higher than the test accuracy, this suggests some level of overfitting. In other words, the model may have learned patterns that are specific to the training data and do not generalize perfectly to new reviews. Despite this, a test accuracy of around 85% is still strong performance for a relatively simple deep learning model without extensive tuning.

Overall, the results show that the model successfully learned to classify sentiment with solid accuracy. While there is room for improvement, the performance demonstrates that LSTMs are effective for text classification tasks.





## 4. Lessons & Experience Learned
This project gave us practical experience working with deep learning models for natural language processing. We learned how to preprocess raw text data into a format suitable for neural networks, including tokenization, sequence conversion, and padding. Understanding this pipeline helped us appreciate how much preparation is required before a model can even begin training.

We also gained experience training and evaluating an LSTM model, interpreting accuracy and loss curves, and diagnosing potential overfitting. One challenge we faced was managing the size of the dataset and the training time, especially when running experiments on limited computational resources. 

Additionally, debugging environment issues and file paths took more time than expected, which reminded us that machine learning workflows involve much more than just writing model code.

If we were to continue improving this project, we would experiment with hyperparameter tuning, such as adjusting the LSTM size, embedding dimension, or learning rate. Adding regularization techniques like dropout could also help reduce overfitting. It would also be interesting to compare performance with other architectures, such as CNNs for text classification or more advanced Transformer-based models.

Overall, this project strengthened our understanding of how deep learning models process text and gave us a clearer picture of real-world machine learning workflows from data preparation to evaluation. It was a valuable hands-on experience that connected theoretical concepts to practical implementation.


