SENTIMENT_ANALYSIS
Welcome! This project demonstrates a basic Sentiment Analysis pipeline built using the Sentiment140 dataset. The goal is to classify tweets as either positive or negative based on their textual content using Logistic Regression.

📌 Project Overview In this project, we:

Load and preprocess the Sentiment140 dataset

Clean the tweet text to reduce noise

Use TF-IDF for feature extraction

Train a Logistic Regression model

Evaluate model performance using accuracy, classification report, and confusion matrix

📁 Dataset Source: https://www.google.com/url?q=https://www.kaggle.com/datasets/kazanova/sentiment140&sa=D&source=editors&ust=1745057332296059&usg=AOvVaw3ketJYTpX9ArehEVFiTr7l

The dataset consists of 1.6 million tweets labeled as either 0 (negative) or 4 (positive).

We convert 4 → 1 to make it a binary classification problem.

⚙ Steps in the Pipeline Loading the Dataset Using pandas, we import the CSV data and assign appropriate column names.

Cleaning Tweets A custom function is used to remove:

URLs, mentions, hashtags

Special characters and numbers

Text is converted to lowercase

Sampling and Visualization To manage processing time, we sample 200,000 tweets. Sentiment distribution is visualized using seaborn.

Train-Test Split Data is split into 50% training and 50% testing, maintaining class balance via stratification.

TF-IDF Vectorization We use TF-IDF (including bigrams) to convert text into numerical features for training.

Model Training We train a Logistic Regression classifier using Scikit-learn.

Model Evaluation The model achieves an accuracy of ~86%. We also display:

Classification report

Confusion matrix
