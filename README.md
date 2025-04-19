# SENTIMENT-ANALYSIS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset with proper error handling
df = pd.read_csv('/content/orginial_data.csv.csv',
                 encoding='latin-1',
                 header=None,
                 engine='python',
                 on_bad_lines='skip')

# Assign column names
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Convert target values: 4 (positive) -> 1, 0 stays 0
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Clean the tweet text
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)  # remove URLs
    tweet = re.sub(r"@\w+", "", tweet)     # remove mentions
    tweet = re.sub(r"#\w+", "", tweet)     # remove hashtags
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)  # keep only letters
    tweet = tweet.lower().strip()  # lowercase and strip whitespace
    return tweet

df['text'] = df['text'].apply(clean_tweet)

# Sample a subset for faster training (~200k rows)
df = df.sample(n=200000, random_state=42)

# Visualize class distribution
sns.countplot(x='target', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Train-test split (50-50)
train_df, test_df = train_test_split(
    df,
    test_size=0.5,
    stratify=df['target'],
    random_state=42
)

# Features and labels
X_train = train_df['text']
y_train = train_df['target']
X_test = test_df['text']
y_test = test_df['target']

# TF-IDF vectorizer (with bigrams)
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")  # Expected: ~0.8600

# Detailed performance metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
