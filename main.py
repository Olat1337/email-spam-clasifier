import pandas as pd
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
urllib.request.urlretrieve(url, "smsspamcollection.zip")

with zipfile.ZipFile("smsspamcollection.zip", "r") as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

print(f"Dataset loaded. Shape: {df.shape}")
print(df.head())

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label_num'],
    test_size=0.2,
    random_state=42
)

print(f"\nData split done:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"\nVectorization complete:")
print(f"Train matrix shape: {X_train_vectorized.shape}")
print(f"Test matrix shape: {X_test_vectorized.shape}")