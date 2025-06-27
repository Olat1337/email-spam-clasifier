import pandas as pd
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

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

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel trained. Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

def classify_message(msg):
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    print(f"\nYour message: {msg}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")

classify_message("You've won a lamborghini")