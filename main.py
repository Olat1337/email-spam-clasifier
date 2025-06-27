import pandas as pd
import urllib.request
import zipfile

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
urllib.request.urlretrieve(url, "smsspamcollection.zip")

with zipfile.ZipFile("smsspamcollection.zip", "r") as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

print(df.head())
