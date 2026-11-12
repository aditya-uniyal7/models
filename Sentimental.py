import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv("H.csv")

df = df.dropna(subset=["Review", "actual_rating"])

x=df["Review"]

y=df["actual_rating"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

vectorizer = TfidfVectorizer(max_features=5000)
x_train_nums = vectorizer.fit_transform(x_train)
x_test_nums = vectorizer.transform(x_test)


model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train_nums, y_train)

predictions = model.predict(x_test_nums)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))



def sentiment_score(text):
    nums = vectorizer.transform([text])
    probs = model.predict_proba(nums)[0]       
    expected_stars = probs @ model.classes_    
    return round(expected_stars, 2)



import joblib
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
pipe.fit(x_train, y_train)

predictions = pipe.predict(x_test)

joblib.dump(pipe, "sentiment_model.joblib")