# Import the required libraries

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import joblib

# Read the dataset

data = pd.read_csv("data/complaints.csv")

print("\nTotal number of complaints:", len(data))

print("\nExamples per category:")
print(data["category"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate descriptions:")
print(data["description"].duplicated().sum())

print(data.head())

# Separate input and output

X = data["description"]
y = data["category"]

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Convert complaint descriptions into numerical features that Logistic Regression can understand

# Create the vectorizer
vectorizer = TfidfVectorizer()

# Learn and convert training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Convert testing data
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression

# Creating Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Model training
model.fit(X_train_tfidf, y_train)

# Save the trained vectorizer and model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "complaint_classifier.pkl")

print("Model and vectorizer saved successfully!")

print("Model training completed successfully!")

# Evaluate the model

# Predict categories for testing data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Generate classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Identify exactly which complaints were classified incorrectly

'''results = pd.DataFrame({
    "Actual Category": y_test.values,
    "Predicted Category": y_pred
})

print("\nPrediction Comparison:")
print(results.to_string(index=False))'''

comparison = pd.DataFrame({
    "Description": X_test.reset_index(drop=True),
    "Actual Category": y_test.reset_index(drop=True),
    "Predicted Category": y_pred
})

comparison["Correct"] = (
    comparison["Actual Category"] == comparison["Predicted Category"]
)

print("\nIncorrect Predictions:")

print(
    comparison[comparison["Correct"] == False]
    .to_string(index=False)
)

# Make predictions

new_complaint = ["There is a large pothole near Garia station"]

new_complaint_tfidf = vectorizer.transform(new_complaint)

prediction = model.predict(new_complaint_tfidf)

print("New complaint:", new_complaint[0])
print("Predicted category:", prediction[0])

