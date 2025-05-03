import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ========== task 1 ==========

# 1. Load the dataset
students = pd.read_csv('./assets/internship_candidates_final_numeric.csv')
df = pd.DataFrame(students)

X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
Y = df['Accepted']

# 2. Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Create a model Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X_test['EnglishLevel'], X_test['EntryTestScore'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')
plt.colorbar(label='Predicted Class (Accepted)')
plt.grid(True)
plt.show()

# ========== task 2 ==========

# 1. Load the dataset
students = pd.read_csv('./assets/internship_candidates_cefr_final.csv')
df = pd.DataFrame(students)

english_level_map = {
    'Elementary': 1,
    'Pre-Intermediate': 2,
    'Intermediate': 3,
    'Upper-Intermediate': 4,
    'Advanced': 5
}

df['EnglishLevel'] = df['EnglishLevel'].map(english_level_map)

X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
Y = df['Accepted']

# 2. Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Create a model Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X_test['EnglishLevel'], X_test['EntryTestScore'], c=y_pred, cmap='viridis', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')
plt.colorbar(label='Predicted Class (Accepted)')
plt.grid(True)
plt.show()