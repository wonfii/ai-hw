# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('energy_usage.csv')  

# Step 2: Prepare features 
X = df[['temperature', 'humidity', 'hour', 'is_weekend']]  # Input features
y = df['consumption']  # Target variable 

# Step 3: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = model.predict(X_test)

# Step 6: Plot Actual vs Predicted consumption
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Sample Index")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Mean Absolute Percentage Error (MAPE): {round(mape, 2)}%")

# --------------------------------------------------------

# Step 1: Load the dataset
df = pd.read_csv('energy_usage_plus.csv')

# Encode the categorical features using OneHotEncoder
df_encoded = pd.get_dummies(df, columns=['season', 'district_type'], drop_first=True)

# Step 3: View updated DataFrame
print(df_encoded.head())