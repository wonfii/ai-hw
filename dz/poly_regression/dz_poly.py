import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# 1. Load the dataset
df = pd.read_csv('fuel_consumption_vs_speed.csv')

X = df[['speed_kmh']].values
Y = df['fuel_consumption_l_per_100km'].values

# 2. Create a model Polynomial Regression
degree = 4
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 3. Train the model
model.fit(X, Y)

# 4. Make predictions
X_test = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_pred = model.predict(X_test)

# 5. Evaluate the model
y_actual_pred = model.predict(X)
mae = mean_absolute_error(Y, y_actual_pred)
mse = mean_squared_error(Y, y_actual_pred)
r2 = r2_score(Y, y_actual_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6: Predict fuel consumption for specific speeds
speed = np.array([[35], [95], [140]]) 
predicted_fuel = model.predict(speed)

for s, p in zip(speed.flatten(), predicted_fuel):
    print(f"Prediction of fuel consumption at {s} km/h — {p:.2f} L/100km")


# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Data', color='blue')
plt.plot(X_test, y_pred, label= f'Polynomial Regression (degree {degree})', color='red')
plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel Consumption (L/100km)')
plt.title('Prediction of Fuel Consumption based on Speed')
plt.legend()
plt.grid(True)
plt.show()