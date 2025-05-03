import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# 1. Load the dataset
rides = pd.read_csv('./assets/rides.csv')
df = pd.DataFrame(rides)

X = df[['Hour']].values
Y = df['Duration'].values

# 2. Create a model Polynomial Regression
degree = 7
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

# 6. Special hours prediction
def time_to_decimal(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes / 60

hours = ['10:30', '00:00', '02:40']

decimal_hours = np.array([time_to_decimal(hour) for hour in hours]).reshape(-1, 1)
predicted_durations = model.predict(decimal_hours)

def minutes_to_hours_minutes(minutes):
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    return f"{hours}:{mins:02d}"


for time_str, duration in zip(hours, predicted_durations):
    print(f"Prediction of duration at {time_str} — {minutes_to_hours_minutes(duration)}")

# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Data', color='blue')
plt.plot(X_test, y_pred, label= f'Polynomial Regression (degree {degree})', color='red')
plt.xlabel('Hour')
plt.ylabel('Duration')
plt.title('Prediction of Duration based on Hour')
plt.legend()
plt.grid(True)
plt.show()