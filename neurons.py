import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# 1. Load and scale data
def load_and_scale_data(filename):
    df = pd.read_csv(filename)
    x = df[['Area_m2']].values
    y = df[['Price_USD']].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    return x, y, x_scaled, y_scaled, scaler_x, scaler_y


# 2. Neural network model
def build_nn_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(1,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# 3. Train neural network
def train_nn(model, x, y, epochs=300):
    model.fit(x, y, epochs=epochs, verbose=0)


# 4. Predict and compare with linear regression
def compare_models(x_train, y_train, x_train_scaled, y_train_scaled, x_test, scaler_x, scaler_y):
    x_test_scaled = scaler_x.transform(x_test)

    # Neural network
    nn_model = build_nn_model()
    train_nn(nn_model, x_train_scaled, y_train_scaled)
    nn_pred_scaled = nn_model.predict(x_test_scaled)
    nn_pred = scaler_y.inverse_transform(nn_pred_scaled)

    # Linear regression
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    linreg_pred = linreg.predict(x_test)

    return nn_pred, linreg_pred


# 5. Visualization
def plot_results(x_train, y_train, x_test, nn_preds, linreg_preds):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, label="Training Data", alpha=0.8, color="green")
    plt.plot(x_test, nn_preds, label="Neural Network", color="red", alpha=0.6)
    plt.plot(x_test, linreg_preds, label="Linear Regression", color="blue", linestyle="dashed", alpha=0.6)
    plt.xlabel("Area (m²)")
    plt.ylabel("Price ($)")
    plt.title("Housing Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()


# 6. Main execution
if __name__ == "__main__":

    x_train, y_train, x_train_scaled, y_train_scaled, scaler_x, scaler_y = load_and_scale_data('./assets/house_prices_simple.csv')

    x_test = np.random.uniform(30, 150, 10).reshape(-1, 1)

    nn_preds, linreg_preds = compare_models(x_train, y_train, x_train_scaled, y_train_scaled, x_test, scaler_x, scaler_y)

    for i in range(len(x_test)):
        print(f"Area: {x_test[i][0]:.1f} m² | NN Price: ${nn_preds[i][0]:.2f} | LR Price: ${linreg_preds[i][0]:.2f}")

    x_plot = np.linspace(30, 150, 100).reshape(-1, 1)
    nn_plot, linreg_plot = compare_models(x_train, y_train, x_train_scaled, y_train_scaled, x_plot, scaler_x, scaler_y)
    
    plot_results(x_train, y_train, x_plot, nn_plot, linreg_plot)
