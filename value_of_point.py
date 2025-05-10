import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

def true_function(x):
    return np.sin(x) + 0.1 * x**2

def generate_training_data(seed=42, num_samples=500):
    np.random.seed(seed)
    x_train = np.random.uniform(-20, 20, num_samples).reshape(-1, 1)
    y_train = true_function(x_train) + np.random.normal(0, 0.1, size=x_train.shape)
    return x_train, y_train

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='tanh', input_shape=(1,)),
        layers.Dense(64, activation='tanh'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="mse", metrics=["mae"])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=100, batch_size=32):

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    return model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

def evaluate_model(model, x_test):
    return model.predict(x_test)

def visualize_results(x_train, y_train, x_test, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, label="Training Data", alpha=0.3)
    plt.plot(x_test, true_function(x_test), label="True Function", color="green", linestyle="dashed")
    plt.plot(x_test, y_pred, label="NN Prediction", color="red")
    plt.legend()
    plt.title("Function Approximation using Neural Network")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

def predict(model, x_new):
    return model.predict(np.array([[x_new]]))[0,0]

# Main Execution
if __name__ == "__main__":
    # Generate Data
    x_train, y_train = generate_training_data()
    x_test = np.linspace(-20, 20, 100).reshape(-1, 1)
    
    # Build and Train Model
    model = build_model()
    train_model(model, x_train, y_train, epochs=500)
    
    # Evaluate Model
    y_pred = evaluate_model(model, x_test)
    
    # Visualize Results
    visualize_results(x_train, y_train, x_test, y_pred)
    
    # Predict a new value
    x_new = 40
    y_new_pred = predict(model, x_new)
    print(f"Predicted value at x={x_new}: {y_new_pred:.4f}")