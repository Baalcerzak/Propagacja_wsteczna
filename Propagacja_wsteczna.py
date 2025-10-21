# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === 1. Wczytanie danych ===
data = np.loadtxt('dane2.txt')
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

# === 2. Podział danych ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# === 3. Parametry sieci ===
S1 = 3      # liczba neuronów w warstwie ukrytej
lr = 0.01   # współczynnik uczenia
epochs = 500
np.random.seed(0)

# === 4. Inicjalizacja wag ===
W1 = np.random.randn(S1, 1) * 0.5
B1 = np.random.randn(S1, 1) * 0.5
W2 = np.random.randn(1, S1) * 0.5
B2 = np.random.randn(1, 1) * 0.5

# === 5. Funkcje aktywacji ===
def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return np.where(x > 0, 1, 0)

# === 6. Trenowanie (tanh) ===
for epoch in range(epochs):
    Z1 = W1 @ X_train.T + B1
    A1 = tanh(Z1)
    Z2 = W2 @ A1 + B2
    A2 = Z2.T
    E = Y_train - A2
    mse = np.mean(E**2)
    dZ2 = -2 * E / len(X_train)
    dW2 = dZ2.T @ A1.T
    dB2 = np.sum(dZ2.T, axis=1, keepdims=True)
    dA1 = dZ2 @ W2
    dZ1 = dA1.T * tanh_derivative(Z1)
    dW1 = dZ1 @ X_train
    dB1 = np.sum(dZ1, axis=1, keepdims=True)
    W1 -= lr * dW1
    B1 -= lr * dB1
    W2 -= lr * dW2
    B2 -= lr * dB2
    if epoch % 50 == 0:
        print(f"[tanh] Epoka {epoch}: MSE = {mse:.5f}")

# === 7. Testowanie (tanh) ===
Z1_test = W1 @ X_test.T + B1
A1_test = tanh(Z1_test)
Z2_test = W2 @ A1_test + B2
Y_pred_tanh = Z2_test.T
mse_tanh = mean_squared_error(Y_test, Y_pred_tanh)
print("\nBłąd testowy (tanh):", mse_tanh)

# === 8. Trenowanie (ReLU) ===
# Nowe wagi (żeby zaczynać od zera)
W1 = np.random.randn(S1, 1) * 0.5
B1 = np.random.randn(S1, 1) * 0.5
W2 = np.random.randn(1, S1) * 0.5
B2 = np.random.randn(1, 1) * 0.5

for epoch in range(epochs):
    Z1 = W1 @ X_train.T + B1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + B2
    A2 = Z2.T
    E = Y_train - A2
    mse = np.mean(E**2)
    dZ2 = -2 * E / len(X_train)
    dW2 = dZ2.T @ A1.T
    dB2 = np.sum(dZ2.T, axis=1, keepdims=True)
    dA1 = dZ2 @ W2
    dZ1 = dA1.T * relu_derivative(Z1)
    dW1 = dZ1 @ X_train
    dB1 = np.sum(dZ1, axis=1, keepdims=True)
    W1 -= lr * dW1
    B1 -= lr * dB1
    W2 -= lr * dW2
    B2 -= lr * dB2
    if epoch % 50 == 0:
        print(f"[ReLU] Epoka {epoch}: MSE = {mse:.5f}")

# === 9. Testowanie (ReLU) ===
Z1_test = W1 @ X_test.T + B1
A1_test = relu(Z1_test)
Z2_test = W2 @ A1_test + B2
Y_pred_relu = Z2_test.T
mse_relu = mean_squared_error(Y_test, Y_pred_relu)
print("\nBłąd testowy (ReLU):", mse_relu)

# === 10. Porównanie ===
print("\nPorównanie wyników:")
print(f"MSE tanh = {mse_tanh:.5f}")
print(f"MSE ReLU = {mse_relu:.5f}")

# === 11. Wizualizacja ===
plt.scatter(X_test, Y_test, color='red', label='Dane rzeczywiste')
plt.plot(X_test, Y_pred_tanh, color='blue', label='tanh')
plt.plot(X_test, Y_pred_relu, color='green', label='ReLU')
plt.legend()
plt.show()


