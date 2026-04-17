import numpy as np

# ==============================================
# FUNKCJE AKTYWACJI
# x - tablica wartości wyjścia z poprzedniej warstwy

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float) # pochodna

def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def leaky_relu_derivative(x): return np.where(x > 0, 1, 0.01)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

def linear(x): return x
def linear_derivative(x): return np.ones_like(x)

# ==============================================
# Klasa Regresyjnej Sieci neuronowej

class NeuralNetwork:
    def __init__(self, layers: list[int], activation, learning_rate=0.001, multiplier=0.01, task="regression", rng_coef=10):
        self.task = task # przeznaczenie sieci
        self.layers = layers # lista liczby neuronów w każdej warstwie
        self.learning_rate = learning_rate # współczynnik uczenia

        self.weights = [] # wagi
        self.biases = [] # biasy

        # inicjalizacja wag
        for i in range(len(layers) - 1): # bo warstwa wyjściowa nie łączy się już z żadną
            rnd = np.random.default_rng(rng_coef) # dla powtarzalności wyników
            w = rnd.normal(layers[i], layers[i + 1]) * multiplier # losujemy wagi dla danej warstwy
            b = np.zeros((1, layers[i + 1])) # ustawiamy biasy

            self.weights.append(w)
            self.biases.append(b)

        # wybór funkcji aktywacji
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'leaky_relu':
            self.activation = linear
            self.activation_derivative = linear_derivative


    # -------------------------------------------------
    # Forward pass

    def forward(self, X):
        """
        Forward pass - przetwarza dane wejściowe, warstwa po warstwie,
        aby wygenerować wyjście

        :param self: aktualna instancja klasy NeuralNetwork
        :param X: tablica danych wejściowych dla sieci neuronowej
        :return: tablica wartości wyjściowych
        """
        # lista tablic wartości wyjściowych po zastosowaniu funkcji aktywacji
        self.a = [X]
        # lista tablic wartości przed przetworzeniem przez funkcję aktywacji
        self.z = []

        for i in range(len(self.weights) - 1): # liczba połączeń między danymi warstwami
            # (pomijamy połączenia z wyjściową warstwą)

            z = self.a[-1] @ self.weights[i] + self.biases[i] # bierzemy ostatnią macierz z listy
            a = self.activation(z)

            self.z.append(z)
            self.a.append(a)

        # ostatnia warstwa (REGRESJA = liniowa)
        z = self.a[-1] @ self.weights[-1] + self.biases[-1]

        if self.task == "regression":
            a = linear(z)
        elif self.task == "classification":
            a = sigmoid(z)

        self.z.append(z)
        self.a.append(a)

        return a

    # -------------------------------------------------
    # Loss MSE - średni błąd kwadratowy

    def compute_loss(self, y_pred, y_true):
        if self.task == "regression":
            # MSE - Mean Squared Error
            return np.mean((y_pred - y_true) ** 2)

        elif self.task == "classification":
            # BCE - Binary Cross Entropy
            epsilon = 1e-15  # zabezpieczenie przed log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

            return -np.mean(
                y_true * np.log(y_pred) +
                (1 - y_true) * np.log(1 - y_pred)
            )

    # -------------------------------------------------
    # Propagacja wsteczna

    def backward(self, y_pred, y_true):
        m = y_true.shape[0]

        if self.task == "regression":
            dz = (y_pred - y_true) / m # gradient błędu w ostatniej warstwie
        elif self.task == "classification":
            dz = y_pred - y_true

        for i in reversed(range(len(self.weights))):
            # gradient funkcji straty względem wag warstwy i
            dw = self.a[i].T @ dz
            # gradient względem biasów
            db = np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.activation_derivative(self.z[i - 1])

            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    # -------------------------------------------------
    # Trenowanie

    def fit(self, X, y, epochs=1000, min_change=1e-6, target_loss=1e-4):
        prev_loss = float('inf')
        metric_name = "MSE" if self.task == "regression" else "BCE"

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)

            self.backward(y_pred, y)

            # wczesny stop, jeśli zmiana straty jest minimalna
            if abs(prev_loss - loss) < min_change:
                print(f"Zmiana błędu mniejsza niż wymagana na epoce {epoch}, {metric_name}: {loss}")
                break

            # wczesny stop, jeśli strata spadła poniżej celu
            if loss < target_loss:
                print(f"Błąd mniejszy od docelowego na epoce {epoch}, {metric_name}: {loss}")
                break

            if epoch % 100 == 0:
                print(f"Epoka {epoch}, {metric_name}: {loss}")

            prev_loss = loss

    # -------------------------------------------------
    # Predykcja

    def predict(self, X):
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.activation(z)

        z = a @ self.weights[-1] + self.biases[-1]

        if self.task == "regression":
            return linear(z)
        elif self.task == "classification":
            return sigmoid(z)

    def predict_classes(self, X, threshold=0.5):
        probs = self.predict(X)
        return (probs > threshold).astype(int)
