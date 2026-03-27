import numpy as np
import pandas as pd

def custom_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Ręczna implementacja podziału na zbiór uczący i testowy.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Tworzymy tablicę indeksów i tasujemy ją
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Obliczamy punkt odcięcia
    test_samples_count = int(X.shape[0] * test_size)

    test_idx = indices[:test_samples_count]
    train_idx = indices[test_samples_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ========================================================
# 1. ŁADOWANIE I PREPROCESSING DANYCH
# ========================================================

df = pd.read_csv('credit_risk_dataset.csv')


# Preprocessing: wypełnianie pustych wartości średnią i one-hot encoding
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)

# Przygotowanie zmiennych docelowych i wejściowych
y_clf = df['loan_status'].values.reshape(-1, 1)  # Klasyfikacja
y_reg = df['loan_int_rate'].values.reshape(-1, 1)  # Regresja

X_clf = df.drop(['loan_status'], axis=1).values
X_reg = df.drop(['loan_int_rate'], axis=1).values

# Ręczna normalizacja Z-score
X_clf = (X_clf - np.mean(X_clf, axis=0)) / (np.std(X_clf, axis=0) + 1e-8)
X_reg = (X_reg - np.mean(X_reg, axis=0)) / (np.std(X_reg, axis=0) + 1e-8)

# Wykorzystanie naszej własnej funkcji do podziału zbioru
X_c_tr, X_c_te, y_c_tr, y_c_te = custom_train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
X_r_tr, X_r_te, y_r_tr, y_r_te = custom_train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)


# ========================================================
# 2. IMPLEMENTACJA SIECI NEURONOWEJ OD ZERA
# ========================================================

def relu(x): return np.maximum(0, x)


def relu_deriv(x): return (x > 0).astype(float)


def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_deriv(x): return sigmoid(x) * (1 - sigmoid(x))


def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)


class CustomNeuralNetwork:
    def __init__(self, layers, task='classification', lr=0.01):
        self.layers = layers
        self.task = task
        self.lr = lr
        self.weights = []
        self.biases = []
        # Inicjalizacja wag (He initialization)
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]))
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, X):
        self.A = [X]
        self.Z = []
        for i in range(len(self.weights)):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            # Funkcja aktywacji: ReLU na warstwach ukrytych, na wyjściu zależnie od problemu
            if i == len(self.weights) - 1:
                A = sigmoid(Z) if self.task == 'classification' else Z
            else:
                A = relu(Z)
            self.A.append(A)
        return self.A[-1]

    def backward(self, y_true):
        m = y_true.shape[0]
        y_pred = self.A[-1]

        if self.task == 'classification':
            dZ = y_pred - y_true
        else:
            dZ = 2 * (y_pred - y_true)

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.A[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            # 🔥 CLIPPING
            dW = np.clip(dW, -1, 1)
            db = np.clip(db, -1, 1)

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * relu_deriv(self.Z[i - 1])

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(y)

# ========================================================
# 3. ROZSZERZONE TESTOWANIE PARAMETRÓW (ZGODNE Z WYMAGANIAMI)
# ========================================================

def evaluate_classification(layers, lr, repeats=3):
    acc_train_list = []
    acc_test_list = []

    for _ in range(repeats):
        nn = CustomNeuralNetwork(layers, task='classification', lr=lr)
        nn.train(X_c_tr, y_c_tr, epochs=500)

        p_tr = (nn.forward(X_c_tr) > 0.5).astype(int)
        p_te = (nn.forward(X_c_te) > 0.5).astype(int)

        acc_train_list.append(np.mean(p_tr == y_c_tr))
        acc_test_list.append(np.mean(p_te == y_c_te))

    return np.mean(acc_train_list), np.mean(acc_test_list)


def evaluate_regression(layers, lr, repeats=3):
    mse_train_list = []
    mse_test_list = []

    for _ in range(repeats):
        nn = CustomNeuralNetwork(layers, task='regression', lr=lr)
        nn.train(X_r_tr, y_r_tr, epochs=500)

        mse_train_list.append(mse(y_r_tr, nn.forward(X_r_tr)))
        mse_test_list.append(mse(y_r_te, nn.forward(X_r_te)))

    return np.mean(mse_train_list), np.mean(mse_test_list)


print("\n===== KLASYFIKACJA =====")

layers_list = [
    [X_c_tr.shape[1], 4, 1],
    [X_c_tr.shape[1], 8, 1],
    [X_c_tr.shape[1], 16, 1],
    [X_c_tr.shape[1], 16, 8, 1]  # więcej warstw
]

learning_rates = [0.001, 0.01, 0.05, 0.1]

for layers in layers_list:
    for lr in learning_rates:
        acc_tr, acc_te = evaluate_classification(layers, lr)

        print(f"Layers={layers}, LR={lr} | Train={acc_tr:.3f} | Test={acc_te:.3f}")


print("\n===== REGRESJA =====")

layers_list_reg = [
    [X_r_tr.shape[1], 4, 1],
    [X_r_tr.shape[1], 8, 1],
    [X_r_tr.shape[1], 16, 1],
    [X_r_tr.shape[1], 16, 8, 1]
]

for layers in layers_list_reg:
    for lr in learning_rates:
        mse_tr, mse_te = evaluate_regression(layers, lr)

        print(f"Layers={layers}, LR={lr} | Train={mse_tr:.3f} | Test={mse_te:.3f}")


