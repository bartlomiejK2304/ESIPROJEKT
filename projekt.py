import numpy as np
import pandas as pd

# ========================================================
# 1. SPLIT
# ========================================================

def custom_train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    split = int(len(idx) * test_size)
    test_idx = idx[:split]
    train_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ========================================================
# 2. AKTYWACJE
# ========================================================

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
def sigmoid_deriv(x): return sigmoid(x) * (1 - sigmoid(x))

def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - np.tanh(x)**2

def linear(x): return x
def linear_deriv(x): return np.ones_like(x)


# ========================================================
# 3. SIEĆ
# ========================================================

class CustomNeuralNetwork:
    def __init__(self, layers, task='classification', lr=0.01, activation='relu'):
        self.layers = layers
        self.task = task
        self.lr = lr
        self.activation_name = activation

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
            self.biases.append(np.zeros((1, layers[i+1])))

    def activation(self, x):
        if self.activation_name == 'relu': return relu(x)
        if self.activation_name == 'tanh': return tanh(x)
        if self.activation_name == 'linear': return linear(x)
        if self.activation_name == 'sigmoid': return sigmoid(x)

    def activation_deriv(self, x):
        if self.activation_name == 'relu': return relu_deriv(x)
        if self.activation_name == 'tanh': return tanh_deriv(x)
        if self.activation_name == 'linear': return linear_deriv(x)
        if self.activation_name == 'sigmoid': return sigmoid_deriv(x)

    def forward(self, X):
        self.A = [X]
        self.Z = []

        for i in range(len(self.weights)):
            Z = self.A[-1] @ self.weights[i] + self.biases[i]
            self.Z.append(Z)

            if i == len(self.weights) - 1:
                A = sigmoid(Z) if self.task == 'classification' else Z
            else:
                A = self.activation(Z)

            self.A.append(A)

        return self.A[-1]

    def backward(self, y):
        m = y.shape[0]
        y_pred = self.A[-1]

        dZ = y_pred - y if self.task == 'classification' else 2 * (y_pred - y)

        for i in reversed(range(len(self.weights))):
            dW = (self.A[i].T @ dZ) / m
            db = np.mean(dZ, axis=0, keepdims=True)

            dW = np.clip(dW, -1, 1)
            db = np.clip(db, -1, 1)

            if i > 0:
                dZ = (dZ @ self.weights[i].T) * self.activation_deriv(self.Z[i-1])

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def train(self, X, y, epochs):
        for _ in range(epochs):
            self.forward(X)
            self.backward(y)


# ========================================================
# 4. DANE
# ========================================================

df = pd.read_csv('credit_risk_dataset.csv')

df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)   # 🔥 KLUCZOWA POPRAWKA

y_clf = df['loan_status'].values.reshape(-1, 1)
y_reg = df['loan_int_rate'].values.reshape(-1, 1)

X_clf = df.drop(['loan_status'], axis=1).values
X_reg = df.drop(['loan_int_rate'], axis=1).values


def normalize(X_tr, X_te):
    X_tr = X_tr.astype(float)
    X_te = X_te.astype(float)

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_te - mean) / std


def mse(y, yhat):
    return np.mean((y - yhat) ** 2)


# ========================================================
# 5. PARAMETRY
# ========================================================

layers_list = [
    [X_clf.shape[1], 4, 1],
    [X_clf.shape[1], 8, 1],
    [X_clf.shape[1], 16, 1],
    [X_clf.shape[1], 16, 8, 1]
]

learning_rates = [0.001, 0.01, 0.05, 0.1]
activations = ['relu', 'tanh', 'linear', 'sigmoid']
epochs_list = [100, 300, 500, 1000]
test_sizes = [0.1, 0.2, 0.3, 0.4]


# ========================================================
# 6. EWALUACJA
# ========================================================

def eval_clf(X_tr, X_te, y_tr, y_te, layers, lr, act, ep):
    tr_scores, te_scores = [], []

    for _ in range(3):
        nn = CustomNeuralNetwork(layers, 'classification', lr, act)
        nn.train(X_tr, y_tr, ep)

        p_tr = (nn.forward(X_tr) > 0.5).astype(int)
        p_te = (nn.forward(X_te) > 0.5).astype(int)

        tr_scores.append(np.mean(p_tr == y_tr))
        te_scores.append(np.mean(p_te == y_te))

    return np.mean(tr_scores), np.mean(te_scores), np.max(te_scores)


def eval_reg(X_tr, X_te, y_tr, y_te, layers, lr, act, ep):
    tr_scores, te_scores = [], []

    for _ in range(3):
        nn = CustomNeuralNetwork(layers, 'regression', lr, act)
        nn.train(X_tr, y_tr, ep)

        tr_scores.append(mse(y_tr, nn.forward(X_tr)))
        te_scores.append(mse(y_te, nn.forward(X_te)))

    return np.mean(tr_scores), np.mean(te_scores), np.min(te_scores)


# ========================================================
# 7. KLASYFIKACJA
# ========================================================

print("\n===== KLASYFIKACJA =====")

base_layers = layers_list[1]
base_lr = 0.01
base_act = 'relu'
base_ep = 300
base_ts = 0.3

print("\n--- TEST SIZE ---")
for ts in test_sizes:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_clf, y_clf, ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_clf(X_tr, X_te, y_tr, y_te, base_layers, base_lr, base_act, base_ep)
    print(f"TS={ts} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- ARCHITECTURE (LAYERS) ---")
for layers in layers_list:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_clf, y_clf, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_clf(X_tr, X_te, y_tr, y_te, layers, base_lr, base_act, base_ep)
    print(f"LAYERS={layers} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- ACTIVATION ---")
for act in activations:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_clf, y_clf, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_clf(X_tr, X_te, y_tr, y_te, base_layers, base_lr, act, base_ep)
    print(f"ACT={act} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- LEARNING RATE ---")
for lr in learning_rates:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_clf, y_clf, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_clf(X_tr, X_te, y_tr, y_te, base_layers, lr, base_act, base_ep)
    print(f"LR={lr} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- EPOCHS ---")
for ep in epochs_list:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_clf, y_clf, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_clf(X_tr, X_te, y_tr, y_te, base_layers, base_lr, base_act, ep)
    print(f"EP={ep} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")


# ========================================================
# 8. REGRESJA
# ========================================================

print("\n===== REGRESJA =====")

base_layers_reg = layers_list[1]

print("\n--- TEST SIZE ---")
for ts in test_sizes:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_reg, y_reg, ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_reg(X_tr, X_te, y_tr, y_te, base_layers_reg, base_lr, base_act, base_ep)
    print(f"TS={ts} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- ARCHITECTURE (LAYERS) ---")
for layers in layers_list:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_reg, y_reg, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_reg(X_tr, X_te, y_tr, y_te, layers, base_lr, base_act, base_ep)
    print(f"LAYERS={layers} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- ACTIVATION ---")
for act in activations:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_reg, y_reg, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_reg(X_tr, X_te, y_tr, y_te, base_layers_reg, base_lr, act, base_ep)
    print(f"ACT={act} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- LEARNING RATE ---")
for lr in learning_rates:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_reg, y_reg, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_reg(X_tr, X_te, y_tr, y_te, base_layers_reg, lr, base_act, base_ep)
    print(f"LR={lr} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")

print("\n--- EPOCHS ---")
for ep in epochs_list:
    X_tr, X_te, y_tr, y_te = custom_train_test_split(X_reg, y_reg, base_ts)
    X_tr, X_te = normalize(X_tr, X_te)

    tr, te, best = eval_reg(X_tr, X_te, y_tr, y_te, base_layers_reg, base_lr, base_act, ep)
    print(f"EP={ep} | Train={tr:.3f} | Test={te:.3f} | Best={best:.3f}")