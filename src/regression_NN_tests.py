from neuralNetwork import NeuralNetwork
from config import PARAM_GRID
import pandas as pd

X = None # dane do wczytania
y = None # dane do wczytania
results = []

# ==========================================
# WARTOŚCI DOMYŚLNE (punkt odniesienia)
# ==========================================

default_params = {
    "layers": [3, 8, 1],
    "activation": "relu",
    "learning_rate": 0.01,
    "multiplier": 0.01,
    "division_coefficient": 1,
    "epochs": 1000,
    "min_change": 1e-6,
    "target_loss": 1e-4
}

# ==========================================
# FUNKCJA TESTUJĄCA
# ==========================================

def run_experiment(params, test_id):
    try:
        X_scaled = X / params["division_coefficient"]

        model = NeuralNetwork(
            layers=params["layers"],
            activation=params["activation"],
            learning_rate=params["learning_rate"],
            multiplier=params["multiplier"],
            task="regression"
        )

        model.fit(
            X_scaled,
            y,
            epochs=params["epochs"],
            min_change=params["min_change"],
            target_loss=params["target_loss"]
        )

        y_pred = model.predict(X_scaled)
        loss = model.compute_loss(y_pred, y)

        results.append({
            **params,
            "final_loss": loss,
            "num_layers": len(params["layers"]),
            "num_neurons": sum(params["layers"])
        })

        print(f"[OK] Test {test_id} | loss={loss}")

    except Exception as e:
        print(f"[FAIL] Test {test_id}:", e)


# ==========================================
# TESTY POJEDYNCZYCH PARAMETRÓW
# ==========================================

test_id = 0

for param_name, values in PARAM_GRID.items():
    for value in values:
        params = default_params.copy()
        params[param_name] = value

        print(f"Test {test_id}: {param_name} = {value}")

        run_experiment(params, test_id)
        test_id += 1


# ==========================================
# KILKA SENSOWNYCH KOMBINACJI
# ==========================================

custom_tests = [
    # głębsza sieć + mniejszy learning rate
    {"layers": [3, 16, 8, 1], "learning_rate": 0.001},

    # mała sieć + duży learning rate
    {"layers": [3, 5, 1], "learning_rate": 0.1},

    # tanh + średnie LR
    {"activation": "tanh", "learning_rate": 0.01},

    # sigmoid + mały LR
    {"activation": "sigmoid", "learning_rate": 0.001},
]

for custom in custom_tests:
    params = default_params.copy()
    params.update(custom)
    run_experiment(params, test_id)
    test_id += 1


# ==========================================
# TABELA WYNIKÓW
# ==========================================

df = pd.DataFrame(results)
df = df.sort_values(by="final_loss")

print("\nTOP 10 WYNIKÓW:")
print(df.head(10))


# ==========================================
# ZAPIS DO CSV
# ==========================================

df.to_csv("results_regression.csv", index=False)
print("\nZapisano do results_regression.csv")