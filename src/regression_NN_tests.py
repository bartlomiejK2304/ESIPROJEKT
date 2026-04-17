from email.errors import NonASCIILocalPartDefect

from neuralNetwork import NeuralNetwork
import pandas as pd

# ==========================================
# Wartości parametrów do testowania (8 parametrów)

PARAM_GRID = {
    "division_coefficient": [ # proporcja podziału danych - treningowy/testowy
        0.6,
        0.7,
        0.8,
        0.9
    ],
    "layers": [ # warstwy
        [3, 5, 1],
        [3, 10, 1],
        [3, 8, 4, 1],
        [3, 16, 8, 1]
    ],
    "activation": [ # funkcja aktywacji
        'relu',
        'tanh',
        'sigmoid',
        'linear'
    ],
    "learning_rate": [ # współczynnik uczenia
        0.1,
        0.01,
        0.001,
        0.0001
    ],
    "multiplier": [ # mnożnik do przeskalowania wag
        0.1,
        0.01,
        0.001,
        0.0001
    ],
    "epochs": [ # liczba epok
        100,
        500,
        1000,
        2000
    ],
    "min_change": [ # minimalna zmiana błędu
        1e-2,
        1e-4,
        1e-6,
        1e-8
    ],
    "target_loss": [ # docelowa zmiana błędu
        1e-1,
        1e-2,
        1e-3,
        1e-4
    ]
}

# ==========================================
# Wczytanie i przygotowanie danych

data = None # dane do wczytania
results = []

# ==========================================
# WARTOŚCI DOMYŚLNE (punkt odniesienia)

default_params = {
    "layers": [11, 11, 1],
    "activation": "sigmoid",
    "learning_rate": 0.001,
    "multiplier": 0.01,
    "division_coefficient": 0.8,
    "epochs": 1000,
    "min_change": 1e-6,
    "target_loss": 1e-4
}

# ==========================================
# FUNKCJA TESTUJĄCA

def run_experiment(params, test_id, task="regression"):
    try:
        # -----------------------------------
        # Podział danych na zbiór uczący i testowy
        data_learn = None
        data_test = None

        if task == "regression":
            X_learn = None
            y_learn = None
            X_test = None
            y_test = None
        if task == "classification":
            X_learn = None
            y_learn = None
            X_test = None
            y_test = None


        model = NeuralNetwork(
            layers=params["layers"],
            activation=params["activation"],
            learning_rate=params["learning_rate"],
            multiplier=params["multiplier"],
            task=task
        )

        model.fit(
            X_learn,
            y_learn,
            epochs=params["epochs"],
            min_change=params["min_change"],
            target_loss=params["target_loss"]
        )

        y_pred = model.predict(X_test)
        loss = model.compute_loss(y_pred, y_learn)

        if params["task"] == "regression":
            results.append({
                **params,
                "final_MSE": loss,
            })

            print(f"Test {test_id} | MSE={loss}")

        if params["task"] == "classification":
            results.append({
                **params,
                "final_BCE": loss,
            })

            print(f"Test {test_id} | BCE={loss}")

    except Exception as e:
        print(f"[FAIL] Test {test_id}:", e)


# ==========================================
# TESTY POJEDYNCZYCH PARAMETRÓW

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

df = pd.DataFrame(results)
df = df.sort_values(by="final_loss")

print("\nTOP 10 WYNIKÓW:")
print(df.head(10))


# ==========================================
# ZAPIS DO CSV

df.to_csv("results_regression.csv", index=False)
print("\nZapisano do results_regression.csv")