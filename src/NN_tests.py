from neuralNetwork import NeuralNetwork
from data_preparation import prepare_data, split_and_prepare_for_network
import pandas as pd
import numpy as np
import copy

# ==========================================
# Wartości parametrów do testowania (8 parametrów)

PARAM_GRID = {
    "division_coefficient": [ # proporcja podziału danych - treningowy/testowy
        0.6,
        0.7,
        0.8,
        0.9
    ],
    "layers": [ # warstwy          #zmienione z 11 na 21
        [21, 5, 1],
        [21, 24, 1],
        [21, 21, 1],
        [21, 21, 4, 1]
    ],
    "activation": [ # funkcja aktywacji
        'relu',
        'tanh',
        'sigmoid',
        'leaky_relu'
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

# Wczytujemy dane i od razu dzielimy je na te do klasyfikacji i regresji.
# Robimy to tutaj, żeby nie wczytywać pliku CSV setki razy w pętli.
X_clf_all, y_clf_all, X_reg_all, y_reg_all = prepare_data("../data/credit_risk_dataset.csv")
# ==========================================

# ==========================================
# WARTOŚCI DOMYŚLNE (dla zachowania zasady ceteris paribus)

default_params = {
    "layers": [21, 11, 1],       # Zmienione z [11, 11, 1]
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

def run_experiment(params, test_id, param_name, value, task, rng_coef):
    try:
        # -----------------------------------
        # Podział danych na zbiór uczący i testowy ze standaryzacją

        if task == "regression":
            X_learn, y_learn, X_test, y_test = split_and_prepare_for_network(
                X_reg_all, y_reg_all, params["division_coefficient"]
            )


        elif task == "classification":
            X_learn,  y_learn, X_test, y_test = split_and_prepare_for_network(
                X_clf_all, y_clf_all, params["division_coefficient"]
            )
        else:
            raise ValueError(f"Unknown task: {task}")


        model = NeuralNetwork(
            layers=params["layers"],
            activation=params["activation"],
            learning_rate=params["learning_rate"],
            multiplier=params["multiplier"],
            task=task,
            rng_coef=rng_coef,
        )

        model.fit(
            X_learn,
            y_learn,
            epochs=params["epochs"],
            min_change=params["min_change"],
            target_loss=params["target_loss"]
        )

        y_pred = model.predict(X_test)
        loss = model.compute_loss(y_pred, y_test)


    except Exception as e:
        print(f"[FAIL] Test {test_id}:", e)
        return None

    return loss

# ==========================================
# TESTY POJEDYNCZYCH PARAMETRÓW


for task in ["regression", "classification"]:
    print(f"============================================\n"
          f"TEST: {task} NN\n"
          f"============================================")
    results = []

    test_id = 0
    for param_name, values in PARAM_GRID.items():
        for value in values:
            print(f"Test id: {test_id}, param: {param_name}, value: {value}")

            params = copy.deepcopy(default_params)
            params[param_name] = value

            est_err = 0
            count = 0
            for rng_coef in [10, 20, 30, 40, 50]:
                loss = run_experiment(params, test_id, param_name=param_name, value=value, task=task, rng_coef=rng_coef)

                if loss is not None:
                    est_err += loss
                    count += 1

            if count > 0:
                est_err /= count
            else:
                est_err = np.nan

            if task == "regression":
                results.append({
                    "param": param_name,
                    "value": value,
                    "final_MSE": est_err,
                })
            elif task == "classification":
                results.append({
                    "param": param_name,
                    "value": value,
                    "final_BCE": est_err,
                })

            test_id += 1

    # ==========================================
    # TABELA WYNIKÓW

    df = pd.DataFrame(results)
    metric = "final_MSE" if task == "regression" else "final_BCE"
    df = df.sort_values(by=["param", metric])

    groups = []

    for param, group in df.groupby("param"):
        group = group.copy()
        group = group.sort_values(by=metric)

        group["value"] = group["value"].astype(str)

        group = group.rename(columns={
            "value": f"{param}_value",
            metric: f"{param}_{metric}"
        })

        groups.append(group[[f"{param}_value", f"{param}_{metric}"]].reset_index(drop=True))

    max_len = max(len(g) for g in groups)
    empty_col = pd.DataFrame({"": [""] * max_len})

    final_with_space = []

    for g in groups:
        final_with_space.append(g)
        final_with_space.append(empty_col)

    final_df = pd.concat(final_with_space, axis=1).fillna("")

    # ==========================================
    # ZAPIS DO CSV

    final_df.to_excel(f"results_{task}.xlsx")
    print(f"\nZapisano do results_{task}.xlsx")