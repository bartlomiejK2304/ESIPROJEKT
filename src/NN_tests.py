from neuralNetwork import NeuralNetwork
from data_preparation import prepare_data, split_and_prepare_for_network
import pandas as pd
import numpy as np
import copy
import time

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

results = {}

for task in ["regression", "classification"]:
    print(f"============================================\n"
          f"TEST: {task} NN\n"
          f"============================================")
    errors_results = []
    times_results = []

    test_id = 0
    for param_name, values in PARAM_GRID.items():
        for value in values:
            print(f"Test id: {test_id}, param: {param_name}, value: {value}")

            params = copy.deepcopy(default_params)
            params[param_name] = value

            est_err = 0
            est_time = 0
            count = 0


            for rng_coef in [10, 20, 30, 40, 50]:
                start_time = time.perf_counter()  # START
                loss = run_experiment(params, test_id, param_name=param_name, value=value, task=task, rng_coef=rng_coef)
                end_time = time.perf_counter()  # STOP

                if loss is not None:
                    est_err += loss
                    est_time += end_time - start_time
                    count += 1

            if count > 0:
                est_err /= count
                est_time /= count
            else:
                est_err = np.nan
                est_time = np.nan

            if task == "regression":
                errors_results.append({
                    "param": param_name,
                    "value": value,
                    "final_MSE": est_err,
                })
            elif task == "classification":
                errors_results.append({
                    "param": param_name,
                    "value": value,
                    "final_BCE": est_err,
                })

            times_results.append({
                "param": param_name,
                "value": value,
                "time_sec": est_time
            })

            test_id += 1

    # ==========================================
    # TABELA WYNIKÓW

    df_errors = pd.DataFrame(errors_results)
    df_times = pd.DataFrame(times_results)

    metric = "final_MSE" if task == "regression" else "final_BCE"
    df_errors = df_errors.sort_values(by=["param", metric])
    df_times = df_times.sort_values(by=["param", "time_sec"])


    def format_param_groups(df, metr):
        groups = []

        df = df.sort_values(by=["param", metr])

        for param, group in df.groupby("param"):
            group = group.copy()
            group = group.sort_values(by=metr)

            group["value"] = group["value"].astype(str)

            group = group.rename(columns={
                "value": f"{param}_value",
                metr: f"{param}_{metr}"
            })

            groups.append(group[[f"{param}_value", f"{param}_{metr}"]].reset_index(drop=True))

        max_len = max(len(g) for g in groups)
        empty_col = pd.DataFrame({"": [""] * max_len})

        final_with_space = []

        for g in groups:
            final_with_space.append(g)
            final_with_space.append(empty_col)

        return pd.concat(final_with_space, axis=1)


    df_errors_formatted = format_param_groups(df_errors, metric)
    df_times_formatted = format_param_groups(df_times, "time_sec")

    if task == "regression":
        results["regression"] = []
        results["regression"].append(df_errors_formatted)
        results["regression"].append(df_times_formatted)
    elif task == "classification":
        results["classification"] = []
        results["classification"].append(df_errors_formatted)
        results["classification"].append(df_times_formatted)

# ==========================================
# ZAPIS

with pd.ExcelWriter(f"results.xlsx") as writer:

    for task in ["regression", "classification"]:
        results[task][0].to_excel(writer, sheet_name=task, index=False, startrow=0)

        startrow_times = len(results[task][0]) + 3

        results[task][1].to_excel(writer, sheet_name=task, index=False, startrow=startrow_times)