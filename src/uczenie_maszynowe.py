import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("===== PROJEKT CZ. 2: UCZENIE MASZYNOWE =====")
print("=" * 60)

# ============================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================
print("\nTrwa wczytywanie i przygotowywanie danych...")

df = pd.read_csv('../data/credit_risk_dataset.csv')

# Usuwanie anomalii
df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 60]

# Uzupełnienie braków i kodowanie zmiennych kategorycznych
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)

# Zmienne docelowe:
#   - klasyfikacja: loan_status   (0/1 - czy pożyczka spłacona)
#   - regresja:     loan_int_rate (wysokość oprocentowania)
TARGET_CLF = 'loan_status'
TARGET_REG = 'loan_int_rate'

y_clf = df[TARGET_CLF].values.ravel()
y_reg = df[TARGET_REG].values.ravel()

# Usuwamy OBIE kolumny docelowe z X (uniknięcie target leakage)
X = df.drop([TARGET_CLF, TARGET_REG], axis=1).values

# Podział 70% train / 30% test
C_X_train, C_X_test, C_y_train, C_y_test = train_test_split(
    X, y_clf, test_size=0.3, stratify=y_clf, random_state=42)
R_X_train, R_X_test, R_y_train, R_y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=42)

print("Dane przygotowane.")
print(f"  Rozmiar zbioru uczącego  (clf): {C_X_train.shape}")
print(f"  Rozmiar zbioru testowego (clf): {C_X_test.shape}")
print(f"  Rozmiar zbioru uczącego  (reg): {R_X_train.shape}")
print(f"  Rozmiar zbioru testowego (reg): {R_X_test.shape}")

# Katalog na wyniki
os.makedirs('Python/wyniki', exist_ok=True)


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================
def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def apply_scaler(scaler, X_train, X_test):
    """Zwraca przeskalowane kopie X_train i X_test (lub surowe, jeśli scaler=None)."""
    if scaler is None:
        return X_train.copy(), X_test.copy()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    return X_tr, X_te


# Zestaw skalerów używany dla KAŻDEJ metody (bo każda reaguje inaczej)
def get_scalers():
    return {
        'Standard': StandardScaler(),  # Średnia 0, odchylenie 1
        'MinMax': MinMaxScaler(),      # Wszystko w przedziale 0-1
        'Robust': RobustScaler(),      # Odporny na wartości odstające
        'None': None                   # Dane surowe
    }


# ============================================================
# 2. PROBLEM KLASYFIKACYJNY
# ============================================================
print_header("PROBLEM KLASYFIKACYJNY  |  Metryka: Accuracy")

# ------------------------------------------------------------
# 2.1 KNN - KLASYFIKACJA
# ------------------------------------------------------------
print("\n--- KLASYFIKACJA: k-Nearest Neighbors ---")

n_neighbors = [3, 5, 10, 20]
weights = ['uniform', 'distance']
p_metric = [1, 2, 3]
scalers = get_scalers()

wyniki_s, wyniki_k, wyniki_w, wyniki_p = [], [], [], []

# 1. Scaler (NAJPIERW - KNN opiera się na odległościach!)
for nazwa, scaler in scalers.items():
    X_tr, X_te = apply_scaler(scaler, C_X_train, C_X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_tr, C_y_train)
    wyniki_s.append({
        'scaler': nazwa, 'n_neighbors': '-', 'weights': '-', 'p': '-',
        'train_accuracy[%]': round(knn.score(X_tr, C_y_train) * 100, 2),
        'accuracy[%]': round(knn.score(X_te, C_y_test) * 100, 2)
    })

df_s = pd.DataFrame(wyniki_s)
best_scaler_name = df_s.loc[df_s['accuracy[%]'].idxmax(), 'scaler']
best_scaler = scalers[best_scaler_name]
X_tr_best, X_te_best = apply_scaler(best_scaler, C_X_train, C_X_test)

# 2. n_neighbors
for k in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr_best, C_y_train)
    wyniki_k.append({
        'scaler': best_scaler_name, 'n_neighbors': k, 'weights': '-', 'p': '-',
        'train_accuracy[%]': round(knn.score(X_tr_best, C_y_train) * 100, 2),
        'accuracy[%]': round(knn.score(X_te_best, C_y_test) * 100, 2)
    })

df_k = pd.DataFrame(wyniki_k)
best_k = int(df_k.loc[df_k['accuracy[%]'].idxmax(), 'n_neighbors'])

# 3. weights
for w in weights:
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=w)
    knn.fit(X_tr_best, C_y_train)
    wyniki_w.append({
        'scaler': best_scaler_name, 'n_neighbors': best_k, 'weights': w, 'p': '-',
        'train_accuracy[%]': round(knn.score(X_tr_best, C_y_train) * 100, 2),
        'accuracy[%]': round(knn.score(X_te_best, C_y_test) * 100, 2)
    })

df_w = pd.DataFrame(wyniki_w)
best_w = df_w.loc[df_w['accuracy[%]'].idxmax(), 'weights']

# 4. p
for p_val in p_metric:
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, p=p_val)
    knn.fit(X_tr_best, C_y_train)
    wyniki_p.append({
        'scaler': best_scaler_name, 'n_neighbors': best_k, 'weights': best_w, 'p': p_val,
        'train_accuracy[%]': round(knn.score(X_tr_best, C_y_train) * 100, 2),
        'accuracy[%]': round(knn.score(X_te_best, C_y_test) * 100, 2)
    })

knn_c_df = pd.concat([
    pd.DataFrame(wyniki_s),
    pd.DataFrame(wyniki_k),
    pd.DataFrame(wyniki_w),
    pd.DataFrame(wyniki_p)
], ignore_index=True)
knn_c_df.to_csv('Python/wyniki/KNN_Classification.csv', index=False)
print(knn_c_df.to_string(index=False))


# ------------------------------------------------------------
# 2.2 SVM - KLASYFIKACJA
# ------------------------------------------------------------
print("\n--- KLASYFIKACJA: Support Vector Machine ---")

C_values = [0.1, 1.0, 10.0, 100.0]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
scalers = get_scalers()

wyniki_s, wyniki_c, wyniki_kern = [], [], []

# 1. Scaler (SVM z RBF też jest wrażliwy na skalę)
for nazwa, scaler in scalers.items():
    X_tr, X_te = apply_scaler(scaler, C_X_train, C_X_test)
    svm = SVC(random_state=42)
    svm.fit(X_tr, C_y_train)
    wyniki_s.append({
        'scaler': nazwa, 'C': '-', 'kernel': '-',
        'train_accuracy[%]': round(svm.score(X_tr, C_y_train) * 100, 2),
        'accuracy[%]': round(svm.score(X_te, C_y_test) * 100, 2)
    })

df_s = pd.DataFrame(wyniki_s)
best_scaler_name = df_s.loc[df_s['accuracy[%]'].idxmax(), 'scaler']
best_scaler = scalers[best_scaler_name]
X_tr_best, X_te_best = apply_scaler(best_scaler, C_X_train, C_X_test)

# 2. C
for c in C_values:
    svm = SVC(C=c, random_state=42)
    svm.fit(X_tr_best, C_y_train)
    wyniki_c.append({
        'scaler': best_scaler_name, 'C': c, 'kernel': '-',
        'train_accuracy[%]': round(svm.score(X_tr_best, C_y_train) * 100, 2),
        'accuracy[%]': round(svm.score(X_te_best, C_y_test) * 100, 2)
    })

df_c = pd.DataFrame(wyniki_c)
best_c = float(df_c.loc[df_c['accuracy[%]'].idxmax(), 'C'])

# 3. kernel
for kern in kernels:
    svm = SVC(C=best_c, kernel=kern, random_state=42)
    svm.fit(X_tr_best, C_y_train)
    wyniki_kern.append({
        'scaler': best_scaler_name, 'C': best_c, 'kernel': kern,
        'train_accuracy[%]': round(svm.score(X_tr_best, C_y_train) * 100, 2),
        'accuracy[%]': round(svm.score(X_te_best, C_y_test) * 100, 2)
    })

svm_c_df = pd.concat([
    pd.DataFrame(wyniki_s),
    pd.DataFrame(wyniki_c),
    pd.DataFrame(wyniki_kern)
], ignore_index=True)
svm_c_df.to_csv('Python/wyniki/SVM_Classification.csv', index=False)
print(svm_c_df.to_string(index=False))


# ------------------------------------------------------------
# 2.3 DRZEWO DECYZYJNE - KLASYFIKACJA
# ------------------------------------------------------------
print("\n--- KLASYFIKACJA: Drzewo Decyzyjne ---")

max_depths = [3, 5, 10, None]
min_samples_leaf = [1, 5, 20, 50]
# Drzewa NIE wymagają skalowania - pomijamy krok skalera.

wyniki_d, wyniki_msl = [], []

# 1. max_depth
for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(C_X_train, C_y_train)
    wyniki_d.append({
        'max_depth': str(depth), 'min_samples_leaf': '-',
        'train_accuracy[%]': round(tree.score(C_X_train, C_y_train) * 100, 2),
        'accuracy[%]': round(tree.score(C_X_test, C_y_test) * 100, 2)
    })

df_d = pd.DataFrame(wyniki_d)
best_depth_str = df_d.loc[df_d['accuracy[%]'].idxmax(), 'max_depth']
best_depth = None if best_depth_str == 'None' else int(best_depth_str)

# 2. min_samples_leaf
for msl in min_samples_leaf:
    tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=msl, random_state=42)
    tree.fit(C_X_train, C_y_train)
    wyniki_msl.append({
        'max_depth': str(best_depth), 'min_samples_leaf': msl,
        'train_accuracy[%]': round(tree.score(C_X_train, C_y_train) * 100, 2),
        'accuracy[%]': round(tree.score(C_X_test, C_y_test) * 100, 2)
    })

tree_c_df = pd.concat([
    pd.DataFrame(wyniki_d),
    pd.DataFrame(wyniki_msl)
], ignore_index=True)
tree_c_df.to_csv('Python/wyniki/Tree_Classification.csv', index=False)
print(tree_c_df.to_string(index=False))


# ------------------------------------------------------------
# 2.4 LAS LOSOWY - KLASYFIKACJA
# ------------------------------------------------------------
print("\n--- KLASYFIKACJA: Las Losowy ---")

n_estimators = [10, 50, 100, 200]
max_depths = [3, 5, 10, None]
# Las losowy również nie wymaga skalowania.

wyniki_n, wyniki_d = [], []

# 1. n_estimators
for n_est in n_estimators:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(C_X_train, C_y_train)
    wyniki_n.append({
        'n_estimators': n_est, 'max_depth': '-',
        'train_accuracy[%]': round(rf.score(C_X_train, C_y_train) * 100, 2),
        'accuracy[%]': round(rf.score(C_X_test, C_y_test) * 100, 2)
    })

df_n = pd.DataFrame(wyniki_n)
best_n = int(df_n.loc[df_n['accuracy[%]'].idxmax(), 'n_estimators'])

# 2. max_depth
for depth in max_depths:
    rf = RandomForestClassifier(n_estimators=best_n, max_depth=depth,
                                random_state=42, n_jobs=-1)
    rf.fit(C_X_train, C_y_train)
    wyniki_d.append({
        'n_estimators': best_n, 'max_depth': str(depth),
        'train_accuracy[%]': round(rf.score(C_X_train, C_y_train) * 100, 2),
        'accuracy[%]': round(rf.score(C_X_test, C_y_test) * 100, 2)
    })

rf_c_df = pd.concat([
    pd.DataFrame(wyniki_n),
    pd.DataFrame(wyniki_d)
], ignore_index=True)
rf_c_df.to_csv('Python/wyniki/RF_Classification.csv', index=False)
print(rf_c_df.to_string(index=False))


# ============================================================
# 3. PROBLEM REGRESYJNY
# ============================================================
print_header("PROBLEM REGRESYJNY  |  Metryka: MSE (błąd średniokwadratowy)")


def mse_score(model, X, y):
    """Zwraca MSE predykcji modelu (im mniej tym lepiej)."""
    return mean_squared_error(y, model.predict(X))


# ------------------------------------------------------------
# 3.1 KNN - REGRESJA
# ------------------------------------------------------------
print("\n--- REGRESJA: k-Nearest Neighbors ---")

n_neighbors = [3, 5, 10, 20]
weights = ['uniform', 'distance']
p_metric = [1, 2, 3]
scalers = get_scalers()

wyniki_s, wyniki_k, wyniki_w, wyniki_p = [], [], [], []

# 1. Scaler
for nazwa, scaler in scalers.items():
    X_tr, X_te = apply_scaler(scaler, R_X_train, R_X_test)
    knn = KNeighborsRegressor()
    knn.fit(X_tr, R_y_train)
    wyniki_s.append({
        'scaler': nazwa, 'n_neighbors': '-', 'weights': '-', 'p': '-',
        'train_MSE': round(mse_score(knn, X_tr, R_y_train), 4),
        'test_MSE': round(mse_score(knn, X_te, R_y_test), 4)
    })

df_s = pd.DataFrame(wyniki_s)
# W regresji chcemy MINIMALIZOWAĆ MSE -> idxmin
best_scaler_name = df_s.loc[df_s['test_MSE'].idxmin(), 'scaler']
best_scaler = scalers[best_scaler_name]
X_tr_best, X_te_best = apply_scaler(best_scaler, R_X_train, R_X_test)

# 2. n_neighbors
for k in n_neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_tr_best, R_y_train)
    wyniki_k.append({
        'scaler': best_scaler_name, 'n_neighbors': k, 'weights': '-', 'p': '-',
        'train_MSE': round(mse_score(knn, X_tr_best, R_y_train), 4),
        'test_MSE': round(mse_score(knn, X_te_best, R_y_test), 4)
    })

df_k = pd.DataFrame(wyniki_k)
best_k = int(df_k.loc[df_k['test_MSE'].idxmin(), 'n_neighbors'])

# 3. weights
for w in weights:
    knn = KNeighborsRegressor(n_neighbors=best_k, weights=w)
    knn.fit(X_tr_best, R_y_train)
    wyniki_w.append({
        'scaler': best_scaler_name, 'n_neighbors': best_k, 'weights': w, 'p': '-',
        'train_MSE': round(mse_score(knn, X_tr_best, R_y_train), 4),
        'test_MSE': round(mse_score(knn, X_te_best, R_y_test), 4)
    })

df_w = pd.DataFrame(wyniki_w)
best_w = df_w.loc[df_w['test_MSE'].idxmin(), 'weights']

# 4. p
for p_val in p_metric:
    knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_w, p=p_val)
    knn.fit(X_tr_best, R_y_train)
    wyniki_p.append({
        'scaler': best_scaler_name, 'n_neighbors': best_k, 'weights': best_w, 'p': p_val,
        'train_MSE': round(mse_score(knn, X_tr_best, R_y_train), 4),
        'test_MSE': round(mse_score(knn, X_te_best, R_y_test), 4)
    })

knn_r_df = pd.concat([
    pd.DataFrame(wyniki_s),
    pd.DataFrame(wyniki_k),
    pd.DataFrame(wyniki_w),
    pd.DataFrame(wyniki_p)
], ignore_index=True)
knn_r_df.to_csv('Python/wyniki/KNN_Regression.csv', index=False)
print(knn_r_df.to_string(index=False))


# ------------------------------------------------------------
# 3.2 SVR - REGRESJA
# ------------------------------------------------------------
print("\n--- REGRESJA: Support Vector Regression ---")

C_values = [0.1, 1.0, 10.0, 100.0]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
scalers = get_scalers()

wyniki_s, wyniki_c, wyniki_kern = [], [], []

# 1. Scaler
for nazwa, scaler in scalers.items():
    X_tr, X_te = apply_scaler(scaler, R_X_train, R_X_test)
    svr = SVR()
    svr.fit(X_tr, R_y_train)
    wyniki_s.append({
        'scaler': nazwa, 'C': '-', 'kernel': '-',
        'train_MSE': round(mse_score(svr, X_tr, R_y_train), 4),
        'test_MSE': round(mse_score(svr, X_te, R_y_test), 4)
    })

df_s = pd.DataFrame(wyniki_s)
best_scaler_name = df_s.loc[df_s['test_MSE'].idxmin(), 'scaler']
best_scaler = scalers[best_scaler_name]
X_tr_best, X_te_best = apply_scaler(best_scaler, R_X_train, R_X_test)

# 2. C
for c in C_values:
    svr = SVR(C=c)
    svr.fit(X_tr_best, R_y_train)
    wyniki_c.append({
        'scaler': best_scaler_name, 'C': c, 'kernel': '-',
        'train_MSE': round(mse_score(svr, X_tr_best, R_y_train), 4),
        'test_MSE': round(mse_score(svr, X_te_best, R_y_test), 4)
    })

df_c = pd.DataFrame(wyniki_c)
best_c = float(df_c.loc[df_c['test_MSE'].idxmin(), 'C'])

# 3. kernel
for kern in kernels:
    svr = SVR(C=best_c, kernel=kern)
    svr.fit(X_tr_best, R_y_train)
    wyniki_kern.append({
        'scaler': best_scaler_name, 'C': best_c, 'kernel': kern,
        'train_MSE': round(mse_score(svr, X_tr_best, R_y_train), 4),
        'test_MSE': round(mse_score(svr, X_te_best, R_y_test), 4)
    })

svr_r_df = pd.concat([
    pd.DataFrame(wyniki_s),
    pd.DataFrame(wyniki_c),
    pd.DataFrame(wyniki_kern)
], ignore_index=True)
svr_r_df.to_csv('Python/wyniki/SVR_Regression.csv', index=False)
print(svr_r_df.to_string(index=False))


# ------------------------------------------------------------
# 3.3 DRZEWO DECYZYJNE - REGRESJA
# ------------------------------------------------------------
print("\n--- REGRESJA: Drzewo Decyzyjne ---")

max_depths = [3, 5, 10, None]
min_samples_leaf = [1, 5, 20, 50]

wyniki_d, wyniki_msl = [], []

# 1. max_depth
for depth in max_depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(R_X_train, R_y_train)
    wyniki_d.append({
        'max_depth': str(depth), 'min_samples_leaf': '-',
        'train_MSE': round(mse_score(tree, R_X_train, R_y_train), 4),
        'test_MSE': round(mse_score(tree, R_X_test, R_y_test), 4)
    })

df_d = pd.DataFrame(wyniki_d)
best_depth_str = df_d.loc[df_d['test_MSE'].idxmin(), 'max_depth']
best_depth = None if best_depth_str == 'None' else int(best_depth_str)

# 2. min_samples_leaf
for msl in min_samples_leaf:
    tree = DecisionTreeRegressor(max_depth=best_depth, min_samples_leaf=msl, random_state=42)
    tree.fit(R_X_train, R_y_train)
    wyniki_msl.append({
        'max_depth': str(best_depth), 'min_samples_leaf': msl,
        'train_MSE': round(mse_score(tree, R_X_train, R_y_train), 4),
        'test_MSE': round(mse_score(tree, R_X_test, R_y_test), 4)
    })

tree_r_df = pd.concat([
    pd.DataFrame(wyniki_d),
    pd.DataFrame(wyniki_msl)
], ignore_index=True)
tree_r_df.to_csv('Python/wyniki/Tree_Regression.csv', index=False)
print(tree_r_df.to_string(index=False))


# ------------------------------------------------------------
# 3.4 LAS LOSOWY - REGRESJA
# ------------------------------------------------------------
print("\n--- REGRESJA: Las Losowy ---")

n_estimators = [10, 50, 100, 200]
max_depths = [3, 5, 10, None]

wyniki_n, wyniki_d = [], []

# 1. n_estimators
for n_est in n_estimators:
    rf = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(R_X_train, R_y_train)
    wyniki_n.append({
        'n_estimators': n_est, 'max_depth': '-',
        'train_MSE': round(mse_score(rf, R_X_train, R_y_train), 4),
        'test_MSE': round(mse_score(rf, R_X_test, R_y_test), 4)
    })

df_n = pd.DataFrame(wyniki_n)
best_n = int(df_n.loc[df_n['test_MSE'].idxmin(), 'n_estimators'])

# 2. max_depth
for depth in max_depths:
    rf = RandomForestRegressor(n_estimators=best_n, max_depth=depth,
                               random_state=42, n_jobs=-1)
    rf.fit(R_X_train, R_y_train)
    wyniki_d.append({
        'n_estimators': best_n, 'max_depth': str(depth),
        'train_MSE': round(mse_score(rf, R_X_train, R_y_train), 4),
        'test_MSE': round(mse_score(rf, R_X_test, R_y_test), 4)
    })

rf_r_df = pd.concat([
    pd.DataFrame(wyniki_n),
    pd.DataFrame(wyniki_d)
], ignore_index=True)
rf_r_df.to_csv('Python/wyniki/RF_Regression.csv', index=False)
print(rf_r_df.to_string(index=False))


print("\n" + "=" * 60)
print("Zakończono analizę metod uczenia maszynowego.")
print("Wyniki zapisano w folderze: Python/wyniki/")
print("=" * 60)