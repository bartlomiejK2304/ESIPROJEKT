import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("===== PROJEKT CZ. 2: UCZENIE MASZYNOWE =====")
print("=" * 60)

# ============================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================
print("\nTrwa wczytywanie i przygotowywanie danych...")

df = pd.read_csv('../data/credit_risk_dataset.csv')

# Uzupełnienie braków i kodowanie zmiennych kategorycznych
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)

# Zmienne docelowe:
#   - klasyfikacja: loan_status (0/1 - czy pożyczka spłacona)
#   - regresja:     loan_int_rate (wysokość oprocentowania)
TARGET_CLF = 'loan_status'
TARGET_REG = 'loan_int_rate'

y_clf = df[TARGET_CLF].values.ravel()
y_reg = df[TARGET_REG].values.ravel()

X_clf = df.drop([TARGET_CLF], axis=1).values
X_reg = df.drop([TARGET_REG, TARGET_CLF], axis=1).values

# Podział 70% train / 30% test
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, stratify=y_clf, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# Skalowanie (tylko kolumny numeryczne – bez 0/1)
# wykrycie kolumn binarnych (0/1) i numerycznych
binary_cols = [i for i in range(X_train_clf.shape[1])
               if np.isin(X_train_clf[:, i], [0, 1]).all()
               ]
numeric_cols = [i for i in range(X_train_clf.shape[1])
                if i not in binary_cols]

# ===== KLASYFIKACJA =====
scaler_clf = StandardScaler()
X_train_clf_s = X_train_clf.copy()
X_test_clf_s  = X_test_clf.copy()
X_train_clf_s[:, numeric_cols] = scaler_clf.fit_transform(X_train_clf[:, numeric_cols])
X_test_clf_s[:, numeric_cols]  = scaler_clf.transform(X_test_clf[:, numeric_cols])

# ===== REGRESJA =====
binary_cols_reg = [i for i in range(X_train_reg.shape[1])
                   if np.isin(X_train_reg[:, i], [0, 1]).all()
                   ]
numeric_cols_reg = [i for i in range(X_train_reg.shape[1])
                    if i not in binary_cols_reg]

scaler_reg = StandardScaler()
X_train_reg_s = X_train_reg.copy()
X_test_reg_s  = X_test_reg.copy()
X_train_reg_s[:, numeric_cols_reg] = scaler_reg.fit_transform(X_train_reg[:, numeric_cols_reg])
X_test_reg_s[:, numeric_cols_reg]  = scaler_reg.transform(X_test_reg[:, numeric_cols_reg])

print("Dane przygotowane.")
print(f"  Rozmiar zbioru uczącego  (clf): {X_train_clf_s.shape}")
print(f"  Rozmiar zbioru testowego (clf): {X_test_clf_s.shape}")

# ============================================================
# POMOCNICZA FUNKCJA DO WYŚWIETLANIA WYNIKÓW
# ============================================================
def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_param_header(desc):
    print(f"\n--- {desc} ---")

# ============================================================
# 2. PROBLEM KLASYFIKACYJNY
# ============================================================
print_header("PROBLEM KLASYFIKACYJNY  |  Metryka: Accuracy")

# ------------------------------------------------------------------
# PARAMETR 1: KNN – liczba sąsiadów (n_neighbors)
# ------------------------------------------------------------------
print_param_header("PARAMETR 1 | KNN – liczba sąsiadów (n_neighbors)")
print(f"  {'n_neighbors':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for k in [3, 5, 7, 9]:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  k = {k:<11} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 2: KNN – metryka odległości (metric)
# ------------------------------------------------------------------
print_param_header("PARAMETR 2 | KNN – metryka odległości (metric)")
print(f"  {'metric':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
    m = KNeighborsClassifier(n_neighbors=5, metric=metric)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  {metric:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 3: SVM – parametr regularyzacji C
# ------------------------------------------------------------------
print_param_header("PARAMETR 3 | SVM – parametr regularyzacji (C)")
print(f"  {'C':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for c in [0.1, 1.0, 10.0, 100.0]:
    m = SVC(C=c, kernel='rbf', random_state=42)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  C = {c:<11} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 4: SVM – rodzaj jądra (kernel)
# ------------------------------------------------------------------
print_param_header("PARAMETR 4 | SVM – rodzaj jądra (kernel)")
print(f"  {'kernel':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    m = SVC(C=1.0, kernel=kernel, random_state=42)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  {kernel:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 5: Drzewo Decyzyjne – maksymalna głębokość (max_depth)
# ------------------------------------------------------------------
print_param_header("PARAMETR 5 | Drzewo Decyzyjne – maksymalna głębokość (max_depth)")
print(f"  {'max_depth':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for depth in [3, 5, 10, None]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=42)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    label = str(depth) if depth is not None else 'None (brak)'
    print(f"  {label:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 6: Drzewo Decyzyjne – min. liczba próbek w liściu (min_samples_leaf)
# ------------------------------------------------------------------
print_param_header("PARAMETR 6 | Drzewo Decyzyjne – min. próbki w liściu (min_samples_leaf)")
print(f"  {'min_samples_leaf':<18} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*18} {'-'*12} {'-'*12}")
for msl in [1, 5, 20, 50]:
    m = DecisionTreeClassifier(min_samples_leaf=msl, random_state=42)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  {msl:<18} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 7: Las Losowy – liczba drzew (n_estimators)
# ------------------------------------------------------------------
print_param_header("PARAMETR 7 | Las Losowy – liczba drzew (n_estimators)")
print(f"  {'n_estimators':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for n_est in [10, 50, 100, 200]:
    m = RandomForestClassifier(n_estimators=n_est, random_state=42)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  {n_est:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 8: Las Losowy – liczba cech przy podziale (max_features)
# ------------------------------------------------------------------
print_param_header("PARAMETR 8 | Las Losowy – liczba cech przy podziale (max_features)")
print(f"  {'max_features':<15} {'Train Acc':>12} {'Test Acc':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for mf in ['sqrt', 'log2', 0.3, 0.5]:
    m = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42)
    m.fit(X_train_clf_s, y_train_clf)
    tr = accuracy_score(y_train_clf, m.predict(X_train_clf_s))
    te = accuracy_score(y_test_clf,  m.predict(X_test_clf_s))
    print(f"  {str(mf):<15} {tr:>12.4f} {te:>12.4f}")


# ============================================================
# 3. PROBLEM REGRESYJNY
# ============================================================
print_header("PROBLEM REGRESYJNY  |  Metryka: MSE (błąd średniokwadratowy)")

# ------------------------------------------------------------------
# PARAMETR 1: KNN – liczba sąsiadów (n_neighbors)
# ------------------------------------------------------------------
print_param_header("PARAMETR 1 | KNN – liczba sąsiadów (n_neighbors)")
print(f"  {'n_neighbors':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for k in [3, 5, 7, 9]:
    m = KNeighborsRegressor(n_neighbors=k)
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    print(f"  k = {k:<11} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 2: KNN – sposób ważenia sąsiadów (weights)
# ------------------------------------------------------------------
print_param_header("PARAMETR 2 | KNN – ważenie sąsiadów (weights)")
print(f"  {'weights':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for weights in ['uniform', 'distance']:
    for k in [3, 7]:
        m = KNeighborsRegressor(n_neighbors=k, weights=weights)
        m.fit(X_train_reg_s, y_train_reg)
        tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
        te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
        print(f"  {weights:<10} k={k:<4} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 3: SVR – parametr regularyzacji C
# ------------------------------------------------------------------
print_param_header("PARAMETR 3 | SVR – parametr regularyzacji (C)")
print(f"  {'C':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for c in [0.1, 1.0, 10.0, 100.0]:
    m = SVR(C=c, kernel='rbf')
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    print(f"  C = {c:<11} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 4: SVR – rodzaj jądra (kernel)
# ------------------------------------------------------------------
print_param_header("PARAMETR 4 | SVR – rodzaj jądra (kernel)")
print(f"  {'kernel':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    m = SVR(C=1.0, kernel=kernel)
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    print(f"  {kernel:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 5: Drzewo Decyzyjne – maksymalna głębokość (max_depth)
# ------------------------------------------------------------------
print_param_header("PARAMETR 5 | Drzewo Decyzyjne – maksymalna głębokość (max_depth)")
print(f"  {'max_depth':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for depth in [3, 5, 10, None]:
    m = DecisionTreeRegressor(max_depth=depth, random_state=42)
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    label = str(depth) if depth is not None else 'None (brak)'
    print(f"  {label:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 6: Drzewo Decyzyjne – min. próbki w liściu (min_samples_leaf)
# ------------------------------------------------------------------
print_param_header("PARAMETR 6 | Drzewo Decyzyjne – min. próbki w liściu (min_samples_leaf)")
print(f"  {'min_samples_leaf':<18} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*18} {'-'*12} {'-'*12}")
for msl in [1, 5, 20, 50]:
    m = DecisionTreeRegressor(min_samples_leaf=msl, random_state=42)
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    print(f"  {msl:<18} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 7: Las Losowy – liczba drzew (n_estimators)
# ------------------------------------------------------------------
print_param_header("PARAMETR 7 | Las Losowy – liczba drzew (n_estimators)")
print(f"  {'n_estimators':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for n_est in [10, 50, 100, 200]:
    m = RandomForestRegressor(n_estimators=n_est, random_state=42)
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    print(f"  {n_est:<15} {tr:>12.4f} {te:>12.4f}")

# ------------------------------------------------------------------
# PARAMETR 8: Las Losowy – liczba cech przy podziale (max_features)
# ------------------------------------------------------------------
print_param_header("PARAMETR 8 | Las Losowy – liczba cech przy podziale (max_features)")
print(f"  {'max_features':<15} {'Train MSE':>12} {'Test MSE':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
for mf in ['sqrt', 'log2', 0.3, 0.5]:
    m = RandomForestRegressor(n_estimators=100, max_features=mf, random_state=42)
    m.fit(X_train_reg_s, y_train_reg)
    tr = mean_squared_error(y_train_reg, m.predict(X_train_reg_s))
    te = mean_squared_error(y_test_reg,  m.predict(X_test_reg_s))
    print(f"  {str(mf):<15} {tr:>12.4f} {te:>12.4f}")

print("\n" + "=" * 60)
print("Zakończono analizę metod uczenia maszynowego.")
print("=" * 60)