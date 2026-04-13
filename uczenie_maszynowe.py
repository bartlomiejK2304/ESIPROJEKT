import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

# Ignorowanie ostrzeżeń z bibliotek dla przejrzystości wyników
warnings.filterwarnings('ignore')

print("=" * 50)
print("===== PROJEKT CZ. 2: UCZENIE MASZYNOWE =====")
print("=" * 50)

# ========================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ========================================================
print("Trwa wczytywanie i przygotowywanie danych...")

# Wczytanie zbioru (upewnij się, że plik csv jest w tym samym folderze co skrypt)
df = pd.read_csv('credit_risk_dataset.csv')

# Czyszczenie i transformacja (zgodnie z częścią 1)
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)

# Podział na X i y dla obu problemów
# Ravel() spłaszcza tablice, czego wymaga scikit-learn
y_clf = df['loan_status'].values.ravel()
y_reg = df['loan_int_rate'].values.ravel()

X_clf = df.drop(['loan_status'], axis=1).values
X_reg = df.drop(['loan_int_rate'], axis=1).values

# Podział na próbę uczącą i testową (70% train, 30% test)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Skalowanie danych (bardzo ważne dla metod opartych na dystansie jak KNN i SVM)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# ========================================================
# 2. DEFINICJA PARAMETRÓW DO PRZETESTOWANIA (Grupa 4 os.)
# ========================================================
params = {
    'KNN_neighbors': [3, 5, 7, 9],  # Liczba sąsiadów
    'SVM_C': [0.1, 1.0, 10.0, 100.0],  # Margines / Waga kary (C)
    'Tree_max_depth': [3, 5, 10, None],  # Maksymalna głębokość drzewa
    'Forest_n_estimators': [10, 50, 100, 200]  # Liczba drzew w lesie
}

# ========================================================
# 3. PROBLEM KLASYFIKACYJNY
# ========================================================
print("\n" + "*" * 50)
print(">>> PROBLEM KLASYFIKACYJNY (Metryka: Accuracy) <<<")
print("*" * 50)

# 1. K-Najbliższych Sąsiadów
print("\n1. K-Najbliższych Sąsiadów (Zmienny parametr: liczba sąsiadów)")
for k in params['KNN_neighbors']:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_clf_scaled, y_train_clf)

    train_acc = accuracy_score(y_train_clf, model.predict(X_train_clf_scaled))
    test_acc = accuracy_score(y_test_clf, model.predict(X_test_clf_scaled))
    print(f"   K={k:<4} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# 2. Maszyny Wektorów Nośnych (SVM)
print("\n2. Maszyny Wektorów Nośnych (Zmienny parametr: C - margines)")
for c in params['SVM_C']:
    model = SVC(C=c, random_state=42)
    model.fit(X_train_clf_scaled, y_train_clf)

    train_acc = accuracy_score(y_train_clf, model.predict(X_train_clf_scaled))
    test_acc = accuracy_score(y_test_clf, model.predict(X_test_clf_scaled))
    print(f"   C={c:<4} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# 3. Drzewa Decyzyjne
print("\n3. Drzewa Decyzyjne (Zmienny parametr: maksymalna głębokość)")
for depth in params['Tree_max_depth']:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_clf_scaled, y_train_clf)

    train_acc = accuracy_score(y_train_clf, model.predict(X_train_clf_scaled))
    test_acc = accuracy_score(y_test_clf, model.predict(X_test_clf_scaled))
    print(f"   Głębokość={str(depth):<4} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# 4. Lasy Losowe
print("\n4. Lasy Losowe (Zmienny parametr: liczba drzew)")
for n_est in params['Forest_n_estimators']:
    model = RandomForestClassifier(n_estimators=n_est, random_state=42)
    model.fit(X_train_clf_scaled, y_train_clf)

    train_acc = accuracy_score(y_train_clf, model.predict(X_train_clf_scaled))
    test_acc = accuracy_score(y_test_clf, model.predict(X_test_clf_scaled))
    print(f"   Drzewa={n_est:<4} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# ========================================================
# 4. PROBLEM REGRESYJNY
# ========================================================
print("\n\n" + "*" * 50)
print(">>> PROBLEM REGRESYJNY (Metryka: Błąd Średniokwadratowy - MSE) <<<")
print("*" * 50)

# 1. K-Najbliższych Sąsiadów
print("\n1. KNN Regressor (Zmienny parametr: liczba sąsiadów)")
for k in params['KNN_neighbors']:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_reg_scaled, y_train_reg)

    train_mse = mean_squared_error(y_train_reg, model.predict(X_train_reg_scaled))
    test_mse = mean_squared_error(y_test_reg, model.predict(X_test_reg_scaled))
    print(f"   K={k:<4} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

# 2. Maszyny Wektorów Nośnych (SVR)
print("\n2. SVR (Zmienny parametr: C - margines)")
# Uwaga: dla SVR duże C i duży zbiór danych mogą wydłużyć czas uczenia
for c in params['SVM_C']:
    model = SVR(C=c)
    model.fit(X_train_reg_scaled, y_train_reg)

    train_mse = mean_squared_error(y_train_reg, model.predict(X_train_reg_scaled))
    test_mse = mean_squared_error(y_test_reg, model.predict(X_test_reg_scaled))
    print(f"   C={c:<4} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

# 3. Drzewa Decyzyjne
print("\n3. Drzewa Decyzyjne Regresja (Zmienny parametr: maksymalna głębokość)")
for depth in params['Tree_max_depth']:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train_reg_scaled, y_train_reg)

    train_mse = mean_squared_error(y_train_reg, model.predict(X_train_reg_scaled))
    test_mse = mean_squared_error(y_test_reg, model.predict(X_test_reg_scaled))
    print(f"   Głębokość={str(depth):<4} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

# 4. Lasy Losowe
print("\n4. Lasy Losowe Regresja (Zmienny parametr: liczba drzew)")
for n_est in params['Forest_n_estimators']:
    model = RandomForestRegressor(n_estimators=n_est, random_state=42)
    model.fit(X_train_reg_scaled, y_train_reg)

    train_mse = mean_squared_error(y_train_reg, model.predict(X_train_reg_scaled))
    test_mse = mean_squared_error(y_test_reg, model.predict(X_test_reg_scaled))
    print(f"   Drzewa={n_est:<4} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

print("\nZakończono analizę metod uczenia maszynowego.")