import pandas as pd
import numpy as np


def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # 1. USUWANIE ANOMALII (To o czym pisałeś!)
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 60]

    # 2. CZYSZCZENIE I TRANSFORMACJA
    df = df.fillna(df.mean(numeric_only=True))
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(float)

    # 3. PODZIAŁ I NAPRAWA WYCIEKU DANYCH (Target Leakage)
    # Dla klasyfikacji usuwamy obie kolumny docelowe z X
    y_clf = df['loan_status'].values.reshape(-1, 1)
    X_clf = df.drop(['loan_status', 'loan_int_rate'], axis=1).values

    # Dla regresji usuwamy obie kolumny docelowe z X
    y_reg = df['loan_int_rate'].values.reshape(-1, 1)
    X_reg = df.drop(['loan_int_rate', 'loan_status'], axis=1).values

    return X_clf, y_clf, X_reg, y_reg


def standardize_data(X_learn, X_test):
    # Obliczamy średnią i odchylenie tylko na zbiorze treningowym (X_learn)
    # zapobiega to "wyciekowi" informacji ze zbioru testowego.
    mean = np.mean(X_learn, axis=0)
    std = np.std(X_learn, axis=0)
    
    # Unikamy dzielenia przez zero, jeśli odchylenie wynosi 0
    std = np.where(std == 0, 1e-8, std)
    
    X_learn_std = (X_learn - mean) / std
    X_test_std = (X_test - mean) / std
    return X_learn_std, X_test_std

def split_and_prepare_for_network(X, y, division_coefficient):
    # Prosty podział na zbiór uczący i testowy
    split_idx = int(len(X) * division_coefficient)
    
    X_learn, X_test = X[:split_idx], X[split_idx:]
    y_learn, y_test = y[:split_idx], y[split_idx:]
    
    # Standaryzacja cech (Z-score)
    X_learn, X_test = standardize_data(X_learn, X_test)
    
    return X_learn, y_learn, X_test, y_test

if __name__ == "__main__":
    # Usuwamy dwie kropki, bo terminal jest już w głównym folderze projektu
    X_clf, y_clf, X_reg, y_reg = prepare_data("data/credit_risk_dataset.csv")
    
    print("--- TEST WCZYTYWANIA DANYCH ---")
    print(f"Liczba cech (kolumn X): {X_clf.shape[1]}")
    print("-------------------------------")