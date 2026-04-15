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