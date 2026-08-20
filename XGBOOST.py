"""
Treinamento e Inferência de Modelo de Predição de Atraso de Voo
Modelo: XGBoost Classifier
Etapas: Preparação dos dados, treinamento, persistência, inferência online e em lote.
"""

# Importações
import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

# Ferramentas do Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump, load
import numpy as np


def main():
    # 1. Preparação e Carregamento da Base
    print(" 1. Carregando e Preparando os Dados...")

    file_name = 'flights_delays_120.csv'
    df = pd.read_csv(file_name)

    print("\nDataFrame Head:")
    print(df.head())

    # Separação de Features (X) e Target (y)
    X = df.drop('delayed', axis=1)
    y = df['delayed']

    # Definição de colunas categóricas para One-Hot Encoding
    categorical_features = ['airline', 'origin', 'destination', 'weather']

    # Pipeline de Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Divisão dos dados em Treinamento e Teste
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Aplicação do pré-processamento
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(f"\nShape de X_train (processado): {X_train.shape}")

    # 2. Treinamento e Salvamento
    print("\n 2. Treinando o Modelo XGBoost...")

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False
    )

    xgb_model.fit(X_train, y_train)

    # Persistência do Modelo e Pré-processador
    model_filename = 'modelo_atraso_voo.joblib'
    preprocessor_filename = 'pre_processador.joblib'
    dump(xgb_model, model_filename)
    dump(preprocessor, preprocessor_filename)

    print(f"Modelo e Pré-processador salvos: '{model_filename}' e '{preprocessor_filename}'")

    # 3. Simulação de Inferência em Tempo Real (Online)
    print("\n 3. Simulação de Inferência Online (Registro Único)")

    loaded_model = load(model_filename)
    loaded_preprocessor = load(preprocessor_filename)

    # Seleção de um registro de teste
    voo_online_raw = X_test_raw.iloc[0:1]

    # Transformação e Predição
    voo_online_processed = loaded_preprocessor.transform(voo_online_raw)
    pred_online = loaded_model.predict(voo_online_processed)[0]
    proba_online = loaded_model.predict_proba(voo_online_processed)[0][1]

    true_label = y_test.iloc[0]

    print(f"\nRegistro de Entrada: {voo_online_raw.to_dict(orient='records')[0]}")
    print("--- Resultado Online ---")
    print(f"Predição: {pred_online}")
    print(f"Probabilidade de Atraso (Classe 1): {proba_online:.4f}")
    print(f"Rótulo Verdadeiro: {true_label}")

    # 4. Simulação de Inferência em Lote (Batch)
    print("\n 4. Simulação de Inferência em Lote (Conjunto de Registros)")

    # Seleção de 5 registros para inferência em lote
    voos_batch_raw = X_test_raw.iloc[0:5]

    # Transformação e Predição
    voos_batch_processed = loaded_preprocessor.transform(voos_batch_raw)
    pred_batch = loaded_model.predict(voos_batch_processed)
    proba_batch = loaded_model.predict_proba(voos_batch_processed)[:, 1]

    # Organização dos Resultados
    results_df = voos_batch_raw.copy()
    results_df['Predição (Batch)'] = pred_batch
    results_df['Prob. Atraso (Batch)'] = proba_batch.round(4)
    results_df['Rótulo Verdadeiro'] = y_test.head(5).values

    print("\n--- Resultados em Lote ---")
    print(results_df)

    # 5. Comparação e Conclusões
    print("\n 5. Comparação e Conclusões")
    print("----------------------------")

    print(f"Comparação do 1º Registro (Online vs. Lote):")
    print(f"Predição Online: {pred_online} | Predição Lote: {results_df['Predição (Batch)'].iloc[0]}")
    print(f"Prob. Online: {proba_online:.4f} | Prob. Lote: {results_df['Prob. Atraso (Batch)'].iloc[0]:.4f}")

    print("""\nCONCLUSÃO:
Os resultados de predição (rótulo e probabilidade) para os registros são **idênticos** tanto na inferência Online quanto na Batch, pois ambos utilizam o mesmo modelo serializado (`.joblib`) e o mesmo pré-processador.

**Aplicações:**
* **Online:** Baixa latência, predição sob demanda (e.g., API REST).
* **Batch:** Alta eficiência, processamento periódico de grandes volumes de dados.
""")


if __name__ == '__main__':
    main()