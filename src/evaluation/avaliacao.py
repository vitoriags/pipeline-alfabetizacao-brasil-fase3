
"""Avaliação do modelo supervisionado de alfabetização.

Realiza a avaliação temporal de 2024, calcula métricas de
classificação e gera relatórios JSON e PNG.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing.features import FEATURES
from src.visualization.relatorio_visual import salvar_relatorio_visual

# ============================================================
# 1. PROBABILIDADES DO MODELO
# ============================================================

def obter_probabilidades_classe_zero(modelo, X):
    """Obtém a probabilidade da classe 0 (não alfabetizado)."""

    classes = list(modelo.classes_)

    if 0 not in classes:
        raise ValueError(
            f"Classe 0 não encontrada no modelo: {classes}"
        )

    indice_classe_zero = classes.index(0)

    probabilidades = modelo.predict_proba(X)[
        :, indice_classe_zero
    ]

    return probabilidades

# ============================================================
# 2. CÁLCULO DAS MÉTRICAS
# ============================================================

def avaliar_modelo(
    modelo,
    X_teste,
    y_teste,
    caminho_relatorio_visual=None,
):
    """Calcula as métricas com foco na classe 0."""

    previsoes = modelo.predict(X_teste)

    probabilidades = obter_probabilidades_classe_zero(
        modelo,
        X_teste,
    )

    y_teste = np.asarray(y_teste)

    # Classe positiva de interesse: não alfabetizado (0)
    alvo_risco = (y_teste == 0).astype(int)
    previsoes_risco = (np.asarray(previsoes) == 0).astype(int)

    # Matriz com labels=[0, 1]:
    # [[TP, FN],
    #  [FP, TN]]
    tp, fn, fp, tn = confusion_matrix(
        y_teste,
        previsoes,
        labels=[0, 1],
    ).ravel()

    metricas = {
        "accuracy": accuracy_score(
            y_teste,
            previsoes,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_teste,
            previsoes,
        ),
        "roc_auc": roc_auc_score(
            alvo_risco,
            probabilidades,
        ),
        "average_precision": average_precision_score(
            alvo_risco,
            probabilidades,
        ),
        "precision_nao_alfabetizado": precision_score(
            alvo_risco,
            previsoes_risco,
            zero_division=0,
        ),
        "recall_nao_alfabetizado": recall_score(
            alvo_risco,
            previsoes_risco,
            zero_division=0,
        ),
        "f1_nao_alfabetizado": f1_score(
            alvo_risco,
            previsoes_risco,
            zero_division=0,
        ),
        "precision_alfabetizado": precision_score(
            y_teste,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "recall_alfabetizado": recall_score(
            y_teste,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "f1_alfabetizado": f1_score(
            y_teste,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "specificity_nao_alfabetizado": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        ),
        "prevalencia_nao_alfabetizado": float(
            alvo_risco.mean()
        ),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "registros": int(len(y_teste)),
    }

    print(
        classification_report(
            y_teste,
            previsoes,
            zero_division=0,
        )
    )

    if caminho_relatorio_visual is not None:
        salvar_relatorio_visual(
            y_real=y_teste,
            y_predito=previsoes,
            metricas=metricas,
            destino=caminho_relatorio_visual,
        )

    return metricas


# ============================================================
# 3. EXECUÇÃO DA AVALIAÇÃO TEMPORAL
# ============================================================

if __name__ == "__main__":

    raiz = Path(__file__).resolve().parents[2]

    caminho_dados = (
        raiz / "data/gold/base_analitica_alfabetizacao.parquet"
    )

    caminho_modelo = (
        raiz / "models/histgradientboosting_alfabetizacao_2023.joblib"
    )

    # Verifica os arquivos necessários
    if not caminho_dados.is_file():
        raise FileNotFoundError(
            f"Base Gold não encontrada: {caminho_dados}"
        )

    if not caminho_modelo.is_file():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute o treinamento primeiro."
        )

    # Carrega apenas as colunas necessárias
    colunas = list(
        dict.fromkeys(
            ["ano", "alfabetizado"] + FEATURES
        )
    )

    # Avaliação temporal com dados de 2024
    dados_teste = pd.read_parquet(
        caminho_dados,
        columns=colunas,
        filters=[
            ("ano", "==", 2024),
            ("rede_codigo", "in", ["2", "3"]),
        ],
    )

    if dados_teste.empty:
        raise ValueError(
            "Nenhum registro encontrado para 2024."
        )

    if dados_teste["alfabetizado"].isna().any():
        raise ValueError(
            "A variável-alvo contém valores ausentes."
        )

    X_teste = dados_teste[FEATURES]
    y_teste = dados_teste["alfabetizado"]

    # Carrega o modelo treinado em 2023
    modelo = joblib.load(caminho_modelo)

    print("AVALIAÇÃO TEMPORAL — 2024")
    print("Registros:", len(dados_teste))

    # Calcula métricas e gera a figura
    metricas = avaliar_modelo(
        modelo,
        X_teste,
        y_teste,
        caminho_relatorio_visual=(
            raiz / "reports/figures/avaliacao_modelo_2024.png"
        ),
    )

    # --------------------------------------------------------
    # Salva métricas em JSON
    # --------------------------------------------------------

    resultado = {
        "ano_teste": 2024,
        **{
            nome: (
                float(valor)
                if isinstance(valor, (np.floating, float))
                else int(valor)
                if isinstance(valor, (np.integer, int))
                else valor
            )
            for nome, valor in metricas.items()
        },
    }

    destino_json = (
        raiz / "reports/avaliacao_temporal_2024.json"
    )

    destino_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destino_json.open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            resultado,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    print("\nMétricas:")

    for nome, valor in metricas.items():
        if isinstance(valor, float):
            print(f"{nome}: {valor:.4f}")
        else:
            print(f"{nome}: {valor}")

    print("\nRelatório JSON salvo em:", destino_json)
