
from pathlib import Path

import joblib
import pandas as pd

# Caminho do modelo treinado, relativo à raiz do projeto.
CAMINHO_MODELO = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "histgradientboosting_alfabetizacao_2023.joblib"
)


# Variáveis utilizadas durante o treinamento.
from src.preprocessing.features import FEATURES


def carregar_modelo():
    """Carrega a pipeline treinada e salva em disco."""
    return joblib.load(CAMINHO_MODELO)


def prever_risco(dados: pd.DataFrame, modelo=None) -> pd.DataFrame:
    """Estima o risco de não alfabetização para os registros recebidos."""

    if modelo is None:
        modelo = carregar_modelo()

    # Verifica se todas as variáveis necessárias estão disponíveis.
    colunas_ausentes = [
        coluna for coluna in FEATURES
        if coluna not in dados.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"Variáveis ausentes: {colunas_ausentes}"
        )

    # Identifica a posição da classe 0 (não alfabetizado).
    indice_classe_0 = list(modelo.classes_).index(0)

    # Aplica o pré-processamento e o modelo treinado.
    probabilidades = modelo.predict_proba(
        dados[FEATURES]
    )[:, indice_classe_0]

    # Retorna os registros com uma nova coluna de risco.
    resultado = dados.copy()
    resultado["risco_nao_alfabetizacao"] = probabilidades

    return resultado
