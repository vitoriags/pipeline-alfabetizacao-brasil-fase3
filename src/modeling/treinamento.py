
"""Treinamento do modelo final de alfabetização."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.features import FEATURES

# Caminhos relativos à raiz do projeto.
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_DADOS = (
    RAIZ_PROJETO / "data/gold/base_analitica_alfabetizacao.parquet"
)

CAMINHO_MODELO = (
    RAIZ_PROJETO
    / "models/histgradientboosting_alfabetizacao_2023.joblib"
)


# Mantendo as mesmas features utilizadas nos experimentos.
FEATURES_CATEGORICAS = ["rede_codigo"]

FEATURES_NUMERICAS = [
    coluna
    for coluna in FEATURES
    if coluna not in FEATURES_CATEGORICAS
]


def carregar_dados_treinamento():
    """Carrega os alunos das redes públicas avaliados em 2023."""

    colunas = ["ano", "alfabetizado"] + FEATURES

    dados = pd.read_parquet(
        CAMINHO_DADOS,
        columns=colunas,
    )

    # Mantém apenas 2023 e as redes estadual e municipal.
    dados = dados.loc[
        (dados["ano"] == 2023)
        & (dados["rede_codigo"].isin(["2", "3"]))
    ].copy()

    if dados.empty:
        raise ValueError(
            "Nenhum registro válido de treinamento encontrado."
        )

    X = dados[FEATURES].copy()
    y = dados["alfabetizado"].copy()

    return X, y


def criar_pipeline():
    """Reproduz o pré-processamento e o modelo selecionado."""

    # Tratamento das variáveis numéricas.
    transformador_numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
    ])

    # Tratamento da variável categórica de rede de ensino.
    transformador_categorico = Pipeline([
        (
            "imputacao",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "codificacao",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    # Combina os dois tipos de transformação.
    preprocessador = ColumnTransformer([
        (
            "numericas",
            transformador_numerico,
            FEATURES_NUMERICAS,
        ),
        (
            "categoricas",
            transformador_categorico,
            FEATURES_CATEGORICAS,
        ),
    ])

    # Hiperparâmetros selecionados na validação de 2023.
    classificador = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=7,
        learning_rate=0.05,
        min_samples_leaf=100,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=42,
    )

    return Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", classificador),
    ])


def treinar_modelo():
    """Treina a pipeline com todos os registros válidos de 2023."""

    X_treino, y_treino = carregar_dados_treinamento()

    pipeline = criar_pipeline()
    pipeline.fit(X_treino, y_treino)

    return pipeline


if __name__ == "__main__":

    # Evita sobrescrever o artefato final já validado.
    if CAMINHO_MODELO.exists():
        print("O modelo final já está salvo.")
        print("Nenhum treinamento foi executado.")

    else:
        modelo = treinar_modelo()

        CAMINHO_MODELO.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(modelo, CAMINHO_MODELO)

        print("Treinamento concluído!")
        print("Modelo salvo em:", CAMINHO_MODELO)
