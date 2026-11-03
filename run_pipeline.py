
"""Pipeline principal do Tech Challenge — Fase 3.

Cada execução gera um novo modelo, relatório JSON
e gráfico de avaliação, sem sobrescrever resultados anteriores.
"""

import argparse
import json
import subprocess
import sys
from itertools import count
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

BASE_GOLD = (
    RAIZ / "data/gold/base_analitica_alfabetizacao.parquet"
)

PASTA_MODELOS = RAIZ / "models"
PASTA_RELATORIOS = RAIZ / "reports"
PASTA_FIGURAS = PASTA_RELATORIOS / "figures"


# ============================================================
# 1. VERSIONAMENTO DOS ARTEFATOS
# ============================================================

def caminhos(numero: int):
    """Define os arquivos da execução, usando o mesmo número."""

    nome = f"modelo_{numero:03d}"

    return (
        PASTA_MODELOS / f"{nome}.joblib",
        PASTA_RELATORIOS / f"metricas_{nome}.json",
        PASTA_FIGURAS / f"avaliacao_{nome}.png",
    )


def proximo_numero():
    """Encontra o próximo número disponível sem sobrescrever arquivos."""

    for numero in count(1):
        arquivos = caminhos(numero)

        if not any(arquivo.exists() for arquivo in arquivos):
            return numero


# ============================================================
# 2. TREINAMENTO
# ============================================================

def treinar(numero: int):
    """Treina e salva uma nova versão do modelo."""

    import joblib

    from src.modeling.treinamento import treinar_modelo

    caminho_modelo, _, _ = caminhos(numero)

    caminho_modelo.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nTREINAMENTO — MODELO {numero:03d}", flush=True)

    # Executa a função de treinamento existente
    modelo = treinar_modelo()

    # Salva o modelo com o número da execução
    joblib.dump(modelo, caminho_modelo)

    print(f"Modelo salvo em: {caminho_modelo}", flush=True)


# ============================================================
# 3. AVALIAÇÃO
# ============================================================

def avaliar(numero: int):
    """Avalia o modelo da execução e salva métricas e gráfico."""

    import joblib
    import numpy as np
    import pandas as pd

    from src.evaluation.avaliacao import avaliar_modelo
    from src.preprocessing.features import FEATURES

    caminho_modelo, caminho_json, caminho_imagem = caminhos(numero)

    if not caminho_modelo.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}"
        )

    print(f"\nAVALIAÇÃO — MODELO {numero:03d}", flush=True)

    # Leitura somente das colunas necessárias
    colunas = list(
        dict.fromkeys(["ano", "alfabetizado", *FEATURES])
    )

    # Seleção dos dados de teste de 2024
    dados = pd.read_parquet(
        BASE_GOLD,
        columns=colunas,
        filters=[
            ("ano", "==", 2024),
            ("rede_codigo", "in", ["2", "3"]),
        ],
    )

    if dados.empty:
        raise ValueError("Base de avaliação vazia.")

    if dados["alfabetizado"].isna().any():
        raise ValueError(
            "Variável-alvo contém valores ausentes."
        )

    X_teste = dados[FEATURES]
    y_teste = dados["alfabetizado"]

    # Carrega o modelo desta execução
    modelo = joblib.load(caminho_modelo)

    # Calcula métricas e gera a imagem
    metricas = avaliar_modelo(
        modelo,
        X_teste,
        y_teste,
        caminho_relatorio_visual=caminho_imagem,
    )

    # Inclui metadados para rastrear o experimento
    resultado = {
        "modelo": f"modelo_{numero:03d}",
        "ano_treino": 2023,
        "ano_teste": 2024,
        **{
            chave: (
                float(valor)
                if isinstance(valor, (float, np.floating))
                else int(valor)
            )
            for chave, valor in metricas.items()
        },
    }

    caminho_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with caminho_json.open("w", encoding="utf-8") as arquivo:
        json.dump(
            resultado,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Métricas salvas em: {caminho_json}", flush=True)
    print(f"Gráfico salvo em: {caminho_imagem}", flush=True)


# ============================================================
# 4. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    """Coordena o treinamento e a avaliação de uma nova execução."""

    parser = argparse.ArgumentParser()

    # Opções internas usadas para separar processos
    parser.add_argument(
        "--worker",
        choices=["treino", "avaliacao"],
    )
    parser.add_argument("--numero", type=int)

    args = parser.parse_args()

    if args.worker:
        if args.numero is None or args.numero < 1:
            parser.error("Informe um número de execução válido.")

        if args.worker == "treino":
            treinar(args.numero)
        else:
            avaliar(args.numero)

        return

    # Verifica a existência da base Gold
    if not BASE_GOLD.is_file():
        raise FileNotFoundError(
            f"Base Gold não encontrada: {BASE_GOLD}"
        )

    # Identifica automaticamente a próxima versão
    numero = proximo_numero()

    print(
        f"\nTECH CHALLENGE — EXECUÇÃO MODELO {numero:03d}",
        flush=True,
    )

    # Cada etapa utiliza um processo separado para
    # liberar memória RAM entre treinamento e avaliação.
    for etapa in ("treino", "avaliacao"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                etapa,
                "--numero",
                str(numero),
            ],
            cwd=RAIZ,
            check=True,
        )

    print(
        f"\nExecução modelo_{numero:03d} concluída!",
        flush=True,
    )


if __name__ == "__main__":
    main()
