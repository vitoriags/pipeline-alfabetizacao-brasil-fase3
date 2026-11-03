
"""Geração de relatórios visuais de avaliação dos modelos."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def salvar_relatorio_visual(y_real, y_predito, metricas, destino):
    """Gera matriz de confusão, métricas e interpretação automática."""

    # Matriz de confusão
    matriz = confusion_matrix(
        y_real,
        y_predito,
        labels=[0, 1],
    )

    totais = matriz.sum(axis=1, keepdims=True)

    matriz_normalizada = np.divide(
        matriz,
        totais,
        out=np.zeros_like(matriz, dtype=float),
        where=totais != 0,
    )

    classes = ["Não alfabetizado", "Alfabetizado"]

    # Configuração da figura
    fig = plt.figure(figsize=(16, 8.5))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.1, 1],
        height_ratios=[1, 0.5],
        wspace=0.38,
        hspace=0.30,
        left=0.08,
        right=0.96,
        top=0.89,
        bottom=0.10,
    )

    fig.suptitle(
        "Avaliação temporal do modelo — 2024",
        fontsize=18,
        fontweight="bold",
    )

    # ========================================================
    # 1. MATRIZ DE CONFUSÃO
    # ========================================================

    ax1 = fig.add_subplot(gs[:, 0])

    imagem = ax1.imshow(
        matriz_normalizada,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    ax1.set_xticks([0, 1], labels=classes)
    ax1.set_yticks([0, 1], labels=classes)

    ax1.set_xlabel("Classe predita")
    ax1.set_ylabel("Classe real")

    ax1.set_title(
        "Matriz de confusão",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    for i in range(2):
        for j in range(2):
            quantidade = f"{matriz[i, j]:,}".replace(",", ".")
            percentual = f"{matriz_normalizada[i, j]:.1%}"

            ax1.text(
                j,
                i,
                f"{quantidade}\n({percentual})",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=(
                    "white"
                    if matriz_normalizada[i, j] > 0.5
                    else "black"
                ),
            )

    fig.colorbar(
        imagem,
        ax=ax1,
        fraction=0.046,
        pad=0.04,
        label="Proporção por classe real",
    )

    # ========================================================
    # 2. MÉTRICAS DO MODELO
    # ========================================================

    ax2 = fig.add_subplot(gs[0, 1])

    nomes = [
        "Accuracy",
        "Balanced Accuracy",
        "ROC-AUC",
        "Average Precision",
        "Precisão (classe 0)",
        "Recall (classe 0)",
        "F1-score (classe 0)",
    ]

    valores = [
        metricas["accuracy"],
        metricas["balanced_accuracy"],
        metricas["roc_auc"],
        metricas["average_precision"],
        metricas["precision_nao_alfabetizado"],
        metricas["recall_nao_alfabetizado"],
        metricas["f1_nao_alfabetizado"],
    ]

    barras = ax2.barh(nomes, valores)

    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Valor da métrica")

    ax2.set_title(
        "Principais métricas",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax2.invert_yaxis()
    ax2.bar_label(barras, fmt="%.4f", padding=4)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ========================================================
    # 3. INTERPRETAÇÃO DOS RESULTADOS
    # ========================================================

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    recall = metricas["recall_nao_alfabetizado"]

    identificados = recall * 100
    nao_identificados = (1 - recall) * 100

    ax3.text(
        0.0,
        0.95,
        "Interpretação — alunos não alfabetizados",
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax3.transAxes,
    )

    ax3.text(
        0.02,
        0.69,
        f"{identificados:.1f}%",
        fontsize=25,
        fontweight="bold",
        color="#237A57",
        va="center",
        transform=ax3.transAxes,
    )

    ax3.text(
        0.02,
        0.49,
        "Identificados corretamente",
        fontsize=10,
        transform=ax3.transAxes,
    )

    ax3.text(
        0.52,
        0.69,
        f"{nao_identificados:.1f}%",
        fontsize=25,
        fontweight="bold",
        color="#C44E52",
        va="center",
        transform=ax3.transAxes,
    )

    ax3.text(
        0.52,
        0.49,
        "Não identificados",
        fontsize=10,
        transform=ax3.transAxes,
    )

    if recall < 0.5:
        conclusao = (
            "Com o limiar atual, o modelo deixa de identificar "
            "a maioria dos alunos não alfabetizados."
        )
    else:
        conclusao = (
            "Com o limiar atual, o modelo identifica "
            "a maioria dos alunos não alfabetizados."
        )

    ax3.text(
        0.02,
        0.24,
        textwrap.fill(conclusao, width=62),
        fontsize=10,
        color="#444444",
        va="top",
        linespacing=1.5,
        transform=ax3.transAxes,
    )

    # ========================================================
    # 4. EXPORTAÇÃO DA FIGURA
    # ========================================================

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    # Salva a versão principal da execução
    fig.savefig(
        destino,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    # Salva também uma versão fixa para documentação/README
    raiz_projeto = destino.parents[2]
    caminho_documentacao = (
        raiz_projeto / "images" / "avaliacao_modelo_atual.png"
    )
    caminho_documentacao.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        caminho_documentacao,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Relatório visual salvo em: {destino}")
    print(
        "Imagem de documentação salva em:",
        caminho_documentacao,
    )
