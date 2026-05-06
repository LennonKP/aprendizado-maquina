import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_distribuicao_target(df, arquivo="distribuicao_target.png"):
    contagem = df["acesso_agua"].value_counts()
    cores = ["#2ecc71", "#e67e22", "#e74c3c"]

    plt.figure(figsize=(8, 4))
    contagem.plot(kind="bar", color=cores, edgecolor="black")
    plt.title("Distribuição do Target — Acesso à Água")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de amostras")
    plt.xticks(rotation=15)
    for i, v in enumerate(contagem):
        plt.text(i, v + 20, f"{v}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()


def plot_heatmap_regiao(df, coluna_regional, arquivo="mapa_calor_correlacao.png"):
    import pandas as pd

    dist_regiao = pd.crosstab(
        df[coluna_regional],
        df["acesso_agua"],
        normalize="index",
    ).sort_index()

    plt.figure(figsize=(12, 8))
    sns.heatmap(dist_regiao, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"Distribuição do Target por {coluna_regional}")
    plt.xlabel("Classe")
    plt.ylabel(coluna_regional)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()


def plot_matrizes_confusao(
    y_test, resultados, melhor_k, nomes_classes, arquivo="matrizes_confusao.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, k, titulo in zip(
        axes, [1, melhor_k], ["NN (K=1)", f"Melhor KNN (K={melhor_k})"]
    ):
        cm = confusion_matrix(y_test, resultados[k]["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        ax.set_xticklabels(nomes_classes, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(nomes_classes, rotation=0, fontsize=8)

    plt.suptitle("Matrizes de Confusão — NN vs Melhor KNN", fontsize=14)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
