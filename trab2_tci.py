"""
Trabalho 2 — TCI: Análise e Classificação de Acesso à Água
KNN / NN com dados do IBGE
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# 1. CARREGAMENTO E ANÁLISE EXPLORATÓRIA
# =============================================================================
df = pd.read_csv("dataset.csv")
df_municipios = pd.read_csv("br_bd_municipio.csv")

colunas_municipio = [
    "id_municipio",
    "sigla_uf",
    "nome_uf",
    "nome_regiao",
    "capital_uf",
    "nome_regiao_saude",
    "nome_regiao_imediata",
    "ddd",
    "amazonia_legal",
]
df = df.merge(df_municipios[colunas_municipio], on="id_municipio", how="left")

print("=== PRIMEIRAS LINHAS DO DATASET ===")
print(df.head())

print("\n=== FORMATO DO DATASET (linhas x colunas) ===")
print(df.shape)

print("\n=== COLUNAS DISPONÍVEIS ===")
print(df.columns.tolist())

print("\n=== INFORMAÇÕES GERAIS DO DATASET ===")
print(df.info())

print("\n=== VALORES NULOS POR COLUNA ===")
print(df.isnull().sum())

print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
print(df.describe())

print("\n=== CLASSES DO ATRIBUTO-ALVO (tipo_ligacao_rede_geral) ===")
print(df["tipo_ligacao_rede_geral"].value_counts())

print("\n=== DISTRIBUIÇÃO POR COR/RAÇA ===")
print(df["cor_raca"].value_counts())

print("\n=== DISTRIBUIÇÃO POR GRUPO DE IDADE ===")
print(df["grupo_idade"].value_counts())


# =============================================================================
# 2. ENGENHARIA DO ATRIBUTO-ALVO
# =============================================================================
def classificar_agua(x):
    x = str(x)
    if "utiliza como forma principal" in x:
        return "Possui e usa"
    elif "Possui" in x:
        return "Possui mas não usa"
    else:
        return "Não possui"


df["acesso_agua"] = df["tipo_ligacao_rede_geral"].apply(classificar_agua)

print("\n=== TARGET COM 3 CLASSES ===")
print(df["acesso_agua"].value_counts())

# Extração do código UF a partir do ID do município para conferência
df["codigo_uf"] = df["id_municipio"].astype(str).str[:2].astype(int)
print("\n=== DISTRIBUIÇÃO POR UF ===")
print(df["codigo_uf"].value_counts())

print("\n=== DADOS TERRITORIAIS AGREGADOS ===")
print(df[["nome_regiao_saude"]].head())

print("\n=== DICIONÁRIO DE DADOS ===")
dicionario = {
    "Atributo": [
        "cor_raca",
        "grupo_idade",
        "populacao",
        "nome_regiao_saude",
        "acesso_agua",
    ],
    "Tipo": [
        "Categórico",
        "Categórico",
        "Numérico",
        "Categórico",
        "Categórico",
    ],
    "Papel": [
        "Previsor",
        "Previsor",
        "Previsor",
        "Previsor",
        "TARGET",
    ],
}
df_dict = pd.DataFrame(dicionario)
print(df_dict.to_string(index=False))

print("\n=== FIM DA ANÁLISE EXPLORATÓRIA ===")
print(f"Total de amostras: {len(df)}")
print(f"Total de atributos: {len(df.columns)}")

# =============================================================================
# 3. VISUALIZAÇÕES
# =============================================================================

# ---- 3.1 Distribuição do Target ----
plt.figure(figsize=(8, 4))
contagem = df["acesso_agua"].value_counts()
cores = ["#2ecc71", "#e67e22", "#e74c3c"]
contagem.plot(kind="bar", color=cores, edgecolor="black")
plt.title("Distribuição do Target — Acesso à Água")
plt.xlabel("Classe")
plt.ylabel("Quantidade de amostras")
plt.xticks(rotation=15)
for i, v in enumerate(contagem):
    plt.text(i, v + 20, f"{v}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("distribuicao_target.png", dpi=150)
plt.close()
print("⚠️  Classes desbalanceadas: 'Possui e usa' representa apenas ~10% dos dados.")

# ---- 3.2 Mapa de calor de correlação ----
df_corr = df.copy()
le = LabelEncoder()
target_encoder = LabelEncoder()

# Preenche valores faltantes para a etapa exploratória.
df_corr["cor_raca"] = df_corr["cor_raca"].fillna("Desconhecido")
df_corr["grupo_idade"] = df_corr["grupo_idade"].fillna("Desconhecido")
df_corr["sigla_uf"] = df_corr["sigla_uf"].fillna("Desconhecido")
df_corr["nome_regiao"] = df_corr["nome_regiao"].fillna("Desconhecido")
df_corr["nome_regiao_saude"] = df_corr["nome_regiao_saude"].fillna("Desconhecido")
df_corr["populacao"] = df_corr["populacao"].fillna(df_corr["populacao"].median())
df_corr["capital_uf"] = df_corr["capital_uf"].fillna(0)
df_corr["ddd"] = df_corr["ddd"].fillna(df_corr["ddd"].median())
df_corr["amazonia_legal"] = df_corr["amazonia_legal"].fillna(0)

df_corr["cor_raca_enc"] = le.fit_transform(df_corr["cor_raca"])
df_corr["grupo_idade_enc"] = le.fit_transform(df_corr["grupo_idade"])
df_corr["sigla_uf_enc"] = le.fit_transform(df_corr["sigla_uf"])
df_corr["nome_regiao_enc"] = le.fit_transform(df_corr["nome_regiao"])
df_corr["nome_regiao_saude_enc"] = le.fit_transform(df_corr["nome_regiao_saude"])
df_corr["target_enc"] = target_encoder.fit_transform(df_corr["acesso_agua"])

cols_corr = [
    "cor_raca_enc",
    "grupo_idade_enc",
    "populacao",
    "sigla_uf_enc",
    "nome_regiao_enc",
    "nome_regiao_saude_enc",
    "capital_uf",
    "ddd",
    "amazonia_legal",
    "target_enc",
]
matriz_corr = df_corr[cols_corr].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(
    matriz_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    xticklabels=[
        "Cor/Raça",
        "Grupo Idade",
        "População",
        "UF",
        "Região",
        "Região Saúde",
        "Capital UF",
        "DDD",
        "Amazônia Legal",
        "Target",
    ],
    yticklabels=[
        "Cor/Raça",
        "Grupo Idade",
        "População",
        "UF",
        "Região",
        "Região Saúde",
        "Capital UF",
        "DDD",
        "Amazônia Legal",
        "Target",
    ],
)
plt.title("Mapa de Calor — Correlação entre Features e Target")
plt.tight_layout()
plt.savefig("mapa_calor_correlacao.png", dpi=150)
plt.close()

# =============================================================================
# 4. PRÉ-PROCESSAMENTO E DIVISÃO TREINO/TESTE
# =============================================================================
features_modelo = ["cor_raca", "grupo_idade", "nome_regiao_saude", "populacao"]

# Se a população estiver ausente em grande parte das linhas, ela tende a virar
# mais ruído do que sinal após imputação. Nesse caso, removemos a coluna.
taxa_nulos_pop = df["populacao"].isnull().mean()
if taxa_nulos_pop > 0.5:
    features_modelo.remove("populacao")
    print(
        f"\n⚠️  'populacao' removida do modelo por excesso de nulos ({taxa_nulos_pop * 100:.1f}%)."
    )

X = df[features_modelo].copy()
y = df["acesso_agua"].copy()

# O KNN não aceita NaN. Para manter o código simples:
# - categorias faltantes viram "Desconhecido"
# - população faltante vira a mediana da coluna
X["cor_raca"] = X["cor_raca"].fillna("Desconhecido")
X["grupo_idade"] = X["grupo_idade"].fillna("Desconhecido")
X["nome_regiao_saude"] = X["nome_regiao_saude"].fillna("Desconhecido")
if "populacao" in X.columns:
    X["populacao"] = X["populacao"].fillna(X["populacao"].median())

print("\n=== VALORES NULOS NAS FEATURES DO MODELO APÓS TRATAMENTO ===")
print(X.isnull().sum())

X = pd.get_dummies(X, columns=["cor_raca", "grupo_idade", "nome_regiao_saude"])

# Divisão 80/20 estratificada para manter proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nomes_classes = ["Não possui", "Possui e usa", "Possui mas não usa"]
print(f"\nFeatures usadas no modelo: {features_modelo}")
print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
print(f"Classes: {nomes_classes}")

# =============================================================================
# 5. TREINAMENTO — NN (K=1) e KNN (K=3, 5, 7)
# =============================================================================
resultados = {}
ks = [1, 3, 5, 7]

for k in ks:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultados[k] = {"modelo": modelo, "y_pred": y_pred, "acuracia": acc}
    nome = "NN (K=1)" if k == 1 else f"KNN (K={k})"
    print(f"{nome:12s} → Acurácia: {acc * 100:.2f}%")

# =============================================================================
# 6. MATRIZES DE CONFUSÃO
# =============================================================================
melhor_k = max([3, 5, 7], key=lambda k: resultados[k]["acuracia"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, k, titulo in zip(
    axes, [1, melhor_k], [f"NN (K=1)", f"Melhor KNN (K={melhor_k})"]
):
    cm = confusion_matrix(y_test, resultados[k]["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xticklabels(nomes_classes, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(nomes_classes, rotation=0, fontsize=8)

plt.suptitle("Matrizes de Confusão — NN vs Melhor KNN", fontsize=14)
plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=150)
plt.close()

# =============================================================================
# 7. TABELA COMPARATIVA E ANÁLISE CRÍTICA
# =============================================================================
linhas = []
for k in ks:
    nome = "NN (K=1)" if k == 1 else f"KNN (K={k})"
    acc = resultados[k]["acuracia"]
    melhor = "✅ Melhor" if k == melhor_k else ""
    linhas.append(
        {"Modelo": nome, "Acurácia (%)": f"{acc * 100:.2f}", "Observação": melhor}
    )

tabela = pd.DataFrame(linhas)
print("\n=== TABELA COMPARATIVA DE ACURÁCIA ===")
print(tabela.to_string(index=False))

acc_nn = resultados[1]["acuracia"]
acc_best = resultados[melhor_k]["acuracia"]
ganho = (acc_best - acc_nn) * 100
melhor_modelo_geral = max(ks, key=lambda k: resultados[k]["acuracia"])
nome_melhor_modelo = (
    "NN (K=1)" if melhor_modelo_geral == 1 else f"KNN (K={melhor_modelo_geral})"
)

print(
    f"""
=== ANÁLISE CRÍTICA ===
- NN (K=1) alcançou {acc_nn * 100:.2f}% de acurácia.
- Melhor KNN (K={melhor_k}) alcançou {acc_best * 100:.2f}% ({ganho:.2f} pp em relação ao NN).
- Melhor modelo geral: {nome_melhor_modelo}.

O aumento de K {"melhorou" if ganho > 0 else "não melhorou significativamente"} a acurácia.
Com K=1, o modelo memoriza os dados de treino e é sensível a ruídos (overfitting).
Com K maiores, as previsões se tornam mais estáveis por considerar a vizinhança,
compensando o desequilíbrio entre as classes do dataset.
A classe 'Possui e usa' (~10% dos dados) é a mais difícil de prever corretamente —
verifique os Falsos Negativos dessa classe na matriz de confusão.
"""
)
