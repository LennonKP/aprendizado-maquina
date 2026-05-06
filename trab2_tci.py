import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from plots import plot_distribuicao_target, plot_heatmap_regiao, plot_matrizes_confusao

COLUNA_REGIONAL = "sigla_uf"  # opções: "nome_regiao_saude", "sigla_uf", "nome"


def classificar_agua(x):
    x = str(x)
    if "utiliza como forma principal" in x:
        return "Possui e usa"
    if "Possui" in x:
        return "Possui mas não usa"
    return "Não possui"


def escolher_features_modelo(coluna_regional):
    return [coluna_regional]


def preparar_dados_modelo(df, features_modelo):
    X = df[features_modelo].copy()
    y = df["acesso_agua"].copy()

    for coluna in features_modelo:
        X[coluna] = X[coluna].fillna("Desconhecido")

    X = pd.get_dummies(X, columns=features_modelo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def treinar_modelos_knn(X_train, X_test, y_train, y_test, ks):
    resultados = {}
    for k in ks:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        resultados[k] = {"modelo": modelo, "y_pred": y_pred, "acuracia": acc}
    return resultados


# -- carregamento --
df = pd.read_csv("dataset.csv")
print(f"\n{df.shape[0]} linhas, {df.shape[1]} colunas")

# -- target simplificado em 3 classes --
df["acesso_agua"] = df["tipo_ligacao_rede_geral"].apply(classificar_agua)

print("\nclasses do target:")
print(df["acesso_agua"].value_counts().to_string())

if "sigla_uf" in df.columns:
    print("\namostras por UF:")
    print(df["sigla_uf"].value_counts().to_string())

# dicionário de atributos usado no modelo
dicionario = {
    "Atributo": [COLUNA_REGIONAL, "acesso_agua"],
    "Tipo": ["Categórico", "Categórico"],
    "Papel": ["Previsor", "TARGET"],
}
print("\natributos do modelo:")
print(pd.DataFrame(dicionario).to_string(index=False))

print(f"\ntotal: {len(df)} amostras")

# -- visualizações --
plot_distribuicao_target(df)
plot_heatmap_regiao(df, COLUNA_REGIONAL)

# -- pré-processamento --
features_modelo = escolher_features_modelo(COLUNA_REGIONAL)
X_train, X_test, y_train, y_test = preparar_dados_modelo(df, features_modelo)

nomes_classes = ["Não possui", "Possui e usa", "Possui mas não usa"]
print(f"\nfeatures: {features_modelo}")
print(f"treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# -- treinamento --
ks = [1, 3, 5, 7]
resultados = treinar_modelos_knn(X_train, X_test, y_train, y_test, ks)

# -- matrizes de confusão --
melhor_k = max([3, 5, 7], key=lambda k: resultados[k]["acuracia"])
plot_matrizes_confusao(y_test, resultados, melhor_k, nomes_classes)

# -- resultados finais --
linhas = []
for k in ks:
    nome = "NN (K=1)" if k == 1 else f"KNN (K={k})"
    acc = resultados[k]["acuracia"]
    melhor = "← melhor" if k == melhor_k else ""
    linhas.append({"Modelo": nome, "Acurácia (%)": f"{acc * 100:.2f}", "": melhor})

print("\n" + pd.DataFrame(linhas).to_string(index=False))
