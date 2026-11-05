"""Classificação de locutor por kNN, igual para as duas representações.

Para cada divisão (semente):
  1. divide os ARQUIVOS em treino/teste (estratificado por locutor) — cada
     arquivo cai em um só lado;
  2. z-score com média e desvio estimados só no treino;
  3. kNN, distância euclidiana, mesmo k;
  4. acurácia no teste.
As mesmas divisões (mesmas sementes -> mesmos índices) servem para MFCC e YAMNet.

Uso:
    python -m src.classify --features data/features --k 5 --divisoes 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def gerar_divisoes(rotulos: np.ndarray, n_divisoes: int = 5,
                   frac_teste: float = 0.3, semente: int = 0):
    """Lista de (idx_treino, idx_teste), reutilizada por todas as representações."""
    sss = StratifiedShuffleSplit(n_splits=n_divisoes, test_size=frac_teste,
                                 random_state=semente)
    return list(sss.split(np.zeros(len(rotulos)), rotulos))


def acuracia_knn(X: np.ndarray, y: np.ndarray, idx_treino, idx_teste, k: int = 5) -> float:
    escala = StandardScaler().fit(X[idx_treino])          # só treino
    Xtr, Xte = escala.transform(X[idx_treino]), escala.transform(X[idx_teste])
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(Xtr, y[idx_treino])
    return float((knn.predict(Xte) == y[idx_teste]).mean())


def avaliar(representacoes: dict[str, np.ndarray], y: np.ndarray, divisoes, k: int = 5) -> pd.DataFrame:
    """Acurácia de cada representação em cada divisão (tabela longa)."""
    linhas = []
    for nome, X in representacoes.items():
        for d, (tr, te) in enumerate(divisoes):
            linhas.append({"representacao": nome, "divisao": d,
                           "acuracia": acuracia_knn(X, y, tr, te, k)})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--divisoes", type=int, default=5)
    a = p.parse_args()

    pasta = Path(a.features)
    meta = pd.read_csv(pasta / "meta.csv", dtype={"locutor": str})
    y = meta["locutor"].to_numpy()
    reps = {"MFCC": np.load(pasta / "X_mfcc.npy"),
            "YAMNet": np.load(pasta / "X_yamnet.npy")}

    divisoes = gerar_divisoes(y, a.divisoes)
    res = avaliar(reps, y, divisoes, a.k)
    print(f"Sem degradação · {len(y)} arquivos · {len(set(y))} locutores "
          f"(acaso = {1 / len(set(y)):.0%}) · k = {a.k} · {a.divisoes} divisões\n")
    print(res.groupby("representacao")["acuracia"].agg(["mean", "std"]).round(4))
    res.to_csv(pasta / f"resultados_sem_degradacao_k{a.k}.csv", index=False)
