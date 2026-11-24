"""Curvas de acurácia × RT60 para as duas representações.

Condições:
  • "principal"  (treino LIMPO, teste REVERBERADO): z-score e kNN ajustados nos
    vetores limpos do treino; o teste usa os vetores reverberados dos MESMOS
    arquivos de teste. Mede a robustez: o modelo nunca viu reverberação.
  • "pareada"    (treino e teste reverberados no mesmo RT60): mede quanta
    informação de locutor sobra no sinal reverberado.

As divisões (5 sementes) e o k são os mesmos para tudo. Em cada divisão cada
arquivo está só no treino ou só no teste.

Uso:
    python -m src.curves --k 5 --divisoes 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.classify import gerar_divisoes
from src.degradation import NIVEIS_RT60
from src.extract_reverb import pasta_nivel

REPS = {"MFCC": "X_mfcc.npy", "YAMNet": "X_yamnet.npy"}


def carregar(pasta_limpa="data/features", pasta_reverb="data/features_reverb", niveis=NIVEIS_RT60):
    """meta, y e um dicionário X[(rep, rt60)] -> matriz (N, d)."""
    meta = pd.read_csv(Path(pasta_limpa) / "meta.csv", dtype={"locutor": str})
    X = {}
    for rep, arq in REPS.items():
        X[(rep, 0.0)] = np.load(Path(pasta_limpa) / arq)
        for rt in niveis:
            if rt > 0:
                X[(rep, rt)] = np.load(pasta_nivel(pasta_reverb, rt) / arq)
    return meta, meta["locutor"].to_numpy(), X


def ajustar(Xtr, ytr, k):
    escala = StandardScaler().fit(Xtr)                         # só treino
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    return escala, knn.fit(escala.transform(Xtr), ytr)


def prever(modelo, Xte):
    escala, knn = modelo
    return knn.predict(escala.transform(Xte))


def avaliar_curvas(y, X, divisoes, niveis=NIVEIS_RT60, k=5, guardar_predicoes=False):
    """Tabela longa: condicao, representacao, rt60, divisao, acuracia.

    Se guardar_predicoes, devolve também {(cond, rep, rt60, divisao): (idx_teste, y_pred)}.
    """
    linhas, preds = [], {}
    for rep in REPS:
        for d, (tr, te) in enumerate(divisoes):
            modelo_limpo = ajustar(X[(rep, 0.0)][tr], y[tr], k)
            for rt in niveis:
                Xr = X[(rep, rt)]
                for cond in ("principal", "pareada"):
                    modelo = modelo_limpo if (cond == "principal" or rt == 0) else ajustar(Xr[tr], y[tr], k)
                    yp = prever(modelo, Xr[te])
                    linhas.append({"condicao": cond, "representacao": rep, "rt60": rt,
                                   "divisao": d, "acuracia": float((yp == y[te]).mean())})
                    if guardar_predicoes:
                        preds[(cond, rep, rt, d)] = (te, yp)
    df = pd.DataFrame(linhas)
    return (df, preds) if guardar_predicoes else df


def resumo_pareado(df: pd.DataFrame, cond: str = "principal") -> pd.DataFrame:
    """Por nível: média ± desvio de cada representação e da diferença PAREADA
    (MFCC − YAMNet na mesma divisão). 'diferenca_supera_dispersao' é verdadeiro
    quando |Δ médio| > 2 × desvio de Δ entre divisões e também > maior desvio
    das duas curvas."""
    t = df[df.condicao == cond].pivot_table(index=["rt60", "divisao"],
                                            columns="representacao", values="acuracia")
    t["delta"] = t["MFCC"] - t["YAMNet"]
    g = t.groupby("rt60")
    r = pd.DataFrame({
        "MFCC_media": g["MFCC"].mean(), "MFCC_desvio": g["MFCC"].std(),
        "YAMNet_media": g["YAMNet"].mean(), "YAMNet_desvio": g["YAMNet"].std(),
        "delta_media": g["delta"].mean(), "delta_desvio": g["delta"].std(),
    })
    maior_desvio = r[["MFCC_desvio", "YAMNet_desvio"]].max(axis=1)
    r["diferenca_supera_dispersao"] = (r["delta_media"].abs() > 2 * r["delta_desvio"]) & \
                                      (r["delta_media"].abs() > maior_desvio)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--divisoes", type=int, default=5)
    p.add_argument("--saida", default="data/resultados")
    a = p.parse_args()

    meta, y, X = carregar()
    divisoes = gerar_divisoes(y, a.divisoes)
    df = avaliar_curvas(y, X, divisoes, k=a.k)
    Path(a.saida).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.saida) / f"curvas_k{a.k}.csv", index=False)
    for cond in ("principal", "pareada"):
        print(f"\n=== Condição {cond} (k = {a.k}) ===")
        print((resumo_pareado(df, cond) * [100, 100, 100, 100, 100, 100, 1]).round(2).to_string())
