"""Extrai MFCC médio e YAMNet de TODOS os áudios em cada nível de RT60.

Para cada nível salva data/features_reverb/rt<RT60>/X_mfcc.npy e X_yamnet.npy,
na MESMA ordem de linhas de data/features/meta.csv. Reverberamos todos os
arquivos (e não só os de teste) porque assim a mesma extração serve:
  • à condição principal (treino limpo, teste reverberado) e
  • à condição secundária (treino e teste reverberados).

Uso:
    python -m src.extract_reverb --meta data/features/meta.csv
    python -m src.extract_reverb --niveis 0.5 1.0      # só alguns níveis
    python -m src.extract_reverb --cortar-cauda --saida data/features_reverb_semcauda  # ablação
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.degradation import NIVEIS_RT60, reverberar, semente_arquivo
from src.features import carregar_audio, mfcc_medio, yamnet_embedding


def pasta_nivel(raiz: str | Path, rt60: float) -> Path:
    return Path(raiz) / f"rt{rt60:.1f}"


def extrair_nivel(meta: pd.DataFrame, rt60: float, raiz_saida: str, cortar_cauda: bool = False) -> None:
    saida = pasta_nivel(raiz_saida, rt60)
    if (saida / "X_yamnet.npy").exists():
        print(f"RT60 = {rt60:.1f} s já extraído, pulando")
        return
    n = len(meta)
    X_mfcc = np.zeros((n, 20), dtype=np.float32)
    X_yamnet = np.zeros((n, 1024), dtype=np.float32)
    inicio = time.time()
    for i, caminho in enumerate(meta["caminho"]):
        y = reverberar(carregar_audio(caminho), rt60, semente=semente_arquivo(i, rt60),
                       cortar_cauda=cortar_cauda)
        X_mfcc[i] = mfcc_medio(y)
        X_yamnet[i] = yamnet_embedding(y)
        if (i + 1) % 2500 == 0 or i + 1 == n:
            print(f"  RT60 = {rt60:.1f} s · {i + 1}/{n} ({(time.time() - inicio) / 60:.1f} min)", flush=True)
    saida.mkdir(parents=True, exist_ok=True)
    np.save(saida / "X_mfcc.npy", X_mfcc)
    np.save(saida / "X_yamnet.npy", X_yamnet)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="data/features/meta.csv")
    p.add_argument("--saida", default="data/features_reverb")
    p.add_argument("--niveis", type=float, nargs="*", default=NIVEIS_RT60)
    p.add_argument("--cortar-cauda", action="store_true",
                   help="ablação: corta a cauda (saída do mesmo tamanho da entrada)")
    a = p.parse_args()
    meta = pd.read_csv(a.meta, dtype={"locutor": str})
    for rt60 in a.niveis:
        extrair_nivel(meta, rt60, a.saida, a.cortar_cauda)
