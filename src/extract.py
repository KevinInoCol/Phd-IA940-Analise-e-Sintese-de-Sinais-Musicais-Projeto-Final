"""Extrai MFCC médio e embedding YAMNet de todo o ESD (inglês) e salva em disco.

Uso:
    python -m src.extract --esd /caminho/para/ESD --saida data/features

Gera, na pasta de saída:
    X_mfcc.npy   (N, 20)
    X_yamnet.npy (N, 1024)
    meta.csv     (N linhas: caminho, arquivo, locutor, emocao, frase, split_oficial)
A linha i dos três arquivos corresponde ao mesmo áudio.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.dataset import indexar_esd
from src.features import carregar_audio, mfcc_medio, yamnet_embedding


def extrair(raiz_esd: str, pasta_saida: str, limite: int | None = None) -> None:
    meta = indexar_esd(raiz_esd)
    if limite:
        # amostra estratificada por locutor, para testes rápidos
        meta = meta.groupby("locutor").head(limite // 10 or 1).reset_index(drop=True)
    n = len(meta)
    print(f"{n} arquivos para processar")

    X_mfcc = np.zeros((n, 20), dtype=np.float32)
    X_yamnet = np.zeros((n, 1024), dtype=np.float32)
    inicio = time.time()
    for i, caminho in enumerate(meta["caminho"]):
        y = carregar_audio(caminho)
        X_mfcc[i] = mfcc_medio(y)
        X_yamnet[i] = yamnet_embedding(y)
        if (i + 1) % 500 == 0 or i + 1 == n:
            dt = time.time() - inicio
            print(f"  {i + 1}/{n}  ({dt / 60:.1f} min, faltam ~{dt / (i + 1) * (n - i - 1) / 60:.1f} min)")

    saida = Path(pasta_saida)
    saida.mkdir(parents=True, exist_ok=True)
    np.save(saida / "X_mfcc.npy", X_mfcc)
    np.save(saida / "X_yamnet.npy", X_yamnet)
    meta.to_csv(saida / "meta.csv", index=False)
    print(f"Salvo em {saida}/  ->  X_mfcc {X_mfcc.shape}, X_yamnet {X_yamnet.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esd", required=True, help="pasta raiz do ESD")
    p.add_argument("--saida", default="data/features")
    p.add_argument("--limite", type=int, default=None,
                   help="processar só ~N arquivos (teste rápido)")
    a = p.parse_args()
    extrair(a.esd, a.saida, a.limite)
