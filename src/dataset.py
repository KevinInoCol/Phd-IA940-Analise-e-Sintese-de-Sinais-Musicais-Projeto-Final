"""Indexa o ESD: uma linha por arquivo .wav com locutor, emoção e split oficial.

Só usamos os 10 locutores de inglês (0011–0020). O rótulo da tarefa é o
LOCUTOR, tirado do nome do arquivo (0011_000351.wav -> "0011"), e não o nome
da pasta (que no ESD é a emoção).
"""

import re
from pathlib import Path

import pandas as pd

LOCUTORES_INGLES = [f"{i:04d}" for i in range(11, 21)]
EMOCOES = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
FRASES_POR_EMOCAO = 350
PADRAO_ARQUIVO = re.compile(r"^(\d{4})_(\d{6})\.wav$")


def _split_oficial(indice_na_emocao: int) -> str:
    """Split oficial do ESD por locutor×emoção: 20 evaluation / 30 test / 300 train.

    Usado só quando o caminho não traz a pasta train/evaluation/test.
    (Suposição: as 20 primeiras frases são evaluation e as 30 seguintes test.)
    """
    if indice_na_emocao <= 20:
        return "evaluation"
    if indice_na_emocao <= 50:
        return "test"
    return "train"


def indexar_esd(raiz_esd: str | Path, locutores=LOCUTORES_INGLES) -> pd.DataFrame:
    """Percorre a pasta do ESD e devolve um DataFrame ordenado.

    Colunas: caminho, arquivo, locutor, emocao, frase (1–350 dentro da emoção),
    split_oficial.
    """
    raiz_esd = Path(raiz_esd)
    linhas = []
    for caminho in raiz_esd.rglob("*.wav"):
        m = PADRAO_ARQUIVO.match(caminho.name)
        if not m or m.group(1) not in locutores:
            continue
        partes = caminho.parts
        emocao = next((p for p in partes if p in EMOCOES), None)
        if emocao is None:
            raise ValueError(f"Emoção não encontrada no caminho: {caminho}")
        numero = int(m.group(2))
        frase = (numero - 1) % FRASES_POR_EMOCAO + 1
        split = next((p for p in partes if p in ("train", "evaluation", "test")),
                     _split_oficial(frase))
        linhas.append({
            "caminho": str(caminho),
            "arquivo": caminho.name,
            "locutor": m.group(1),
            "emocao": emocao,
            "frase": frase,
            "split_oficial": split,
        })
    if not linhas:
        raise FileNotFoundError(f"Nenhum .wav de {locutores[0]}–{locutores[-1]} em {raiz_esd}")
    df = pd.DataFrame(linhas).sort_values("arquivo").reset_index(drop=True)
    if df["arquivo"].duplicated().any():
        raise ValueError("Há arquivos duplicados no índice (mesmo nome em pastas diferentes).")
    return df


if __name__ == "__main__":
    import sys

    df = indexar_esd(sys.argv[1])
    print(df.head())
    print(f"\nTotal: {len(df)} arquivos")
    print(df.groupby(["locutor", "emocao"]).size().unstack())
