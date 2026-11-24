"""Ferramentas para a análise de falhas no nível em que as curvas se separam.

  • f0_mediana_por_locutor: F0 mediana (pYIN) de alguns áudios limpos por locutor,
    para saber se os erros cruzam ou não o gênero / registro da voz.
  • categorias_acerto: para uma divisão e um nível, separa os arquivos de teste em
    "os dois acertam", "só MFCC acerta", "só YAMNet acerta", "os dois erram".
  • exportar_exemplos: grava o áudio limpo e o reverberado de cada exemplo em WAV
    (os mesmos que vão no notebook).
  • deslocamento_mfcc: média, sobre os arquivos, de MFCC(reverb) − MFCC(limpo) —
    mostra que a reverberação empurra o vetor MFCC médio numa direção comum.
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.degradation import reverberar, semente_arquivo
from src.features import SR, carregar_audio


def f0_mediana_por_locutor(meta: pd.DataFrame, n_por_locutor: int = 20, semente: int = 0) -> pd.Series:
    rng = np.random.default_rng(semente)
    f0 = {}
    for loc, grupo in meta[meta.emocao == "Neutral"].groupby("locutor"):
        caminhos = rng.choice(grupo["caminho"].to_numpy(), size=n_por_locutor, replace=False)
        valores = []
        for c in caminhos:
            f, voz, _ = librosa.pyin(carregar_audio(c), fmin=60, fmax=400, sr=SR)
            valores.append(f[voz])
        f0[loc] = float(np.nanmedian(np.concatenate(valores)))
    return pd.Series(f0, name="F0_mediana_Hz")


def categorias_acerto(y: np.ndarray, te: np.ndarray, pred_mfcc: np.ndarray, pred_yam: np.ndarray) -> pd.DataFrame:
    ok_m, ok_y = pred_mfcc == y[te], pred_yam == y[te]
    cat = np.select([ok_m & ok_y, ok_m & ~ok_y, ~ok_m & ok_y],
                    ["os dois acertam", "só MFCC acerta", "só YAMNet acerta"], "os dois erram")
    return pd.DataFrame({"idx": te, "verdadeiro": y[te], "pred_MFCC": pred_mfcc,
                         "pred_YAMNet": pred_yam, "categoria": cat})


def exportar_exemplos(meta: pd.DataFrame, exemplos: pd.DataFrame, rt60: float,
                      pasta: str | Path = "data/fallas") -> pd.DataFrame:
    """Grava <categoria>/<arquivo>_limpo.wav e _rt<RT60>.wav. Devolve a tabela com os caminhos."""
    pasta = Path(pasta)
    linhas = []
    for _, ex in exemplos.iterrows():
        i = int(ex["idx"])
        y = carregar_audio(meta.loc[i, "caminho"])
        yr = reverberar(y, rt60, semente=semente_arquivo(i, rt60))
        sub = pasta / ex["categoria"].replace(" ", "_")
        sub.mkdir(parents=True, exist_ok=True)
        base = Path(meta.loc[i, "arquivo"]).stem
        p_limpo, p_rev = sub / f"{base}_limpo.wav", sub / f"{base}_rt{rt60:.1f}.wav"
        sf.write(p_limpo, y, SR)
        sf.write(p_rev, yr / max(1.0, np.abs(yr).max()), SR)
        linhas.append({**ex.to_dict(), "emocao": meta.loc[i, "emocao"],
                       "wav_limpo": str(p_limpo), "wav_reverb": str(p_rev)})
    return pd.DataFrame(linhas)


def deslocamento_mfcc(X_limpo: np.ndarray, X_reverb: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Δ_c = média(MFCC_rev − MFCC_limpo) por coeficiente, e quanto dele é comum
    a todos os locutores (razão entre a variância do Δ entre locutores e o Δ médio)."""
    d = X_reverb - X_limpo
    por_loc = pd.DataFrame(d).groupby(y).mean()
    return pd.DataFrame({"delta_medio": d.mean(0),
                         "desvio_entre_locutores": por_loc.std(0).to_numpy(),
                         "desvio_entre_arquivos_limpos": X_limpo.std(0)})


def geometria(X_limpo: np.ndarray, X_rev: np.ndarray, y: np.ndarray, tr, te) -> dict:
    """No espaço z-score (média/desvio do treino limpo), compara:
      • desloc   : ‖média(rev) − média(limpo)‖ nos arquivos de teste — o quanto a
                   reverberação empurra TODOS os pontos numa direção comum;
      • entre    : distância média entre centróides de locutores (treino limpo);
      • dentro   : distância média de um ponto limpo ao centróide do seu locutor.
    Se desloc ≳ entre, o kNN passa a decidir pela direção do deslocamento e não
    pela identidade do locutor."""
    mu, sd = X_limpo[tr].mean(0), X_limpo[tr].std(0) + 1e-8
    Z0, Zr = (X_limpo - mu) / sd, (X_rev - mu) / sd
    locs = np.unique(y)
    cent = np.stack([Z0[tr][y[tr] == l].mean(0) for l in locs])
    entre = np.mean([np.linalg.norm(cent[i] - cent[j])
                     for i in range(len(locs)) for j in range(i + 1, len(locs))])
    dentro = np.mean([np.linalg.norm(Z0[tr][y[tr] == l] - cent[i], axis=1).mean()
                      for i, l in enumerate(locs)])
    desloc = np.linalg.norm(Zr[te].mean(0) - Z0[te].mean(0))
    return {"deslocamento": desloc, "dist_entre_locutores": entre,
            "dist_dentro_locutor": dentro, "razao_desloc_entre": desloc / entre}


CLASSES_SALA = ["Speech", "Reverberation", "Echo", "Inside, small room",
                "Inside, large room or hall", "Silence"]


def scores_yamnet(meta: pd.DataFrame, niveis, n: int = 200, semente: int = 0,
                  classes=CLASSES_SALA) -> pd.DataFrame:
    """Score médio (saída de classificação do YAMNet, 521 classes do AudioSet)
    de algumas classes relacionadas à sala, em n arquivos, para cada RT60."""
    import csv
    import tensorflow as tf
    from src.features import _modelo_yamnet
    modelo = _modelo_yamnet()
    nomes = [r["display_name"] for r in csv.DictReader(
        tf.io.gfile.GFile(modelo.class_map_path().numpy().decode()))]
    idx = [nomes.index(c) for c in classes]
    rng = np.random.default_rng(semente)
    amostra = rng.choice(len(meta), size=n, replace=False)
    linhas = []
    for rt in niveis:
        acc = np.zeros(len(classes))
        for i in amostra:
            yr = reverberar(carregar_audio(meta.loc[i, "caminho"]), rt, semente=semente_arquivo(i, rt))
            scores, _, _ = modelo(yr)
            acc += scores.numpy().mean(0)[idx]
        linhas.append({"rt60": rt, **dict(zip(classes, acc / n))})
    return pd.DataFrame(linhas).set_index("rt60")
