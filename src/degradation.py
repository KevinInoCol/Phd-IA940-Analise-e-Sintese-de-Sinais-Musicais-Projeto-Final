"""Degradação: reverberação de sala por convolução com uma resposta ao impulso
(RI) sintética.

Modelo da RI (amostra n, taxa fs, tempo t = n / fs):

    h[n] = δ[n] + g · b[n] · exp(-6,908 · n / (RT60 · fs)),   n ≥ n0

    • δ[n]  : som direto (amplitude 1, atraso 0).
    • b[n]  : ruído branco gaussiano, média 0, variância 1 (campo difuso:
              muitas reflexões de fase aleatória).
    • exp(…): envoltória de amplitude. Como 20·log10(e^{-6,908}) = -60 dB,
              a energia cai 60 dB em t = RT60 — a definição de RT60.
              (6,908 = ln(1000); a amplitude cai 1000× = 60 dB.)
    • n0    : pré-atraso de 2,5 ms antes da primeira reflexão.
    • g     : ganho da cauda, escolhido pela relação direto/reverberante (DRR).

Energia da cauda (soma geométrica, ignorando n0):
    E_cauda ≈ g² · Σ e^{-13,816 n / (RT60·fs)} ≈ g² · RT60 · fs / 13,816
Com E_direto = 1:
    DRR(RT60) = 10·log10(13,816 / (g² · RT60 · fs))
Fixamos g de modo que DRR = 0 dB em RT60 = 0,5 s (RT60_REF). Então
    DRR(RT60) = 10·log10(RT60_REF / RT60)   →  +7 dB em 0,1 s, -6 dB em 2 s.
Ou seja: mesma sala e mesma distância fonte-microfone; só a absorção das
paredes muda. Quanto maior o RT60, mais energia reverberante (como na
fórmula de Sabine, em que a energia reverberante é ∝ RT60).

Sinal reverberado:
    x_r = (x * h)  (convolução completa — a cauda é mantida; o áudio fica
                    RT60 segundos mais longo, como numa gravação real na sala)
    x_r ← x_r · RMS(x) / RMS(x_r)   (mesma energia do original, para que a
                                      reverberação não vire também uma
                                      mudança de volume)

Cada arquivo recebe uma realização própria de b[n] (semente = índice do
arquivo + nível), de modo que não existe uma RI única compartilhada por todos
os áudios de teste.
"""

import numpy as np
from scipy.signal import fftconvolve

SR = 16000
RT60_REF = 0.5          # s — RT60 em que DRR = 0 dB
PRE_ATRASO = 0.0025     # s — atraso da primeira reflexão
DECAIMENTO_60DB = np.log(1000.0)   # 6,908: amplitude cai 60 dB em t = RT60
NIVEIS_RT60 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def ganho_cauda(sr: int = SR, rt60_ref: float = RT60_REF) -> float:
    """g tal que DRR(rt60_ref) = 0 dB."""
    return float(np.sqrt(2 * DECAIMENTO_60DB / (rt60_ref * sr)))


def drr_db(rt60: float, rt60_ref: float = RT60_REF) -> float:
    """Relação direto/reverberante teórica do modelo, em dB."""
    return float(10 * np.log10(rt60_ref / rt60)) if rt60 > 0 else np.inf


def resposta_impulso(rt60: float, sr: int = SR, semente: int | None = None) -> np.ndarray:
    """RI sintética com o RT60 pedido. rt60 = 0 → impulso unitário (sala seca)."""
    if rt60 <= 0:
        return np.ones(1, dtype=np.float32)
    rng = np.random.default_rng(semente)
    n_total = int(np.ceil(rt60 * sr))          # até -60 dB
    n = np.arange(n_total)
    envoltoria = np.exp(-DECAIMENTO_60DB * n / (rt60 * sr))
    h = ganho_cauda(sr) * rng.standard_normal(n_total) * envoltoria
    h[: int(PRE_ATRASO * sr)] = 0.0             # antes da 1ª reflexão só há o som direto
    h[0] = 1.0                                  # som direto
    return h.astype(np.float32)


def rt60_medido(h: np.ndarray, sr: int = SR) -> float:
    """Verificação: RT60 pela integral de Schroeder (ajuste de -5 a -25 dB, T20×3)."""
    edc = np.cumsum(h[::-1] ** 2)[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-20)
    i5, i25 = np.argmax(edc_db <= -5), np.argmax(edc_db <= -25)
    t = np.arange(i5, i25) / sr
    inclinacao = np.polyfit(t, edc_db[i5:i25], 1)[0]      # dB/s
    return float(-60 / inclinacao)


def reverberar(y: np.ndarray, rt60: float, sr: int = SR, semente: int | None = None,
               cortar_cauda: bool = False) -> np.ndarray:
    """Convolui y com a RI e devolve o sinal com a mesma energia RMS de y.

    cortar_cauda=True mantém só as len(y) primeiras amostras (ablação: o áudio
    não fica mais longo; some a cauda depois da última palavra)."""
    if rt60 <= 0:
        return y.astype(np.float32)
    h = resposta_impulso(rt60, sr, semente)
    yr = fftconvolve(y, h, mode="full")
    if cortar_cauda:
        yr = yr[: len(y)]
    yr *= np.sqrt(np.mean(y ** 2) / (np.mean(yr ** 2) + 1e-20))
    return yr.astype(np.float32)


def semente_arquivo(indice: int, rt60: float) -> int:
    """Semente reprodutível e diferente para cada (arquivo, nível)."""
    return int(indice) * 1000 + int(round(rt60 * 100))
