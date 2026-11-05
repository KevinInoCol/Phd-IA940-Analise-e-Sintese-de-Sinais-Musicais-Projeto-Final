"""As duas representações do projeto, calculadas a partir da FORMA DE ONDA.

Recebem `y` (e não o caminho do arquivo) para que a reverberação possa ser
aplicada antes: features(reverberar(y, rt60)).
"""

import librosa
import numpy as np

SR = 16000          # YAMNet exige 16 kHz mono; o ESD já vem a 16 kHz
N_MFCC = 20
N_FFT = 512         # janela de 25 ms (400 amostras) dentro de uma FFT de 512
WIN_LENGTH = 400
HOP_LENGTH = 160    # salto de 10 ms
N_MELS = 40
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

_yamnet = None


def carregar_audio(caminho: str) -> np.ndarray:
    """Lê o arquivo como float32 mono a 16 kHz, em [-1, 1]."""
    y, _ = librosa.load(caminho, sr=SR, mono=True)
    return y.astype(np.float32)


def mfcc_medio(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Baseline clássico: média temporal dos MFCC -> vetor (N_MFCC,)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
                                n_mels=N_MELS)
    return mfcc.mean(axis=1).astype(np.float32)


def _modelo_yamnet():
    """Carrega o YAMNet uma única vez (import tardio: TensorFlow é pesado)."""
    global _yamnet
    if _yamnet is None:
        import tensorflow_hub as hub
        _yamnet = hub.load(YAMNET_URL)
    return _yamnet


def yamnet_embedding(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Embedding aprendido: média dos vetores de 1024-d (penúltima camada).

    O YAMNet devolve um embedding por janela de 0,96 s com salto de 0,48 s;
    tiramos a média sobre as janelas -> vetor (1024,).
    """
    if sr != SR:
        raise ValueError(f"YAMNet precisa de {SR} Hz, recebeu {sr} Hz")
    _, embeddings, _ = _modelo_yamnet()(y.astype(np.float32))
    return embeddings.numpy().mean(axis=0).astype(np.float32)
