# IA940 · Projeto Final — Identificação de locutor sob reverberação

**Análise e Síntese de Sinais Musicais (IA940/EG940) · UNICAMP · 2º semestre de 2026 · Prof. Bruno Masiero**

Grupo: Kevin Inofuente Colque, Luiza Viana Souza e Guilherme Maia Masetto · Trilha **Voz**

## Objetivo

Comparar uma representação clássica de áudio com uma representação aprendida e medir como cada uma se degrada quando o som é alterado de forma controlada.

- **Baseline clássico:** vetor de MFCCs médios (20 coeficientes).
- **Embedding aprendido:** YAMNet, penúltima camada (vetor de 1024 dimensões).
- **Tarefa:** identificar o **locutor** com kNN (distância euclidiana, z-score estimado só no treino, mesmo k e mesmas divisões para as duas representações).
- **Degradação:** reverberação da sala, **RT60 de 0 a 2 s**, por convolução com resposta ao impulso sintética, em ≥ 5 níveis.
- **Resultado central:** um único gráfico de acurácia × RT60, com uma curva por representação.

## Dados

[Emotional Speech Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data) (Zhou et al., 2022). Usamos só os **10 locutores de inglês (0011–0020)**: 5 emoções × 350 frases = 1 750 áudios por locutor, 17 500 no total, 16 kHz. Acaso = 10 %.

O rótulo é o locutor, tirado do **nome do arquivo** (`0011_000351.wav` → `0011`). O nome da pasta no ESD é a emoção, não o locutor.

Os áudios não vão para o repositório (estão no `.gitignore`).

## Estrutura

```
src/
├── dataset.py    # indexa o ESD: caminho, locutor, emoção, frase, split oficial
├── features.py   # mfcc_medio(y) -> (20,) e yamnet_embedding(y) -> (1024,)
└── extract.py    # extrai tudo e salva X_mfcc.npy, X_yamnet.npy e meta.csv
```

As funções de `features.py` recebem a **forma de onda** e não o caminho do arquivo. Assim, na etapa de degradação basta fazer `features(reverberar(y, rt60))`.

## Como rodar

O TensorFlow ainda não suporta Python 3.14, então usamos **Python 3.11**.

```bash
uv venv -p 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# teste rápido com ~50 áudios
python -m src.extract --esd /caminho/para/ESD --limite 50

# extração completa
python -m src.extract --esd /caminho/para/ESD --saida data/features
```

Saída em `data/features/`: `X_mfcc.npy` (N × 20), `X_yamnet.npy` (N × 1024) e `meta.csv`. A linha *i* dos três arquivos corresponde ao mesmo áudio.

## Passo a passo

| # | Passo | Entrega | Estado |
|---|---|---|---|
| 1 | Indexar o ESD (locutores 0011–0020) | 06/11 | Escrito, ainda não testado com o ESD |
| 2 | Extrair MFCC médio e YAMNet | 06/11 | Escrito, ainda não testado com o ESD |
| 3 | Dividir treino/teste (5 divisões, cada arquivo em um só lado) | 06/11 | Pendente |
| 4 | Normalizar com z-score (média e desvio só do treino) | 06/11 | Pendente |
| 5 | Classificar com kNN (mesmo k para as duas representações) | 06/11 | Pendente |
| 6 | Medir a acurácia (média ± desvio) | 06/11 | Pendente |
| 7 | Reverberar os áudios, RT60 de 0 a 2 s, ≥ 5 níveis | 13/11 | Pendente |
| 8 | Repetir os passos 2–6 em cada nível → curvas | 13/11 | Pendente |
| 9 | Variabilidade: a diferença é maior que a dispersão? | 27/11 | Pendente |
| 10 | Análise de falhas com exemplos audíveis | 27/11 | Pendente |
| 11 | Notebook, apresentação e defesa oral | 04/12 | Pendente |

## Pontos em aberto

- Reverberação só no teste ou no treino e no teste?
- Cortar ou não a cauda da reverberação (a convolução alonga o áudio em até ~RT60)?
- Confirmar a regra do split oficial do ESD (20 evaluation / 30 test / 300 train por locutor × emoção).
