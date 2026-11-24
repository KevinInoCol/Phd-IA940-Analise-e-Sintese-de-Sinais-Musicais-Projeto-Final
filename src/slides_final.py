"""Gera a apresentação final (Final_PT.html + ia940-final/ com áudios e figuras).

Todos os números vêm de data/resultados/*.csv (gerados pelo notebook) ou são
recalculados aqui; os áudios são gerados com src.degradation.

Uso (da pasta Proyecto, com o .venv):
    PYTHONPATH=. python -m src.slides_final
"""

import json
import shutil
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from src.degradation import NIVEIS_RT60, drr_db, resposta_impulso, reverberar, semente_arquivo
from src.features import SR, carregar_audio

SAIDA = Path("ia940-final")
FONTE = Path("Final_PT.html")
MFCC_C, YAM_C = "var(--mfcc)", "var(--yam)"
N_SLIDES = 16
T_MAX_ESPEC = 5.2   # eixo de tempo fixo do espectrograma ao vivo (o nível 2,0 s dura 5,06 s)

# ── Modal "espectrograma em tempo real" (slide 8) ─────────────────────────────
# Web Audio: o WAV é decodificado, tocado por um AnalyserNode (FFT de 2048 pontos)
# e cada quadro de animação pinta uma coluna nova do espectrograma (0–8 kHz,
# eixo de tempo fixo em T_MAX_ESPEC para que as caudas sejam comparáveis).
# À direita, o espectro médio (LTAS) acumulado durante a reprodução; o do seco
# fica em cinza para comparação — mostra que a cauda quase não muda o espectro
# médio, mesmo borrando o espectrograma.
# ── Equações: KaTeX local (ia940-final/katex/, funciona sem internet) ──────────
KATEX_HEAD = r"""<link rel="stylesheet" href="katex/katex.min.css">
<script defer src="katex/katex.min.js"></script>
<script defer src="katex/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters: [{left: '\\[', right: '\\]', display: true}, {left: '\\(', right: '\\)', display: false}]})"></script>"""

RT = r"\mathrm{RT}_{60}"
EQ = {
    "ri": r"\[ h[n] \;=\; \underbrace{\delta[n]}_{\text{som direto}} \;+\; \underbrace{g\; b[n]\; e^{-6{,}908\,n/(" + RT + r"\,f_s)}}_{\text{cauda reverberante}} \qquad\qquad x_r[n] = (x * h)[n] \]",
    "h": r"\(h[n]\)",
    "delta": r"\(\delta[n]\)",
    "b": r"\(b[n] \sim \mathcal{N}(0,1)\)",
    "ln": r"\(6{,}908 = \ln 1000\)",
    "rt_def": r"\(e^{-6{,}908} = 10^{-3}\)",
    "g": r"\(g\)",
    "drr_rt": r"\(\mathrm{DRR} = 10\log_{10}\!\left(0{,}5\,\text{s}/" + RT + r"\right)\)",
    "drr_def": r"\[ \mathrm{DRR} \;=\; 10\,\log_{10}\!\left(\frac{E_{\text{direto}}}{E_{\text{reverberante}}}\right)\ \text{dB} \]",
    "e_dir": r"\(E_{\text{direto}} = 1\)",
    "e_rev": r"\[ E_{\text{rev}} \;=\; g^2 \sum_{n} e^{-13{,}8\,n/(" + RT + r"\,f_s)} \;\approx\; \frac{g^2\, " + RT + r"\, f_s}{13{,}8} \quad\Longrightarrow\quad \mathrm{DRR} = 10\log_{10}\frac{0{,}5\ \text{s}}{" + RT + r"} \]",
    "ltas": r"\[ |X_r(f)|^2 \;\approx\; |X(f)|^2 \cdot \mathbb{E}\,|H(f)|^2 \]",
    "EH": r"\(\mathbb{E}\,|H(f)|^2\)",
    "c0": r"\(c_0\)", "c1": r"\(c_1\)",
}

MODAL_CSS = """
  .pipe { background: var(--card); border: 1px solid var(--line); border-left: 5px solid var(--c); border-radius: 12px; padding: 10px 14px; margin-bottom: 10px;
    display: grid; grid-template-columns: 104px 1fr; grid-template-rows: auto auto; gap: 6px 12px; }
  .pnome { grid-row: 1 / 3; display: flex; flex-direction: column; justify-content: center; }
  .pnome b { color: var(--c); font-size: 21px; line-height: 1.1; } .pnome small { font-size: 13px; color: var(--muted); font-weight: 700; margin-top: 4px; }
  .etps { display: flex; align-items: stretch; gap: 4px; }
  .etp { flex: 1; background: #faf9f6; border: 1px solid var(--line); border-radius: 9px; padding: 6px 9px; display: flex; flex-direction: column; }
  .etp h4 { margin: 0; font-size: 14.5px; } .etp p { margin: 2px 0 4px; font-size: 12.5px; line-height: 1.3; color: #3c4043; flex: 1; }
  .etp span { font: 700 12px/1 ui-monospace, Menlo, monospace; color: var(--c); background: #fff; border: 1px solid var(--line); border-radius: 99px; padding: 3px 7px; align-self: flex-start; }
  .etp.fim { background: var(--c); border-color: var(--c); } .etp.fim h4, .etp.fim p { color: #fff; } .etp.fim span { color: var(--c); border-color: #fff; }
  .seta { align-self: center; color: #adb5bd; font-size: 18px; font-weight: 800; }
  .gp { display: flex; gap: 20px; font-size: 14px; color: #3c4043; } .gp p { margin: 0; flex: 1; }
  .passos { display: flex; flex-direction: column; gap: 8px; }
  .passo { display: flex; gap: 12px; align-items: flex-start; background: var(--card); border: 1px solid var(--line); border-radius: 11px; padding: 8px 14px; }
  .passo > b { flex: 0 0 28px; height: 28px; border-radius: 50%; background: var(--accent); color: #fff; display: grid; place-items: center; font-size: 15px; margin-top: 2px; }
  .passo h4 { margin: 0; font-size: 16px; } .passo p { margin: 1px 0 0; font-size: 14.5px; line-height: 1.35; color: #3c4043; }
  .katex-display { margin: .45em 0; } .katex { font-size: 1.12em; }
  .sigl { display: flex; gap: 10px; margin: 0 0 12px; font-size: 14px; color: #3c4043; }
  .sigl > span { background: #f1efe8; border-radius: 99px; padding: 5px 14px; white-space: nowrap; }
  .sigl > span > b { color: var(--accent); margin-right: 6px; } .sigl .katex { font-size: 1.05em; }
  .casos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .caso { border-radius: 10px; padding: 9px 12px; display: flex; flex-direction: column; gap: 2px; border-top: 4px solid; }
  .caso b { font-size: 20px; } .caso span { font-size: 14.5px; color: #3c4043; line-height: 1.3; } .caso em { font-size: 14.5px; font-weight: 700; font-style: normal; }
  .caso.pos { background: var(--ok-soft); border-color: var(--ok); } .caso.pos b, .caso.pos em { color: var(--ok); }
  .caso.zer { background: #f1efe8; border-color: #80868b; } .caso.zer b, .caso.zer em { color: #5f6368; }
  .caso.neg { background: var(--bad-soft); border-color: var(--bad); } .caso.neg b, .caso.neg em { color: var(--bad); }
  .aud { grid-template-columns: 56px 1fr 36px !important; }
  .spec-btn { grid-column: 3; grid-row: 1 / 3; width: 34px; height: 34px; border-radius: 9px; border: 1px solid var(--line);
    background: var(--accent-soft); color: var(--accent); cursor: pointer; display: grid; place-items: center; padding: 0; }
  .spec-btn:hover { background: var(--accent); color: #fff; }
  .spec-big { position: absolute; top: 34px; right: 56px; border: 0; border-radius: 99px; background: var(--accent); color: #fff;
    font: 700 14px/1 inherit; padding: 9px 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,.25); }
  .modal { position: fixed; inset: 0; z-index: 50; background: rgba(10,11,14,.82); display: grid; place-items: center; }
  .modal[hidden] { display: none; }
  .mbox { width: min(1240px, 96vw); background: #101114; color: #e8eaed; border-radius: 16px; padding: 20px 26px 18px;
    box-shadow: 0 20px 60px rgba(0,0,0,.5); font-size: 15px; }
  .mhead { display: flex; justify-content: space-between; align-items: flex-start; }
  .mhead .kick { color: #91a7ff; }
  .mhead h3 { margin: 2px 0 0; font-size: 30px; }
  .mhead h3 small { font-size: 17px; color: #9aa0a6; font-weight: 600; margin-left: 10px; }
  .mx { background: none; border: 0; color: #9aa0a6; font-size: 34px; line-height: 1; cursor: pointer; }
  .mlev { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 14px; }
  .mlev button { border: 1px solid #3c4043; background: #1c1d21; color: #e8eaed; border-radius: 99px; padding: 6px 13px;
    font: 600 14px/1 inherit; cursor: pointer; }
  .mlev button.on { background: #91a7ff; border-color: #91a7ff; color: #101114; }
  .mplot { display: grid; grid-template-columns: 46px 1fr 210px; gap: 0 12px; }
  .yax { position: relative; font-size: 12px; color: #9aa0a6; }
  .yax span { position: absolute; right: 2px; transform: translateY(-50%); }
  .cvw, .ltw { position: relative; line-height: 0; }
  .cvw canvas, .ltw canvas { width: 100%; height: auto; border-radius: 6px; background: #000; }
  .ltw canvas { position: absolute; inset: 0; height: 100%; }
  .ph { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; opacity: .8; left: 0; display: none; }
  .xax { position: relative; height: 20px; font-size: 12px; color: #9aa0a6; }
  .xax span { position: absolute; transform: translateX(-50%); top: 4px; }
  .lttit { font-size: 12px; color: #9aa0a6; text-align: center; margin-top: 4px; }
  .mfoot { display: flex; align-items: center; gap: 18px; margin-top: 8px; font-size: 13px; color: #9aa0a6; }
  .mfoot button { border: 0; border-radius: 99px; background: #91a7ff; color: #101114; font: 800 15px/1 inherit; padding: 10px 20px; cursor: pointer; min-width: 110px; }
  .mfoot i { display: inline-block; width: 18px; border-top: 3px solid; vertical-align: middle; margin-right: 5px; }
"""

ICONE_ESPEC = ('<svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor" aria-hidden="true">'
               '<rect x="1" y="7" width="2.4" height="9" rx="1"/><rect x="5" y="3" width="2.4" height="13" rx="1"/>'
               '<rect x="9" y="9" width="2.4" height="7" rx="1"/><rect x="13" y="5" width="2.4" height="11" rx="1"/></svg>')

MODAL_HTML = """
<div id="spec" class="modal" hidden>
  <div class="mbox">
    <div class="mhead">
      <div><div class="kick">Espectrograma em tempo real · locutor 0013, Neutral</div><h3 id="spec-tit"></h3></div>
      <button class="mx" id="spec-x" title="Fechar (Esc)">×</button>
    </div>
    <div class="mlev" id="spec-lev"></div>
    <div class="mplot">
      <div class="yax" id="spec-y"></div>
      <div class="cvw"><canvas id="spec-cv" width="960" height="340"></canvas><div class="ph" id="spec-ph"></div></div>
      <div class="ltw"><canvas id="spec-lt" width="210" height="340"></canvas></div>
      <div></div><div class="xax" id="spec-xa"></div><div class="lttit">espectro médio (LTAS), dB</div>
    </div>
    <div class="mfoot">
      <button id="spec-play">▶ Tocar</button>
      <span><i style="border-color:#91a7ff"></i>nível atual · <i style="border-color:#80868b"></i>seco (toque o seco uma vez para comparar)</span>
      <span style="margin-left:auto">Esc fecha · espaço toca/para · ← → muda o nível</span>
    </div>
  </div>
</div>
"""

MODAL_JS = """
  // ── espectrograma em tempo real ──
  const SP = (() => {
    const NIV = __NIVEIS__, TMAX = __TMAX__, FMAX = 8000, DBMIN = -115, DBMAX = -30;
    const $ = id => document.getElementById(id);
    const modal = $('spec'), cv = $('spec-cv'), g = cv.getContext('2d'), lt = $('spec-lt'), gl = lt.getContext('2d');
    const W = cv.width, H = cv.height, btn = $('spec-play'), ph = $('spec-ph');
    const stops = [[0,0,0,4],[.13,28,16,68],[.25,79,18,123],[.38,129,37,129],[.5,181,54,122],
                   [.63,229,80,100],[.75,251,135,97],[.88,254,194,135],[1,252,253,191]];
    const LUT = Array.from({length: 256}, (_, i) => {
      const t = i / 255; let k = 0; while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
      const [a, b] = [stops[k], stops[k + 1]], u = (t - a[0]) / (b[0] - a[0]);
      return [1, 2, 3].map(c => Math.round(a[c] + u * (b[c] - a[c])));
    });
    let ac, an, data, rowBin, nb, src = null, t0 = 0, lastX = 0, raf, cur = 0, acc, nacc = 0;
    const buf = {}, ltas = {};
    // eixos
    $('spec-y').innerHTML = [0, 2, 4, 6, 8].map(k => `<span style="top:${100 - k / 8 * 100}%">${k ? k + ' kHz' : '0'}</span>`).join('');
    $('spec-xa').innerHTML = Array.from({length: Math.floor(TMAX) + 1}, (_, s) => `<span style="left:${s / TMAX * 100}%">${s} s</span>`).join('');
    $('spec-lev').innerHTML = NIV.map((n, k) => `<button data-k="${k}">${n.rotulo}</button>`).join('');
    $('spec-lev').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { escolher(+b.dataset.k); tocar(); } });
    $('spec-x').onclick = fechar;
    btn.onclick = () => src ? parar() : tocar();
    modal.addEventListener('click', e => { if (e.target === modal) fechar(); });

    function iniciar() {
      if (ac) return;
      ac = new (window.AudioContext || window.webkitAudioContext)();
      an = ac.createAnalyser(); an.fftSize = 2048; an.smoothingTimeConstant = 0;
      an.connect(ac.destination);
      data = new Float32Array(an.frequencyBinCount);
      const hzBin = ac.sampleRate / an.fftSize;
      nb = Math.ceil(FMAX / hzBin) + 1;
      rowBin = Array.from({length: H}, (_, y) => Math.min(nb - 1, Math.round((1 - (y + .5) / H) * FMAX / hzBin)));
    }
    async function carregar(k) {
      if (!buf[k]) buf[k] = await ac.decodeAudioData(await (await fetch(NIV[k].src)).arrayBuffer());
      return buf[k];
    }
    function limpar() {
      g.fillStyle = '#000'; g.fillRect(0, 0, W, H); lastX = 0;
      acc = new Float64Array(nb); nacc = 0; desenharLtas();
    }
    function escolher(k) {
      parar(); cur = (k + NIV.length) % NIV.length;
      $('spec-tit').innerHTML = NIV[cur].titulo + (NIV[cur].drr ? `<small>${NIV[cur].drr}</small>` : '');
      [...$('spec-lev').children].forEach((b, j) => b.classList.toggle('on', j === cur));
    }
    async function tocar() {
      iniciar(); ac.resume(); parar();
      let b;
      try { b = await carregar(cur); }
      catch (e) { g.fillStyle = '#fff'; g.font = '18px sans-serif'; g.fillText('Não foi possível carregar o áudio (abra pela URL publicada, não como arquivo local).', 20, 40); return; }
      limpar();
      src = ac.createBufferSource(); src.buffer = b; src.connect(an);
      src.onended = () => { src = null; terminar(); };
      t0 = ac.currentTime + 0.02; src.start(t0);
      btn.textContent = '■ Parar'; ph.style.display = 'block';
      raf = requestAnimationFrame(quadro);
    }
    function parar() {
      if (src) { src.onended = null; try { src.stop(); } catch (e) {} src = null; }
      terminar();
    }
    function terminar() {
      if (nacc > 10) ltas[cur] = Array.from(acc, v => 10 * Math.log10(v / nacc + 1e-20));
      cancelAnimationFrame(raf); btn.textContent = '▶ Tocar'; ph.style.display = 'none'; desenharLtas(); }
    function quadro() {
      if (!src) return;
      const t = ac.currentTime - t0;
      if (t >= 0) {
        an.getFloatFrequencyData(data);
        const x = Math.min(W, Math.round(t / TMAX * W));
        if (x > lastX) {
          const w = x - lastX, img = g.createImageData(w, H);
          for (let y = 0; y < H; y++) {
            const v = (data[rowBin[y]] - DBMIN) / (DBMAX - DBMIN);
            const c = LUT[Math.max(0, Math.min(255, Math.round(v * 255)))];
            for (let i = 0; i < w; i++) { const p = 4 * (y * w + i); img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255; }
          }
          g.putImageData(img, lastX, 0); lastX = x;
        }
        for (let k = 0; k < nb; k++) acc[k] += Math.pow(10, data[k] / 10);
        nacc++;
        ph.style.left = (100 * x / W) + '%';
        if (nacc % 6 === 0) desenharLtas();
      }
      raf = requestAnimationFrame(quadro);
    }
    function curva(db, cor, largura) {
      gl.strokeStyle = cor; gl.lineWidth = largura; gl.beginPath();
      for (let y = 0; y < H; y++) {
        const x = (db[rowBin[y]] - DBMIN) / (DBMAX - DBMIN) * lt.width;
        y ? gl.lineTo(x, y) : gl.moveTo(x, y);
      }
      gl.stroke();
    }
    function desenharLtas() {
      gl.fillStyle = '#000'; gl.fillRect(0, 0, lt.width, H);
      gl.strokeStyle = '#26282c'; gl.lineWidth = 1;
      for (let d = DBMIN + 15; d < DBMAX; d += 20) { const x = (d - DBMIN) / (DBMAX - DBMIN) * lt.width; gl.beginPath(); gl.moveTo(x, 0); gl.lineTo(x, H); gl.stroke(); }
      if (!rowBin) return;
      if (ltas[0] && cur !== 0) curva(ltas[0], '#80868b', 2);
      const atual = nacc ? Array.from(acc, v => 10 * Math.log10(v / nacc + 1e-20)) : ltas[cur];
      if (atual) curva(atual, '#91a7ff', 2.5);
    }
    function abrir(k, auto) { modal.hidden = false; escolher(k); if (!ltas[cur]) { g.fillStyle = '#000'; g.fillRect(0, 0, W, H); desenharLtas(); } if (auto) tocar(); }
    function fechar() { parar(); modal.hidden = true; }
    function tecla(e) {
      if (modal.hidden) return false;
      if (e.key === 'Escape') fechar();
      else if (e.key === ' ') { e.preventDefault(); src ? parar() : tocar(); }
      else if (e.key === 'ArrowRight') { escolher(cur + 1); tocar(); }
      else if (e.key === 'ArrowLeft') { escolher(cur - 1); tocar(); }
      return true;
    }
    document.addEventListener('click', e => { const b = e.target.closest('[data-spec]'); if (b) abrir(+b.dataset.spec, true); });
    return { tecla, aberto: () => !modal.hidden };
  })();
"""


def fmt(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


# ------------------------------------------------------------------ gráfico SVG
def grafico(series, W=620, H=380, xmax=2.0, titulo_x="RT60 (s)", acaso=True, legenda=True):
    L, R, T, B = 52, 16, 14, 46
    pw, ph = W - L - R, H - T - B
    x = lambda v: L + v / xmax * pw
    y = lambda v: T + ph - v / 100 * ph
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for v in range(0, 101, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--line)"/>'
                 f'<text x="{L-8}" y="{y(v)+5:.1f}" text-anchor="end">{v}%</text>')
    for v in np.arange(0, xmax + 1e-9, 0.5):
        o.append(f'<text x="{x(v):.1f}" y="{H-24}" text-anchor="middle">{fmt(v)}</text>')
    o.append(f'<text x="{L+pw/2}" y="{H-4}" text-anchor="middle">{titulo_x}</text>')
    if acaso:
        o.append(f'<line x1="{x(0)}" x2="{x(xmax)}" y1="{y(10):.1f}" y2="{y(10):.1f}" stroke="var(--muted)" stroke-dasharray="6 5"/>'
                 f'<text x="{x(xmax)-4}" y="{y(10)-7:.1f}" text-anchor="end" fill="var(--muted)">acaso 10%</text>')
    for s in series:
        xs, m = np.asarray(s["x"]), np.asarray(s["m"])
        if s.get("sd") is not None:
            sd = np.asarray(s["sd"])
            pts = [f"{x(a):.1f},{y(b+c):.1f}" for a, b, c in zip(xs, m, sd)] + \
                  [f"{x(a):.1f},{y(b-c):.1f}" for a, b, c in zip(xs[::-1], m[::-1], sd[::-1])]
            o.append(f'<polygon points="{" ".join(pts)}" fill="{s["c"]}" opacity=".22"/>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(a):.1f} {y(b):.1f}" for i, (a, b) in enumerate(zip(xs, m)))
        dash = ' stroke-dasharray="9 7"' if s.get("dash") else ""
        o.append(f'<path d="{d}" fill="none" stroke="{s["c"]}" stroke-width="{s.get("w", 3.5)}"{dash} stroke-linecap="round" opacity="{s.get("op", 1)}"/>')
        if s.get("pts", True):
            o += [f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="5" fill="{s["c"]}"/>' for a, b in zip(xs, m)]
        if s.get("rotulo"):
            a, b = s.get("rotulo_xy", (xs[-1], m[-1]))
            o.append(f'<text x="{x(a):.1f}" y="{y(b):.1f}" fill="{s["c"]}" font-weight="700" text-anchor="end">{s["rotulo"]}</text>')
    o.append("</svg>")
    return "\n".join(o)


# ------------------------------------------------------------------ dados
def resumo(df, cond):
    t = df[df.condicao == cond].groupby(["representacao", "rt60"]).acuracia.agg(["mean", "std"]) * 100
    return {rep: t.loc[rep] for rep in ("MFCC", "YAMNet")}


def gerar_midia(meta):
    shutil.rmtree(SAIDA / "audios", ignore_errors=True)
    (SAIDA / "audios").mkdir(parents=True, exist_ok=True)
    (SAIDA / "img").mkdir(parents=True, exist_ok=True)
    i_ex = int(meta.index[meta.arquivo == "0013_000020.wav"][0])
    y0 = carregar_audio(meta.caminho[i_ex])
    for rt in NIVEIS_RT60:
        yr = reverberar(y0, rt, semente=semente_arquivo(i_ex, rt))
        sf.write(SAIDA / "audios" / f"nivel_{rt:.1f}.wav", yr / max(1, np.abs(yr).max()), SR)
    # espectrogramas
    fig, ax = plt.subplots(1, 3, figsize=(12, 2.35), sharey=True)
    for a, rt in zip(ax, [0.0, 0.5, 2.0]):
        yr = reverberar(y0, rt, semente=semente_arquivo(i_ex, rt))
        S = librosa.amplitude_to_db(np.abs(librosa.stft(yr, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=160, x_axis="time", y_axis="hz", ax=a, vmin=-70, cmap="magma")
        a.set_title(f"RT60 = {fmt(rt)} s", fontsize=12); a.set_xlabel("")
    ax[1].set_ylabel(""); ax[2].set_ylabel("")
    plt.tight_layout(); plt.savefig(SAIDA / "img" / "espectrogramas.png", dpi=130, facecolor="white"); plt.close()
    # RI
    h = resposta_impulso(0.5, semente=1)
    fig, a = plt.subplots(figsize=(5.4, 2.4))
    a.plot(np.arange(len(h)) / SR * 1000, h, lw=.5, c="#3b5bdb"); a.set_ylim(-.12, .12)
    a.annotate(r"som direto $\delta[n] = 1$", (2, .115), (90, .09), arrowprops=dict(arrowstyle="->"), fontsize=10)
    t = np.arange(len(h)) / SR
    a.plot(t * 1000, 0.04 * np.exp(-6.908 * t / 0.5) * 2.2, c="#d9480f", lw=1.5, ls="--", label=r"envoltória $e^{-6{,}908\,t/\mathrm{RT}_{60}}$")
    a.set(xlabel="t (ms)", title=r"Resposta ao impulso sintética, $\mathrm{RT}_{60}$ = 0,5 s"); a.legend(fontsize=9, loc="lower right"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(SAIDA / "img" / "ri.png", dpi=150, facecolor="white"); plt.close()
    # exemplos de falha
    exemplos = []
    for p in sorted(Path("data/fallas").glob("*/*_limpo.wav")):
        cat = p.parent.name
        asc = cat.replace("ó", "o").replace("ó", "o")       # nomes ASCII nas URLs
        destino = SAIDA / "audios" / f"{asc}__{p.name}"
        shutil.copy(p, destino)
        rev = next(p.parent.glob(p.name.replace("_limpo.wav", "_rt*.wav")))
        shutil.copy(rev, SAIDA / "audios" / f"{asc}__{rev.name}")
        exemplos.append((cat, p.name.replace("_limpo.wav", ""), destino.name, f"{asc}__{rev.name}"))
    return exemplos


# ------------------------------------------------------------------ HTML
def grafico_drr(W=440, H=380):
    """Barras horizontais divergentes do DRR por nível (o seco não tem cauda: DRR = ∞)."""
    x0, x1, top, row = 70, W - 50, 14, (H - 40) / len(NIVEIS_RT60)
    esc = lambda d: x0 + (d + 8) / 16 * (x1 - x0)
    z = esc(0)
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}">']
    for d in (-6, -3, 0, 3, 6):
        p.append(f'<line x1="{esc(d):.1f}" x2="{esc(d):.1f}" y1="{top}" y2="{H-28}" stroke="#e3e1da" stroke-width="{2 if d == 0 else 1}"/>'
                 f'<text x="{esc(d):.1f}" y="{H-10}" text-anchor="middle">{d:+d} dB</text>'.replace("+0 dB", "0 dB"))
    for k, rt in enumerate(NIVEIS_RT60):
        y = top + k * row + row / 2
        rot = "seco" if rt == 0 else f"{fmt(rt)} s"
        p.append(f'<text x="{x0-10}" y="{y+5:.1f}" text-anchor="end" style="font-weight:700;fill:#202124">{rot}</text>')
        if rt == 0:
            p.append(f'<text x="{z+8:.1f}" y="{y+5:.1f}">∞ (sem cauda)</text>')
            continue
        d = drr_db(rt)
        cor = "var(--ok)" if d > 0.05 else ("var(--bad)" if d < -0.05 else "var(--muted)")
        xa, xb = sorted((z, esc(d)))
        p.append(f'<rect x="{xa:.1f}" y="{y-row*0.32:.1f}" width="{max(xb-xa, 2):.1f}" height="{row*0.64:.1f}" rx="3" fill="{cor}" opacity=".85"/>')
        tx, anc = (xb + 6, "start") if d >= 0 else (xa - 6, "end")
        p.append(f'<text x="{tx:.1f}" y="{y+5:.1f}" text-anchor="{anc}" style="fill:{cor};font-weight:700">{"0" if abs(d) < 0.05 else f"{d:+.1f}".replace(".", ",")}</text>')
    p.append("</svg>")
    return "".join(p)


def pipeline(cor, nome, dim, passos, guarda, perde):
    """Linha de etapas (caixas com seta) do áudio até o vetor final."""
    caixas = []
    for k, (tit, txt, forma) in enumerate(passos):
        ult = k == len(passos) - 1
        caixas.append(f'<div class="etp{" fim" if ult else ""}" style="--c:{cor}"><h4>{tit}</h4><p>{txt}</p><span>{forma}</span></div>')
        if not ult:
            caixas.append('<div class="seta">→</div>')
    return (f'<div class="pipe" style="--c:{cor}"><div class="pnome"><b>{nome}</b><small>{dim}</small></div>'
            f'<div class="etps">{"".join(caixas)}</div>'
            f'<div class="gp"><p><b>Guarda:</b> {guarda}</p><p><b>Perde:</b> {perde}</p></div></div>')


def knn_svg(W=440, H=280):
    """Ilustração do kNN com posições fixas: 3 locutores, um áudio de teste (★)
    com os 5 vizinhos (4 de A, 1 de B) e o mesmo áudio deslocado pela sala (☆)."""
    A = [(95, 115), (120, 100), (140, 125), (130, 150), (105, 165), (80, 140), (150, 165), (70, 180), (60, 110), (115, 195)]
    B = [(205, 95), (260, 60), (290, 40), (300, 85), (330, 55), (250, 30), (335, 100), (275, 110)]
    C = [(290, 200), (320, 225), (350, 195), (300, 240), (345, 240), (325, 180), (375, 215), (270, 228)]
    cores = {"A": "#2b8a3e", "B": "#7048e8", "C": "#868e96"}
    pts = [("A", q) for q in A] + [("B", q) for q in B] + [("C", q) for q in C]
    tx, ty = 170, 130
    d2 = lambda q, x, y: (q[0] - x) ** 2 + (q[1] - y) ** 2
    viz = sorted(pts, key=lambda t: d2(t[1], tx, ty))[:5]
    assert [v[0] for v in viz].count("A") == 4, viz
    raio = max(d2(v[1], tx, ty) for v in viz) ** .5 + 9
    rx, ry = 312, 208   # o mesmo áudio, reverberado
    o = [f'<svg class="chart" viewBox="0 0 {W} {H}" style="margin:2px 0">']
    o.append(f'<circle cx="{tx}" cy="{ty}" r="{raio:.1f}" fill="#3b5bdb0d" stroke="#3b5bdb" stroke-dasharray="5 4"/>')
    for _, (x, y) in viz:
        o.append(f'<line x1="{tx}" y1="{ty}" x2="{x}" y2="{y}" stroke="#495057" stroke-width="1.6"/>')
    for nome, (x, y) in pts:
        o.append(f'<circle cx="{x}" cy="{y}" r="6.5" fill="{cores[nome]}"/>')
    o.append(f'<text x="40" y="232" style="fill:{cores["A"]};font-weight:800;font-size:15px">locutor A</text>')
    o.append(f'<text x="420" y="30" text-anchor="end" style="fill:{cores["B"]};font-weight:800;font-size:15px">locutor B</text>')
    o.append(f'<text x="430" y="172" text-anchor="end" style="fill:{cores["C"]};font-weight:800;font-size:15px">locutor C</text>')
    o.append(f'<path d="M{tx + 10},{ty + 12} Q {tx + 70},{ty + 70} {rx - 14},{ry - 6}" fill="none" stroke="#c92a2a" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ponta)"/>')
    o.append('<defs><marker id="ponta" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c92a2a"/></marker></defs>')
    o.append(f'<text x="{tx}" y="{ty + 10}" text-anchor="middle" style="font-size:30px;fill:#1d1d1f">★</text>')
    o.append(f'<text x="{rx}" y="{ry + 10}" text-anchor="middle" style="font-size:30px;fill:#c92a2a">☆</text>')
    o.append(f'<text x="{tx}" y="{ty - raio - 8:.0f}" text-anchor="middle" style="font-size:13px;font-weight:700;fill:#1d1d1f">teste limpo → A ✓</text>')
    o.append(f'<text x="{rx - 20}" y="{H - 8}" text-anchor="middle" style="font-size:13px;font-weight:700;fill:#c92a2a">o mesmo áudio na sala → C ✗</text>')
    o.append("</svg>")
    return "".join(o)


def foot(n, esq="IA940 · Projeto Final · Trilha Voz"):
    return f'<div class="foot"><span>{esq}</span><span>{n} / {N_SLIDES}</span></div>'


def main():
    meta = pd.read_csv("data/features/meta.csv", dtype={"locutor": str})
    df5 = pd.read_csv("data/resultados/curvas_k5.csv")
    P, Q = resumo(df5, "principal"), resumo(df5, "pareada")
    exemplos = gerar_midia(meta)
    niv = P["MFCC"].index.to_numpy()

    # previsão registrada (logísticas do slide de 23/10)
    t = np.linspace(0, 2, 81)
    logi = lambda a0, t50: 100 * (0.1 + (a0 - 0.1) / (1 + (t / t50) ** 2))

    g_central = grafico([
        dict(x=niv, m=P["MFCC"]["mean"], sd=P["MFCC"]["std"], c=MFCC_C, rotulo="MFCC médio", rotulo_xy=(2.0, 86)),
        dict(x=niv, m=P["YAMNet"]["mean"], sd=P["YAMNet"]["std"], c=YAM_C, rotulo="YAMNet", rotulo_xy=(0.62, 30)),
    ])
    g_prev = grafico([
        dict(x=t, m=logi(0.82, 0.7), c=MFCC_C, dash=True, pts=False, w=2.5, op=.8),
        dict(x=t, m=logi(0.78, 1.4), c=YAM_C, dash=True, pts=False, w=2.5, op=.8),
        dict(x=niv, m=P["MFCC"]["mean"], c=MFCC_C, rotulo="MFCC", rotulo_xy=(2.0, 86)),
        dict(x=niv, m=P["YAMNet"]["mean"], c=YAM_C, rotulo="YAMNet", rotulo_xy=(0.55, 4)),
    ], W=600, H=370)
    g_par = grafico([
        dict(x=niv, m=Q["MFCC"]["mean"], sd=Q["MFCC"]["std"], c=MFCC_C, rotulo="MFCC", rotulo_xy=(2.0, 84)),
        dict(x=niv, m=Q["YAMNet"]["mean"], sd=Q["YAMNet"]["std"], c=YAM_C, rotulo="YAMNet pareada", rotulo_xy=(2.0, 50)),
        dict(x=niv, m=P["YAMNet"]["mean"], c=YAM_C, dash=True, w=2, op=.6, pts=False, rotulo="YAMNet principal (treino limpo)", rotulo_xy=(1.45, 22)),
    ], W=560, H=350)

    # tabela de variabilidade (diferença pareada)
    t_ = df5[df5.condicao == "principal"].pivot_table(index=["rt60", "divisao"], columns="representacao", values="acuracia") * 100
    t_["d"] = t_["MFCC"] - t_["YAMNet"]
    g = t_.groupby("rt60")
    linhas_var = "".join(
        f"<tr><td>{fmt(rt)}</td><td>{fmt(g['MFCC'].mean()[rt])} ± {fmt(g['MFCC'].std()[rt])}</td>"
        f"<td>{fmt(g['YAMNet'].mean()[rt])} ± {fmt(g['YAMNet'].std()[rt])}</td>"
        f"<td><b>{fmt(g['d'].mean()[rt])}</b> ± {fmt(g['d'].std()[rt])}</td><td class='ok'>sim</td></tr>"
        for rt in niv)

    audios_niveis = "".join(
        f'<div class="aud"><span>{"seco" if rt == 0 else fmt(rt) + " s"}</span>'
        f'<small>{"" if rt == 0 else f"DRR {drr_db(rt):+.1f} dB".replace(".", ",")}</small>'
        f'<audio controls preload="none" src="audios/nivel_{rt:.1f}.wav"></audio>'
        f'<button class="spec-btn" data-spec="{k}" title="Espectrograma em tempo real">{ICONE_ESPEC}</button></div>'
        for k, rt in enumerate(NIVEIS_RT60))
    niveis_js = json.dumps([
        {"src": f"audios/nivel_{rt:.1f}.wav",
         "rotulo": "seco" if rt == 0 else f"{fmt(rt)} s",
         "titulo": "Seco (RT60 = 0)" if rt == 0 else f"RT60 = {fmt(rt)} s",
         "drr": "" if rt == 0 else f"DRR {drr_db(rt):+.1f} dB".replace(".", ",")}
        for rt in NIVEIS_RT60], ensure_ascii=False)
    modal_js = MODAL_JS.replace("__NIVEIS__", niveis_js).replace("__TMAX__", str(T_MAX_ESPEC))

    info = {  # da análise de falhas no notebook (divisão 0, RT60 = 0,2 s)
        "só_MFCC_acerta": ("Só MFCC acerta", "3 601 (69 %)"),
        "só_YAMNet_acerta": ("Só YAMNet acerta", "41 (< 1 %)"),
        "os_dois_erram": ("Os dois erram", "72"),
        "os_dois_acertam": ("Os dois acertam", "1 536"),
    }
    import unicodedata
    pred = {}
    for cat, arq, lim, rev in exemplos:
        pred.setdefault(unicodedata.normalize("NFC", cat), []).append((arq, lim, rev))
    caixas = ""
    for cat in ["só_MFCC_acerta", "só_YAMNet_acerta", "os_dois_erram", "os_dois_acertam"]:
        nome, n = info[cat]
        ex = pred.get(unicodedata.normalize("NFC", cat), [])[:1]
        corpo = "".join(f'<div class="mono" style="font-size:14px">{a}.wav</div>'
                        f'<div class="aud2"><small>limpo</small><audio controls preload="none" src="audios/{l}"></audio></div>'
                        f'<div class="aud2"><small>0,2 s</small><audio controls preload="none" src="audios/{r}"></audio></div>'
                        for a, l, r in ex)
        caixas += f'<div class="card fx"><h3 style="font-size:19px">{nome}<br><span class="cnt" style="margin:4px 0 0">{n} arquivos</span></h3>{corpo}</div>'

    grafico_drr_svg = grafico_drr()
    pipe_mfcc = pipeline("var(--mfcc)", "MFCC médio", "20 números", [
        ("Áudio", "3 s a 16 kHz", "48 000 amostras"),
        ("Quadros", "janelas de 25 ms a cada 10 ms: a voz é estável nesse trecho", "≈ 300 quadros"),
        ("Espectro mel", "FFT + 40 filtros espaçados como o ouvido (finos no grave)", "300 × 40"),
        ("log + DCT", "log ≈ sensação de volume; a DCT resume a forma do espectro", "300 × 20"),
        ("Média no tempo", "um valor médio por coeficiente", "vetor de 20"),
    ], "a forma média do espectro → o <b>trato vocal</b>, o timbre de cada pessoa.",
       "a ordem no tempo (quais sons vieram antes ou depois).")
    pipe_yam = pipeline("var(--yam)", "YAMNet", "1024 números", [
        ("Áudio", "o mesmo arquivo, 3 s a 16 kHz", "48 000 amostras"),
        ("Log-mel", "espectrograma com 64 bandas mel, quadros de 10 ms", "≈ 300 × 64"),
        ("Janelas de 0,96 s", "recortes de ~1 s, um a cada 0,48 s", "≈ 5 × (96 × 64)"),
        ("Rede neural (CNN)", "MobileNet treinada no AudioSet (2 M vídeos, 521 tipos de som); paramos na penúltima camada", "5 × 1024"),
        ("Média no tempo", "média das janelas", "vetor de 1024"),
    ], "padrões tempo-frequência aprendidos para dizer <b>que som é</b> (fala, música, cachorro…).",
       "foi treinada para <i>que som</i>, não para <i>quem fala</i>: a identidade é um efeito colateral.")
    knn_svg_txt = knn_svg()
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quem está falando? · IA940</title>
{KATEX_HEAD}
<style>
  :root {{
    --bg: #f7f6f2; --card: #ffffff; --ink: #1d1d1f; --muted: #5f6368; --line: #e3e1da;
    --accent: #3b5bdb; --accent-soft: #e8edff;
    --mfcc: #d9480f; --mfcc-soft: #fff0e6; --yam: #3b5bdb; --yam-soft: #e8edff;
    --ok: #2b8a3e; --ok-soft: #e6f6ea; --bad: #c92a2a; --bad-soft: #fff0f0;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #16171a; }}
  body {{ color: var(--ink); font: 20px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; background: #f1efe8; padding: 1px 6px; border-radius: 5px; }}
  .slide {{ display: none; position: absolute; left: 50%; top: 50%; width: 1280px; height: 720px;
    transform: translate(-50%, -50%) scale(var(--s, 1)); background: var(--bg);
    padding: 40px 56px 56px; overflow: hidden; border-radius: 4px; flex-direction: column; }}
  .slide.active {{ display: flex; }}
  .kick {{ color: var(--accent); font-weight: 700; font-size: 15px; text-transform: uppercase; letter-spacing: .06em; }}
  h2 {{ font-size: 36px; line-height: 1.15; margin: 4px 0 22px; }}
  .foot {{ position: absolute; left: 56px; right: 56px; bottom: 20px; display: flex; justify-content: space-between; font-size: 14px; color: var(--muted); }}
  .note {{ border-left: 4px solid var(--accent); background: var(--accent-soft); border-radius: 8px; padding: 12px 18px; font-size: 18px; }}
  .note.warn {{ border-left-color: var(--mfcc); background: var(--mfcc-soft); }}
  .note.bad {{ border-left-color: var(--bad); background: var(--bad-soft); }}
  .note.ok {{ border-left-color: var(--ok); background: var(--ok-soft); }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 22px; }}
  .card h3 {{ margin: 0 0 8px; font-size: 21px; }}
  .card p, .card li {{ margin: 0; font-size: 17.5px; color: #3c4043; }}
  .card ul {{ margin: 0; padding-left: 20px; }} .card li {{ margin-bottom: 6px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .wl {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 22px; align-items: start; }}
  .big {{ font-size: 44px; font-weight: 800; line-height: 1; }}
  .lbl {{ font-size: 15px; color: var(--muted); margin-top: 6px; }}
  .m {{ color: var(--mfcc); font-weight: 700; }} .y {{ color: var(--yam); font-weight: 700; }}
  .title-slide {{ justify-content: center; padding: 70px 90px; }}
  .title-slide h1 {{ font-size: 62px; line-height: 1.05; margin: 12px 0 18px; }}
  .t-sub {{ font-size: 26px; line-height: 1.35; margin: 0 0 30px; color: #3c4043; max-width: 1050px; }}
  .t-people {{ padding-top: 18px; border-top: 2px solid var(--line); display: flex; gap: 34px; flex-wrap: wrap; font-size: 22px; font-weight: 600; }}
  svg.chart {{ width: 100%; height: auto; font-size: 14px; fill: var(--muted); }}
  svg.chart text {{ font-family: inherit; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 16.5px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }}
  td {{ white-space: nowrap; }} .vs td, .card td.wrap {{ white-space: normal; }}
  th {{ font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }}
  td.ok {{ color: var(--ok); font-weight: 700; }}
  .eq {{ font-family: "Times New Roman", serif; font-size: 27px; text-align: center; padding: 10px 0 8px; }}
  .eq i {{ font-style: italic; }}
  .auds {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px 16px; }}
  .aud {{ display: grid; grid-template-columns: 56px 1fr; grid-template-rows: auto auto; column-gap: 8px; align-items: center; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 6px 10px; }}
  .aud span {{ font-weight: 800; font-size: 17px; }} .aud small {{ grid-column: 1; font-size: 12px; color: var(--muted); }}
  .aud audio {{ grid-column: 2; grid-row: 1 / 3; width: 100%; height: 34px; }}
  .aud2 {{ display: grid; grid-template-columns: 44px 1fr; align-items: center; gap: 6px; margin-top: 4px; }}
  .aud2 small {{ font-size: 12px; color: var(--muted); }} .aud2 audio {{ width: 100%; height: 32px; }}
  .cnt {{ display: inline-block; white-space: nowrap; font-size: 14px; font-weight: 700; color: var(--muted); background: #f1efe8; border-radius: 99px; padding: 2px 10px; margin-left: 6px; }}
  .leg {{ display: flex; gap: 22px; font-size: 15px; color: var(--muted); margin: 0 0 4px 52px; }}
  .leg i {{ display: inline-block; width: 26px; height: 0; border-top: 3px solid; vertical-align: middle; margin-right: 6px; }}
  .leg i.d {{ border-top-style: dashed; }}
  .vs td:first-child {{ font-weight: 700; color: var(--muted); width: 22%; }}
  .bad {{ color: var(--bad); font-weight: 700; }} .good {{ color: var(--ok); font-weight: 700; }}
  .nav {{ position: fixed; right: 18px; bottom: 14px; color: #9aa0a6; font: 13px/1 -apple-system, sans-serif; }}
  .prog {{ position: fixed; left: 0; bottom: 0; height: 3px; background: var(--accent); transition: width .2s; }}
{MODAL_CSS}</style>
</head>
<body>

<!-- 1 · Capa -->
<section class="slide title-slide active">
  <div class="kick">IA940 · Análise e Síntese de Sinais Musicais · Projeto Final</div>
  <h1>Quem está falando?<br>MFCC × YAMNet sob reverberação</h1>
  <p class="t-sub">Trilha Voz · identificação de locutor com kNN no ESD (10 locutores de inglês), com a sala reverberando de RT60 = 0 a 2 s.</p>
  <div class="t-people"><span>Kevin Inofuente Colque</span><span>Luiza Viana Souza</span><span>Guilherme Maia Masetto</span></div>
</section>

<!-- 2 · As duas representações -->
<section class="slide">
  <div class="kick">A pergunta · qual vetor ainda reconhece <i>quem fala</i> quando a sala reverbera?</div>
  <h2>Duas formas de transformar 3 s de voz em um vetor</h2>
  {pipe_mfcc}
  {pipe_yam}
  <div class="note" style="margin-top:14px">Os dois terminam com uma <b>média no tempo</b>: um arquivo → um vetor. É sobre esses vetores que o classificador decide quem está falando.</div>
  {foot(2)}
</section>

<!-- 3 · O experimento -->
<section class="slide">
  <div class="kick">Como medimos · o mesmo experimento para os dois vetores</div>
  <h2>Cadastrar no estúdio, reconhecer na sala</h2>
  <div class="wl" style="grid-template-columns:1.2fr 1fr">
    <div class="passos">
      <div class="passo"><b>1</b><div><h4>Dados</h4><p>17 500 áudios do ESD: <b>10 locutores</b> × 1 750 frases. Acertar por sorte = <b>10 %</b>.</p></div></div>
      <div class="passo"><b>2</b><div><h4>Dividir 70 / 30</h4><p>12 250 de <b>treino</b> e 5 250 de <b>teste</b>, a mesma proporção de cada locutor. Um arquivo nunca está nos dois lados.</p></div></div>
      <div class="passo"><b>3</b><div><h4>Reverberar só o teste</h4><p>Treino limpo (o “cadastro” da voz); o teste passa pela sala nos <b>9 níveis</b> de RT60, de 0 a 2 s.</p></div></div>
      <div class="passo"><b>4</b><div><h4>Classificar com kNN (k = 5)</h4><p>z-score com média e desvio <b>do treino</b>; cada áudio de teste recebe o locutor <b>mais votado entre seus 5 vizinhos</b> mais próximos.</p></div></div>
      <div class="passo"><b>5</b><div><h4>Medir e repetir</h4><p>Acurácia = % do teste com o locutor certo. Repetimos com <b>5 divisões</b> diferentes → média ± desvio.</p></div></div>
    </div>
    <div class="card" style="padding:14px 18px"><h3 style="font-size:18px;margin-bottom:2px">Como o kNN decide</h3>{knn_svg_txt}
      <p style="font-size:14.5px">Cada ponto é <b>um arquivo de treino</b> (um vetor). A <b>★</b> é um áudio de teste: dos 5 vizinhos mais próximos, <b>4 são de A</b> → o kNN diz “A”. Se a sala <b class="bad">desloca</b> o vetor (<span class="bad">☆</span>), os vizinhos passam a ser de outro locutor e o voto erra. <b>Medir esse deslocamento é o experimento.</b></p></div>
  </div>
  {foot(3)}
</section>

<!-- 4 · Previsão registrada -->
<section class="slide">
  <div class="kick">Previsão registrada · 23/10 (antes de qualquer degradação)</div>
  <h2>Apostamos que o YAMNet cairia mais devagar que o MFCC</h2>
  <div class="wl">
    <div class="card" style="padding:12px 14px">
      <div class="leg"><span><i style="border-color:var(--mfcc)" class="d"></i>MFCC</span><span><i style="border-color:var(--yam)" class="d"></i>YAMNet</span><span>(curvas desenhadas à mão)</span></div>
      {grafico([dict(x=t, m=logi(0.82, 0.7), c=MFCC_C, dash=True, pts=False, rotulo="MFCC", rotulo_xy=(0.55, 52)),
                dict(x=t, m=logi(0.78, 1.4), c=YAM_C, dash=True, pts=False, rotulo="YAMNet", rotulo_xy=(1.95, 44))], W=600, H=360)}
    </div>
    <div>
      <div class="card"><h3>Justificativa (texto de 23/10)</h3>
        <ul>
          <li><span class="m">MFCC</span> usa janelas de ≈ 25 ms; a cauda da reverberação dura centenas de ms e <b>mistura quadros vizinhos</b>.</li>
          <li><span class="y">YAMNet</span> foi treinado no AudioSet: ≈ 2 milhões de clipes do YouTube, <b>gravados em ambientes muito variados</b> → deveria ser mais robusto.</li>
        </ul></div>
      <div class="note warn" style="margin-top:14px"><b>06/11, sem degradação:</b> MFCC 98 %, YAMNet só 55 %. Mantivemos a aposta na <i>inclinação</i>: a pergunta virou “o MFCC cai até cruzar o YAMNet?”.</div>
    </div>
  </div>
  {foot(4)}
</section>

<!-- 5 · Dados -->
<section class="slide">
  <div class="kick">Banco de dados</div>
  <h2>ESD: 10 locutores × 5 emoções × 350 frases = 17 500 áudios</h2>
  <div class="grid3" style="margin-bottom:18px">
    <div class="card"><div class="big">10</div><div class="lbl">locutores de inglês (0011–0020) · acaso = 10 %</div></div>
    <div class="card"><div class="big">1 750</div><div class="lbl">áudios por locutor: 350 por emoção (Neutral, Angry, Happy, Sad, Surprise)</div></div>
    <div class="card"><div class="big">2,8 s</div><div class="lbl">duração média · 16 kHz · estúdio, SNR &gt; 20 dB</div></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Sem vazamento</h3>
      <ul><li>A divisão é por <b>arquivo inteiro</b>: nenhum arquivo (nem trecho) aparece no treino e no teste — verificado nas 5 divisões.</li>
          <li>12 250 treino / 5 250 teste; <b>525 de teste por locutor</b>.</li>
          <li>O rótulo sai do nome do arquivo (<code>0013_000020.wav</code>); a pasta é a emoção.</li></ul></div>
    <div class="card"><h3>Emoções misturadas nos dois lados</h3>
      <ul><li>Queremos reconhecer o locutor <b>apesar</b> da emoção: o classificador não pode usar a emoção como atalho.</li>
          <li><b>Limitação:</b> cada locutor gravou em sessões próprias — o canal pode ajudar a identificar. Vale igual para as duas representações.</li></ul></div>
  </div>
  {foot(5, "Zhou et al., Emotional voice conversion: Theory, databases and ESD, Speech Communication 137 (2022)")}
</section>

<!-- 6 · Degradação: equações -->
<section class="slide">
  <div class="kick">Degradação · src/degradation.py</div>
  <h2 style="margin-bottom:10px">Reverberação = convolução com uma resposta ao impulso sintética</h2>
  <div class="sigl"><span><b>RI</b> resposta ao impulso da sala, o filtro {EQ['h']}</span><span><b>RT60</b> tempo para a energia cair 60 dB</span><span><b>DRR</b> energia direta ÷ reverberante</span></div>
  <div class="card" style="padding:4px 22px">{EQ['ri']}</div>
  <div class="wl" style="margin-top:12px">
    <div class="card"><ul>
      <li>{EQ['delta']}: som direto. {EQ['b']}: campo difuso, milhares de reflexões com fase aleatória.</li>
      <li>{EQ['ln']}, pois {EQ['rt_def']}: a amplitude cai 1000× = <b>60 dB em t = RT60</b>, a definição de RT60.</li>
      <li>{EQ['g']} escolhido para <b>DRR = 0 dB em 0,5 s</b> ⇒ {EQ['drr_rt']}: mesma sala e distância, só a absorção muda (slide 7).</li>
      <li>Convolução completa (a cauda fica) e <b>mesmo RMS</b> do original; uma RI diferente por arquivo.</li>
    </ul></div>
    <div class="card" style="padding:10px"><img src="img/ri.png" style="width:100%">
      <p style="font-size:14px;margin-top:6px">Verificação (integral de Schroeder): RT60 medido 0,108 · 0,209 · 0,302 · 0,488 · 0,696 · 0,993 · 1,503 · 1,992 s para os 8 níveis pedidos.</p></div>
  </div>
  {foot(6)}
</section>

<!-- 7 · O que é DRR -->
<section class="slide">
  <div class="kick">Degradação · o segundo número de cada nível</div>
  <h2>DRR: quanto da voz chega direto × quanto chega pela sala</h2>
  <div class="wl">
    <div>
      <div class="card">
        {EQ['drr_def']}
        <p style="text-align:center;font-size:15px;margin:0 0 10px"><b>Direct-to-Reverberant Ratio</b> · som direto boca → microfone ÷ todas as reflexões</p>
        <div class="casos">
          <div class="caso pos"><b>DRR &gt; 0</b><span>domina a voz direta</span><em>soa perto e clara</em></div>
          <div class="caso zer"><b>DRR = 0</b><span>energias iguais</span><em>distância crítica</em></div>
          <div class="caso neg"><b>DRR &lt; 0</b><span>domina a sala</span><em>soa longe</em></div>
        </div>
      </div>
      <div class="card" style="margin-top:12px;padding:12px 22px"><h3 style="margin-bottom:2px">Na nossa resposta ao impulso (RI)</h3>
        <p style="font-size:16px">{EQ['e_dir']} (o {EQ['delta']}); a energia da cauda <b>cresce com o RT60</b>, e com {EQ['g']} fixo para DRR = 0 dB em 0,5 s:</p>
        <div style="font-size:15px">{EQ['e_rev']}</div></div>
      <div class="note" style="margin-top:12px"><b>RT60</b> = <b>quanto tempo</b> a cauda dura · <b>DRR</b> = <b>quão forte</b> ela é. Microfone a distância fixa; só a absorção da sala muda.</div>
    </div>
    <div class="card"><h3>DRR de cada nível</h3>{grafico_drr_svg}
      <p style="font-size:14px;margin-top:6px">A partir de 0,5 s a energia da sala supera a da voz direta.</p></div>
  </div>
  {foot(7, "DRR = 0 dB define a distância crítica de uma sala")}
</section>

<!-- 8 · Ouça -->
<section class="slide">
  <div class="kick">Exemplos audíveis · locutor 0013, Neutral</div>
  <h2>Os 9 níveis: 0 · 0,1 · 0,2 · 0,3 · 0,5 · 0,7 · 1,0 · 1,5 · 2,0 s</h2>
  <div class="auds">{audios_niveis}</div>
  <div class="card" style="margin-top:14px;padding:8px"><img src="img/espectrogramas.png" style="width:100%"></div>
  <button class="spec-big" data-spec="0">▶ Espectrograma em tempo real</button>
  {foot(8, "A cauda preenche as pausas e borra as transições entre fonemas; o espectro médio quase não muda")}
</section>

<!-- 9 · Resultado central -->
<section class="slide">
  <div class="kick">Resultado central · treino limpo, teste reverberado</div>
  <h2>O <span class="y">YAMNet</span> desaba já em 0,2 s; o <span class="m">MFCC médio</span> resiste até 2 s</h2>
  <div class="wl">
    <div class="card" style="padding:12px 14px">{g_central}
      <p style="font-size:14px;margin:2px 0 0 52px">média de 5 divisões · faixa = ± 1 desvio · k = 5</p></div>
    <div>
      <div class="card"><div class="grid2" style="gap:10px">
        <div><div class="big m">{fmt(P['MFCC']['mean'][2.0], 0)}%</div><div class="lbl">MFCC em RT60 = 2 s</div></div>
        <div><div class="big y">{fmt(P['YAMNet']['mean'][0.2], 0)}%</div><div class="lbl">YAMNet já em 0,2 s (partia de 55 %)</div></div>
      </div></div>
      <div class="card" style="margin-top:14px"><ul>
        <li><span class="y">YAMNet</span>: −6 pontos em 0,1 s, metade em 0,2 s, <b>acaso a partir de ≈ 0,7 s</b>.</li>
        <li><span class="m">MFCC</span>: intacto até 0,3 s (97,5 %), 96 % em 0,5 s, cai quase linear depois.</li>
        <li><b>As curvas não se cruzam.</b> A distância entre elas cresce até 0,7 s (83 pontos) e só diminui porque o YAMNet já está no acaso.</li>
      </ul></div>
    </div>
  </div>
  {foot(9)}
</section>

<!-- 10 · Variabilidade -->
<section class="slide">
  <div class="kick">Variabilidade · 5 divisões por ponto</div>
  <h2>Em todos os níveis a diferença é dezenas de vezes maior que a dispersão</h2>
  <div class="wl">
    <div class="card" style="padding:10px 16px"><table>
      <tr><th>RT60 (s)</th><th class="m">MFCC (%)</th><th class="y">YAMNet (%)</th><th>Δ pareado (pontos)</th><th>Δ &gt; dispersão?</th></tr>
      {linhas_var}
    </table></div>
    <div>
      <div class="card"><h3>Como comparamos</h3>
        <p>As duas representações usam <b>as mesmas divisões</b>, então calculamos Δ = acc(MFCC) − acc(YAMNet) <b>em cada divisão</b> e tomamos média ± desvio de Δ.</p>
        <p style="margin-top:8px">Critério: |Δ médio| &gt; 2 × desvio de Δ <b>e</b> &gt; o maior desvio das duas curvas.</p></div>
      <div class="card" style="margin-top:14px"><h3>Robustez da conclusão</h3>
        <ul><li>Desvios entre divisões &lt; 1 ponto em todos os níveis.</li>
            <li>Com <b>k = 1</b>: mesma forma, 1–7 pontos abaixo.</li></ul></div>
    </div>
  </div>
  {foot(10)}
</section>

<!-- 11 · Previsto x observado -->
<section class="slide">
  <div class="kick">Previsto × observado</div>
  <h2>A previsão errou a direção: quem cai primeiro é o YAMNet</h2>
  <div class="wl">
    <div class="card" style="padding:12px 14px">
      <div class="leg"><span><i style="border-color:var(--ink)" class="d"></i>previsto (23/10)</span><span><i style="border-color:var(--ink)"></i>observado</span></div>
      {g_prev}</div>
    <div class="card"><table class="vs">
      <tr><th></th><th>Previsto</th><th>Observado</th></tr>
      <tr><td>Cai primeiro</td><td>MFCC</td><td class="bad">YAMNet</td></tr>
      <tr><td>A partir de</td><td>MFCC ≈ 0,3–0,5 s</td><td>YAMNet já em 0,1–0,2 s</td></tr>
      <tr><td>Forma</td><td>duas logísticas</td><td>YAMNet em degrau; MFCC quase linear</td></tr>
      <tr><td>Em 2 s</td><td>MFCC ≈ 18 %, YAMNet ≈ 32 %</td><td>MFCC <b>80 %</b>, YAMNet <b>10 %</b></td></tr>
    </table>
    <p style="margin-top:12px">Os próximos slides explicam <b>o que estava errado no raciocínio</b>.</p></div>
  </div>
  {foot(11)}
</section>

<!-- 12 · Por quê: o embedding se desloca -->
<section class="slide">
  <div class="kick">Por quê · 1</div>
  <h2>No YAMNet a informação do locutor não some: ela é <i>deslocada</i></h2>
  <div class="wl">
    <div class="card" style="padding:12px 14px">
      <p style="font-size:15px;margin:0 0 4px 52px"><b>Pareada</b>: treino e teste reverberados no mesmo RT60</p>
      {g_par}</div>
    <div>
      <div class="card"><h3>Treino e teste na mesma sala</h3><p>O YAMNet mantém <b>42–55 %</b> → a identidade continua no embedding. O colapso é um <b>descasamento de domínio</b> treino-teste.</p></div>
      <div class="card" style="margin-top:14px"><h3>Quanto a reverberação empurra os pontos?</h3>
        <table><tr><th>RT60</th><th>0,2 s</th><th>0,5 s</th><th>2 s</th></tr>
          <tr><td class="m">MFCC</td><td>0,14×</td><td>0,30×</td><td>0,61×</td></tr>
          <tr><td class="y">YAMNet</td><td><b>3,9×</b></td><td><b>15×</b></td><td><b>39×</b></td></tr></table>
        <p style="font-size:14.5px;margin-top:6px">deslocamento médio ÷ distância média entre centróides de locutores (espaço z-score). Acima de 1×, o vizinho mais próximo é decidido pela sala, não pela pessoa.</p></div>
    </div>
  </div>
  {foot(12)}
</section>

<!-- 13 · Por quê: MFCC médio e YAMNet -->
<section class="slide">
  <div class="kick">Por quê · 2</div>
  <h2>A média do MFCC descarta exatamente o que a reverberação estraga</h2>
  <div class="grid2">
    <div class="card" style="border-top:4px solid var(--mfcc)"><h3 class="m">MFCC médio ≈ DCT do log-espectro de longo prazo</h3>
      {EQ['ltas']}
      <ul><li>Nossa cauda é ruído <b>branco</b> → {EQ['EH']} é <b>constante em f</b>: a forma do espectro médio quase não muda.</li>
          <li>A reverberação estraga o <b>tempo</b> (transições, pausas) — que a média joga fora.</li>
          <li>Sobra um deslocamento em {EQ['c0']} (energia) e {EQ['c1']} (inclinação), <b>quase igual para todos</b> os locutores.</li></ul></div>
    <div class="card" style="border-top:4px solid var(--yam)"><h3 class="y">YAMNet olha padrões tempo-frequência de 0,96 s</h3>
      <ul><li>Ataques e transições de formantes são o que a CNN usa — e o que a cauda borra.</li>
          <li>Treinado para dizer <b>que som é</b>; a identidade é um resíduo fraco (55 % no limpo).</li>
          <li>Até a classe da própria rede degrada: P(<i>Speech</i>) <b>0,98 → 0,79 → 0,33</b> em 0 / 0,5 / 2 s.</li></ul></div>
  </div>
  <div class="note ok" style="margin-top:16px"><b>Controles:</b> sem z-score ou com normalização L2 o YAMNet colapsa igual (55 → 16 → 9 %) · cortando a cauda final do áudio: 34 / 22 / 16 / 11 % em 0,2 / 0,5 / 1 / 2 s — a cauda pesa pouco, o problema é a reverberação <i>dentro</i> da fala.</div>
  {foot(13)}
</section>

<!-- 14 · Análise de falhas -->
<section class="slide">
  <div class="kick">Análise de falhas · RT60 = 0,2 s (onde as curvas se separam) · divisão 0</div>
  <h2>Ouça: acertos e erros de cada representação</h2>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">{caixas}</div>
  <div class="grid2" style="margin-top:14px">
    <div class="card"><h3 class="y">Erros do YAMNet: “atratores”</h3>
      <p>4 locutores (0014, 0012, 0017, 0020) recebem <b>64 %</b> dos erros; em 1 s, só o 0014 recebe metade. Todos os pontos migram para a mesma região — acerto por locutor vai de 52 % (0020) a 3 % (0015).</p></div>
    <div class="card"><h3 class="m">Erros do MFCC: vizinhos</h3>
      <p>Só 113 de 5 250; <b>39 são 0016 ↔ 0018</b>, as duas vozes mais agudas (F0 ≈ 204 e 243 Hz). Os dois erram sobretudo em emoções de alta excitação (Surprise, Angry, Happy: 56 de 72).</p></div>
  </div>
  {foot(14)}
</section>

<!-- 15 · Conclusões -->
<section class="slide">
  <div class="kick">Conclusões</div>
  <h2>Para <i>quem fala</i>, o MFCC médio é muito mais robusto à sala que o YAMNet</h2>
  <div class="grid2">
    <div class="card"><h3>O que aprendemos</h3><ul>
      <li>MFCC <b>98 → 80 %</b> de 0 a 2 s; YAMNet <b>55 → 10 %</b> (acaso a partir de ≈ 0,7 s). Diferença ≫ dispersão em todos os níveis.</li>
      <li>Erramos a previsão por dois motivos: esquecemos que o baseline é uma <b>média</b> (que ignora o tempo) e supusemos que a robustez do YAMNet para <b>“que som é”</b> valeria para <b>“quem fala”</b>.</li>
      <li>A informação de locutor no YAMNet existe, mas a sala a desloca 4–40× mais do que os locutores se separam.</li></ul></div>
    <div class="card"><h3>Limitações e próximos passos</h3><ul>
      <li>RI sintética com decaimento <b>igual em todas as frequências</b>. Em salas reais os agudos decaem mais rápido → prevemos que o MFCC cairia mais cedo.</li>
      <li>DRR acoplado ao RT60 (uma sala, uma distância); uma RI por arquivo e nível.</li>
      <li>ESD em estúdio, sessões por locutor (canal).</li>
      <li>Próximos: RIs medidas, CMN no MFCC, embeddings de locutor (x-vector/ECAPA) como 3ª curva.</li></ul></div>
  </div>
  {foot(15, "Notebook completo: Projeto_Final.ipynb · código em src/")}
</section>

<!-- 16 · Obrigado -->
<section class="slide title-slide">
  <div class="kick">IA940 · Projeto Final · Trilha Voz</div>
  <h1>Obrigado!<br>Perguntas?</h1>
  <p class="t-sub">Código e notebook: github.com/KevinInoCol/Phd-IA940-Analise-e-Sintese-de-Sinais-Musicais-Projeto-Final</p>
  <div class="t-people"><span>Kevin Inofuente Colque</span><span>Luiza Viana Souza</span><span>Guilherme Maia Masetto</span></div>
</section>

{MODAL_HTML}
<div class="prog"></div>
<div class="nav">← → para navegar</div>
<script>
  const slides = [...document.querySelectorAll('.slide')];
  const prog = document.querySelector('.prog');
  let i = 0;
  function show(k) {{
    i = Math.max(0, Math.min(slides.length - 1, k));
    slides.forEach((s, j) => s.classList.toggle('active', j === i));
    document.querySelectorAll('audio').forEach(a => a.pause());
    prog.style.width = (100 * (i + 1) / slides.length) + '%';
  }}
  function fit() {{ const s = Math.min(innerWidth / 1280, innerHeight / 720) * 0.96; document.documentElement.style.setProperty('--s', s); }}
{modal_js}
  addEventListener('keydown', e => {{
    if (SP.tecla(e)) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) {{ e.preventDefault(); show(i + 1); }}
    if (['ArrowLeft', 'PageUp'].includes(e.key)) show(i - 1);
  }});
  addEventListener('click', e => {{
    if (SP.aberto() || e.target.closest('audio, button')) return;
    show(i + (e.clientX > innerWidth / 2 ? 1 : -1));
  }});
  addEventListener('resize', fit); fit();
  show((parseInt(location.hash.slice(1)) || 1) - 1);
  addEventListener('hashchange', () => show((parseInt(location.hash.slice(1)) || 1) - 1));
</script>
</body>
</html>
"""
    # a cópia na raiz do projeto aponta para os arquivos dentro de ia940-final/
    FONTE.write_text(html.replace('src="img/', f'src="{SAIDA}/img/')
                         .replace('src="audios/', f'src="{SAIDA}/audios/')
                         .replace('="katex/', f'="{SAIDA}/katex/')
                         .replace('"src": "audios/', f'"src": "{SAIDA}/audios/'), encoding="utf-8")
    (SAIDA / "index.html").write_text(html, encoding="utf-8")
    print(f"ok: {FONTE} e {SAIDA}/index.html · {len(exemplos)} exemplos de falha")


if __name__ == "__main__":
    main()
