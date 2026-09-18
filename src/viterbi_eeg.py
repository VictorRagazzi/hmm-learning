"""
Exemplo didático do algoritmo de Viterbi aplicado a sinais de EEG sintéticos.

Pipeline:
    1. Gera um sinal de EEG sintético com estágios de sono conhecidos.
    2. Segmenta o sinal em janelas de 30 s (padrão clínico).
    3. Calcula a FFT de cada janela e identifica a banda dominante.
    4. Usa essa sequência de bandas como observações discretas para o HMM.
    5. Roda o Viterbi para inferir a sequência de estágios mais provável (MPE).
    6. Gera 5 figuras didáticas explicando cada etapa do pipeline.

Estados ocultos (estados do HMM):
    0 = Vigília   — alta atividade, predomínio de beta
    1 = NREM leve — transição, predomínio de alpha/theta
    2 = NREM prof — sono profundo, predomínio de delta
    3 = REM       — semelhante à vigília, mas com theta dominante

Observações (banda dominante na janela):
    0 = delta  (0.5 – 4 Hz)
    1 = theta  (4  – 8 Hz)
    2 = alpha  (8  – 13 Hz)
    3 = beta   (13 – 30 Hz)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paletas — usadas em todos os plots para consistência visual
# ---------------------------------------------------------------------------
STATE_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # por estado
BAND_COLORS   = ["#8172B2", "#937860", "#CCB974", "#64B5CD"]   # por banda
STATE_LABELS  = ["Vigília", "NREM leve", "NREM prof", "REM"]
BAND_LABELS   = ["delta", "theta", "alpha", "beta"]

# ---------------------------------------------------------------------------
# Parâmetros do sinal sintético
# ---------------------------------------------------------------------------
FS = 256          # frequência de amostragem (Hz) — padrão em EEG clínico
WINDOW_SEC = 30   # duração de cada janela/época (segundos)
WINDOW_SAMPLES = FS * WINDOW_SEC

# Limites das bandas de frequência (Hz).
BANDS = [
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
]

# ---------------------------------------------------------------------------
# Parâmetros do HMM
# ---------------------------------------------------------------------------
states  = STATE_LABELS
symbols = BAND_LABELS

pi = [0.7, 0.2, 0.05, 0.05]

A = [
    [0.60,  0.35,  0.03,  0.02],
    [0.10,  0.50,  0.35,  0.05],
    [0.05,  0.20,  0.55,  0.20],
    [0.15,  0.25,  0.10,  0.50],
]

B = [
    [0.05,  0.10,  0.30,  0.55],
    [0.10,  0.35,  0.40,  0.15],
    [0.70,  0.20,  0.07,  0.03],
    [0.10,  0.55,  0.25,  0.10],
]


# ---------------------------------------------------------------------------
# 1. Geração do sinal de EEG sintético
# ---------------------------------------------------------------------------
def generate_eeg_segment(state_index, duration_sec, fs):
    """
    Gera um segmento de EEG sintético para um estado de sono dado.
    O sinal é uma soma de senoides nas 4 bandas, com amplitudes que
    refletem a banda dominante de cada estado, mais ruído gaussiano.
    """
    n_samples = duration_sec * fs
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    band_amplitudes = {
        0: np.array([0.5,  1.0,  3.0,  5.0]),   # Vigília
        1: np.array([1.0,  3.5,  4.0,  1.5]),   # NREM leve
        2: np.array([7.0,  2.0,  0.7,  0.3]),   # NREM prof
        3: np.array([1.0,  5.5,  2.5,  1.0]),   # REM
    }
    amplitudes = band_amplitudes[state_index]
    band_freqs = [2.0, 6.0, 10.0, 20.0]

    signal = np.zeros(n_samples)
    rng = np.random.default_rng()
    for amp, freq in zip(amplitudes, band_freqs):
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    signal += rng.normal(0, 0.5, n_samples)
    return signal


def generate_full_eeg(stage_sequence, window_sec, fs):
    """Concatena segmentos de EEG para cada estágio em stage_sequence."""
    segments = [generate_eeg_segment(s, window_sec, fs) for s in stage_sequence]
    return np.concatenate(segments), stage_sequence


# ---------------------------------------------------------------------------
# 2. Extração de feature: banda dominante por janela (via FFT)
# ---------------------------------------------------------------------------
def dominant_band_index(window, fs):
    """
    Calcula a FFT de uma janela e retorna o índice da banda com maior potência.
    Também retorna as potências por banda (para plotar).
    """
    n = len(window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(window)) ** 2

    band_powers = []
    for _, f_min, f_max in BANDS:
        mask = (freqs >= f_min) & (freqs < f_max)
        band_powers.append(power[mask].sum())

    return int(np.argmax(band_powers)), band_powers, freqs, power


def extract_observations(signal, window_samples, fs):
    """
    Segmenta em janelas e retorna observações + potências por banda por janela.
    """
    n_windows = len(signal) // window_samples
    observations = []
    all_band_powers = []   # shape: (n_windows, 4)

    for i in range(n_windows):
        start = i * window_samples
        window = signal[start:start + window_samples]
        obs, band_powers, _, _ = dominant_band_index(window, fs)
        observations.append(obs)
        all_band_powers.append(band_powers)

    return observations, all_band_powers


# ---------------------------------------------------------------------------
# 3. Algoritmo de Viterbi
# ---------------------------------------------------------------------------
def mpe_inference(pi, A, B, observations):
    """
    Viterbi: retorna (path, score, trellis, backpointers).
    Retornamos o trellis completo para poder plotar depois.
    """
    if not observations:
        raise ValueError("A sequência de observações não pode ser vazia.")

    number_of_states = len(pi)
    number_of_times  = len(observations)

    trellis      = [[0.0]  * number_of_times for _ in range(number_of_states)]
    backpointers = [[None] * number_of_times for _ in range(number_of_states)]

    for state in range(number_of_states):
        trellis[state][0] = pi[state] * B[state][observations[0]]

    for time in range(1, number_of_times):
        observation = observations[time]
        for state in range(number_of_states):
            best_previous_state = 0
            best_probability = (
                trellis[0][time - 1] * A[0][state] * B[state][observation]
            )
            for previous_state in range(1, number_of_states):
                probability = (
                    trellis[previous_state][time - 1]
                    * A[previous_state][state]
                    * B[state][observation]
                )
                if probability > best_probability:
                    best_probability = probability
                    best_previous_state = previous_state

            trellis[state][time]      = best_probability
            backpointers[state][time] = best_previous_state

    final_state = 0
    score = trellis[0][number_of_times - 1]
    for state in range(1, number_of_states):
        if trellis[state][number_of_times - 1] > score:
            score = trellis[state][number_of_times - 1]
            final_state = state

    path = [0] * number_of_times
    path[-1] = final_state
    for time in range(number_of_times - 1, 0, -1):
        path[time - 1] = backpointers[path[time]][time]

    return path, score, trellis, backpointers


# ---------------------------------------------------------------------------
# 4. Funções de plot — uma figura por etapa do pipeline
# ---------------------------------------------------------------------------

def plot_fig1_sinal_bruto(signal, true_stages, fs, window_sec):
    """
    Figura 1 — Sinal de EEG bruto com fundo colorido por estágio verdadeiro.
    Mostra apenas os primeiros 4 segundos de cada janela para não poluir.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Eixo de tempo em minutos
    n_samples = len(signal)
    t = np.arange(n_samples) / fs / 60.0
    ax.plot(t, signal, color="#333333", lw=0.4, alpha=0.8)

    # Fundo colorido: uma faixa por janela, cor = estado verdadeiro
    window_min = window_sec / 60.0
    for i, state in enumerate(true_stages):
        ax.axvspan(i * window_min, (i + 1) * window_min,
                   color=STATE_COLORS[state], alpha=0.18, lw=0)

    # Bordas verticais entre janelas
    for i in range(1, len(true_stages)):
        ax.axvline(i * window_min, color="gray", lw=0.5, ls="--", alpha=0.5)

    # Legenda dos estados
    patches = [mpatches.Patch(color=STATE_COLORS[s], alpha=0.6, label=STATE_LABELS[s])
               for s in range(4)]
    ax.legend(handles=patches, loc="upper right", fontsize=9, ncol=4)

    ax.set_xlabel("Tempo (minutos)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Fig 1 — Sinal de EEG bruto\n"
                 "Fundo colorido = estágio verdadeiro (oculto ao HMM)", fontsize=11)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    fig.savefig("./figs/fig1_sinal_bruto.png", dpi=130)
    plt.close(fig)
    print("  Salva: fig1_sinal_bruto.png")


def plot_fig2_fft_exemplos(signal, true_stages, fs, window_samples):
    """
    Figura 2 — FFT de 4 janelas representativas (uma por estado).
    Para cada estado, pega a primeira janela daquele estado e plota o espectro
    até 35 Hz, destacando as regiões de cada banda.
    """
    # Encontra a primeira janela de cada estado
    first_window_of_state = {}
    for i, state in enumerate(true_stages):
        if state not in first_window_of_state:
            first_window_of_state[state] = i

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=False)
    axes = axes.flatten()

    band_regions = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
    band_names   = ["δ delta", "θ theta", "α alpha", "β beta"]

    for plot_idx, state in enumerate(range(4)):
        ax = axes[plot_idx]
        win_idx = first_window_of_state[state]
        start   = win_idx * window_samples
        window  = signal[start:start + window_samples]

        n      = len(window)
        freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
        power  = np.abs(np.fft.rfft(window)) ** 2

        # Plota o espectro até 35 Hz
        mask_plot = freqs <= 35
        ax.plot(freqs[mask_plot], power[mask_plot],
                color=STATE_COLORS[state], lw=1.2)

        # Fundo por banda
        for b_idx, (f_lo, f_hi) in enumerate(band_regions):
            ax.axvspan(f_lo, f_hi, color=BAND_COLORS[b_idx], alpha=0.15, lw=0)
            # Rótulo da banda no centro — usa transform do eixo (0-1 em y)
            ax.text((f_lo + f_hi) / 2, 0.97,
                    band_names[b_idx], ha="center", va="top",
                    fontsize=8, color=BAND_COLORS[b_idx],
                    transform=ax.get_xaxis_transform())

        # Calcula potência por banda para mostrar no título
        band_powers = []
        for _, f_min, f_max in BANDS:
            m = (freqs >= f_min) & (freqs < f_max)
            band_powers.append(power[m].sum())
        dominant = int(np.argmax(band_powers))

        ax.set_title(f"Estado: {STATE_LABELS[state]}  →  banda dominante: "
                     f"{BAND_LABELS[dominant].upper()}",
                     fontsize=10, color=STATE_COLORS[state], fontweight="bold")
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Potência")
        ax.set_xlim(0, 35)

    fig.suptitle("Fig 2 — Espectro de potência (FFT) por estado de sono\n"
                 "A banda com maior área colorida vira a observação do HMM",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig("./figs/fig2_fft_exemplos.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Salva: fig2_fft_exemplos.png")


def plot_fig3_potencias_por_janela(all_band_powers, observations, true_stages):
    """
    Figura 3 — Potência relativa de cada banda ao longo das janelas (stacked bar)
    + linha mostrando a banda dominante escolhida.
    Deixa claro COMO a observação discreta é extraída do espectro.
    """
    n_windows = len(observations)
    powers = np.array(all_band_powers, dtype=float)

    # Normaliza para potência relativa (soma = 1 por janela)
    row_sums = powers.sum(axis=1, keepdims=True)
    powers_rel = powers / row_sums

    fig, (ax_bar, ax_obs) = plt.subplots(
        2, 1, figsize=(13, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    x = np.arange(n_windows)
    bottom = np.zeros(n_windows)
    for b in range(4):
        ax_bar.bar(x, powers_rel[:, b], bottom=bottom,
                   color=BAND_COLORS[b], label=BAND_LABELS[b], width=0.8)
        bottom += powers_rel[:, b]

    ax_bar.set_ylabel("Potência relativa")
    ax_bar.set_ylim(0, 1)
    ax_bar.legend(loc="upper right", ncol=4, fontsize=9, title="Banda")
    ax_bar.set_title("Fig 3 — Composição espectral por janela\n"
                     "Cada coluna = uma época de 30 s; a banda com maior fatia "
                     "vira a observação discreta do HMM", fontsize=11)

    # Painel inferior: observação discreta escolhida
    obs_colors = [BAND_COLORS[o] for o in observations]
    ax_obs.bar(x, [1] * n_windows, color=obs_colors, width=0.8)
    for i, o in enumerate(observations):
        ax_obs.text(i, 0.5, BAND_LABELS[o][0].upper(),
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax_obs.set_yticks([])
    ax_obs.set_ylabel("Obs.\ndiscreta", fontsize=9)
    ax_obs.set_xlabel("Janela (época de 30 s)")
    ax_obs.set_xticks(x)
    ax_obs.set_xticklabels([str(i + 1) for i in x])

    fig.tight_layout()
    fig.savefig("./figs/fig3_potencias_janelas.png", dpi=130)
    plt.close(fig)
    print("  Salva: fig3_potencias_janelas.png")


def plot_fig4_trellis(trellis, backpointers, path, observations):
    """
    Figura 4 — Trellis do Viterbi.
    Cada célula = log10(probabilidade) naquele (estado, tempo).
    O caminho ótimo é destacado com bordas brancas e setas de backpointer.
    """
    n_states = len(trellis)
    n_times  = len(trellis[0])

    # Monta matriz de log-probabilidades para o heatmap
    log_mat = np.zeros((n_states, n_times))
    for s in range(n_states):
        for t in range(n_times):
            v = trellis[s][t]
            log_mat[s, t] = np.log10(v) if v > 0 else -40

    fig, ax = plt.subplots(figsize=(max(10, n_times * 0.7), 4.5))
    im = ax.imshow(log_mat, aspect="auto", cmap="YlOrRd_r",
                   origin="upper", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log₁₀(probabilidade)", fontsize=9)

    # Rótulos dos eixos
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(STATE_LABELS, fontsize=10)
    ax.set_xticks(range(n_times))
    ax.set_xticklabels(
        [f"t={t+1}\n({BAND_LABELS[observations[t]][0].upper()})" for t in range(n_times)],
        fontsize=8
    )
    ax.set_xlabel("Tempo (janela) — letra = observação discreta")
    ax.set_title("Fig 4 — Trellis do Viterbi\n"
                 "Cor = log₁₀(prob do melhor caminho até aqui). "
                 "Quadros brancos = caminho ótimo escolhido.", fontsize=11)

    # Destaca células do caminho ótimo
    for t, s in enumerate(path):
        rect = plt.Rectangle((t - 0.5, s - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="white", facecolor="none")
        ax.add_patch(rect)

    # Setas de backpointer (de onde veio cada célula do caminho)
    for t in range(1, n_times):
        s_curr = path[t]
        s_prev = backpointers[s_curr][t]
        if s_prev is not None:
            ax.annotate(
                "", xy=(t - 0.5, s_curr), xytext=(t - 1 + 0.5, s_prev),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5)
            )

    fig.tight_layout()
    fig.savefig("./figs/fig4_trellis.png", dpi=130)
    plt.close(fig)
    print("  Salva: fig4_trellis.png")


def plot_fig5_hipnograma(true_stages, observations, predicted_path):
    """
    Figura 5 — Hipnograma: comparação entre estágio verdadeiro e predito.
    Hipnograma é o gráfico padrão clínico para estágios de sono.
    Também mostra a observação discreta em cada janela.
    """
    n = len(true_stages)
    x = np.arange(n)

    # Ordem clínica do hipnograma (vigília no topo, sono profundo embaixo)
    clinical_order = [0, 3, 1, 2]  # Vigília, REM, NREM leve, NREM prof
    y_pos = {state: clinical_order.index(state) for state in range(4)}
    y_labels = ["Vigília", "REM", "NREM leve", "NREM prof"]

    fig = plt.figure(figsize=(14, 6))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.05,
                   height_ratios=[3, 3, 1])

    ax_true = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1], sharex=ax_true)
    ax_obs  = fig.add_subplot(gs[2], sharex=ax_true)

    # --- Hipnograma verdadeiro ---
    y_true = [y_pos[s] for s in true_stages]
    ax_true.step(x, y_true, where="post", color="#333333", lw=2)
    ax_true.fill_between(x, y_true, step="post", alpha=0.15, color="#333333")
    for i, s in enumerate(true_stages):
        ax_true.plot(i, y_pos[s], "o", color=STATE_COLORS[s], ms=7, zorder=5)
    ax_true.set_yticks(range(4))
    ax_true.set_yticklabels(y_labels, fontsize=9)
    ax_true.set_ylabel("Estado\nverdadeiro", fontsize=9)
    ax_true.set_title("Fig 5 — Hipnograma: estágio verdadeiro vs. Viterbi predito", fontsize=11)
    ax_true.grid(axis="x", ls="--", alpha=0.4)

    # --- Hipnograma predito ---
    y_pred = [y_pos[s] for s in predicted_path]
    ax_pred.step(x, y_pred, where="post", color="#C44E52", lw=2)
    ax_pred.fill_between(x, y_pred, step="post", alpha=0.12, color="#C44E52")
    for i, s in enumerate(predicted_path):
        correct = (s == true_stages[i])
        marker = "o" if correct else "X"
        ax_pred.plot(i, y_pos[s], marker, color=STATE_COLORS[s],
                     ms=8, zorder=5,
                     markeredgecolor="black" if not correct else "none",
                     markeredgewidth=1.5)
    ax_pred.set_yticks(range(4))
    ax_pred.set_yticklabels(y_labels, fontsize=9)
    ax_pred.set_ylabel("Estado\npredito\n(Viterbi)", fontsize=9)
    ax_pred.grid(axis="x", ls="--", alpha=0.4)

    # Marca janelas erradas com fundo vermelho claro
    for i in range(n):
        if predicted_path[i] != true_stages[i]:
            ax_true.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)
            ax_pred.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)

    # --- Linha de observações ---
    obs_colors = [BAND_COLORS[o] for o in observations]
    ax_obs.bar(x, [1] * n, color=obs_colors, width=0.9)
    for i, o in enumerate(observations):
        ax_obs.text(i, 0.5, BAND_LABELS[o][0].upper(),
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
    ax_obs.set_yticks([])
    ax_obs.set_ylabel("Obs.", fontsize=9)
    ax_obs.set_xlabel("Janela (época de 30 s)")
    ax_obs.set_xticks(x)
    ax_obs.set_xticklabels([str(i + 1) for i in x], fontsize=8)

    plt.setp(ax_true.get_xticklabels(), visible=False)
    plt.setp(ax_pred.get_xticklabels(), visible=False)

    # Legenda de acerto/erro
    ok_patch  = mpatches.Patch(color="none", label="● = acerto")
    err_patch = mpatches.Patch(color="red", alpha=0.3, label="✕ / fundo vermelho = erro")
    ax_pred.legend(handles=[ok_patch, err_patch], loc="upper right",
                   fontsize=8, framealpha=0.8)

    fig.savefig("./figs/fig5_hipnograma.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Salva: fig5_hipnograma.png")


def plot_fig6_matrizes_hmm():
    """
    Figura 6 — Visualiza as matrizes A (transição) e B (emissão) do HMM.
    Essencial para entender o modelo antes de ver o Viterbi rodar.
    """
    fig, (ax_a, ax_b, ax_pi) = plt.subplots(1, 3, figsize=(14, 4),
                                              gridspec_kw={"width_ratios": [4, 4, 1]})

    # --- Matriz A ---
    mat_a = np.array(A)
    im_a  = ax_a.imshow(mat_a, cmap="Blues", vmin=0, vmax=1)
    ax_a.set_xticks(range(4)); ax_a.set_xticklabels(STATE_LABELS, rotation=30, ha="right")
    ax_a.set_yticks(range(4)); ax_a.set_yticklabels(STATE_LABELS)
    ax_a.set_xlabel("Próximo estado")
    ax_a.set_ylabel("Estado atual")
    ax_a.set_title("Matriz A — Transição\nA[i][j] = P(j | i)", fontsize=10)
    for i in range(4):
        for j in range(4):
            ax_a.text(j, i, f"{mat_a[i,j]:.2f}", ha="center", va="center",
                      fontsize=10, color="white" if mat_a[i,j] > 0.4 else "black")
    fig.colorbar(im_a, ax=ax_a, shrink=0.8)

    # --- Matriz B ---
    mat_b = np.array(B)
    im_b  = ax_b.imshow(mat_b, cmap="Greens", vmin=0, vmax=1)
    ax_b.set_xticks(range(4)); ax_b.set_xticklabels(BAND_LABELS, rotation=30, ha="right")
    ax_b.set_yticks(range(4)); ax_b.set_yticklabels(STATE_LABELS)
    ax_b.set_xlabel("Banda dominante (observação)")
    ax_b.set_ylabel("Estado oculto")
    ax_b.set_title("Matriz B — Emissão\nB[i][k] = P(banda k | estado i)", fontsize=10)
    for i in range(4):
        for j in range(4):
            ax_b.text(j, i, f"{mat_b[i,j]:.2f}", ha="center", va="center",
                      fontsize=10, color="white" if mat_b[i,j] > 0.4 else "black")
    fig.colorbar(im_b, ax=ax_b, shrink=0.8)

    # --- pi ---
    mat_pi = np.array(pi).reshape(-1, 1)
    im_pi  = ax_pi.imshow(mat_pi, cmap="Purples", vmin=0, vmax=1)
    ax_pi.set_xticks([0]); ax_pi.set_xticklabels(["π"], fontsize=12)
    ax_pi.set_yticks(range(4)); ax_pi.set_yticklabels(STATE_LABELS)
    ax_pi.set_title("π\n(inicial)", fontsize=10)
    for i in range(4):
        ax_pi.text(0, i, f"{pi[i]:.2f}", ha="center", va="center",
                   fontsize=10, color="white" if pi[i] > 0.4 else "black")
    fig.colorbar(im_pi, ax=ax_pi, shrink=0.8)

    fig.suptitle("Fig 6 — Parâmetros do HMM\n"
                 "O Viterbi usa A, B e π para encontrar a sequência de estados mais provável",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig("./figs/fig6_matrizes_hmm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Salva: fig6_matrizes_hmm.png")


# ---------------------------------------------------------------------------
# 5. Print de resultados no terminal
# ---------------------------------------------------------------------------
def print_results(true_stages, observations, predicted_path, score):
    n = len(true_stages)
    print(f"\n{'Janela':>7}  {'Verdadeiro':>12}  {'Observação':>12}  {'Predito':>12}  {'OK':>4}")
    print("-" * 60)
    correct = 0
    for i in range(n):
        true_label = states[true_stages[i]]
        obs_label  = symbols[observations[i]]
        pred_label = states[predicted_path[i]]
        ok = "✓" if predicted_path[i] == true_stages[i] else "✗"
        if predicted_path[i] == true_stages[i]:
            correct += 1
        print(f"{i+1:>7}  {true_label:>12}  {obs_label:>12}  {pred_label:>12}  {ok:>4}")
    accuracy = correct / n * 100
    print("-" * 60)
    print(f"Acurácia: {correct}/{n} janelas corretas ({accuracy:.1f}%)")
    print(f"Score do caminho (Viterbi): {score:.6e}")
    print()
    print("Nota: score muito pequeno é esperado — é o produto de muitas")
    print("probabilidades. Em produção, usa-se log-probabilidade para evitar")
    print("underflow numérico.")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    true_stages = [
        0, 0,
        1, 1, 1,
        2, 2, 2, 2,
        3, 3,
        1, 1,
        2, 2,
        3, 3,
        1,
        0,
    ]

    print("=" * 60)
    print("Gerando sinal de EEG sintético...")
    signal, _ = generate_full_eeg(true_stages, WINDOW_SEC, FS)
    print(f"Sinal: {len(signal)} amostras  "
          f"({len(signal)/FS:.0f} s = {len(true_stages)} janelas de {WINDOW_SEC} s)")

    print("\nExtraindo observações via FFT...")
    observations, all_band_powers = extract_observations(signal, WINDOW_SAMPLES, FS)
    print("Observações:", [symbols[o] for o in observations])

    print("\nRodando Viterbi (MPE)...")
    predicted_path, score, trellis, backpointers = mpe_inference(pi, A, B, observations)

    print_results(true_stages, observations, predicted_path, score)

    print("\nGerando figuras...")
    plot_fig6_matrizes_hmm()
    plot_fig1_sinal_bruto(signal, true_stages, FS, WINDOW_SEC)
    plot_fig2_fft_exemplos(signal, true_stages, FS, WINDOW_SAMPLES)
    plot_fig3_potencias_por_janela(all_band_powers, observations, true_stages)
    plot_fig4_trellis(trellis, backpointers, predicted_path, observations)
    plot_fig5_hipnograma(true_stages, observations, predicted_path)
    print("\nPipeline completo. Veja as figuras em ordem: fig6 → fig1 → fig2 → fig3 → fig4 → fig5")