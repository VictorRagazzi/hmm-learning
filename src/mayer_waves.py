"""
Detecção de Ondas de Mayer em Dados Não-Regularmente Amostrados
usando Empirical Mode Decomposition (EMD) + Periodograma de Lomb-Scargle

Baseado em:
  Ragazzi et al. - "Detection of Mayer waves in non-regularly sampled data
  using Lomb-Scargle periodograms" (UFV / Maynooth University)

Dataset: EuroBaVar — beat-to-beat (arquivos *B.txt)
  Colunas: RR (s) | SBP (mmHg) | DBP (mmHg) | MBP (mmHg)
  Nomenclatura: A001LB = série A, paciente 001, L=supine, S=standing, B=beat-to-beat

Ondas de Mayer: oscilações de ~0.1 Hz na pressão arterial, reflexo
                da atividade simpática do sistema nervoso autônomo.

Pipeline completo:
  Arquivo → acumular RR → RR | EMD → média(IMFs escolhidas) → Lomb-Scargle → Detecção

Dependências: pip install numpy scipy matplotlib EMD-signal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import detrend, lombscargle
from PyEMD import EMD


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_DADOS     = "./data"
TIPO_SINAL      = "RR"            # sinal passado ao classificador no MATLAB
FREQ_MAX_HZ     = 0.4
FAIXA_MAYER_HZ  = (0.05, 0.15)
FP_PERCENT      = 1.0             # fp do MATLAB, em porcentagem
NUM_IMF         = (1, 2)          # índices MATLAB (começam em 1)
SUMA            = 1               # número mínimo de pontos acima do limiar
OFAC            = 4               # sobreamostragem padrão do plomb


# =============================================================================
# PARTE 1 — LEITURA DOS DADOS
# =============================================================================
# Cada arquivo beat-to-beat tem uma linha por batimento cardíaco.
# Coluna 0: intervalo RR em segundos (tempo entre batimentos)
# Coluna 3: MAP/MBP — pressão arterial média em mmHg
#
# Como o LS trabalha com instantes de tempo, precisamos converter os
# intervalos RR em timestamps absolutos: t[i] = sum(RR[0..i])
# Isso é exatamente o que o paper faz — os dados já são irregulares por
# natureza (o sensor lê apenas nos picos R da onda ECG).

def carregar_arquivo(caminho, tipo_sinal=TIPO_SINAL):
    """
    Lê um arquivo beat-to-beat do EuroBaVar.
    Retorna timestamps acumulados e RR (padrão) ou MAP.
    """
    dados = np.loadtxt(caminho)
    rr  = dados[:, 0]   # intervalos RR em segundos
    map_s = dados[:, 3]  # pressão arterial média em mmHg
    if tipo_sinal not in ("MAP", "RR"):
        raise ValueError("tipo_sinal deve ser 'MAP' ou 'RR'")

    # Acumular RR para obter o instante de cada batimento
    # t[0] = RR[0], t[1] = RR[0]+RR[1], ... (T = cumsum(RR) no MATLAB)
    t = np.cumsum(rr)

    return t, map_s if tipo_sinal == "MAP" else rr


def listar_arquivos():
    """
    Varre a pasta ./data e retorna uma lista de dicts com metadados de cada arquivo.
    Exemplo de nome: A001LB.txt → serie=A, paciente=001, posicao=L(supine)/S(standing)
    """
    arquivos = []
    for nome in sorted(os.listdir(PASTA_DADOS)):
        if not nome.endswith("B.txt"):   # só beat-to-beat
            continue
        stem = nome.replace(".txt", "")  # ex: A001LB
        serie    = stem[0]               # A ou B
        paciente = stem[1:4]             # 001, 002, ...
        posicao  = stem[4]               # L (supine) ou S (standing)
        arquivos.append({
            "nome":     nome,
            "caminho":  os.path.join(PASTA_DADOS, nome),
            "serie":    serie,
            "paciente": paciente,
            "posicao":  posicao,          # L=supine, S=standing
            "id":       f"{serie}{paciente}",
        })
    return arquivos


# =============================================================================
# PARTE 2 — EMD: Empirical Mode Decomposition
# =============================================================================
# O EMD decompõe o sinal em IMFs (Intrinsic Mode Functions):
#
#   IMF 1 → frequências MAIS ALTAS (ruído, variabilidade batimento a batimento)
#   IMF 2 → frequências intermediárias
#   IMF 3+ → frequências BAIXAS
#   Resíduo → tendência DC (deriva lenta da PA)
#
# Insight do paper:
#   - IMF1 sozinha: perde o sinal de Mayer (frequências muito altas)
#   - IMF2 sozinha: falsos positivos (amplitude alta na faixa de 0.1 Hz)
#   - MÉDIA(IMF1, IMF2): configuração padrão, ajustável em NUM_IMF

def aplicar_emd(sinal, n_imfs=3):
    emd  = EMD()
    # emd(sinal) no MATLAB usa a ordem das amostras, sem T_S na decomposição.
    imfs = emd.emd(sinal, max_imf=n_imfs)
    return imfs[:n_imfs]


def selecionar_imfs(imfs, num_imf=NUM_IMF):
    selecionadas = np.atleast_1d(num_imf)
    if selecionadas.size == 0 or np.any(selecionadas != np.floor(selecionadas)):
        raise ValueError("NumIMF deve conter índices inteiros")
    indices = selecionadas.astype(int)
    if np.any(indices < 1) or np.any(indices > len(imfs)):
        raise ValueError(f"NumIMF deve selecionar IMFs entre 1 e {len(imfs)}")
    return np.mean(imfs[indices - 1], axis=0)


# =============================================================================
# PARTE 3 — PERIODOGRAMA DE LOMB-SCARGLE
# =============================================================================
# O LS periodograma é análogo à FFT, mas para dados irregulares.
# Para cada frequência ω, ajusta: A·cos(ωt) + B·sin(ωt) e mede a potência.
#
# Normalização de plomb('normalized'):
#   P_N(ω) = potência_bruta / var(sinal_preprocessado, ddof=1)
#   Propriedade: se o sinal for só ruído, P_N segue distribuição exponencial.
#   → prob(P_N > z | ruído) = e^{-z}
#
# Limiar FAP aproximado: a documentação de plomb não especifica a
# implementação numérica de pth. Usamos M ~ fmax * duração (frequências
# independentes), não o número de pontos da grade sobreamostrada.
#
# Detecção: contar pontos >= limiar na faixa pedida, como no MATLAB.

def calcular_lombscargle(t, sinal_preprocessado, fp=FP_PERCENT):
    t = np.asarray(t, dtype=float)
    sinal_preprocessado = np.asarray(sinal_preprocessado, dtype=float)
    if t.ndim != 1 or sinal_preprocessado.shape != t.shape or len(t) < 3:
        raise ValueError("t e sinal devem ser vetores do mesmo tamanho, com ao menos 3 pontos")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(sinal_preprocessado))):
        raise ValueError("t e sinal devem conter somente valores finitos")
    if np.any(np.diff(t) <= 0) or t[0] < 0:
        raise ValueError("t deve ser não negativo e estritamente crescente")
    if not 0 < fp < 100:
        raise ValueError("fp deve estar entre 0 e 100 (porcentagem)")

    n = len(t)
    duracao = t[-1] - t[0]
    tempo_medio = duracao / (n - 1)
    fmin = 1.0 / (OFAC * n * tempo_medio)
    n_freqs = int(np.floor(FREQ_MAX_HZ / fmin + 0.5))  # round positivo do MATLAB
    if n_freqs < 1:
        raise ValueError("duração insuficiente para calcular frequências até 0,4 Hz")
    freqs_hz = fmin * np.arange(1, n_freqs + 1)
    freqs_rad = 2 * np.pi * freqs_hz

    s         = sinal_preprocessado - np.mean(sinal_preprocessado)
    variancia = np.var(s, ddof=1)
    if variancia == 0:
        raise ValueError("sinal constante não possui periodograma normalizado")
    pot_bruta = lombscargle(t, s, freqs_rad, normalize=False)
    pot       = pot_bruta / variancia

    m_independentes = max(1, int(np.floor(FREQ_MAX_HZ * duracao + 0.5)))
    log_pd = np.log1p(-fp / 100.0)
    limiar = -np.log(-np.expm1(log_pd / m_independentes))

    return freqs_hz, pot, limiar


def detectar_mayer(freqs, potencia, limiar, range_f=FAIXA_MAYER_HZ, suma=SUMA):
    if len(range_f) != 2 or range_f[0] > range_f[1]:
        raise ValueError("range_f deve ser um intervalo (mínimo, máximo)")
    if not isinstance(suma, (int, np.integer)) or suma < 1:
        raise ValueError("suma deve ser um inteiro positivo")
    mask = (freqs >= range_f[0]) & (freqs <= range_f[1])
    pico = np.max(potencia[mask]) if np.any(mask) else 0.0
    quantidade = int(np.count_nonzero(potencia[mask] >= limiar))
    return quantidade >= suma, pico, quantidade


def classificador(sinal, t_s, fp=FP_PERCENT, range_f=FAIXA_MAYER_HZ,
                  num_imf=NUM_IMF, suma=SUMA):
    """Classifica MAP ou RR; replica as etapas MATLAB com pth aproximado."""
    imfs = aplicar_emd(np.asarray(sinal, dtype=float), n_imfs=3)
    prep = selecionar_imfs(imfs, num_imf)
    freqs, pot, limiar = calcular_lombscargle(t_s, prep, fp)
    detectou, pico, quantidade = detectar_mayer(freqs, pot, limiar, range_f, suma)
    return {"detectou": detectou, "temounao": "Yes" if detectou else "No",
            "pico": pico, "quantidade": quantidade,
            "limiar": limiar, "freqs": freqs, "pot": pot,
            "imfs": imfs, "prep": prep}


# =============================================================================
# PARTE 4 — PIPELINE COMPLETO PARA TODOS OS ARQUIVOS
# =============================================================================

def calcular_fft_map(t, mapa):
    """Espectro de MAP após interpolação spline e detrend, como no exemplo MATLAB."""
    passo = 2 / 3
    n = int(np.floor((t[-1] - t[0]) / passo)) + 1
    t_regular = t[0] + passo * np.arange(n)
    mapa_regular = CubicSpline(t, mapa)(t_regular)
    mapa_sem_tendencia = detrend(mapa_regular)

    e = n // 2 + 1
    amplitude = 2 * np.abs(np.fft.fft(mapa_sem_tendencia) / n)[:e]
    frequencias = np.arange(1, e + 1) * (1 / passo) / n
    return frequencias, amplitude


def main():
    arquivos = listar_arquivos()
    print(f"Arquivos encontrados: {len(arquivos)}")
    for a in arquivos:
        pos = "supine  " if a["posicao"] == "L" else "standing"
        print(f"  {a['nome']:12}  →  série {a['serie']} | pac {a['paciente']} | {pos}")

    print("\n--- Rodando pipeline EMD-LS em todos os arquivos ---")
    print(f"{'Arquivo':14} | {'Posição':10} | {'Detecção EMD-LS':16} | {'Pico (P_N)':12} | {'Limiar':8}")
    print("-" * 70)

    resultados = []

    for arq in arquivos:
        try:
            t, sinal = carregar_arquivo(arq["caminho"], tipo_sinal=TIPO_SINAL)
            _, mapa = carregar_arquivo(arq["caminho"], tipo_sinal="MAP")
            resultado = classificador(sinal, t)
            detectou = resultado["detectou"]
            pico = resultado["pico"]
            lim = resultado["limiar"]

            posicao_str = "supine  " if arq["posicao"] == "L" else "standing"
            icone       = "✓ SIM" if detectou else "✗ NÃO"

            print(f"{arq['nome']:14} | {posicao_str:10} | {icone:16} | {pico:12.1f} | {lim:.1f}")

            resultados.append({**arq, **resultado, "t": t, "sinal": sinal,
                               "mapa": mapa})

        except Exception as e:
            print(f"{arq['nome']:14} | ERRO: {e}")


    # =============================================================================
    # PARTE 5 — VISUALIZAÇÃO: um subplot por arquivo
    # =============================================================================
    # Cada arquivo gera três painéis:
    #   - Esquerda: sinal bruto + pré-processamento (IMFs selecionadas)
    #   - Centro: periodograma LS com limiar e faixa de Mayer destacada
    #   - Direita: FFT da MAP interpolada

    n = len(resultados)
    fig, axes = plt.subplots(n, 3, figsize=(21, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # garantir shape (n, 3) mesmo com 1 arquivo

    fig.suptitle(
        "Detecção de Ondas de Mayer — EuroBaVar Dataset\n"
        "EMD (IMFs selecionadas) + Periodograma de Lomb-Scargle\n"
        "Baseado em Ragazzi et al. (UFV / Maynooth University)",
        fontsize=13, fontweight='bold'
    )

    COR_STANDING = "#1565C0"
    COR_SUPINE   = "#C62828"
    COR_LIM      = "#E65100"
    COR_MAY      = "#2E7D32"
    COR_PREP     = "#6A1B9A"

    for i, res in enumerate(resultados):
        cor = COR_STANDING if res["posicao"] == "S" else COR_SUPINE
        pos_str = "standing" if res["posicao"] == "S" else "supine"

        # ── Painel esquerdo: sinal bruto e pré-processamento ──
        ax_sig = axes[i, 0]
        ax_sig.plot(res["t"], res["sinal"], color=cor, lw=0.8, alpha=0.7,
                    label=f"{TIPO_SINAL} bruto")

        # Pré-processamento reescalado para sobrepor no mesmo eixo
        sinal_range = np.ptp(res["sinal"])
        prep_norm = res["prep"] / (np.std(res["prep"]) + 1e-9) * (sinal_range * 0.3)
        prep_off  = np.mean(res["sinal"])
        ax_sig.plot(res["t"], prep_norm + prep_off, color=COR_PREP, lw=1.2,
                    linestyle='--', label=f"Média das IMFs {NUM_IMF}", alpha=0.9)

        ax_sig.set_title(f"{res['nome']} — {pos_str} | {'✓ Mayer detectado' if res['detectou'] else '✗ Mayer não detectado'}",
                         fontweight='bold', color=cor)
        ax_sig.set_xlabel("Tempo (s)")
        ax_sig.set_ylabel("RR (s)" if TIPO_SINAL == "RR" else "MAP (mmHg)")
        ax_sig.legend(fontsize=7, loc='upper right')
        ax_sig.text(0.02, 0.04, f"{len(res['t'])} batimentos · {res['t'][-1]:.0f}s",
                    transform=ax_sig.transAxes, fontsize=8, color='gray')

        # ── Painel direito: periodograma LS ──
        ax_ls = axes[i, 1]
        ax_ls.plot(res["freqs"], res["pot"], color=cor, lw=1.3, label="Potência LS")
        ax_ls.axhline(res["limiar"], color=COR_LIM, lw=1.5, linestyle='--',
                      label=f"Limiar FAP {FP_PERCENT:g}% (aprox.) = {res['limiar']:.1f}")
        ax_ls.axvspan(*FAIXA_MAYER_HZ, alpha=0.12, color=COR_MAY,
                      label="Faixa Mayer (0.05–0.15 Hz)")
        ax_ls.axvline(0.10, color=COR_MAY, lw=0.9, linestyle=':', alpha=0.8)
        ax_ls.set_xlim(0, FREQ_MAX_HZ)
        ax_ls.set_ylim(0, 50)
        ax_ls.set_xlabel("Frequência (Hz)")
        ax_ls.set_ylabel("Potência norm. (P_N)")
        ax_ls.legend(fontsize=7, loc='upper right')

        # Marcar o pico na faixa de Mayer
        mask_may = (res["freqs"] >= FAIXA_MAYER_HZ[0]) & (res["freqs"] <= FAIXA_MAYER_HZ[1])
        if np.any(mask_may):
            idx_pico = np.argmax(res["pot"][mask_may])
            f_pico   = res["freqs"][mask_may][idx_pico]
            p_pico   = res["pot"][mask_may][idx_pico]
            ax_ls.plot(f_pico, p_pico, 'o', color=COR_MAY, ms=6, zorder=5,
                       label=f"Pico: {f_pico:.3f} Hz")
            ax_ls.legend(fontsize=7, loc='upper right')

        # ── Painel direito: FFT da MAP, como no exemplo MATLAB ──
        ax_fft = axes[i, 2]
        f_fft, amplitude_fft = calcular_fft_map(res["t"], res["mapa"])
        ax_fft.plot(f_fft, amplitude_fft, color=cor, lw=1)
        ax_fft.grid(True)
        ax_fft.set_xlabel("Frequência (Hz)")
        ax_fft.set_ylabel("Amplitude Espectral")
        ax_fft.set_title(f"FFT MAP — {res['nome']} — {pos_str}")
        ax_fft.set_xlim(0, 0.6)

    plt.tight_layout()
    os.makedirs("./figs", exist_ok=True)
    plt.savefig("./figs/resultados_mayer.png", dpi=120, bbox_inches='tight')
    print("\nFigura salva em ./figs/resultados_mayer.png")


    # =============================================================================
    # PARTE 6 — RESUMO FINAL
    # =============================================================================

    print("\n=== RESUMO ===")
    standing = [r for r in resultados if r["posicao"] == "S"]
    supine   = [r for r in resultados if r["posicao"] == "L"]

    det_standing = sum(r["detectou"] for r in standing)
    det_supine   = sum(r["detectou"] for r in supine)

    print(f"Standing: {det_standing}/{len(standing)} detectaram ondas de Mayer")
    print(f"Supine:   {det_supine}/{len(supine)} detectaram ondas de Mayer")
    print("Limiar FAP aproximado; para equivalência numérica de pth, compare com MATLAB.")


if __name__ == "__main__":
    main()
