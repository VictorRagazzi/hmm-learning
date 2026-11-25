"""Gera as figuras do relatório IEEE em português (sufixo _pt): densidades de emissão e taxas de detecção.

Uso (da raiz do repositório):
    .venv/bin/python docs/relatorios/Final_Project/figuras/gerar_figuras_pt.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- configuração
PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[3]
sys.path[:0] = [str(RAIZ / "src/rayleigh_hmm"), str(RAIZ / "src/rayleigh_hmm/monte_carlo")]
import emissao  # noqa: E402
import experimento as ex  # noqa: E402
import mc_comum as mc  # noqa: E402

JSON_HIPER = RAIZ / "results/rayleigh_hmm/exploratorio_monte_carlo/completo/completo.json"

AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#333333"
GRADE = "#d9d9d9"
LARGURA, ALTURA = 7.16, 2.3
DPI_PNG = 200
Z_MAX = 12.0
BIN = 0.25
N_EPOCAS = [30, 120, 480]
ESTILOS_N = {30: ":", 120: "--", 480: "-"}
ALFA = 0.05

# Contagens de detecção (de 88 sequências por intensidade, já conferidas).
# Painel (a): results/rayleigh_hmm/exploratorio_monte_carlo/loso (variante GIBBS)
#   e Rayleigh de referência.
# Painel (b): results/rayleigh_hmm/exploratorio_detectores_mcmc/tabelas.md
#   (Rayleigh, emissão Presente por MCMC completo, teto de 6% de falso positivo).
TOTAL = 88
CONTAGENS = {
    "Registros isolados": {"Rayleigh": [39, 63, 63, 47, 34], "HMM": [37, 58, 62, 47, 34]},
    "Protocolo concatenado": {"Rayleigh": [36, 55, 58, 44, 34], "HMM": [32, 54, 59, 57, 52]},
}
ROTULOS_X = ["30 dB\n8 min", "40 dB\n8 min", "50 dB\n5 min", "60 dB\n2–3 min", "70 dB\n1 min"]
LARG_BARRA = 0.38
FOLGA = 0.02

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42,
    "text.color": CINZA, "axes.labelcolor": CINZA, "axes.edgecolor": CINZA,
    "xtick.color": CINZA, "ytick.color": CINZA,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRADE, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})


def estilizar(ax, letra):
    """Título '(a)'/'(b)' pequeno à esquerda do painel."""
    ax.set_title(letra, loc="left", fontsize=8)


def salvar(fig, nome):
    for ext, kw in (("pdf", {}), ("png", {"dpi": DPI_PNG})):
        fig.savefig(PASTA / f"{nome}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


# ------------------------------------------------------------------- figura 1
def figura_emissoes():
    seq, _ = ex.carregar_sequencias()
    z_esp = np.concatenate([s["z"] for s in seq if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    assert z_esp.size == 1160, z_esp.size

    aus = emissao.ajustar_mle(z_esp)
    print(f"Ausente: k = {aus['k']:.4f}, mu = {aus['mu']:.4f}")
    hiper = json.load(open(JSON_HIPER))["completo"]["media_posteriori"]
    hiper = {c: hiper[c] for c in ("a_k", "b_k", "a_rho", "b_rho")}
    print("Hiperparâmetros:", {c: round(v, 4) for c, v in hiper.items()})
    pontos = mc.pontos_mc(hiper)

    z = np.linspace(0.01, Z_MAX, 600)
    b_aus = np.exp(emissao.log_pdf_gamma(z, aus["k"], aus["mu"]))
    b_pres = {n: np.exp(emissao.log_pdf_preditiva(z, n, pontos)) for n in N_EPOCAS}
    razao = {n: b_pres[n] / b_aus for n in N_EPOCAS}
    for zz, n in ((6, 120), (1, 120), (0.5, 480)):
        r = np.exp(emissao.log_pdf_preditiva(np.array([float(zz)]), n, pontos)
                   - emissao.log_pdf_gamma(np.array([float(zz)]), aus["k"], aus["mu"]))
        print(f"Razão Present/Absent em Z={zz}, N={n}: {r[0]:.4f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGURA, ALTURA))

    # (a) densidades
    a.hist(z_esp, bins=np.arange(0, Z_MAX + BIN, BIN), density=True, color=AZUL, alpha=0.3,
           linewidth=0, label="Registros sem estímulo")
    a.plot(z, b_aus, color=AZUL, lw=1.6, label="Ausente")
    for n in N_EPOCAS:
        a.plot(z, b_pres[n], color=LARANJA, ls=ESTILOS_N[n], lw=1.4, label=f"Presente, N = {n}")
    lim = -np.log(ALFA)
    a.axvline(lim, color="#888888", lw=0.6, ls=(0, (4, 3)))
    a.text(lim + 0.15, 0.62, r"Limiar de Rayleigh ($\alpha$ = 0,05)", rotation=90,
           fontsize=6.5, color="#666666", va="center", ha="left")
    a.set_xlim(0, Z_MAX); a.set_ylim(0, 1.1)
    a.set_xlabel("Z (estatística de Rayleigh acumulada)"); a.set_ylabel("Densidade")
    a.legend(loc="upper right", frameon=False)
    estilizar(a, "(a) Densidades de emissão")

    # (b) razão de verossimilhança
    for n in N_EPOCAS:
        b.plot(z, razao[n], color=LARANJA, ls=ESTILOS_N[n], lw=1.4, label=f"N = {n}")
    b.axhline(1, color="#888888", lw=0.6)
    b.set_yscale("log"); b.set_xlim(0, Z_MAX); b.set_ylim(0.1, 1000)
    b.text(0.3, 30, "favorece Presente \u2191", fontsize=6.5, color="#666666", va="center")
    b.text(4.5, 0.55, "favorece Ausente \u2193", fontsize=6.5, color="#666666", va="center")
    b.set_xlabel("Z (estatística de Rayleigh acumulada)"); b.set_ylabel("Razão de verossimilhança Presente / Ausente")
    b.legend(loc="lower right", frameon=False)
    estilizar(b, "(b) Evidência por observação")

    fig.tight_layout(w_pad=1.5)
    salvar(fig, "fig_emissoes_pt")


# ------------------------------------------------------------------- figura 2
def figura_deteccao():
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, ALTURA), sharey=True)
    x = np.arange(len(ROTULOS_X))
    for ax, letra, (titulo, dados) in zip(eixos, "ab", CONTAGENS.items()):
        d = (LARG_BARRA + FOLGA) / 2  # deslocamento de cada barra em relação ao centro do grupo
        for desloc, nome, cor, hach in ((-d, "Rayleigh", AZUL, None), (d, "HMM", LARANJA, "///")):
            pct = 100 * np.array(dados[nome]) / TOTAL
            plt.rcParams["hatch.linewidth"] = 0.8
            ax.bar(x + desloc, pct, LARG_BARRA, color=cor, hatch=hach,
                   edgecolor="white" if hach else cor, linewidth=0, label=nome)
            for xi, p in zip(x + desloc, pct):
                ax.text(xi, p + 1, f"{p:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=6.5, color=CINZA)
        ax.set_xticks(x); ax.set_xticklabels(ROTULOS_X)
        ax.set_xlabel("Intensidade do estímulo (duração do registro)")
        ax.set_ylim(0, 100)
        estilizar(ax, f"({letra}) {titulo}")
    eixos[0].set_ylabel("Taxa de detecção (%)")
    eixos[0].legend(loc="upper right", frameon=False)
    fig.tight_layout(w_pad=1.5)
    salvar(fig, "fig_deteccao_pt")


if __name__ == "__main__":
    figura_deteccao()
    figura_emissoes()
