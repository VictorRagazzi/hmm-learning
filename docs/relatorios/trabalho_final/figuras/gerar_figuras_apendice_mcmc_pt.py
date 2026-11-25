"""Gera as figuras de diagnóstico MCMC por fold (LOSO) para o apêndice do relatório IEEE em português.

Para cada um dos 11 folds (um por participante deixado de fora), re-executa apenas o
amostrador Metropolis-dentro-de-Gibbs (`mcmc_completo.ajustar_completo`) com a mesma
semente do LOSO (`mc_comum.SEMENTE`), confere a reprodução exata dos diagnósticos
salvos em `results/rayleigh_hmm/exploratorio_monte_carlo/loso/resultados.json` e gera
uma figura composta 3x4 (trace, autocorrelação e distribuição a posteriori dos quatro
hiperparâmetros a_k, b_k, a_rho, b_rho), com R-hat, ESS e média ± dp anotados.

Uso (da raiz do repositório):
    .venv/bin/python docs/relatorios/trabalho_final/figuras/gerar_figuras_apendice_mcmc_pt.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------- configuração
PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[3]
sys.path[:0] = [str(RAIZ / "src/rayleigh_hmm"), str(RAIZ / "src/rayleigh_hmm/monte_carlo")]

import experimento as ex  # noqa: E402
import mc_comum as mc  # noqa: E402
import mcmc_completo as mcc  # noqa: E402

JSON_LOSO = RAIZ / "results/rayleigh_hmm/exploratorio_monte_carlo/loso/resultados.json"
N_PROCESSOS = 6
MAX_LAG = 120
DPI_PNG = 180
LARGURA, ALTURA = 7.16, 4.15

CINZA = "#333333"
GRADE = "#d9d9d9"
CORES_CADEIA = ["#2a78d6", "#eb6834", "#2ca02c", "#9467bd"]
ROTULOS_TEX = {"a_k": r"$a_k$", "b_k": r"$b_k$", "a_rho": r"$a_\rho$", "b_rho": r"$b_\rho$"}

SEQ = None
DIAG_REF = None


def configurar_estilo():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "pdf.fonttype": 42,
        "text.color": CINZA, "axes.labelcolor": CINZA, "axes.edgecolor": CINZA,
        "xtick.color": CINZA, "ytick.color": CINZA,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.35,
        "axes.axisbelow": True,
    })


def fmt_pt(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def plotar_fold(participante, res, diag):
    configurar_estilo()
    cad = np.exp(res["cadeias"])  # (4, 20000, 4)
    n_cad, n_iter, _ = cad.shape
    passos_k = np.arange(1, n_iter + 1) / 1000.0
    burn_k = mcc.BURN_IN / 1000.0
    bruto_log = res["cadeias"][:, mcc.BURN_IN:]
    pos = cad[:, mcc.BURN_IN::mcc.THIN]

    fig, eixos = plt.subplots(3, 4, figsize=(LARGURA, ALTURA))

    for j, nome in enumerate(mc.NOMES_PARAM):
        rhat = diag["rhat"][nome]
        ess_v = diag["ess_thin"][nome]
        media = diag["media_posteriori"][nome]
        dp = diag["desvio_posteriori"][nome]
        casas_m = 1 if nome == "b_rho" else 2

        # Linha 0: trace das 4 cadeias
        ax = eixos[0, j]
        for i in range(n_cad):
            ax.plot(passos_k, cad[i, :, j], color=CORES_CADEIA[i], lw=0.32, alpha=0.75,
                    label=f"C{i + 1}" if j == 0 else None)
        ax.axvline(burn_k, color="#222222", ls="--", lw=0.75, label="Burn-in" if j == 0 else None)
        ax.set_yscale("log")
        ax.set_xlim(0, n_iter / 1000.0)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_title(ROTULOS_TEX[nome], fontsize=8.5, pad=3)
        ax.text(0.97, 0.93, rf"$\hat{{R}} = {fmt_pt(rhat, 3)}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82))
        if j == 0:
            ax.set_ylabel("Valor (escala log)")
        ax.set_xlabel(r"Iteração ($\times 10^3$)")

        # Linha 1: autocorrelação pós-burn-in
        ax = eixos[1, j]
        ac = mc.autocorrelacao(bruto_log[:, :, j], max_lag=MAX_LAG)
        ax.plot(np.arange(len(ac)), ac, color="#2a78d6", lw=1.0)
        ax.axvline(mcc.THIN, color="#eb6834", ls="--", lw=0.8, label=f"Thin ({mcc.THIN})" if j == 0 else None)
        ax.axhline(0.0, color="#666666", lw=0.5)
        ax.set_xlim(0, MAX_LAG)
        ax.set_ylim(-0.15, 1.05)
        ax.text(0.97, 0.93, rf"$\mathrm{{ESS}} = {ess_v:.0f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82))
        if j == 0:
            ax.set_ylabel("Autocorrelação")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Defasagem (lag)")

        # Linha 2: distribuição a posteriori (pós-burn-in, thinned)
        ax = eixos[2, j]
        v = pos[:, :, j].ravel()
        ax.hist(v, bins=36, density=True, color="#2a78d6", alpha=0.55, edgecolor="white", linewidth=0.25)
        ax.axvline(media, color="#eb6834", ls="-", lw=1.0, label="Média" if j == 0 else None)
        ax.axvline(media - dp, color="#666666", ls=":", lw=0.75)
        ax.axvline(media + dp, color="#666666", ls=":", lw=0.75)
        ax.text(0.97, 0.93, rf"${fmt_pt(media, casas_m)} \pm {fmt_pt(dp, casas_m)}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.3,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82))
        if j == 0:
            ax.set_ylabel("Densidade")
        ax.set_xlabel(ROTULOS_TEX[nome])

    # Legenda compacta no topo
    handles, labels = eixos[0, 0].get_legend_handles_labels()
    h2, l2 = eixos[1, 0].get_legend_handles_labels()
    h3, l3 = eixos[2, 0].get_legend_handles_labels()
    fig.legend(handles + h2 + h3, labels + l2 + l3, loc="upper center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=1.4, columnspacing=1.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=0.7, w_pad=0.55)
    base = PASTA / f"fig_mcmc_{participante}_pt"
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{base}.png", dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def processar_fold(participante):
    t0 = time.time()
    alvos = [s for s in SEQ if s["participante"] != participante and s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    emi = {
        "z_sintetico": np.concatenate([s["z"] for s in alvos]),
        "n_sintetico": np.concatenate([s["n_ep"] for s in alvos]),
        "id_sintetico": np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)]),
    }
    res = mcc.ajustar_completo(emi, rotulo=f"fold_{participante}")
    diag = mc.diagnosticos(res, mcc.BURN_IN, mcc.THIN)

    # Conferência com o JSON oficial do LOSO GIBBS
    ref = DIAG_REF[participante]
    for nome in mc.NOMES_PARAM:
        diff_rhat = abs(diag["rhat"][nome] - ref["rhat"][nome])
        diff_med = abs(diag["media_posteriori"][nome] - ref["media_posteriori"][nome])
        assert diff_rhat < 1e-9, f"{participante} {nome} rhat divergente: {diag['rhat'][nome]} vs {ref['rhat'][nome]}"
        assert diff_med < 1e-9, f"{participante} {nome} media divergente: {diag['media_posteriori'][nome]} vs {ref['media_posteriori'][nome]}"

    plotar_fold(participante, res, diag)
    return participante, time.time() - t0, diag["rhat_max"], diag["ess_min_thin"]


def main():
    global SEQ, DIAG_REF
    t0 = time.time()
    SEQ, _ = ex.carregar_sequencias()
    loso = json.load(open(JSON_LOSO))
    DIAG_REF = {f["participante"]: f["diag"] for f in loso["folds"]["GIBBS"]}
    participantes = sorted({s["participante"] for s in SEQ})

    print(f"Gerando figuras MCMC para {len(participantes)} folds ({N_PROCESSOS} processos)...", flush=True)
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        for p, seg, rmax, emin in pool.imap_unordered(processar_fold, participantes):
            print(f"  [OK] {p}: {seg:.1f} s | R-hat max = {rmax:.4f} | ESS min = {emin:.0f}", flush=True)
    print(f"Concluído em {time.time() - t0:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
