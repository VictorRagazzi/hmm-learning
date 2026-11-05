"""Figuras do experimento: emissoes, priori x posteriori e deteccao por intensidade."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import emissao

COR_1, COR_2, COR_3 = "#2a78d6", "#eb6834", "#1baf7a"
COR_HIST = "#c3c2b7"
TINTA = "#3d3d3a"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COR_HIST,
        "axes.labelcolor": TINTA,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "axes.grid": True,
        "grid.color": "#e8e7e2",
        "grid.linewidth": 0.8,
        "font.size": 10,
    }
)


def figura_emissoes(pasta, emissoes):
    """Ausente em p-valor (Uniforme sob H0) e Presente: -ln p cresce com N."""
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))

    ax = eixos[0]
    aus = emissoes["ausente"]
    p = np.linspace(1e-4, 1, 400)
    z = -np.log(p)
    # densidade de p a partir da Gamma em Z: f_p(p) = f_Z(-ln p) / p
    ax.hist(np.exp(-emissoes["z_esp"]), bins=20, range=(0, 1), density=True, color=COR_HIST,
            label="p-valor no ESP")
    ax.plot(p, stats.gamma.pdf(z, aus["k"], scale=aus["mu"] / aus["k"]) / p, color=COR_1, lw=2,
            label=f"Ausente ajustada (k={aus['k']:.2f}, μ={aus['mu']:.2f})")
    ax.axhline(1.0, color=TINTA, lw=1.5, ls="--", label="Uniforme teórica (H0)")
    ax.set_ylim(0, 2)
    ax.set_title("Ausente: ESP", loc="left")
    ax.set_xlabel("p-valor de Rayleigh (acumulado)")
    ax.set_ylabel("densidade")
    ax.legend(frameon=False, loc="upper right")

    ax = eixos[1]
    post = emissoes["posteriori"]
    n = emissoes["n_sintetico"]
    jitter = np.random.default_rng(0).uniform(-3, 3, n.size)
    ax.scatter(n + jitter, emissoes["z_sintetico"], s=4, color=COR_HIST, label="alvos com estímulo (real)" if emissoes.get("calibracao") == "real" else "ESP + senoide")
    grade_n = np.arange(10, n.max() + 1)
    ax.plot(grade_n, emissao.media_presente(grade_n, post["map"]["rho"]), color=COR_2, lw=2,
            label=f"média Presente, ρ_MAP={post['map']['rho']:.4f}")
    ax.axhline(1.0, color=COR_1, lw=1.5, label="média Ausente (=1)")
    ax.axhline(-np.log(0.05), color=TINTA, lw=1, ls=":", label="p = 0,05")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_title("Presente sintético", loc="left")
    ax.set_xlabel("épocas acumuladas N")
    ax.set_ylabel("Z = −ln p")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "emissoes.png"), dpi=150)
    plt.close(fig)


def figura_posteriori(pasta, emissoes):
    post = emissoes["posteriori"]
    k, rho = post["grade_k"], post["grade_rho"]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    for ax, massa, titulo, mapa_cor in (
        (eixos[0], post["massa_priori"], "Priori de (k, ρ)", "Blues"),
        (eixos[1], post["massa_posteriori"], "Posteriori de (k, ρ)", "Oranges"),
    ):
        ax.contourf(k, rho, massa.T, levels=12, cmap=mapa_cor)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for eixo in (ax.xaxis, ax.yaxis):
            eixo.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel("forma k")
        ax.set_ylabel("ρ (PLV² da resposta)")
        ax.set_title(titulo, loc="left")
    m = post["map"]
    eixos[1].plot(m["k"], m["rho"], "o", color=TINTA, ms=8, label=f"MAP k={m['k']:.2f}, ρ={m['rho']:.4f}")
    eixos[1].legend(frameon=False)
    # zoom na posteriori, que e muito mais concentrada que a priori
    i, j = np.nonzero(post["massa_posteriori"] > post["massa_posteriori"].max() * 1e-3)
    eixos[1].set_xlim(k[i.min()] / 1.1, k[i.max()] * 1.1)
    eixos[1].set_ylim(rho[j.min()] / 1.1, rho[j.max()] * 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "posteriori.png"), dpi=150)
    plt.close(fig)


def figura_deteccao(pasta, saida):
    series = [
        ("bruto_referencia", "Rayleigh α=0,05, 3 consecutivas", COR_1),
        ("bruto", "Rayleigh α otimizado", COR_3),
        ("hmm_preditiva", "HMM (preditiva)", COR_2),
    ]
    linhas = [l for l in saida["tabela_por_intensidade"] if l["condicao"] != "ESP"]
    condicoes = list(dict.fromkeys(l["condicao"] for l in linhas))
    x = np.arange(len(condicoes))
    largura = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (nome, rotulo, cor) in enumerate(series):
        m = saida["metodos"][nome]
        taxas = [next(l["deteccao"] for l in linhas if l["metodo"] == nome and l["condicao"] == c)
                 for c in condicoes]
        ax.bar(x + (i - 1) * largura, taxas, largura - 0.02, color=cor,
               label=f"{rotulo} (FP lat. {m['metricas']['fp_lateral']:.1%})")
    ax.set_xticks(x, condicoes)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("taxa de detecção (alvos)")
    ax.set_title("Detecção por intensidade", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "deteccao_por_intensidade.png"), dpi=150)
    plt.close(fig)


def figura_paciente(pasta, linhas):
    """Mapa participante x intensidade com detectados/8 para bruto e HMM."""
    participantes = sorted({l["participante"] for l in linhas})
    condicoes = [c for c in dict.fromkeys(l["condicao"] for l in linhas) if c != "ESP"]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, nome, titulo in (
        (eixos[0], "bruto", "Rayleigh α otimizado"),
        (eixos[1], "hmm_preditiva", "HMM (preditiva)"),
    ):
        mapa = np.full((len(participantes), len(condicoes)), np.nan)
        for l in linhas:
            if l["condicao"] in condicoes:
                mapa[participantes.index(l["participante"]), condicoes.index(l["condicao"])] = (
                    l[f"{nome}_alvo_detectados"] / l[f"{nome}_alvo_total"]
                )
        ax.imshow(mapa, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(mapa):
            ax.text(j, i, f"{round(v * 8)}/8", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else TINTA)
        ax.set_xticks(range(len(condicoes)), condicoes)
        ax.set_yticks(range(len(participantes)), participantes)
        ax.set_title(titulo, loc="left")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "deteccao_por_paciente.png"), dpi=150)
    plt.close(fig)


def gerar(pasta, emissoes, saida, linhas):
    os.makedirs(pasta, exist_ok=True)
    figura_emissoes(pasta, emissoes)
    figura_posteriori(pasta, emissoes)
    figura_deteccao(pasta, saida)
    figura_paciente(pasta, linhas)
