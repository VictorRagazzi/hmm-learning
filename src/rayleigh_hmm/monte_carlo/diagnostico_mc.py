"""Diagnostico (EXPLORATORIO) do Monte Carlo e do MCMC nos dados completos, braco REAL.

Calibracao Presente montada como explorar_loso.ajustar_braco("REAL", ...), mas com
os 11 participantes (in-sample; serve para checar a integral, nao como resultado).
Saidas: results/rayleigh_hmm/exploratorio_monte_carlo/diagnostico/
Execucao: .venv/bin/python src/rayleigh_hmm/monte_carlo/diagnostico_mc.py  (usa GPU se houver; 4 fios BLAS)
"""
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "4")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import json
import os
import time

import numpy as np

import mc_comum as mc
from mc_comum import emissao, ex

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_monte_carlo", "diagnostico")
N_ITER_DIAG = 20000
BURN_IN_DIAG = 5000
Z_PARES = [0.5, 2.0, 5.0, 15.0]
N_PARES = [10, 60, 180]
M_LISTA = [100, 250, 500, 1000, 2000, 5000]
N_REPLICAS = 20
NOS_REFERENCIA = (80, 160)  # quadratura fina usada como referencia da integral


def montar_calibracao(seq):
    alvos = [s for s in seq if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    return {"z_sintetico": np.concatenate([s["z"] for s in alvos]),
            "n_sintetico": np.concatenate([s["n_ep"] for s in alvos]),
            "id_sintetico": np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])}


def logp(pontos):
    """Matriz (len(N), len(z)) de log p(z | N) para os pares da grade."""
    z, n = np.meshgrid(Z_PARES, N_PARES)
    return emissao.log_pdf_preditiva(z, n, pontos)


def main():
    t0 = time.time()
    seq, _ = ex.carregar_sequencias()
    emi = montar_calibracao(seq)
    t = time.time()
    marg = ex.ajustar_marginal(emi)
    seg_mlii = time.time() - t
    h = marg["hiper"]
    print(f"{len(np.unique(emi['id_sintetico']))} sequencias; ML-II em {seg_mlii:.1f} s:",
          {k: round(h[k], 4) for k in mc.NOMES_PARAM}, flush=True)

    # --- MCMC (config propria do diagnostico) ---
    res = mc.ajustar_mcmc(emi, n_iter=N_ITER_DIAG, burn_in=BURN_IN_DIAG, rotulo="diagnostico")
    diag = mc.diagnosticos(res, BURN_IN_DIAG, mc.THIN)
    lp = res["logpost"]
    theta_ml = np.log([[h[k] for k in mc.NOMES_PARAM]])
    diag["log_post_mlii"] = float(lp(theta_ml)[0])
    # maximo da log posteriori nas amostras pos burn-in (thin; blocos pequenos de theta)
    pos = res["cadeias"][:, BURN_IN_DIAG::mc.THIN].reshape(-1, 4)
    diag["log_post_max_cadeias"] = float(max(lp(pos[i:i + 16]).max() for i in range(0, len(pos), 16)))
    diag["ml2"] = {k: h[k] for k in mc.NOMES_PARAM}
    diag["segundos_mcmc"] = res["segundos"]
    diag["n_iter"], diag["burn_in"], diag["thin"] = N_ITER_DIAG, BURN_IN_DIAG, mc.THIN
    mc.figuras_mcmc(res, h, PASTA, "mcmc", BURN_IN_DIAG, mc.THIN)
    print("MCMC:", json.dumps({k: diag[k] for k in ("aceitacao", "rhat", "ess_thin", "media_posteriori")}), flush=True)

    # --- log p(z|N): QUAD vs MC vs MCMC vs referencia fina ---
    k_ref, rho_ref = emissao.nos_quadratura(h, *NOS_REFERENCIA)
    ref = logp({"k": k_ref, "rho": rho_ref, "peso": np.full(k_ref.size, 1.0 / k_ref.size)})
    quad = logp(marg["pontos"])
    p_mc = logp(mc.pontos_mc(h))
    pontos_mcmc = mc.pontos_mcmc(res, burn_in=BURN_IN_DIAG)
    p_mcmc = logp(pontos_mcmc)

    # --- convergencia vs M ---
    conv = np.empty((len(M_LISTA), N_REPLICAS, len(N_PARES), len(Z_PARES)))
    for i, m in enumerate(M_LISTA):
        for r in range(N_REPLICAS):
            conv[i, r] = logp(mc.pontos_mc(h, m, semente=1000 * m + r))
    media, desvio = conv.mean(axis=1), conv.std(axis=1, ddof=1)
    i2000 = M_LISTA.index(2000)

    linhas = ["| N | z | referencia fina | QUAD 10x24 | MC (M=2000) | MC dp (20 sementes) | MCMC (M=2000) |",
              "|---|---|---|---|---|---|---|"]
    for a, n in enumerate(N_PARES):
        for b, z in enumerate(Z_PARES):
            linhas.append(f"| {n} | {z} | {ref[a, b]:.4f} | {quad[a, b]:.4f} | {p_mc[a, b]:.4f} | "
                          f"{desvio[i2000, a, b]:.4f} | {p_mcmc[a, b]:.4f} |")

    # figura de convergencia
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eixos = plt.subplots(len(N_PARES), len(Z_PARES), figsize=(14, 8), sharex=True)
    for a, n in enumerate(N_PARES):
        for b, z in enumerate(Z_PARES):
            ax = eixos[a, b]
            ax.errorbar(M_LISTA, media[:, a, b], yerr=desvio[:, a, b], marker="o", capsize=3, label="MC: media +- dp")
            ax.axhline(ref[a, b], color="r", ls=":", label="referencia fina")
            ax.axhline(quad[a, b], color="g", ls="--", label="QUAD 10x24")
            ax.set_xscale("log")
            ax.set_title(f"N={n}, z={z}", fontsize=9)
    eixos[0, 0].legend(fontsize=7)
    fig.supxlabel("M")
    fig.supylabel("log p(z | N)")
    fig.tight_layout()
    os.makedirs(PASTA, exist_ok=True)
    fig.savefig(os.path.join(PASTA, "convergencia_M.png"), dpi=110)
    plt.close(fig)

    # tabela de convergencia
    conv_md = ["| M | " + " | ".join(f"N={n},z={z}" for n in N_PARES for z in Z_PARES) + " |",
               "|---|" + "---|" * (len(N_PARES) * len(Z_PARES))]
    for i, m in enumerate(M_LISTA):
        conv_md.append(f"| {m} | " + " | ".join(
            f"{media[i, a, b] - ref[a, b]:+.4f} +- {desvio[i, a, b]:.4f}"
            for a in range(len(N_PARES)) for b in range(len(Z_PARES))) + " |")

    # medias a posteriori vs ML-II
    pm = ["| Parametro | ML-II | media posteriori | dp posteriori | R-hat | ESS (thin) | ESS (sem thin) |",
          "|---|---|---|---|---|---|---|"]
    for k in mc.NOMES_PARAM:
        pm.append(f"| {k} | {h[k]:.4g} | {diag['media_posteriori'][k]:.4g} | {diag['desvio_posteriori'][k]:.3g} | "
                  f"{diag['rhat'][k]:.4f} | {diag['ess_thin'][k]:.0f} | {diag['ess_sem_thin'][k]:.0f} |")

    md = ["# Diagnostico do Monte Carlo / MCMC (exploratorio, 11 participantes, braco REAL, in-sample)\n",
          f"MCMC: {N_ITER_DIAG} iteracoes, burn-in {BURN_IN_DIAG}, thin {mc.THIN}, 4 cadeias; "
          f"{diag['n_amostras_agrupadas']} amostras agrupadas; {res['segundos']:.0f} s. "
          f"Aceitacao por cadeia: {', '.join(f'{x:.3f}' for x in diag['aceitacao'])}.\n",
          f"log posteriori no ML-II: {diag['log_post_mlii']:.3f}; maximo visitado pelas cadeias: "
          f"{diag['log_post_max_cadeias']:.3f}.\n",
          "## Hiperparametros\n", *pm,
          "\n## log p(z | N)\n", *linhas,
          f"\n## Vies MC - referencia fina por M (media +- dp de {N_REPLICAS} sementes)\n", *conv_md,
          f"\nTempo total: {time.time() - t0:.0f} s."]
    open(os.path.join(PASTA, "diagnostico.md"), "w").write("\n".join(md) + "\n")
    json.dump({"hiper_mlii": diag["ml2"], "diagnosticos": diag, "N": N_PARES, "z": Z_PARES, "M": M_LISTA,
               "referencia": ref.tolist(), "quad": quad.tolist(), "mc_2000": p_mc.tolist(),
               "mcmc_2000": p_mcmc.tolist(), "mc_media": media.tolist(), "mc_desvio": desvio.tolist()},
              open(os.path.join(PASTA, "diagnostico.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
