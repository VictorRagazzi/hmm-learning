"""MCMC "completo" (EXPLORATORIO): Metropolis-dentro-de-Gibbs em (k_s, rho_s) e hiperparametros.

Modelo: z_t | k_s, rho_s ~ Gamma(forma k_s, media 1 + (N_t - 1) rho_s);
k_s ~ Gamma(a_k, taxa b_k), rho_s ~ Gamma(a_rho, taxa b_rho); hiperpriori uniforme
em log de cada hiperparametro dentro de mc.LIMITES_LOG.

Cada iteracao (as 4 cadeias avancam juntas, vetorizado em (cadeias, sequencias)):
  a) Metropolis de passeio aleatorio em log k_s e depois em log rho_s, todas as
     sequencias de uma vez (aceitacao independente por cadeia x sequencia);
  b) b_k | a_k, {k_s} e b_rho | a_rho, {rho_s}: Gibbs exato (Gamma);
  c) a_k | b_k, {k_s} e a_rho | b_rho, {rho_s}: Metropolis em log a.
Passos adaptados so no burn-in. O amostrador colapsado (mc.mcmc_hiper) fica como conferencia.
Saidas: results/rayleigh_hmm/exploratorio_monte_carlo/completo/
Execucao: cd src/rayleigh_hmm/monte_carlo && ../../../.venv/bin/python mcmc_completo.py
"""
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "4")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import json
import os
import time

import numpy as np
from scipy import special

import mc_comum as mc
from mc_comum import emissao, ex

# ---------------- configuracao ----------------
N_ITER = 20000
BURN_IN = 5000
THIN = 10
JANELA_ADAPTACAO = 50        # so durante o burn-in
ACEITACAO_ALVO = 0.44        # passos univariados
N_EXEMPLOS = 3               # sequencias de exemplo (maior Z, mediana, menor Z)
DURACAO_MINIMA_EXEMPLOS_S = 480.0   # exemplos so entre gravacoes longas (30 e 40 dB)
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_monte_carlo", "completo")
N_ITER_COLAPSADO, BURN_IN_COLAPSADO = 20000, 5000   # mesma config do diagnostico_mc
PASSO_INICIAL = 0.5          # desvio inicial dos passos em log
PASSO_INICIAL_HIPER = 0.1
LIMITE_PASSO = (1e-3, 20.0)


# ---------------- amostrador ----------------
def preparar_dados(z, n_ep, ids):
    """Ordena por sequencia e devolve estatisticas suficientes (T_s, soma log z_s) e indices."""
    z = np.clip(np.asarray(z, dtype=float).ravel(), emissao.EPS, None)
    n_ep = np.asarray(n_ep, dtype=float).ravel()
    _, ids = np.unique(np.asarray(ids).ravel(), return_inverse=True)
    ordem = np.argsort(ids, kind="stable")
    z, n_ep, ids = z[ordem], n_ep[ordem], ids[ordem]
    inicios = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    return {"z": z, "n_m1": n_ep - 1.0, "ids": ids, "inicios": inicios,
            "T": np.diff(np.r_[inicios, z.size]).astype(float),
            "soma_log_z": np.add.reduceat(np.log(z), inicios)}


def soma_rho(d, rho):
    """S_s(rho) = soma_t [log mu_t + z_t / mu_t] por sequencia; rho (C, S) -> (C, S)."""
    mu = 1.0 + d["n_m1"][None, :] * rho[:, d["ids"]]
    return np.add.reduceat(np.log(mu) + d["z"][None, :] / mu, d["inicios"], axis=1)


def log_alvo_latente(x, loglik_extra, a, b, d):
    """Alvo em coordenada log: log L_s + a log x - b x (x = k ou rho); loglik_extra ja inclui L_s."""
    return loglik_extra + a[:, None] * np.log(x) - b[:, None] * x


def log_lik_k(k, s_rho, d):
    return d["T"][None, :] * (k * np.log(k) - special.gammaln(k)) + (k - 1.0) * d["soma_log_z"][None, :] - k * s_rho


def passo_hiper(log_a, log_b, soma_log_x, n_seq, passo, rng):
    """Metropolis em log a: alvo S (a log b - lgamma a) + a soma_s log x_s; rejeita fora dos limites."""
    def alvo(la):
        a = np.exp(la)
        return n_seq * (a * log_b - special.gammaln(a)) + a * soma_log_x
    prop = log_a + passo * rng.standard_normal(log_a.shape)
    dentro = (prop >= mc.LIMITES_LOG[0]) & (prop <= mc.LIMITES_LOG[1])
    razao = np.where(dentro, alvo(np.where(dentro, prop, log_a)) - alvo(log_a), -np.inf)
    ok = np.log(rng.uniform(size=log_a.shape)) < razao
    return np.where(ok, prop, log_a), ok


def gibbs_taxa(a, x, rng):
    """b | a, {x_s}: Gamma(forma S a, taxa soma x); reamostra se cair fora de LIMITES_LOG."""
    forma, taxa = x.shape[1] * a, x.sum(axis=1)
    b = rng.gamma(forma, 1.0 / taxa)
    for _ in range(100):
        fora = (np.log(b) < mc.LIMITES_LOG[0]) | (np.log(b) > mc.LIMITES_LOG[1])
        if not fora.any():
            break
        b[fora] = rng.gamma(forma[fora], 1.0 / taxa[fora])
    return np.clip(b, np.exp(mc.LIMITES_LOG[0]), np.exp(mc.LIMITES_LOG[1]))


def mcmc_completo(z, n_ep, ids, n_iter=N_ITER, burn_in=BURN_IN, semente=mc.SEMENTE, rotulo="",
                  idx_exemplos=(), thin=THIN):
    """Retorna dict compativel com mc.diagnosticos / mc.pontos_mcmc / mc.figuras_mcmc.

    cadeias (C, n_iter, 4) em log (a_k, b_k, a_rho, b_rho); latentes_exemplos
    (C, n_iter, len(idx_exemplos), 2) = (log k_s, log rho_s) completos; latentes_thin
    (C, n_pos//thin, S, 2) float32 pos burn-in de todas as sequencias.
    """
    rng = np.random.default_rng(semente)
    d = preparar_dados(z, n_ep, ids)
    n_seq = d["T"].size
    inicios = np.log(np.array(mc.INICIOS_CADEIAS, dtype=float))
    c = inicios.shape[0]
    ak, bk, ar, br = np.exp(inicios).T.copy()
    lk = rng.normal(0.0, 1.0, (c, n_seq))
    lr = rng.normal(np.log(0.01), 1.5, (c, n_seq))
    k, rho = np.exp(lk), np.exp(lr)
    s_rho = soma_rho(d, rho)
    passo = np.full((c, n_seq, 2), PASSO_INICIAL)
    passo_h = np.full((c, 2), PASSO_INICIAL_HIPER)
    acc_lat = np.zeros((c, n_seq, 2))     # janela de adaptacao
    acc_h = np.zeros((c, 2))
    cadeias = np.empty((c, n_iter, 4))
    aceita_h = np.zeros((c, n_iter, 2), bool)
    soma_acc_lat = np.zeros(2)            # aceitacao media dos latentes (pos burn-in)
    soma_acc_lat_bi = np.zeros(2)
    idx_exemplos = list(idx_exemplos)
    ex_chain = np.empty((c, n_iter, len(idx_exemplos), 2))
    n_fino = len(range(burn_in, n_iter, thin))
    fino = np.empty((c, n_fino, n_seq, 2), np.float32)
    i_fino = 0
    passo_log = max(1, n_iter // mc.PASSOS_PROGRESSO)
    t_ini = time.time()
    for t in range(n_iter):
        # a) latentes
        prop = lk + passo[:, :, 0] * rng.standard_normal((c, n_seq))
        kp = np.exp(prop)
        alvo_p = log_alvo_latente(kp, log_lik_k(kp, s_rho, d), ak, bk, d)
        alvo_a = log_alvo_latente(k, log_lik_k(k, s_rho, d), ak, bk, d)
        ok_k = np.log(rng.uniform(size=(c, n_seq))) < alvo_p - alvo_a
        lk, k = np.where(ok_k, prop, lk), np.where(ok_k, kp, k)

        prop = lr + passo[:, :, 1] * rng.standard_normal((c, n_seq))
        rp = np.exp(prop)
        s_p = soma_rho(d, rp)
        alvo_p = log_alvo_latente(rp, log_lik_k(k, s_p, d), ar, br, d)
        alvo_a = log_alvo_latente(rho, log_lik_k(k, s_rho, d), ar, br, d)
        ok_r = np.log(rng.uniform(size=(c, n_seq))) < alvo_p - alvo_a
        lr, rho, s_rho = np.where(ok_r, prop, lr), np.where(ok_r, rp, rho), np.where(ok_r, s_p, s_rho)

        # b) Gibbs nas taxas; c) Metropolis nas formas
        bk = gibbs_taxa(ak, k, rng)
        br = gibbs_taxa(ar, rho, rng)
        lak, ok_a = passo_hiper(np.log(ak), np.log(bk), lk.sum(axis=1), n_seq, passo_h[:, 0], rng)
        lar, ok_b = passo_hiper(np.log(ar), np.log(br), lr.sum(axis=1), n_seq, passo_h[:, 1], rng)
        ak, ar = np.exp(lak), np.exp(lar)

        cadeias[:, t] = np.log(np.stack([ak, bk, ar, br], axis=1))
        aceita_h[:, t, 0], aceita_h[:, t, 1] = ok_a, ok_b
        if idx_exemplos:
            ex_chain[:, t, :, 0], ex_chain[:, t, :, 1] = lk[:, idx_exemplos], lr[:, idx_exemplos]
        acc_lat[:, :, 0] += ok_k
        acc_lat[:, :, 1] += ok_r
        acc_h[:, 0] += ok_a
        acc_h[:, 1] += ok_b
        taxas = np.array([ok_k.mean(), ok_r.mean()])
        if t < burn_in:
            soma_acc_lat_bi += taxas
        else:
            soma_acc_lat += taxas
            if (t - burn_in) % thin == 0:
                fino[:, i_fino, :, 0], fino[:, i_fino, :, 1] = lk, lr
                i_fino += 1

        if t < burn_in and (t + 1) % JANELA_ADAPTACAO == 0:
            passo = np.clip(passo * np.exp(acc_lat / JANELA_ADAPTACAO - ACEITACAO_ALVO), *LIMITE_PASSO)
            passo_h = np.clip(passo_h * np.exp(acc_h / JANELA_ADAPTACAO - ACEITACAO_ALVO), *LIMITE_PASSO)
            acc_lat[:] = 0
            acc_h[:] = 0
        if (t + 1) % passo_log == 0:
            ini = max(0, t + 1 - passo_log)
            print(f"[GIBBS {rotulo}] iter {t + 1}/{n_iter} aceitacao hiper (ultimo bloco) "
                  f"{aceita_h[:, ini:t + 1].mean():.2f} tempo {time.time() - t_ini:.0f} s", flush=True)
    return {"cadeias": cadeias,
            "aceitacao": aceita_h[:, burn_in:].mean(axis=(1, 2)),
            "aceitacao_burn_in": aceita_h[:, :burn_in].mean(axis=(1, 2)),
            "aceitacao_a_k": aceita_h[:, burn_in:, 0].mean(axis=1),
            "aceitacao_a_rho": aceita_h[:, burn_in:, 1].mean(axis=1),
            "aceitacao_k_latente": float(soma_acc_lat[0] / max(n_iter - burn_in, 1)),
            "aceitacao_rho_latente": float(soma_acc_lat[1] / max(n_iter - burn_in, 1)),
            "aceitacao_k_latente_burn_in": float(soma_acc_lat_bi[0] / max(burn_in, 1)),
            "aceitacao_rho_latente_burn_in": float(soma_acc_lat_bi[1] / max(burn_in, 1)),
            "latentes_exemplos": ex_chain, "latentes_thin": fino,
            "segundos": time.time() - t_ini}


def ajustar_completo(emissoes, semente=mc.SEMENTE, n_iter=N_ITER, burn_in=BURN_IN, rotulo="",
                     idx_exemplos=(), thin=THIN):
    """Analogo a mc.ajustar_mcmc, com o amostrador completo."""
    return mcmc_completo(emissoes["z_sintetico"], emissoes["n_sintetico"], emissoes["id_sintetico"],
                         n_iter, burn_in, semente, rotulo, idx_exemplos, thin)


# ---------------- diagnostico ----------------
def resumo_latentes(fino):
    """R-hat e ESS por sequencia para log k e log rho (series ja com thin)."""
    saida = {}
    for j, nome in enumerate(("log_k", "log_rho")):
        x = fino[:, :, :, j].astype(float)
        rhat = np.array([mc.split_rhat(x[:, :, s]) for s in range(x.shape[2])])
        essv = np.array([mc.ess(x[:, :, s]) for s in range(x.shape[2])])
        saida[nome] = {"rhat_max": float(rhat.max()), "rhat_mediano": float(np.median(rhat)),
                       "ess_min": float(essv.min()), "ess_mediano": float(np.median(essv)),
                       "rhat": rhat, "ess": essv}
    return saida


def escolher_exemplos(seq, alvos):
    """Entre as gravacoes longas, sequencias de maior Z, mediana e menor Z (Z do ultimo passo)."""
    dur = np.array([s["duracao_gravacao_s"] for s in alvos])
    cand = np.flatnonzero(dur >= DURACAO_MINIMA_EXEMPLOS_S)
    zf = np.array([alvos[i]["z"][-1] for i in cand])
    ordem = cand[np.argsort(zf)]
    return [int(ordem[-1]), int(ordem[len(ordem) // 2]), int(ordem[0])]


def nome_seq(s):
    return f"{s['participante']} / {s['condicao']} / {s['frequencia_hz']:g} Hz (Z final {s['z'][-1]:.1f}, T={len(s['z'])})"


def figuras_completo(res, col, h, exemplos, titulos, pasta, burn_in=BURN_IN):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cad = np.exp(res["cadeias"])
    n_cad, n_iter, _ = cad.shape
    # media acumulada
    fig, eixos = plt.subplots(1, 4, figsize=(15, 3.4))
    passos = np.arange(1, n_iter + 1)
    for j, (ax, nome) in enumerate(zip(eixos, mc.NOMES_PARAM)):
        for i in range(n_cad):
            ax.plot(np.cumsum(cad[i, :, j]) / passos, lw=0.8, label=f"cadeia {i + 1}")
        ax.axvline(burn_in, color="k", ls="--", lw=1)
        ax.axhline(h[nome], color="r", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteracao (escala log)")
        ax.set_title(f"media acumulada de {nome}")
    eixos[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "media_acumulada.png"), dpi=110)
    plt.close(fig)

    # trace dos latentes
    ex_c = res["latentes_exemplos"]
    for arq, plotar in (("latentes_trace.png", "trace"), ("latentes_autocorrelacao.png", "ac")):
        fig, eixos = plt.subplots(2, len(exemplos), figsize=(5 * len(exemplos), 6), squeeze=False,
                                  sharex=(plotar == "ac"))
        for e in range(len(exemplos)):
            for j, nome in enumerate(("k_s", "rho_s")):
                ax = eixos[j, e]
                if plotar == "trace":
                    for i in range(n_cad):
                        ax.plot(np.arange(1, n_iter + 1), np.exp(ex_c[i, :, e, j]), lw=0.4)
                    ax.axvline(burn_in, color="k", ls="--", lw=1)
                    ax.set_yscale("log")
                    ax.set_xscale("log")
                    ax.set_xlabel("iteracao (escala log)")
                else:
                    ac = mc.autocorrelacao(ex_c[:, burn_in:, e, j], max_lag=200)
                    ax.plot(ac)
                    ax.axhline(0, color="k", lw=0.5)
                    ax.set_xlabel("lag")
                ax.set_ylabel(nome if plotar == "trace" else f"autocorr. log {nome}")
            eixos[0, e].set_title(titulos[e], fontsize=8)
        fig.suptitle("Trace dos latentes (tracejado: fim do burn-in)" if plotar == "trace"
                     else "Autocorrelacao pos burn-in dos latentes")
        fig.tight_layout()
        fig.savefig(os.path.join(pasta, arq), dpi=110)
        plt.close(fig)

    # dispersao
    fig, eixos = plt.subplots(1, len(exemplos), figsize=(5 * len(exemplos), 4.2), squeeze=False)
    for e in range(len(exemplos)):
        a, b = ex_c[:, burn_in:, e, 0].ravel(), ex_c[:, burn_in:, e, 1].ravel()
        eixos[0, e].plot(a[::5], b[::5], ".", ms=1.5, alpha=0.4)
        eixos[0, e].set_xlabel("log k_s")
        eixos[0, e].set_ylabel("log rho_s")
        eixos[0, e].set_title(f"{titulos[e]}\ncorrelacao = {np.corrcoef(a, b)[0, 1]:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "latentes_dispersao.png"), dpi=110)
    plt.close(fig)

    # comparacao completo x colapsado
    fig, eixos = plt.subplots(1, 4, figsize=(15, 3.4))
    for j, (ax, nome) in enumerate(zip(eixos, mc.NOMES_PARAM)):
        a = res["cadeias"][:, burn_in::THIN, j].ravel()
        b = col["cadeias"][:, BURN_IN_COLAPSADO::mc.THIN, j].ravel()
        faixa = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 60)
        ax.hist(a, bins=faixa, density=True, alpha=0.5, label="completo")
        ax.hist(b, bins=faixa, density=True, alpha=0.5, label="colapsado")
        ax.axvline(np.log(h[nome]), color="r", label="ML-II")
        ax.set_xlabel(f"log {nome}")
    eixos[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "comparacao_colapsado.png"), dpi=110)
    plt.close(fig)


def main():
    t0 = time.time()
    from diagnostico_mc import montar_calibracao
    seq, _ = ex.carregar_sequencias()
    alvos = [s for s in seq if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    emi = montar_calibracao(seq)
    h = ex.ajustar_marginal(emi)["hiper"]
    exemplos = escolher_exemplos(seq, alvos)
    titulos = [nome_seq(alvos[i]) for i in exemplos]
    print("exemplos:", titulos, flush=True)

    res = ajustar_completo(emi, rotulo="diagnostico", idx_exemplos=exemplos)
    diag = mc.diagnosticos(res, BURN_IN, THIN)
    col = mc.ajustar_mcmc(emi, n_iter=N_ITER_COLAPSADO, burn_in=BURN_IN_COLAPSADO, rotulo="colapsado")
    dcol = mc.diagnosticos(col, BURN_IN_COLAPSADO, mc.THIN)
    lat = resumo_latentes(res["latentes_thin"])
    n_pos = (N_ITER - BURN_IN) * res["cadeias"].shape[0]
    os.makedirs(PASTA, exist_ok=True)
    mc.figuras_mcmc(res, h, PASTA, "completo", BURN_IN, THIN)
    figuras_completo(res, col, h, exemplos, titulos, PASTA)

    tab = ["| Parametro | ML-II | completo media | completo dp | colapsado media | colapsado dp | R-hat | ESS (thin) | ESS (sem thin) | tau integrado |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in mc.NOMES_PARAM:
        tab.append(f"| {k} | {h[k]:.4g} | {diag['media_posteriori'][k]:.4g} | {diag['desvio_posteriori'][k]:.3g} | "
                   f"{dcol['media_posteriori'][k]:.4g} | {dcol['desvio_posteriori'][k]:.3g} | {diag['rhat'][k]:.4f} | "
                   f"{diag['ess_thin'][k]:.0f} | {diag['ess_sem_thin'][k]:.0f} | {n_pos / diag['ess_sem_thin'][k]:.1f} |")
    exe = ["| Sequencia | k_s media | rho_s media | R-hat log k | R-hat log rho | ESS log k | ESS log rho |",
           "|---|---|---|---|---|---|---|"]
    for e, i in enumerate(exemplos):
        pos = res["latentes_exemplos"][:, BURN_IN:, e]
        fino = pos[:, ::THIN]
        exe.append(f"| {titulos[e]} | {np.exp(pos[..., 0]).mean():.4g} | {np.exp(pos[..., 1]).mean():.4g} | "
                   f"{mc.split_rhat(fino[..., 0]):.4f} | {mc.split_rhat(fino[..., 1]):.4f} | "
                   f"{mc.ess(fino[..., 0]):.0f} | {mc.ess(fino[..., 1]):.0f} |")
    lin = ["| Latente | R-hat max | R-hat mediano | ESS min (thin) | ESS mediano (thin) |", "|---|---|---|---|---|"]
    for nome, v in lat.items():
        lin.append(f"| {nome} | {v['rhat_max']:.4f} | {v['rhat_mediano']:.4f} | {v['ess_min']:.0f} | {v['ess_mediano']:.0f} |")
    md = ["# MCMC completo (Metropolis-dentro-de-Gibbs) - diagnostico exploratorio, 11 participantes, in-sample\n",
          f"Configuracao: {N_ITER} iteracoes, burn-in {BURN_IN}, thin {THIN}, 4 cadeias, "
          f"{len(res['latentes_thin'][0, 0])} sequencias; tempo do completo {res['segundos']:.0f} s "
          f"(colapsado: {col['segundos']:.0f} s, {N_ITER_COLAPSADO} iteracoes).\n",
          "## Hiperparametros (completo x colapsado)\n", *tab,
          f"\nR-hat maximo {diag['rhat_max']:.4f}; ESS minimo (thin) {diag['ess_min_thin']:.0f}.\n",
          "## Aceitacoes (pos burn-in)\n",
          f"- Hiperparametros (media de a_k e a_rho) por cadeia: {', '.join(f'{x:.3f}' for x in res['aceitacao'])}",
          f"- a_k: {', '.join(f'{x:.3f}' for x in res['aceitacao_a_k'])}; a_rho: {', '.join(f'{x:.3f}' for x in res['aceitacao_a_rho'])}",
          f"- Latentes k_s: {res['aceitacao_k_latente']:.3f}; rho_s: {res['aceitacao_rho_latente']:.3f}\n",
          "## Latentes: resumo sobre todas as sequencias\n", *lin,
          "\n## Sequencias de exemplo\n", *exe,
          f"\nTempo total: {time.time() - t0:.0f} s."]
    open(os.path.join(PASTA, "completo.md"), "w").write("\n".join(md) + "\n")
    json.dump({"config": {"n_iter": N_ITER, "burn_in": BURN_IN, "thin": THIN, "cadeias": 4},
               "segundos": res["segundos"], "segundos_colapsado": col["segundos"], "hiper_mlii": h,
               "completo": diag, "colapsado": dcol,
               "tau_integrado": {k: n_pos / diag["ess_sem_thin"][k] for k in mc.NOMES_PARAM},
               "aceitacao": {"hiper": res["aceitacao"].tolist(), "a_k": res["aceitacao_a_k"].tolist(),
                             "a_rho": res["aceitacao_a_rho"].tolist(), "k_latente": res["aceitacao_k_latente"],
                             "rho_latente": res["aceitacao_rho_latente"]},
               "latentes": {n: {k: v[k] for k in v if k not in ("rhat", "ess")} for n, v in lat.items()},
               "exemplos": titulos},
              open(os.path.join(PASTA, "completo.json"), "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
