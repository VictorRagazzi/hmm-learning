"""Experimento LOSO: emissao Presente qui-quadrado nao-central (ncx2) vs Gamma.

Sob H1 (fasor com PLV^2 verdadeiro = rho), Z = N * PLV^2 tem media 1 + (N-1)*rho e
2Z segue aproximadamente qui-quadrado nao-central com 2 graus de liberdade e parametro
de nao-centralidade lambda = 2*(N-1)*rho, cuja densidade em Z e:
    p(Z | N, rho) = exp(-Z - (N-1)*rho) * I_0(2 * sqrt((N-1)*rho*Z))

Diferenca chave para a Gamma(k, mu):
  - Na Gamma ajustada aos dados reais, k < 1 (~0.83 no modelo fixo), o que faz p(Z|Presente)
    divergir para +infinito quando Z -> 0+.
  - Na ncx2, p(0 | N, rho) = exp(-(N-1)*rho) < 1, entao Z pequeno da evidencia limpa a favor
    de Ausente, sem explosao em Z -> 0.

Testa duas variantes ncx2 em LOSO (sinal FC, teto = FP lateral do bruto_referencia no treino):
  - HMM-NCX2-HIER: rho_s ~ Gamma(a_rho, b_rho) por sequencia (ML-II + quadratura de 24 nos)
  - HMM-NCX2-FIXO: um unico rho por maxima verossimilhanca no treino do fold

Saidas: results/rayleigh_hmm/exploratorio_ncx2/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_ncx2.py
"""

import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np
from scipy import optimize, special, stats

import emissao
import experimento as ex
import explorar_loso as el
import explorar_calibracao_50_70 as base
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI
from explorar_loso_fc import teto_referencia

PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_ncx2")
GRADE_RHO = ex.GRADE_HIER_RHO
J_RHO = ex.J_MARGINAL_RHO
SEQ_TODAS = None


def log_pdf_ncx2(z, n_epocas, rho):
    """log p(Z | N, rho) para 2Z ~ ncx2(df=2, nc=2*(N-1)*rho). Aceita broadcast."""
    lam_meio = (n_epocas - 1.0) * rho
    arg = 2.0 * np.sqrt(np.maximum(lam_meio * z, 0.0))
    return -z - lam_meio + arg + np.log(np.maximum(special.i0e(arg), 1e-300))


def log_pdf_preditiva_ncx2(z, n_epocas, pontos):
    """Mistura ponderada de densidades ncx2 nos nos de quadratura de rho."""
    lp = log_pdf_ncx2(z[:, None], n_epocas[:, None], pontos["rho"][None, :])
    return special.logsumexp(lp + np.log(pontos["peso"])[None, :], axis=1)


def ajustar_ncx2_hier(z, n_epocas, ids, grade_rho=GRADE_RHO):
    """ML-II de rho_s ~ Gamma(a_rho, b_rho) integrando rho_s em grade 1D."""
    d_log_r = float(np.log(grade_rho[1] / grade_rho[0]))
    ids_unicos = np.unique(ids)
    # log_verossimilhanca (S, R)
    ll = np.zeros((len(ids_unicos), len(grade_rho)))
    for i, s_id in enumerate(ids_unicos):
        m = ids == s_id
        ll[i] = log_pdf_ncx2(z[m, None], n_epocas[m, None], grade_rho[None, :]).sum(axis=0)
    escala = ll.max(axis=1, keepdims=True)
    ver_rel = np.exp(ll - escala)  # (S, R)
    log_r = np.log(grade_rho)

    def nll(theta):
        a, b = np.exp(theta[0]), np.exp(theta[1])
        log_pr = a * np.log(b) - special.gammaln(a) + a * log_r - b * grade_rho + np.log(d_log_r)
        pr = np.exp(np.clip(log_pr, -700, 700))
        marg = np.maximum(ver_rel @ pr, 1e-300)
        return -float((np.log(marg) + escala[:, 0]).mean())

    melhor = None
    for ini in [(0.3, 10.0), (1.0, 50.0), (0.5, 20.0)]:
        r = optimize.minimize(nll, np.log(ini), method="L-BFGS-B", bounds=[(-5.0, 11.0), (-5.0, 11.0)])
        if melhor is None or r.fun < melhor.fun:
            melhor = r
    a_r, b_r = float(np.exp(melhor.x[0])), float(np.exp(melhor.x[1]))
    q = (np.arange(J_RHO) + 0.5) / J_RHO
    nos_r = stats.gamma.ppf(q, a=a_r, scale=1.0 / b_r)
    return {
        "hiper": {"a_rho": a_r, "b_rho": b_r, "rho_media": a_r / b_r, "nll": float(melhor.fun)},
        "pontos": {"rho": nos_r, "peso": np.full(J_RHO, 1.0 / J_RHO)},
    }


def ajustar_ncx2_fixo(z, n_epocas):
    """MLE de um unico rho para todas as sequencias de calibracao."""
    def nll(log_rho):
        rho = float(np.exp(log_rho[0]))
        return -float(log_pdf_ncx2(z, n_epocas, rho).mean())

    r = optimize.minimize(nll, [np.log(0.015)], method="L-BFGS-B", bounds=[(np.log(1e-5), np.log(0.95))])
    rho = float(np.exp(r.x[0]))
    return {
        "hiper": {"a_rho": np.nan, "b_rho": np.nan, "rho_media": rho, "nll": float(r.fun)},
        "pontos": {"rho": np.array([rho]), "peso": np.array([1.0])},
    }


def grupos_log_em_ncx2(seqs, emi):
    aus = emi["ausente"]
    pts = emi["ncx2"]["pontos"]
    grupos = []
    for idx, dur in ex.indices_por_comprimento(seqs):
        le = []
        for i in idx:
            z, n_ep = seqs[i]["z"], seqs[i]["n_ep"]
            la = emissao.log_pdf_gamma(z, aus["k"], aus["mu"])
            lp = log_pdf_preditiva_ncx2(z, n_ep, pts)
            le.append(np.stack([la, lp], axis=1))
        grupos.append((idx, dur, np.stack(le)))
    return grupos


def rodar_fold_ncx2(modo, fold):
    t0 = time.time()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    h = participantes[fold]
    i_tr = [i for i, s in enumerate(SEQ_TODAS) if s["participante"] != h]
    i_h = [i for i, s in enumerate(SEQ_TODAS) if s["participante"] == h]
    seq_tr, seq_h = [SEQ_TODAS[i] for i in i_tr], [SEQ_TODAS[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    teto = teto_referencia(seq_tr, masc_tr)

    z_esp = np.concatenate([s["z"] for s in seq_tr if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    alvos = [s for s in seq_tr if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    z_cal = np.concatenate([s["z"] for s in alvos])
    n_cal = np.concatenate([s["n_ep"] for s in alvos])
    id_cal = np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])

    emi = {"ausente": emissao.ajustar_mle(z_esp)}
    if modo == "HMM-NCX2-HIER":
        emi["ncx2"] = ajustar_ncx2_hier(z_cal, n_cal, id_cal)
    elif modo == "HMM-NCX2-FIXO":
        emi["ncx2"] = ajustar_ncx2_fixo(z_cal, n_cal)
    else:
        raise ValueError(modo)

    grupos_tr = grupos_log_em_ncx2(seq_tr, emi)
    melhores = {}
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        ex.avaliar_regras(seq_tr, masc_tr, el.detectores_hmm(grupos_tr, pi, A),
                          {"pi": pi.tolist(), "A": A.tolist()}, melhores, teto)
    melhor = ex.melhor_geral(melhores)

    pi, A = np.array(melhor["pi"]), np.array(melhor["A"])
    gd_h = el.detectores_hmm(grupos_log_em_ncx2(seq_h, emi), pi, A)
    det_h, tempo_h = ex.decidir(gd_h, len(seq_h), melhor["regra"])
    masc_h = ex.mascaras(seq_h)
    return {
        "modo": modo,
        "fold": fold,
        "participante": h,
        "indices_h": i_h,
        "detectado_h": det_h,
        "tempo_h": tempo_h,
        "fp_lateral_h": float(det_h[masc_h["lateral"]].mean()),
        "teto_treino": teto,
        "escolha": {"pi": melhor["pi"], "A": melhor["A"], "regra": melhor["regra"],
                    "metricas_treino": melhor["metricas"]},
        "hiper": emi["ncx2"]["hiper"],
        "segundos": time.time() - t0,
    }


def tarefa(args):
    modo, fold = args
    if modo == "HMM-GAMMA":
        return base._rodar_hmm_fold(SEQ_TODAS, fold, modo, base.INTENSIDADES_TODAS)
    return rodar_fold_ncx2(modo, fold)


def main():
    global SEQ_TODAS
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    t0 = time.time()
    SEQ_TODAS, _ = ex.carregar_sequencias()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    n_todas = len(SEQ_TODAS)
    masc_todas = ex.mascaras(SEQ_TODAS)

    modos = ["HMM-GAMMA", "HMM-NCX2-HIER", "HMM-NCX2-FIXO"]
    tarefas = [(m, f) for m in modos for f in range(len(participantes))]
    print(f"Rodando {len(tarefas)} tarefas LOSO em {base.N_PROCESSOS} processos...", flush=True)

    with mp.get_context("fork").Pool(base.N_PROCESSOS) as pool:
        saidas = []
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            print(f"[{k}/{len(tarefas)}] {r['modo']} {r['participante']}: {r['segundos']:.0f}s "
                  f"(decorrido {time.time()-t0:.0f}s)", flush=True)

    por_modo = {m: [r for r in saidas if r["modo"] == m] for m in modos}
    c_todas = {m: el.juntar(por_modo[m], n_todas) for m in modos}
    _, ref_todas = ex.buscar_bruto(SEQ_TODAS, masc_todas)
    c_todas["bruto_referencia"] = {"detectado": ref_todas["detectado"], "tempo_s": ref_todas["tempo_s"]}

    par_todas = {
        "HMM-NCX2-HIER vs HMM-GAMMA": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-NCX2-HIER"], c_todas["HMM-GAMMA"]),
        "HMM-NCX2-FIXO vs HMM-GAMMA": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-NCX2-FIXO"], c_todas["HMM-GAMMA"]),
        "HMM-NCX2-HIER vs bruto_referencia": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-NCX2-HIER"], c_todas["bruto_referencia"]),
        "HMM-NCX2-FIXO vs bruto_referencia": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-NCX2-FIXO"], c_todas["bruto_referencia"]),
    }

    # Razoes de verossimilhanca ilustrativas (11 participantes)
    z_esp = np.concatenate([s["z"] for s in SEQ_TODAS if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    aus = emissao.ajustar_mle(z_esp)
    alvos = [s for s in SEQ_TODAS if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    z_cal = np.concatenate([s["z"] for s in alvos])
    n_cal = np.concatenate([s["n_ep"] for s in alvos])
    id_cal = np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])
    ncx2_h = ajustar_ncx2_hier(z_cal, n_cal, id_cal)
    ncx2_f = ajustar_ncx2_fixo(z_cal, n_cal)
    gamma_ex = base.razoes_verossimilhanca_exemplo(SEQ_TODAS, base.INTENSIDADES_TODAS)

    linhas_razoes = ["| (N, Z) | Gamma hier. | NCX2 hier. | NCX2 fixo |", "|---|---:|---:|---:|"]
    for ex_g in gamma_ex["exemplos"]:
        n_ep, z_val = ex_g["N"], ex_g["Z"]
        la = float(emissao.log_pdf_gamma(np.array([z_val]), aus["k"], aus["mu"])[0])
        lp_h = float(log_pdf_preditiva_ncx2(np.array([z_val]), np.array([n_ep]), ncx2_h["pontos"])[0])
        lp_f = float(log_pdf_preditiva_ncx2(np.array([z_val]), np.array([n_ep]), ncx2_f["pontos"])[0])
        linhas_razoes.append(
            f"| N={n_ep}, Z={z_val} | {ex_g['razao_P_sobre_A']:.2f}x | "
            f"{float(np.exp(lp_h - la)):.2f}x | {float(np.exp(lp_f - la)):.2f}x |"
        )

    met_todas = [*modos, "bruto_referencia"]
    linhas_inten = [
        "| Intensidade | " + " | ".join(f"{m} det / FP lat / tempo" for m in met_todas) + " |",
        "|---|" + "---|" * len(met_todas),
    ]
    for cond in base.INTENSIDADES_TODAS:
        cel = []
        for m in met_todas:
            r = ex.resumo(SEQ_TODAS, c_todas[m], lambda s, cond=cond: s["condicao"] == cond)
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} ({100*r['deteccao']:.1f}%) / "
                       f"{r['fp_lateral_detectados']}/{r['fp_lateral_total']} / {el.fmt_t(r['tempo_medio_s'])}")
        linhas_inten.append(f"| {cond} | " + " | ".join(cel) + " |")

    tempo_total = time.time() - t0
    md = [
        "# Experimento LOSO: emissao Presente qui-quadrado nao-central (NCX2) vs Gamma\n",
        "## 1. Geral (LOSO, 440 alvos / 528 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |",
        "|---|---|---|---|---|",
        base._linha_geral("HMM-GAMMA (baseline oficial)", SEQ_TODAS, masc_todas, c_todas["HMM-GAMMA"]),
        base._linha_geral("HMM-NCX2-HIER (rho ~ Gamma)", SEQ_TODAS, masc_todas, c_todas["HMM-NCX2-HIER"]),
        base._linha_geral("HMM-NCX2-FIXO (rho unico MLE)", SEQ_TODAS, masc_todas, c_todas["HMM-NCX2-FIXO"]),
        base._linha_geral("bruto_referencia", SEQ_TODAS, masc_todas, c_todas["bruto_referencia"]),
        "\n### Testes pareados\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |",
        "|---|---|---|---|---|",
        *[base._fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in par_todas.items()],
        "\n### Por intensidade\n",
        *linhas_inten,
        "\n## 2. Parametros medios de rho (11 folds) e razoes P(Z|Pres)/P(Z|Aus)\n",
        f"- **HMM-GAMMA:** a_rho={np.mean([r['hiper']['a_rho'] for r in por_modo['HMM-GAMMA']]):.3f}, "
        f"b_rho={np.mean([r['hiper']['b_rho'] for r in por_modo['HMM-GAMMA']]):.2f}, "
        f"rho_medio={np.mean([r['hiper']['rho_media'] for r in por_modo['HMM-GAMMA']]):.4f}",
        f"- **HMM-NCX2-HIER:** a_rho={np.mean([r['hiper']['a_rho'] for r in por_modo['HMM-NCX2-HIER']]):.3f}, "
        f"b_rho={np.mean([r['hiper']['b_rho'] for r in por_modo['HMM-NCX2-HIER']]):.2f}, "
        f"rho_medio={np.mean([r['hiper']['rho_media'] for r in por_modo['HMM-NCX2-HIER']]):.4f}",
        f"- **HMM-NCX2-FIXO:** rho={np.mean([r['hiper']['rho_media'] for r in por_modo['HMM-NCX2-FIXO']]):.4f}\n",
        *linhas_razoes,
        f"\nTempo total de execucao: {tempo_total:.0f} s.",
    ]

    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    saida = {
        "pooled": {m: ex.metricas(masc_todas, c["detectado"], c["tempo_s"]) for m, c in c_todas.items()},
        "pareados": par_todas,
        "folds": {m: [el.sem_arrays(r) for r in sorted(por_modo[m], key=lambda x: x["fold"])] for m in modos},
        "tempo_execucao_s": tempo_total,
    }
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
