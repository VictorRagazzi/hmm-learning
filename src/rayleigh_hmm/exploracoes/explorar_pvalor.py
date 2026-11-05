"""Teste exploratorio: o HMM observa o p-valor Rayleigh acumulado p = exp(-Z), nao Z.

Nao altera nenhum arquivo existente; reaproveita experimento/dados/emissao/hmm/
explorar_busca. Resultados in-sample (mesmos 11 participantes), sem validacao.
Saidas: results/rayleigh_hmm/exploratorio_pvalor/{resultados.json,tabelas.md}
Execucao (da raiz): .venv/bin/python src/rayleigh_hmm/exploracoes/explorar_pvalor.py

Duas variantes (so hmm_viterbi, "alguma Presente"):
  B1 (checagem de matematica): emissao em p por mudanca de variaveis do modelo
     Gamma atual, f_p(p|N) = f_Z(-ln p|N) / p. O jacobiano 1/p e comum aos dois
     estados e some no Viterbi: deve reproduzir exatamente o resultado atual.
  B2 (teste real): distribuicoes ajustadas DIRETAMENTE em p.
     Ausente: p ~ Beta(a0, b0) por MLE nos p do ESP (teoria: Beta(1,1)=Uniforme).
     Presente: p ~ Beta(alfa, beta), alfa = 1/mu(N), mu(N) = 1 + (N-1) rho.
     Justificativa: se Z ~ Exponencial de media mu (Gamma com k=1), entao
     p = exp(-Z) e exatamente Beta(1/mu, 1). Logo beta faz o papel da forma k
     (beta = 1 recupera k = 1). Hierarquico como hoje, com (beta, rho) no
     lugar de (k, rho): beta ~ Gamma(a_b, b_b), rho ~ Gamma(a_r, b_r), um par
     (beta, rho) por sequencia sintetica, compartilhado por suas janelas.

Tudo em escala log: log p = -Z; log(1-p) = log(-expm1(-Z)), com p recortado
em [1e-300, 1-1e-12] (p nao e calculado para nao dar underflow).
"""

import json
import sys
import os
import time

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import emissao
import experimento as ex
import explorar_busca as eb
import hmm

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_pvalor")
GRADE_BETA = np.geomspace(0.01, 300.0, 170)  # como GRADE_HIER_K
GRADE_RHO = ex.GRADE_HIER_RHO  # mesma grade de rho de hoje
# 3 inicios de hoje (a, b, a_rho, b_rho) + 1 perto do ajuste Gamma atual
INICIOS_BETA = list(ex.INICIOS_HIER) + [(1.26, 0.225, 0.67, 18.6)]
J_BETA, J_RHO = ex.J_MARGINAL_K, ex.J_MARGINAL_RHO  # 10 x 24 nos
LOG_P_MIN, LOG_P_MAX = np.log(1e-300), np.log1p(-1e-12)
P1 = {"p1": 0.01, "a01": 0.005, "a10": 0.3,
      "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.1, "consecutivas": 2}}


# ---------------- log p, log(1-p) ----------------
def logs_p(z):
    """(log p, log(1-p)) a partir de Z, sem calcular p."""
    lp = np.clip(-np.asarray(z, dtype=float), LOG_P_MIN, LOG_P_MAX)
    return lp, np.log(-np.expm1(lp))


def log_pdf_beta(lp, l1p, a, b):
    """log Beta(p | a, b) dado log p e log(1-p). a, b broadcastaveis."""
    return (special.gammaln(a + b) - special.gammaln(a) - special.gammaln(b)
            + (a - 1.0) * lp + (b - 1.0) * l1p)


# ---------------- ajuste B2 ----------------
def ajustar_ausente_beta(z_esp):
    """MLE de Beta(a0, b0) nos p do ESP (alvos): mesmas observacoes do Ausente atual."""
    lp, l1p = logs_p(z_esp)
    f = lambda t: -np.mean(log_pdf_beta(lp, l1p, np.exp(t[0]), np.exp(t[1])))
    r = optimize.minimize(f, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 5000})
    a0, b0 = np.exp(r.x)
    return {"a0": float(a0), "b0": float(b0), "n": int(lp.size), "log_verossimilhanca": float(-r.fun * lp.size)}


def matriz_verossimilhanca_beta(z, n_epocas, ids, grade_beta, grade_rho):
    """log prod_t Beta(p_st | 1/mu(N_st, rho), beta) em (sequencias, len(beta), len(rho)).

    lgamma(alfa + beta) nao separa em t: laco em rho, vetorizado em (t, beta).
    """
    lp, l1p = logs_p(z)
    n_epocas = np.asarray(n_epocas, dtype=float).ravel()
    _, ids = np.unique(np.asarray(ids).ravel(), return_inverse=True)
    ordem = np.argsort(ids, kind="stable")
    lp, l1p, n_epocas, ids = lp[ordem], l1p[ordem], n_epocas[ordem], ids[ordem]
    inicios = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    t_s = np.diff(np.r_[inicios, lp.size])
    soma_l1p = np.add.reduceat(l1p, inicios)  # (seq,)
    b = grade_beta[None, :]
    saida = np.empty((len(inicios), len(grade_beta), len(grade_rho)))
    for j, rho in enumerate(grade_rho):
        alfa = 1.0 / ex.emissao.media_presente(n_epocas, rho)  # (t,)
        termo_t = -special.gammaln(alfa) + (alfa - 1.0) * lp  # (t,)
        soma_t = np.add.reduceat(termo_t, inicios)  # (seq,)
        gam = np.add.reduceat(special.gammaln(alfa[:, None] + b), inicios, axis=0)  # (seq, beta)
        saida[:, :, j] = (gam + soma_t[:, None] + (b - 1.0) * soma_l1p[:, None]
                          - t_s[:, None] * special.gammaln(b))
    return saida


def ajustar_presente_beta(emi):
    matriz = matriz_verossimilhanca_beta(emi["z_sintetico"], emi["n_sintetico"], emi["id_sintetico"],
                                         GRADE_BETA, GRADE_RHO)
    h = emissao.ajustar_hierarquico(matriz, GRADE_BETA, GRADE_RHO, INICIOS_BETA)
    h["n_sequencias"] = int(matriz.shape[0])
    # nomes proprios: a_k/b_k da rotina generica sao (a_beta, b_beta)
    h = {("a_beta" if k == "a_k" else "b_beta" if k == "b_k" else "beta_media" if k == "k_media"
          else "beta_desvio" if k == "k_desvio" else k): v for k, v in h.items()}
    h["massa_fora_grade"] = {"beta": h["massa_fora_grade"]["k"], "rho": h["massa_fora_grade"]["rho"]}
    # massa da posteriori (media sobre sequencias) nas bordas da grade de beta e rho
    return h, matriz


def nos_beta(h):
    b, rho = emissao.nos_quadratura({"a_k": h["a_beta"], "b_k": h["b_beta"], "a_rho": h["a_rho"], "b_rho": h["b_rho"]},
                                    J_BETA, J_RHO)
    return b, rho


def massa_borda_posteriori(matriz, h):
    """Massa media (sobre sequencias) da posteriori de (beta, rho) na 1a/ultima coluna das grades."""
    lb, lr = np.log(GRADE_BETA), np.log(GRADE_RHO)
    pb = h["a_beta"] * lb - h["b_beta"] * GRADE_BETA
    pr = h["a_rho"] * lr - h["b_rho"] * GRADE_RHO
    lw = matriz + pb[None, :, None] + pr[None, None, :]
    lw -= lw.max(axis=(1, 2), keepdims=True)
    w = np.exp(lw)
    w /= w.sum(axis=(1, 2), keepdims=True)
    mb, mr = w.sum(axis=2).mean(axis=0), w.sum(axis=1).mean(axis=0)
    return {"beta_min": float(mb[0]), "beta_max": float(mb[-1]), "rho_min": float(mr[0]), "rho_max": float(mr[-1])}


# ---------------- emissoes das duas variantes ----------------
def log_emissoes_b1(s, emissoes):
    """Emissao Gamma atual + jacobiano: log f_p = log f_Z(Z) - log p = log f_Z + Z (igual nos 2 estados)."""
    return ex.log_emissoes(s, emissoes, "marginal") + np.asarray(s["z"])[:, None]


def log_emissoes_b2(s, a0, b0, b_nos, rho_nos):
    lp, l1p = logs_p(s["z"])
    aus = log_pdf_beta(lp, l1p, a0, b0)
    alfa = 1.0 / ex.emissao.media_presente(np.asarray(s["n_ep"], dtype=float)[:, None], rho_nos[None, :])
    termos = log_pdf_beta(lp[:, None], l1p[:, None], alfa, b_nos[None, :])
    pres = special.logsumexp(termos, axis=1) - np.log(b_nos.size)
    return np.stack([aus, pres], axis=-1)


def montar_grupos(sequencias, f):
    return [(idx, dur, np.stack([f(sequencias[i]) for i in idx])) for idx, dur in ex.indices_por_comprimento(sequencias)]


# ---------------- verificacoes ----------------
def checar_b1(grupos_atual, grupos_b1, n_seq):
    """Compara as deteccoes brutas do Viterbi (todas as 1296 combinacoes de pi, A) entre Z e p (B1)."""
    divergentes, total = 0, 0
    for p1, a01, a10 in [(p, a, b) for p in eb.GRADE_PI for a in eb.GRADE_A01 for b in eb.GRADE_A10]:
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        for (_, _, le0), (_, _, le1) in zip(grupos_atual, grupos_b1):
            d0 = hmm.viterbi_alguma_presente(le0, pi, A)
            d1 = hmm.viterbi_alguma_presente(le1, pi, A)
            divergentes += int((d0 != d1).sum())
            total += d0.size
    return {"combinacoes_pi_A": 1296, "celulas_comparadas": total, "celulas_divergentes": divergentes}


def indice_regra(regra):
    return next(k for k, r in enumerate(ex.REGRAS) if r == regra)


def pareado_entre(masc, a, b):
    ambos = masc["alvo_estimulo"] & a["detectado"] & b["detectado"]
    d = b["tempo_s"][ambos] - a["tempo_s"][ambos]
    return {"n_pares": int(ambos.sum()), "dif_media_s": float(d.mean())}


# ---------------- principal ----------------
def main():
    t0 = time.time()
    sequencias, esp = ex.carregar_sequencias()
    emi = ex.ajustar_emissoes(sequencias, esp)
    emi["marginal"] = ex.ajustar_marginal(emi)
    masc = ex.mascaras(sequencias)
    _, ref = ex.buscar_bruto(sequencias, masc)
    n_seq = len(sequencias)
    ref_m = ex.metricas(masc, ref["detectado"], ref["tempo_s"])
    sa = {"ausente_gamma": emi["ausente"], "n_z_esp": int(emi["z_esp"].size)}

    # ---- B1
    g_atual = montar_grupos(sequencias, lambda s: ex.log_emissoes(s, emi, "marginal"))
    g_b1 = montar_grupos(sequencias, lambda s: log_emissoes_b1(s, emi))
    b1 = checar_b1(g_atual, g_b1, n_seq)
    print("B1:", b1, f"({time.time()-t0:.0f} s)")
    cand_p1 = {"p1": P1["p1"], "a01": P1["a01"], "a10": P1["a10"], "k": indice_regra(P1["regra"])}
    r_z_p1 = eb.completar(cand_p1, g_atual, sequencias, masc, ref)
    r_b1_p1 = eb.completar(cand_p1, g_b1, sequencias, masc, ref)
    b1["p1_Z"] = {k: r_z_p1[k] for k in ("n_det", "n_fp_lat", "n_fp_esp")} | {"tempo": r_z_p1["metricas"]["tempo_medio_s"]}
    b1["p1_B1"] = {k: r_b1_p1[k] for k in ("n_det", "n_fp_lat", "n_fp_esp")} | {"tempo": r_b1_p1["metricas"]["tempo_medio_s"]}
    b1["identico"] = b1["celulas_divergentes"] == 0 and b1["p1_Z"] == b1["p1_B1"]

    # ---- B2: ajustes
    aus = ajustar_ausente_beta(emi["z_esp"])
    lp, l1p = logs_p(emi["z_esp"])
    ll_unif = float(np.sum(log_pdf_beta(lp, l1p, 1.0, 1.0)))
    ll_gamma_z = float(emissao.log_pdf_gamma(emi["z_esp"], emi["ausente"]["k"], emi["ausente"]["mu"]).sum())
    ll_gamma_em_p = ll_gamma_z + float(np.sum(emi["z_esp"]))  # f_p = f_Z / p, log(1/p) = +Z
    aus["comparacao_mesmos_dados_em_p"] = {"beta_mle": aus["log_verossimilhanca"], "uniforme": ll_unif,
                                          "gamma_Z_com_jacobiano": ll_gamma_em_p}
    h, matriz = ajustar_presente_beta(emi)
    h["massa_posteriori_borda"] = massa_borda_posteriori(matriz, h)
    h["na_borda_posteriori"] = bool(max(h["massa_posteriori_borda"].values()) > 1e-2)
    b_nos, rho_nos = nos_beta(h)
    print(f"B2 ajustado ({time.time()-t0:.0f} s): Ausente {aus['a0']:.4f},{aus['b0']:.4f}; Presente {h['a_beta']:.3f},{h['b_beta']:.3f},{h['a_rho']:.3f},{h['b_rho']:.3f}")
    # verossimilhanca marginal dos dados sinteticos: B2 (em p) x Gamma (em Z, com jacobiano), mesma escala
    ll_gamma_hier = emi["marginal"]["hiper"]["log_verossimilhanca"] + float(np.sum(emi["z_sintetico"]))
    comp_pres = {"beta_hier_em_p": h["log_verossimilhanca"], "gamma_hier_Z_com_jacobiano": ll_gamma_hier,
                 "nota": "log-verossimilhanca marginal por sequencia (grade fixa); Gamma ajustada em Z + soma(Z) = densidade em p"}

    g_b2 = montar_grupos(sequencias, lambda s: log_emissoes_b2(s, aus["a0"], aus["b0"], b_nos, rho_nos))
    r_b2_p1 = eb.completar(cand_p1, g_b2, sequencias, masc, ref)
    pareado_p1 = pareado_entre(masc, r_z_p1["_cand"], r_b2_p1["_cand"])

    # ---- B2: busca
    eb.GRADE_PI, eb.GRADE_A01, eb.GRADE_A10  # mesma grade estendida de explorar_busca
    viaveis, total = eb.varrer(g_b2, sequencias, masc, ref)
    print(f"B2 busca: {total} candidatos, {len(viaveis)} viaveis ({time.time()-t0:.0f} s)")
    a_c = max(viaveis, key=eb.chave_a)
    nao_lento = [c for c in viaveis if c["dif_pareada_s"] <= 0]
    n_ref = int(ref["detectado"][masc["alvo_estimulo"]].sum())
    if nao_lento:
        b_c, nome_b = max(nao_lento, key=eb.chave_b), "b_nao_mais_lento"
    else:
        quase = [c for c in viaveis if round(c["m"]["deteccao"] * 440) >= n_ref]
        b_c = min(quase, key=lambda c: (c["dif_pareada_s"], -c["m"]["deteccao"])) if quase else None
        nome_b = "b_vazio_menor_dif_det_ge_bruto"
    fronteira = eb.pareto(viaveis)
    pontos = eb.representativos(fronteira, 252 / 440)
    escolhidos = {"a_chave_ordem": a_c}
    if b_c:
        escolhidos[nome_b] = b_c
    res = {n: eb.completar(c, g_b2, sequencias, masc, ref) for n, c in escolhidos.items()}
    res_front = [eb.completar(c, g_b2, sequencias, masc, ref) for c in pontos]
    res["B2_em_P1"] = r_b2_p1
    for r in list(res.values()) + res_front:  # pareado contra o HMM Gamma-Z (P1)
        r["pareado_vs_P1_Z"] = pareado_entre(masc, r_z_p1["_cand"], r["_cand"])

    saida = {"aviso": "exploratorio, in-sample (mesmos 11 participantes); nao e validacao",
             "B1": b1, "B2_ausente_beta": aus, "ausente_gamma_atual": emi["ausente"],
             "B2_presente_hier": {k: v for k, v in h.items()}, "comparacao_presente": comp_pres,
             "nos_quadratura": {"beta": b_nos.tolist(), "rho": rho_nos.tolist()},
             "n_candidatos_avaliados": total, "n_viaveis": len(viaveis), "n_nao_mais_lentos": len(nao_lento),
             "n_pontos_fronteira_total": len(fronteira), "bruto_referencia": ref_m,
             "P1_Z_gamma": eb.sem_privado(r_z_p1),
             "escolhidos": {n: eb.sem_privado(r) for n, r in res.items()},
             "fronteira_pareto": [eb.sem_privado(r) for r in res_front]}

    md = ["# Exploratorio: HMM observando o p-valor acumulado (in-sample, 11 participantes)\n",
          "## B1 (checagem: mudanca de variaveis do modelo Gamma)\n",
          f"Viterbi Z vs p(B1) em {b1['combinacoes_pi_A']} combinacoes (pi, A): {b1['celulas_divergentes']} "
          f"divergencias em {b1['celulas_comparadas']} celulas. No ponto P1: Z {b1['p1_Z']} vs B1 {b1['p1_B1']}. "
          f"Identico: {b1['identico']}.\n",
          "## B2: ajustes\n",
          f"Ausente Beta: a0={aus['a0']:.4f}, b0={aus['b0']:.4f} (n={aus['n']}). Log-verossim. em p (mesmos 1160 dados): "
          f"Beta {aus['log_verossimilhanca']:.2f}; Uniforme {ll_unif:.2f}; Gamma(Z)+jacobiano {ll_gamma_em_p:.2f}.\n",
          f"Presente: a_beta={h['a_beta']:.4f}, b_beta={h['b_beta']:.4f} (media beta {h['beta_media']:.3f}, dp {h['beta_desvio']:.3f}); "
          f"a_rho={h['a_rho']:.4f}, b_rho={h['b_rho']:.4f} (media rho {h['rho_media']:.4f}, dp {h['rho_desvio']:.4f}). "
          f"Log-verossim. marginal {h['log_verossimilhanca']:.1f} (Gamma-Z + jacobiano: {ll_gamma_hier:.1f}). "
          f"Massa fora da grade {h['massa_fora_grade']}; hiper no limite {h['hiper_no_limite']}; "
          f"massa da posteriori nas bordas {h['massa_posteriori_borda']}; dispersao entre inicios {h['dispersao_relativa_inicios']:.2g}; "
          f"na_borda={h['na_borda']}.\n",
          "Corridas: " + "; ".join(f"inicio {c['inicio']} -> ll {c['log_verossimilhanca']:.2f}, it {c['iteracoes']}, ok {c['sucesso']}" for c in h["corridas"]) + "\n",
          f"bruto_referencia: det {n_ref}/440, FP lat {ref['detectado'][masc['lateral']].sum()}/528, "
          f"FP ESP {ref['detectado'][masc['alvo_esp']].sum()}/88, tempo medio/mediano "
          f"{eb.fmt_t(ref_m['tempo_medio_s'])}/{eb.fmt_t(ref_m['tempo_mediano_s'])}.\n",
          f"B2 busca: {total} candidatos; viaveis {len(viaveis)}; com dif. pareada <= 0: {len(nao_lento)}; pontos de Pareto {len(fronteira)}.\n",
          "## Vencedores B2 e referencias\n", eb.CABECALHO]
    md.append(eb.linha_md("P1 Gamma-Z (referencia)", r_z_p1))
    for n, r in res.items():
        md.append(eb.linha_md(n, r))
    md += ["\n## Fronteira de Pareto B2\n", eb.CABECALHO]
    for i, r in enumerate(res_front):
        md.append(eb.linha_md(f"P{i+1}", r))
    md += ["\n## Pareado B2 vs P1 Gamma-Z (n pares, dif. media s)\n"]
    for n, r in res.items():
        md.append(f"- {n}: {r['pareado_vs_P1_Z']['n_pares']} pares, {r['pareado_vs_P1_Z']['dif_media_s']:+.1f} s")
    for n in [n for n in res if n.startswith(("a_", "b_"))]:
        li = eb.tabela_intensidade(sequencias, res[n]["_cand"], ref)
        md.append(eb.tabela_intensidade_md(li, f"B2 {n}"))
        saida["escolhidos"][n]["por_intensidade"] = li
    tempo = time.time() - t0
    saida["tempo_execucao_s"] = tempo
    md.append(f"\nTempo de execucao: {tempo:.0f} s.")
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
