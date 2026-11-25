"""LOSO comparativo (EXPLORATORIO) do braco REAL: emissao QUAD (atual), MC, MCMC, GIBBS e FIXO.

Reaproveita explorar_loso.rodar_fold_braco trocando, em memoria, ex.ajustar_marginal
(emissao Presente) e el.teto_fp (teto = FP lateral do Rayleigh de referencia no treino,
como em explorar_loso_fc.py). Nada em disco e alterado fora de results/.../exploratorio_monte_carlo/.
FIXO: um unico (k, rho) por maxima verossimilhanca, sem priori nem hierarquia.
Saidas: results/rayleigh_hmm/exploratorio_monte_carlo/loso/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/monte_carlo/loso_mc.py [--processos N] [--so-agregar]
Cada tarefa (variante, fold) grava folds/<variante>_<participante>.pkl ao terminar; reexecutar
retoma pulando as ja gravadas. --so-agregar monta tabelas so com os folds existentes.
"""
import os

# um fio de BLAS por processo: o paralelismo e por fold (igual ao LOSO oficial)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import mc_comum as mc  # noqa: E402
import mcmc_completo as mcc  # noqa: E402
from mc_comum import ex  # noqa: E402
import explorar_loso as el  # noqa: E402
from explorar_loso_fc import teto_referencia  # noqa: E402

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_monte_carlo", "loso")
OFICIAL = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_loso_fc", "resultados.json")
VARIANTES = ["QUAD", "MC", "MCMC", "GIBBS", "FIXO"]
COM_CADEIA = ("MCMC", "GIBBS")  # variantes com diagnostico de cadeia
N_PROCESSOS = 8   # sobrescrevel por --processos
PASTA_FOLDS = os.path.join(PASTA, "folds")

VARIANTE = None  # variante ativa no processo (definida em cada tarefa)
ROTULO = ""      # fold/participante corrente (para o progresso do MH)
DIAG = {}        # diagnosticos do ajuste do fold corrente (MCMC)
_ajustar_marginal_original = ex.ajustar_marginal


def ajustar_marginal_variante(emissoes):
    """ML-II como hoje; so a integral (e, no MCMC, os hiperparametros) muda."""
    global DIAG
    t0 = time.time()
    marg = _ajustar_marginal_original(emissoes)
    DIAG = {"segundos_mlii": time.time() - t0}
    if VARIANTE == "MC":
        marg["pontos"] = mc.pontos_mc(marg["hiper"])
    elif VARIANTE == "MCMC":
        res = mc.ajustar_mcmc(emissoes, rotulo=ROTULO)
        marg["pontos"] = mc.pontos_mcmc(res)
        d = mc.diagnosticos(res)
        d["segundos_mcmc"] = res["segundos"]
        d["mlii"] = {k: marg["hiper"][k] for k in mc.NOMES_PARAM}
        DIAG.update(d)
    elif VARIANTE == "GIBBS":
        res = mcc.ajustar_completo(emissoes, rotulo=ROTULO)
        marg["pontos"] = mc.pontos_mcmc(res, burn_in=mcc.BURN_IN, thin=mcc.THIN)
        d = mc.diagnosticos(res, mcc.BURN_IN, mcc.THIN)
        d["segundos_mcmc"] = res["segundos"]
        d["mlii"] = {k: marg["hiper"][k] for k in mc.NOMES_PARAM}
        DIAG.update(d)
    elif VARIANTE == "FIXO":
        fixo = mc.ajustar_fixo(emissoes["z_sintetico"], emissoes["n_sintetico"])
        marg["pontos"] = {"k": np.array([fixo["k"]]), "rho": np.array([fixo["rho"]]), "peso": np.array([1.0])}
        DIAG["fixo"] = fixo
    return marg


def arquivo_fold(variante, participante):
    return os.path.join(PASTA_FOLDS, f"{variante}_{participante}.pkl")


def tarefa(args):
    global VARIANTE, ROTULO
    variante, fold, participante = args
    VARIANTE, ROTULO = variante, f"{variante} fold {fold} {participante}"
    t0 = time.time()
    r = el.rodar_fold_braco((fold, "REAL"))
    r["variante"], r["diag"], r["segundos"] = variante, dict(DIAG), time.time() - t0
    destino = arquivo_fold(variante, participante)
    with open(destino + ".tmp", "wb") as f:  # gravacao atomica
        pickle.dump(r, f)
    os.replace(destino + ".tmp", destino)
    return variante, participante, r["segundos"]


def contagens(masc, c):
    det = c["detectado"]
    return {"det": int(det[masc["alvo_estimulo"]].sum()), "fp_lat": int(det[masc["lateral"]].sum()),
            "fp_esp": int(det[masc["alvo_esp"]].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processos", type=int, default=N_PROCESSOS)
    ap.add_argument("--so-agregar", action="store_true", help="so monta as tabelas com os folds ja gravados")
    args = ap.parse_args()
    t0 = time.time()
    el.SEQ, _ = ex.carregar_sequencias()
    el.COEF = el.carregar_coef()
    el.teto_fp = teto_referencia
    ex.ajustar_marginal = ajustar_marginal_variante  # antes do Pool (fork herda)
    n_seq = len(el.SEQ)
    masc = ex.mascaras(el.SEQ)
    participantes = sorted({s["participante"] for s in el.SEQ})
    os.makedirs(PASTA_FOLDS, exist_ok=True)

    if not args.so_agregar:
        # as mais lentas primeiro; pula o que ja esta gravado (retomada)
        tarefas = [(v, f, p) for v in ("GIBBS", "MCMC", "MC", "QUAD", "FIXO") for f, p in enumerate(participantes)
                   if not os.path.exists(arquivo_fold(v, p))]
        print(f"{len(tarefas)} tarefas a rodar ({args.processos} processos)", flush=True)
        if tarefas:
            with mp.get_context("fork").Pool(args.processos) as pool:
                for k, (v, p, seg) in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
                    print(f"[{k}/{len(tarefas)}] {p} {v}: {seg:.0f} s (decorrido {time.time() - t0:.0f} s)", flush=True)

    saidas = []
    for v in VARIANTES:
        for p in participantes:
            if os.path.exists(arquivo_fold(v, p)):
                saidas.append(pickle.load(open(arquivo_fold(v, p), "rb")))
    completas = [v for v in VARIANTES if sum(r["variante"] == v for r in saidas) == len(participantes)]
    if "QUAD" not in completas:
        print("QUAD incompleto; nada a agregar")
        return
    VARIANTES_OK = completas

    c_met, res_var, tempo_var = {}, {}, {}
    for v in VARIANTES_OK:
        rs = sorted((r for r in saidas if r["variante"] == v), key=lambda r: r["fold"])
        c_met[v] = el.juntar(rs, n_seq)
        res_var[v] = rs
        tempo_var[v] = {"soma_folds_s": sum(r["segundos"] for r in rs),
                        "ajuste_emissao_medio_s": float(np.mean([r["diag"]["segundos_mlii"] + r["diag"].get("segundos_mcmc", 0)
                                                                 for r in rs]))}
    oficial = json.load(open(OFICIAL))["pooled"]["HMM-REAL"]

    def linha(nome, m, n=None):
        t = lambda x: "-" if x is None else f"{x:.1f}"
        return (f"| {nome} | {100 * m['deteccao']:.1f}% | {100 * m['fp_lateral']:.1f}% | "
                f"{100 * m['fp_alvo_esp']:.1f}% | {t(m['tempo_medio_s'])} / {t(m['tempo_mediano_s'])} |")

    pooled, cont = {}, {}
    geral = ["| Variante | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |", "|---|---|---|---|---|"]
    for v in VARIANTES_OK:
        pooled[v] = ex.metricas(masc, c_met[v]["detectado"], c_met[v]["tempo_s"])
        cont[v] = contagens(masc, c_met[v])
        geral.append(linha(v, pooled[v]))
    geral.append(linha("HMM-REAL oficial (resultados.json)", oficial))

    dif = ["| Variante vs QUAD | d deteccao | d FP lateral | d FP ESP | d tempo medio (s) | sequencias com decisao diferente |",
           "|---|---|---|---|---|---|"]
    difs = {}
    for v in VARIANTES_OK:
        if v == "QUAD":
            continue
        q = c_met["QUAD"]["detectado"]
        difs[v] = {"det": cont[v]["det"] - cont["QUAD"]["det"], "fp_lat": cont[v]["fp_lat"] - cont["QUAD"]["fp_lat"],
                   "fp_esp": cont[v]["fp_esp"] - cont["QUAD"]["fp_esp"],
                   "tempo_medio_s": pooled[v]["tempo_medio_s"] - pooled["QUAD"]["tempo_medio_s"],
                   "decisoes_diferentes": int((c_met[v]["detectado"] != q).sum())}
        d = difs[v]
        dif.append(f"| {v} | {d['det']:+d} | {d['fp_lat']:+d} | {d['fp_esp']:+d} | {d['tempo_medio_s']:+.1f} | "
                   f"{d['decisoes_diferentes']}/{n_seq} |")
    reproduz = all(abs(pooled["QUAD"][k] - oficial[k]) < 1e-12 for k in ("deteccao", "fp_lateral", "fp_alvo_esp")) and \
        abs(pooled["QUAD"]["tempo_medio_s"] - oficial["tempo_medio_s"]) < 1e-9

    por_fold = ["| Held-out | Variante | det h | FP lat h | segundos | R-hat max | ESS min (thin) | aceitacao (4 cadeias) | MCMC s |",
                "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTES_OK:
        for r in res_var[v]:
            d = r["diag"]
            m = r["metricas_h"]
            extra = (f"{d['rhat_max']:.3f} | {d['ess_min_thin']:.0f} | " + ", ".join(f"{a:.2f}" for a in d["aceitacao"])
                     + f" | {d['segundos_mcmc']:.0f}") if v in COM_CADEIA else "- | - | - | -"
            por_fold.append(f"| {r['participante']} | {v} | {m['deteccao']:.3f} | {r['fp_lateral_h']:.3f} | "
                            f"{r['segundos']:.0f} | {extra} |")
    hip, resumo = [], []
    for v in COM_CADEIA:
        if v not in VARIANTES_OK:
            continue
        hip += [f"\n### {v}\n", "| Held-out | " + " | ".join(f"ML-II {k} / media post." for k in mc.NOMES_PARAM) + " |",
                "|---|---|---|---|---|"]
        for r in res_var[v]:
            d = r["diag"]
            hip.append(f"| {r['participante']} | " + " | ".join(
                f"{d['mlii'][k]:.3g} / {d['media_posteriori'][k]:.3g}" for k in mc.NOMES_PARAM) + " |")
        rhat_max = max(r["diag"]["rhat_max"] for r in res_var[v])
        ess_min = min(r["diag"]["ess_min_thin"] for r in res_var[v])
        ac = [a for r in res_var[v] for a in r["diag"]["aceitacao"]]
        resumo.append(f"- {v}: R-hat maximo entre folds e parametros {rhat_max:.3f}; ESS minimo {ess_min:.0f}; "
                      f"aceitacao (todas as cadeias/folds): min {min(ac):.2f}, max {max(ac):.2f}.")
    fixo_md = []
    if "FIXO" in VARIANTES_OK:
        fixo_md = ["\n## FIXO: (k, rho) por fold\n", "| Held-out | k | rho | na borda |", "|---|---|---|---|"]
        for r in res_var["FIXO"]:
            f = r["diag"]["fixo"]
            fixo_md.append(f"| {r['participante']} | {f['k']:.3f} | {f['rho']:.4g} | {f['na_borda']} |")
    tempo = time.time() - t0

    md = ["# LOSO comparativo da emissao Presente: QUAD x MC x MCMC x GIBBS x FIXO (EXPLORATORIO, braco REAL)\n",
          "Mesmo teto de FP por fold do LOSO oficial (FP lateral do Rayleigh de referencia no treino). "
          f"MC: M={mc.M_AMOSTRAS}. MCMC: {mc.N_ITER} iteracoes x 4 cadeias, burn-in {mc.BURN_IN}, thin {mc.THIN}, "
          f"M={mc.M_AMOSTRAS} sorteios. GIBBS (MCMC completo em k_s, rho_s e hiperparametros): {mcc.N_ITER} iteracoes x 4 cadeias, "
          f"burn-in {mcc.BURN_IN}, thin {mcc.THIN}, M={mc.M_AMOSTRAS} sorteios. FIXO: um unico (k, rho) por maxima verossimilhanca, sem priori nem hierarquia. Nao e resultado oficial.\n",
          "## Geral (pooled, fora da amostra)\n", *geral,
          f"\nQUAD reproduz o HMM-REAL oficial: **{reproduz}**.\n",
          "## Diferencas vs QUAD\n", *dif,
          "\n## MCMC e GIBBS: resumo\n", *(resumo or ["Ainda incompleto."]),
          *fixo_md, "## Por fold\n", *por_fold, "\n## ML-II vs media posteriori por fold\n", *hip,
          "\n## Tempos\n", *[f"- {v}: soma dos folds {t['soma_folds_s']:.0f} s; ajuste da emissao medio por fold "
                             f"{t['ajuste_emissao_medio_s']:.1f} s" for v, t in tempo_var.items()],
          f"\nVariantes agregadas: {', '.join(VARIANTES_OK)}. Tempo desta execucao: {tempo:.0f} s ({args.processos} processos)."]
    saida = {"aviso": "exploratorio; nao oficial", "config": {"M": mc.M_AMOSTRAS, "n_iter": mc.N_ITER,
             "burn_in": mc.BURN_IN, "thin": mc.THIN, "semente": mc.SEMENTE,
             "gibbs": {"n_iter": mcc.N_ITER, "burn_in": mcc.BURN_IN, "thin": mcc.THIN}},
             "pooled": pooled, "contagens": cont, "diferencas_vs_quad": difs, "oficial_HMM_REAL": oficial,
             "quad_reproduz_oficial": bool(reproduz), "tempos": tempo_var, "tempo_total_s": tempo,
             "folds": {v: [{"participante": r["participante"], "metricas_h": r["metricas_h"],
                            "fp_lateral_h": r["fp_lateral_h"], "escolha": r["escolha"], "segundos": r["segundos"],
                            "diag": r["diag"]} for r in res_var[v]] for v in VARIANTES_OK}}
    os.makedirs(PASTA, exist_ok=True)
    json.dump(saida, open(os.path.join(PASTA, "resultados.json"), "w"), indent=1, default=float)
    open(os.path.join(PASTA, "tabelas.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
