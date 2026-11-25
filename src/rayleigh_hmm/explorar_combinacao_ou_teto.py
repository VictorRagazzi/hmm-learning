"""Exploratorio: OU de Rayleigh e HMM-REAL, cada um restrito SOZINHO a FP lateral <= teto individual.

LOSO (fora da amostra), sinal FC. Em cada fold, no treino, escolhe-se separadamente a melhor
config do Rayleigh (alfa, regra) e a do HMM (pi, A, regra) com FP lateral proprio <= teto
(2%, 3%, 4%, 5%; criterio: mais deteccao, menos FP, menor tempo). Depois, OU das duas no
participante retirado. O FP do OU fica <= soma dos dois tetos (no treino).
Saidas: results/rayleigh_hmm/exploratorio_combinacao_ou/tabelas_teto_individual.md (+ .json)
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_combinacao_ou_teto.py
"""
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import explorar_combinacao_ou as oc
import explorar_loso as el
import experimento as ex

TETOS = [0.02, 0.03, 0.04, 0.05]  # FP lateral maximo de CADA teste, no treino


def chave(m):
    return (m["deteccao"], -m["fp_lateral"], -(m["tempo_medio_s"] or np.inf))


def rodar_fold(fold):
    oc_seq, coef = oc.SEQ, oc.COEF
    participantes = sorted({s["participante"] for s in oc_seq})
    h = participantes[fold]
    treino = set(participantes) - {h}
    i_tr = [i for i, s in enumerate(oc_seq) if s["participante"] in treino]
    i_h = [i for i, s in enumerate(oc_seq) if s["participante"] == h]
    seq_tr, seq_h = [oc_seq[i] for i in i_tr], [oc_seq[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    n_tr, n_h = len(seq_tr), len(seq_h)

    emi = el.ajustar_braco("REAL", seq_tr, coef, treino, None)
    grupos_tr = el.grupos_log_em(seq_tr, emi)
    grupos_h = el.grupos_log_em(seq_h, emi)
    z_tr = [(idx, dur, np.stack([seq_tr[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_tr)]
    z_h = [(idx, dur, np.stack([seq_h[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_h)]

    melhor_r = {t: None for t in TETOS}
    for c in oc.candidatos_rayleigh(z_tr, n_tr, masc_tr):
        m = ex.metricas(masc_tr, c["det"], c["tempo"])
        for t in TETOS:
            if m["fp_lateral"] <= t + 1e-12 and (melhor_r[t] is None or chave(m) > chave(melhor_r[t]["m"])):
                melhor_r[t] = {"alfa": c["alfa"], "regra": c["regra"], "m": m}

    melhor_h = {t: None for t in TETOS}
    for p1, a01, a10 in itertools.product(el.GRADE_PI_BUSCA, el.GRADE_A01_BUSCA, el.GRADE_A10_BUSCA):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        gd = el.detectores_hmm(grupos_tr, pi, A)
        for regra in ex.REGRAS:
            det, tempo = ex.decidir(gd, n_tr, regra)
            m = ex.metricas(masc_tr, det, tempo)
            for t in TETOS:
                if m["fp_lateral"] <= t + 1e-12 and (melhor_h[t] is None or chave(m) > chave(melhor_h[t]["m"])):
                    melhor_h[t] = {"pi": pi.tolist(), "A": A.tolist(), "regra": regra, "m": m}

    def rayleigh(alfa, regra):
        gd = [(idx, *ex.indicadores(np.exp(-z) <= alfa), dur) for idx, dur, z in z_h]
        return ex.decidir(gd, n_h, regra)

    def hmm_h(c):
        gd = el.detectores_hmm(grupos_h, np.array(c["pi"]), np.array(c["A"]))
        return ex.decidir(gd, n_h, c["regra"])

    dec = {"referencia": rayleigh(ex.REFERENCIA_BRUTO["alfa"], ex.REFERENCIA_BRUTO["regra"])}
    for t in TETOS:
        r, hm = rayleigh(melhor_r[t]["alfa"], melhor_r[t]["regra"]), hmm_h(melhor_h[t])
        dec[f"Rayleigh-{int(t*100)}%"] = r
        dec[f"HMM-{int(t*100)}%"] = hm
        dec[f"OU-{int(t*100)}%"] = oc.ou(*r, *hm)
    return {"fold": fold, "participante": h, "indices_h": i_h, "decisoes": dec,
            "escolhas": {str(t): {"rayleigh": {k: melhor_r[t][k] for k in ("alfa", "regra")},
                                  "hmm": {k: melhor_h[t][k] for k in ("pi", "A", "regra")}} for t in TETOS}}


def pct(n, d):
    return f"{100 * n / d:.1f}%"


def main():
    t0 = time.time()
    oc.SEQ, _ = ex.carregar_sequencias()
    oc.COEF = el.carregar_coef()
    participantes = sorted({s["participante"] for s in oc.SEQ})
    n_seq = len(oc.SEQ)
    masc = ex.mascaras(oc.SEQ)
    with mp.get_context("fork").Pool(oc.N_PROCESSOS) as pool:
        res = pool.map(rodar_fold, range(len(participantes)))
    nomes = ["referencia"] + [f"{k}-{int(t*100)}%" for t in TETOS for k in ("Rayleigh", "HMM", "OU")]
    c_met = {}
    for nome in nomes:
        det, tempo = np.zeros(n_seq, bool), np.full(n_seq, np.nan)
        for r in res:
            det[r["indices_h"]], tempo[r["indices_h"]] = r["decisoes"][nome]
        c_met[nome] = {"detectado": det, "tempo_s": tempo}

    linhas = ["| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio / mediano (s) |", "|---|---|---|---|---|"]
    for nome in nomes:
        c = c_met[nome]
        m = ex.metricas(masc, c["detectado"], c["tempo_s"])
        linhas.append(f"| {nome} | {100*m['deteccao']:.1f}% | {100*m['fp_lateral']:.1f}% | {100*m['fp_alvo_esp']:.1f}% | "
                      f"{el.fmt_t(m['tempo_medio_s'])} / {el.fmt_t(m['tempo_mediano_s'])} |")
    pares = [(f"OU-{int(t*100)}%", "referencia") for t in TETOS] + [(f"OU-{int(t*100)}%", f"Rayleigh-{int(t*100)}%") for t in TETOS]
    sig, sig_linhas = oc.pareados(oc.SEQ, masc, c_met, pares)
    md = ["# Exploratorio: OU com teto de FP lateral individual (LOSO, FC)\n",
          "Cada teste (Rayleigh e HMM) e escolhido sozinho no treino com FP lateral proprio <= teto; OU no held-out.\n",
          "## Geral\n", *linhas, "\n## Testes pareados (so_A-so_B)\n", *sig_linhas, f"\nTempo: {time.time()-t0:.0f} s."]
    pasta = oc.PASTA
    os.makedirs(pasta, exist_ok=True)
    open(os.path.join(pasta, "tabelas_teto_individual.md"), "w").write("\n".join(md) + "\n")
    json.dump({"pareados": sig, "pooled": {n: ex.metricas(masc, c["detectado"], c["tempo_s"]) for n, c in c_met.items()},
               "escolhas": [{"participante": r["participante"], **r["escolhas"]} for r in res]},
              open(os.path.join(pasta, "resultados_teto_individual.json"), "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
