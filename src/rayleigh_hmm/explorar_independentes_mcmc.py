"""Exploratorio: LOSO em blocos INDEPENDENTES (M=30 e M=10) usando as curvas calibradas no Z acumulado via MCMC (GIBBS).

Em cada fold (deixando 1 participante de fora):
  1) As duas curvas de emissao (Ausente por MLE e Presente pelo MCMC completo GIBBS:
     4 cadeias, 20 000 iteracoes, burn-in 5 000, thin 10, M = 2 000 sorteios) sao
     calibradas no Z ACUMULADO dos 10 participantes de treino — exatamente o mesmo
     ajuste do experimento principal (R-hat <= 1,002 em todos os folds).
  2) As observacoes usadas na busca de (pi, A, regra) nos 10 participantes de treino
     e na avaliacao do participante deixado de fora sao os blocos INDEPENDENTES de
     M epocas (M=30 e M=10, sem acumular; N = M fixo na emissao Presente).
  3) Teto de FP lateral por fold = FP lateral do Rayleigh acumulado de referencia
     nos 10 participantes de treino; nenhuma regra dispara antes de 30 s
     (M=30 -> 1 avaliacao; M=10 -> 3 avaliacoes).

Saidas: results/rayleigh_hmm/exploratorio_independentes/mcmc/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_independentes_mcmc.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

PASTA_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [PASTA_SRC, os.path.join(PASTA_SRC, "monte_carlo")]

import dados  # noqa: E402
import emissao  # noqa: E402
import experimento as ex  # noqa: E402
import explorar_independentes as ei  # noqa: E402
import explorar_loso as el  # noqa: E402
import mc_comum as mc  # noqa: E402
import mcmc_completo as mcc  # noqa: E402

# ---------------- configuracao ----------------
VALORES_M = (30, 10)
TEMPO_MINIMO_S = ei.TEMPO_MINIMO_S
N_PROCESSOS = 6
PASTA_SAIDA = os.path.join(ei.PASTA_BASE, "mcmc")

# Globais herdadas via fork
SEQ_ACUM = None
SEQS_BLOCOS = {}
TETOS = None
MIN_AVS = {}


def calibrar_emissoes_acum(participante_h):
    """Calibra Ausente (MLE) e Presente (MCMC GIBBS) no Z acumulado dos 10 participantes de treino."""
    seq_tr = [s for s in SEQ_ACUM if s["participante"] != participante_h]
    z_esp = np.concatenate([s["z"] for s in seq_tr if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    aus = emissao.ajustar_mle(z_esp)
    alvos = [s for s in seq_tr if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    emi = {
        "ausente": aus,
        "z_sintetico": np.concatenate([s["z"] for s in alvos]),
        "n_sintetico": np.concatenate([s["n_ep"] for s in alvos]),
        "id_sintetico": np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)]),
    }
    res = mcc.ajustar_completo(emi, rotulo=f"acum_{participante_h}")
    pontos = mc.pontos_mcmc(res, burn_in=mcc.BURN_IN, thin=mcc.THIN)
    diag = mc.diagnosticos(res, mcc.BURN_IN, mcc.THIN)
    diag["segundos_mcmc"] = res["segundos"]
    emi["marginal"] = {"pontos": pontos}
    return emi, diag


def rodar_bloco_fold(seq_bloco, emi, participante_h, fold, m_bloco):
    """Busca (pi, A, regra) nos blocos dos 10 de treino e avalia no participante deixado de fora."""
    ex.EPOCAS_POR_JANELA = m_bloco
    ex.MINIMO_AVALIACOES = MIN_AVS[m_bloco]
    teto = TETOS[participante_h]
    i_tr = [i for i, s in enumerate(seq_bloco) if s["participante"] != participante_h]
    i_h = [i for i, s in enumerate(seq_bloco) if s["participante"] == participante_h]
    seq_tr, seq_h = [seq_bloco[i] for i in i_tr], [seq_bloco[i] for i in i_h]
    masc_tr, masc_h = ex.mascaras(seq_tr), ex.mascaras(seq_h)

    # HMM com emissoes calibradas no Z acumulado
    grupos_tr = el.grupos_log_em(seq_tr, emi)
    melhores = {}
    for p1, a01, a10 in itertools.product(el.GRADE_PI_BUSCA, el.GRADE_A01_BUSCA, el.GRADE_A10_BUSCA):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        ex.avaliar_regras(seq_tr, masc_tr, el.detectores_hmm(grupos_tr, pi, A),
                          {"pi": pi.tolist(), "A": A.tolist()}, melhores, teto)
    melhor = ex.melhor_geral(melhores)
    if melhor is None:
        det_h, tempo_h = np.zeros(len(seq_h), bool), np.full(len(seq_h), np.nan)
        escolha = None
    else:
        pi, A = np.array(melhor["pi"]), np.array(melhor["A"])
        gd_h = el.detectores_hmm(el.grupos_log_em(seq_h, emi), pi, A)
        det_h, tempo_h = ex.decidir(gd_h, len(seq_h), melhor["regra"])
        escolha = {"pi": melhor["pi"], "A": melhor["A"], "regra": melhor["regra"],
                   "metricas_treino": melhor["metricas"]}
    r_hmm = {
        "fold": fold, "participante": participante_h, "indices_h": i_h,
        "detectado_h": det_h, "tempo_h": tempo_h,
        "metricas_h": ex.metricas(masc_h, det_h, tempo_h),
        "fp_lateral_h": float(det_h[masc_h["lateral"]].mean()),
        "escolha": escolha,
    }

    # Rayleigh otimizado em blocos (mesmo teto de FP)
    melhores_b, _ = ex.buscar_bruto(seq_tr, masc_tr, teto)
    melhor_b = ex.melhor_geral(melhores_b)
    grupos_b = [(idx, dur, np.stack([seq_h[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_h)]
    gd_b = [(idx, *ex.indicadores(np.exp(-z) <= melhor_b["alfa"]), dur) for idx, dur, z in grupos_b]
    det_b, tempo_b = ex.decidir(gd_b, len(seq_h), melhor_b["regra"])
    r_bruto = {
        "fold": fold, "participante": participante_h, "indices_h": i_h,
        "detectado_h": det_b, "tempo_h": tempo_b,
        "alfa": melhor_b["alfa"], "regra": melhor_b["regra"],
        "fp_lateral_h": float(det_b[masc_h["lateral"]].mean()),
    }
    return r_hmm, r_bruto


def tarefa_participante(args):
    fold, participante_h = args
    t0 = time.time()
    emi, diag = calibrar_emissoes_acum(participante_h)
    por_m = {}
    for M in VALORES_M:
        r_hmm, r_bruto = rodar_bloco_fold(SEQS_BLOCOS[M], emi, participante_h, fold, M)
        r_hmm["diag"] = diag
        por_m[M] = (r_hmm, r_bruto)
    return fold, participante_h, por_m, diag, time.time() - t0


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def seg(x):
    return "-" if x is None else f"{x:.1f}".replace(".", ",")


def resumo_pareado(seq, masc, c_hmm, c_bruto):
    sig = ex.significancia(seq, masc, c_bruto, c_hmm)
    ambos = masc["alvo_estimulo"] & c_bruto["detectado"] & c_hmm["detectado"]
    d = c_hmm["tempo_s"][ambos] - c_bruto["tempo_s"][ambos]
    return {
        "mcnemar": {k: sig[k] for k in ("alvos", "laterais", "esp")},
        "tempo": {
            "n_pares": int(ambos.sum()),
            "hmm_mais_rapido": int((d < 0).sum()),
            "hmm_mais_lento": int((d > 0).sum()),
            "iguais": int((d == 0).sum()),
            "dif_media_s": float(d.mean()) if ambos.sum() else None,
            "p_valor": sig["tempo"]["p_valor"],
        },
    }


def main():
    global SEQ_ACUM, SEQS_BLOCOS, TETOS, MIN_AVS
    t0 = time.time()
    SEQ_ACUM, _ = ex.carregar_sequencias()
    TETOS, _, _ = ei.calcular_tetos(SEQ_ACUM)
    n_acum = len(SEQ_ACUM)

    # Carrega sequencias em blocos para M=30 e M=10 antes do fork
    orig = (dados.rayleigh_z_acumulado, ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES)
    try:
        for M in VALORES_M:
            MIN_AVS[M] = max(1, TEMPO_MINIMO_S // M)
            dados.rayleigh_z_acumulado = ei.rayleigh_z_blocos
            ex.EPOCAS_POR_JANELA = M
            ex.MINIMO_AVALIACOES = MIN_AVS[M]
            seq_m, _ = ex.carregar_sequencias()
            assert len(seq_m) == n_acum, f"M={M}: {len(seq_m)} != {n_acum}"
            SEQS_BLOCOS[M] = seq_m
    finally:
        dados.rayleigh_z_acumulado, ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES = orig

    participantes = sorted({s["participante"] for s in SEQ_ACUM})
    tarefas = list(enumerate(participantes))
    print(f"Rodando 11 folds (emissao GIBBS no Z acumulado -> blocos M={VALORES_M}, {N_PROCESSOS} processos)...",
          flush=True)

    acum_hmm = {M: [] for M in VALORES_M}
    acum_bruto = {M: [] for M in VALORES_M}
    diags = []
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        for k, (fold, p, por_m, diag, dt) in enumerate(pool.imap_unordered(tarefa_participante, tarefas), 1):
            diags.append(diag)
            for M in VALORES_M:
                acum_hmm[M].append(por_m[M][0])
                acum_bruto[M].append(por_m[M][1])
            print(f"  [{k}/{len(tarefas)}] {p}: {dt:.1f} s | "
                  f"R-hat max={diag['rhat_max']:.4f} | ESS min={diag['ess_min_thin']:.0f}", flush=True)

    rhat_max = max(d["rhat_max"] for d in diags)
    ess_min = min(d["ess_min_thin"] for d in diags)

    resultados_m = {"calibracao": "Z_acumulado_GIBBS", "mcmc": {"rhat_max": rhat_max, "ess_min_thin": ess_min}}
    linhas = [
        "# Rayleigh em blocos independentes com curvas calibradas no Z acumulado via MCMC (GIBBS, LOSO)\n",
        f"Emissoes (Ausente MLE + Presente GIBBS) calibradas no Z acumulado dos 10 participantes de treino "
        f"(R-hat max = {rhat_max:.4f}, ESS min = {ess_min:.0f}); observacoes em blocos independentes de M epocas. "
        "Teto de FP lateral por fold = FP lateral do Rayleigh acumulado de referencia no treino. "
        "Nenhuma regra dispara antes de 30 s (M=30 -> 1 avaliacao; M=10 -> 3 avaliacoes).\n",
        "## Geral (pooled, fora da amostra)\n",
        "| Bloco | Metodo | Deteccao | FP lateral | FP ESP | Tempo medio / mediano (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for M in VALORES_M:
        seq_m = SEQS_BLOCOS[M]
        masc = ex.mascaras(seq_m)
        rs_h = sorted(acum_hmm[M], key=lambda r: r["fold"])
        rs_b = sorted(acum_bruto[M], key=lambda r: r["fold"])
        c_h = el.juntar(rs_h, len(seq_m))
        c_b = el.juntar(rs_b, len(seq_m))
        m_h = ex.metricas(masc, c_h["detectado"], c_h["tempo_s"])
        m_b = ex.metricas(masc, c_b["detectado"], c_b["tempo_s"])
        par = resumo_pareado(seq_m, masc, c_h, c_b)
        resultados_m[str(M)] = {
            "M": M,
            "minimo_avaliacoes": MIN_AVS[M],
            "pooled": {"HMM_GIBBS": m_h, "Rayleigh_otimizado": m_b},
            "pareado_hmm_vs_rayleigh": par,
            "folds_hmm": [
                {"participante": r["participante"], "metricas_h": r["metricas_h"],
                 "fp_lateral_h": r["fp_lateral_h"], "escolha": r["escolha"]}
                for r in rs_h
            ],
        }
        linhas.append(f"| M={M} | Rayleigh (blocos, LOSO) | {pct(m_b['deteccao'])} | {pct(m_b['fp_lateral'])} | "
                      f"{pct(m_b['fp_alvo_esp'])} | {seg(m_b['tempo_medio_s'])} / {seg(m_b['tempo_mediano_s'])} |")
        linhas.append(f"| M={M} | HMM bayesiano (GIBBS) | {pct(m_h['deteccao'])} | {pct(m_h['fp_lateral'])} | "
                      f"{pct(m_h['fp_alvo_esp'])} | {seg(m_h['tempo_medio_s'])} / {seg(m_h['tempo_mediano_s'])} |")

    linhas += ["\n## Comparacao pareada (HMM GIBBS vs Rayleigh blocos otimizado)\n",
               "| Bloco | McNemar alvos (so Rayleigh x so HMM, p) | Laterais | ESP | Wilcoxon tempo (n, p) |",
               "|---|---|---|---|---|"]
    for M in VALORES_M:
        par = resultados_m[str(M)]["pareado_hmm_vs_rayleigh"]
        mc_a, mc_l, mc_e, tw = par["mcnemar"]["alvos"], par["mcnemar"]["laterais"], par["mcnemar"]["esp"], par["tempo"]
        linhas.append(
            f"| M={M} | {mc_a['so_bruto']} x {mc_a['so_hmm']}, p={mc_a['p_valor']:.4g} | "
            f"{mc_l['so_bruto']} x {mc_l['so_hmm']}, p={mc_l['p_valor']:.4g} | "
            f"{mc_e['so_bruto']} x {mc_e['so_hmm']}, p={mc_e['p_valor']:.4g} | "
            f"n={tw['n_pares']}, p={tw['p_valor']:.4g} |"
        )
    linhas.append(f"\nTempo total: {time.time() - t0:.1f} s.")

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    with open(os.path.join(PASTA_SAIDA, "resultados.json"), "w") as f:
        json.dump(resultados_m, f, indent=1, default=float)
    with open(os.path.join(PASTA_SAIDA, "tabelas.md"), "w") as f:
        f.write("\n".join(linhas) + "\n")
    print("\n" + "\n".join(linhas))


if __name__ == "__main__":
    main()
