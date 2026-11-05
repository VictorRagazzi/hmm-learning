"""Exploratorio: LOSO (11 folds) do protocolo concatenado ESP->30->...->70 dB, SEM heranca.

Nao altera outros arquivos; importa de explorar_protocolo (ep), experimento (ex), emissao.
Para cada participante retirado h (T = outros 10), tudo que e ajustado/escolhido usa so T:
emissao Ausente (Gamma MLE nos alvos ESP de T), Presente (hierarquico nos alvos com
estimulo de T), M, parametros e regra. A configuracao escolhida e aplicada aos segmentos
de h. O Viterbi roda por protocolo (participante x frequencia x tipo), logo nao ha
vazamento entre participantes.
Metodos: rayleigh_loso (M, alfa, regra), hmm_loso (M, pi, A, regra; Viterbi causal,
decisao pelo estado atual) e bruto_referencia (Rayleigh M=None, alfa 0,05, 3 consecutivas).
Teto de FP lateral no treino = FP lateral do bruto_referencia nos laterais de T.

Execucao: .venv/bin/python src/rayleigh_hmm/explorar_protocolo_loso.py
Saidas: results/rayleigh_hmm/exploratorio_protocolo_loso/{resultados.json,tabelas.md,por_segmento.csv}
"""

import csv
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import emissao
import experimento as ex
import explorar_protocolo as ep
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI

# ---------------- configuracao ----------------
JANELAS_M = ep.JANELAS_M
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_protocolo_loso")
N_PROCESSOS = 10  # a matriz do ajuste hierarquico tem centenas de MB por processo
CABECALHO = "LOSO, sem herança; exploratório; cada intensidade decidida pela própria gravação; estado do HMM atravessa as gravações dentro do protocolo"
METODOS = ["rayleigh_loso", "hmm_loso"]
N_SEG = 1056

ARQ = EST = ZN = PART_SEG = PARTICIPANTES = None  # herdados pelos filhos (fork)


# ---------------- emissoes restritas a T ----------------
def ajustar_emissoes_treino(z, n, no_treino):
    seg_id = EST["seg_id"]
    no_esp = EST["m_alvo_esp"][seg_id] & no_treino
    no_est = EST["m_alvo_estim"][seg_id] & no_treino
    aus = emissao.ajustar_mle(z[no_esp])
    matriz = emissao.matriz_verossimilhanca_sequencias(z[no_est], n[no_est], seg_id[no_est],
                                                       ex.GRADE_HIER_K, ex.GRADE_HIER_RHO)
    hiper = emissao.ajustar_hierarquico(matriz, ex.GRADE_HIER_K, ex.GRADE_HIER_RHO, ex.INICIOS_HIER)
    del matriz
    pontos = ex.pontos_marginal(hiper)
    log_em = np.stack([emissao.log_pdf_gamma(z, aus["k"], aus["mu"]), emissao.log_pdf_preditiva(z, n, pontos)], axis=-1)
    aviso = {"na_borda": bool(hiper["na_borda"]), "hiper_no_limite": bool(hiper["hiper_no_limite"]),
             "dispersao_inicios": float(hiper["dispersao_relativa_inicios"]),
             "a_k": float(hiper["a_k"]), "b_k": float(hiper["b_k"]),
             "ausente_k": float(aus["k"]), "ausente_mu": float(aus["mu"])}
    return log_em, aviso


# ---------------- busca restrita a T ----------------
def contagens_treino(det_seg, tempo, no_treino_seg):
    m = no_treino_seg
    alvo = EST["m_alvo_estim"] & m
    t = tempo[alvo & det_seg]
    return (int(det_seg[alvo].sum()), int(det_seg[EST["m_lateral"] & m].sum()),
            int(det_seg[EST["m_alvo_esp"] & m].sum()), float(t.mean()) if t.size else np.inf)


def varrer(det, teto, treino_seg, melhor, parametros):
    pf, pc = ep.mascaras_regras(EST, det)
    for regra in ex.REGRAS:
        ds, tempo = ep.decidir_com(EST, pf, pc, regra)
        c = contagens_treino(ds, tempo, treino_seg)
        if c[1] > teto:
            continue
        if melhor.get("c") is None or ep.chave(c) > ep.chave(melhor["c"]):
            melhor.update({"c": c, "regra": regra, "detectado": ds, "tempo_s": tempo, **parametros})


def tarefa(args):
    fold, m = args
    t0 = time.time()
    h = PARTICIPANTES[fold]
    treino_seg = PART_SEG != h
    no_treino = treino_seg[EST["seg_id"]]
    z, n = ZN[m]
    log_em, aviso = ajustar_emissoes_treino(z, n, no_treino)
    log_pad = ep.padronizar(EST, log_em)
    teto = ZN["teto"][fold]

    ray = {}
    for alfa in ex.GRADE_ALFA:
        varrer(np.exp(-z) <= alfa, teto, treino_seg, ray, {"alfa": alfa})
    hm = {}
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
        pi = np.array([1 - p1, p1])
        A = np.array([[1 - a01, a01], [a10, 1 - a10]])
        d = ep.viterbi_estado(log_pad, pi, A)[EST["rows"], EST["cols"]]
        varrer(d, teto, treino_seg, hm, {"pi": pi.tolist(), "A": A.tolist(), "pi_presente": p1,
                                          "A_aus_pres": a01, "A_pres_aus": a10})
    return {"fold": fold, "m": m, "aviso": aviso, "rayleigh_loso": ray, "hmm_loso": hm, "segundos": time.time() - t0}


# ---------------- principal ----------------
def main():
    global ARQ, EST, ZN, PART_SEG, PARTICIPANTES
    t_ini = time.time()
    ARQ = ep.carregar_fases()
    EST = ep.montar_estrutura(ARQ)
    assert len(EST["segs"]) == N_SEG
    PARTICIPANTES = sorted({p for p, _ in ARQ})
    PART_SEG = np.array([s["participante"] for s in EST["segs"]])
    z_cum, _ = ep.z_plano(ARQ, EST, None)
    ref = ep.finalizar(EST, {"alfa": ex.REFERENCIA_BRUTO["alfa"], "regra": ex.REFERENCIA_BRUTO["regra"]},
                       np.exp(-z_cum) <= ex.REFERENCIA_BRUTO["alfa"])
    ref["metricas"] = ep.metricas_metodo(EST, ref)
    print(f"bruto_referencia: det {ref['metricas']['det']}/440 FP lat {ref['metricas']['fp_lateral']}/528 "
          f"FP ESP {ref['metricas']['fp_esp']}/88", flush=True)
    ZN = {m: ep.z_plano(ARQ, EST, m) for m in JANELAS_M}
    ZN["teto"] = [int(ref["detectado"][EST["m_lateral"] & (PART_SEG != h)].sum()) for h in PARTICIPANTES]
    print("tetos de FP lateral por fold:", ZN["teto"], flush=True)

    tarefas = [(f, m) for m in JANELAS_M for f in range(len(PARTICIPANTES))]
    saidas = []
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            print(f"[{k}/{len(tarefas)}] fold {PARTICIPANTES[r['fold']]} M={r['m']}: {r['segundos']:.0f}s "
                  f"(decorrido {time.time()-t_ini:.0f}s)", flush=True)

    # escolha de M por fold e metodo (criterio no treino) e juntada fora da amostra
    c_met = {k: {"detectado": np.zeros(N_SEG, bool), "tempo_s": np.full(N_SEG, np.nan),
                 "m_seg": np.full(N_SEG, "", dtype=object), "vezes": np.zeros(N_SEG, int)} for k in METODOS}
    folds = []
    for f, h in enumerate(PARTICIPANTES):
        no_h = PART_SEG == h
        for nome in METODOS:
            melhor, m_esc = None, None
            for m in JANELAS_M:
                r = next(x for x in saidas if x["fold"] == f and x["m"] == m)[nome]
                if r.get("c") is not None and (melhor is None or ep.chave(r["c"]) > ep.chave(melhor["c"])):
                    melhor, m_esc = r, m
            c = c_met[nome]
            c["vezes"][no_h] += 1
            if melhor is None:
                c["m_seg"][no_h] = "-"
                folds.append({"fold": f, "participante": h, "metodo": nome, "m": None, "viavel": False,
                              "teto": ZN["teto"][f]})
                continue
            c["detectado"][no_h] = melhor["detectado"][no_h]
            c["tempo_s"][no_h] = melhor["tempo_s"][no_h]
            c["m_seg"][no_h] = str(m_esc)
            dh = melhor["detectado"]
            folds.append({
                "fold": f, "participante": h, "metodo": nome, "m": str(m_esc), "viavel": True, "teto": ZN["teto"][f],
                "parametros": {k: melhor[k] for k in ("alfa", "pi", "A") if k in melhor},
                "regra": melhor["regra"], "c_treino": {"det": melhor["c"][0], "fp_lat": melhor["c"][1],
                                                       "fp_esp": melhor["c"][2], "tempo_medio_s": melhor["c"][3]},
                "h_det": int(dh[EST["m_alvo_estim"] & no_h].sum()), "h_fp_lat": int(dh[EST["m_lateral"] & no_h].sum()),
                "h_fp_esp": int(dh[EST["m_alvo_esp"] & no_h].sum())})
    for nome in METODOS:
        assert (c_met[nome]["vezes"] == 1).all(), "segmento avaliado 0 ou 2+ vezes"
        c_met[nome]["metricas"] = ep.metricas_metodo(EST, c_met[nome])
    print("sanidade: cada um dos", N_SEG, "segmentos avaliado exatamente uma vez em cada metodo")

    todos = {"bruto_referencia": ref, **c_met}
    testes = {}
    for a, b in (("rayleigh_loso", "hmm_loso"), ("bruto_referencia", "rayleigh_loso"), ("bruto_referencia", "hmm_loso")):
        testes[f"{a} vs {b}"] = ep.testes_pareados(EST, todos[a], todos[b])  # so_ref = so a, so_metodo = so b

    avisos = [{"fold": PARTICIPANTES[r["fold"]], "m": str(r["m"]), **r["aviso"]} for r in sorted(saidas, key=lambda r: (str(r["m"]), r["fold"]))]
    disp = {str(m): int(sum(a["dispersao_inicios"] > 1e-3 for a in avisos if a["m"] == str(m))) for m in JANELAS_M}
    freq_m = {nome: {str(m): sum(1 for x in folds if x["metodo"] == nome and x["m"] == str(m)) for m in JANELAS_M}
              for nome in METODOS}
    total = time.time() - t_ini
    saida = {
        "aviso": CABECALHO, "participantes": PARTICIPANTES,
        "configuracao": {"janelas_m": JANELAS_M, "grade_pi": GRADE_PI, "grade_a_aus_pres": GRADE_A01,
                         "grade_a_pres_aus": GRADE_A10, "grade_alfa": ex.GRADE_ALFA, "n_regras": len(ex.REGRAS),
                         "tetos_por_fold": dict(zip(PARTICIPANTES, ZN["teto"]))},
        "metricas": {"bruto_referencia": ep.sem_arrays(ref)["metricas"], **{k: c_met[k]["metricas"] for k in METODOS}},
        "testes_pareados": testes, "folds": folds, "frequencia_m": freq_m,
        "ajuste_hierarquico": avisos, "folds_com_dispersao_acima_1e-3_por_m": disp,
        "ajuste_hierarquico_na_borda_ou_limite": {str(m): int(sum(a["na_borda"] or a["hiper_no_limite"] for a in avisos if a["m"] == str(m))) for m in JANELAS_M},
        "tempo_total_s": total}
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(PASTA, "por_segmento.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "fold_participante", "tipo", "freq_hz", "intensidade", "detectado", "tempo_s", "M_escolhido"])
        for nome in ["bruto_referencia"] + METODOS:
            c = todos[nome]
            for k, s in enumerate(EST["segs"]):
                t = c["tempo_s"][k]
                w.writerow([nome, s["participante"], s["tipo"], s["freq"], s["cond"], int(c["detectado"][k]),
                            "" if np.isnan(t) else f"{t:.1f}", "None" if nome == "bruto_referencia" else c["m_seg"][k]])
    escrever_md(saida, total)
    print(f"tempo total {total:.0f}s; saidas em {PASTA}")


# ---------------- tabelas ----------------
def pct(x, n):
    return f"{100 * x / n:.1f}%".replace(".", ",")


def f_t(x):
    return "-" if x is None else f"{x:.0f}"


def txt_regra(r):
    return ", ".join(f"{k}={v}" for k, v in r.items() if k != "familia") + f" ({r['familia']})"


def linha(nome, m):
    di = m["det_por_intensidade"]
    return (f"| {nome} | {pct(m['det'], 440)} | " + " | ".join(pct(di[c], 88) for c in ep.ORDEM_CONDICOES[1:])
            + f" | {pct(m['fp_lateral'], 528)} | {pct(m['fp_esp'], 88)} | {pct(m['proto_alvo_det_estimulo'], 88)} "
            f"| {pct(m['proto_lateral_det'], 88)} | {f_t(m['tempo_medio_s'])} / {f_t(m['tempo_mediano_s'])} |")


def escrever_md(s, total):
    nomes = ["bruto_referencia"] + METODOS
    md = [f"# {CABECALHO}", "",
          "Percentuais: detecção geral sobre 440; cada intensidade sobre 88; FP lateral sobre 528; FP ESP e protocolos sobre 88. "
          "Tudo ajustado/escolhido (emissões, M, parâmetros, regra) usa só os 10 participantes de treino; teto de FP lateral no treino = "
          "FP lateral do bruto_referencia nos laterais de T. Viterbi roda por protocolo (um participante × frequência), sem vazamento entre participantes.", "",
          "## Resultado fora da amostra (11 folds)", "",
          "| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    md += [linha(n, s["metricas"][n]) for n in nomes]
    md += ["", "Primeira intensidade detectada por protocolo-alvo (% de 88):", "",
           "| Método | " + " | ".join(ep.ORDEM_CONDICOES + ["nenhuma"]) + " |", "|---|" + "---:|" * 7]
    for n in nomes:
        p = s["metricas"][n]["primeira_intensidade"]
        md.append(f"| {n} | " + " | ".join(pct(p[c], 88) for c in ep.ORDEM_CONDICOES + ["nenhuma"]) + " |")
    md += ["", "## Testes pareados", "",
           "McNemar exato (só_A / só_B, p) e Wilcoxon de tempo nos pares detectados por ambos (n, mediana A/B, p).", "",
           "| A vs B | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |", "|---|---|---|---|---|---|---|"]
    for k, t in s["testes_pareados"].items():
        w = t["tempo"]
        pw = "-" if w["p_valor"] is None else f"{w['p_valor']:.3g}"
        md.append(f"| {k} | " + " | ".join(f"{t[q]['so_ref']}/{t[q]['so_metodo']}, p={t[q]['p_valor']:.3g}"
                  for q in ("alvos", "laterais", "esp", "alvos_proto", "laterais_proto"))
                  + f" | {w['n_pares']}, {f_t(w['mediana_ref_s'])}/{f_t(w['mediana_metodo_s'])}, p={pw} |")
    md += ["", "## Frequência de escolha de M (folds, de 11)", "",
           "| Método | " + " | ".join(f"M={m}" for m in s["frequencia_m"]["hmm_loso"]) + " |", "|---|" + "---:|" * len(JANELAS_M)]
    for n in METODOS:
        md.append(f"| {n} | " + " | ".join(str(v) for v in s["frequencia_m"][n].values()) + " |")
    md += ["", "## Por fold (participante retirado)", "",
           "Det alvos /40, FP lat /48, FP ESP /8 do participante retirado; treino = valores em T.", "",
           "| Fold | Método | M | Parâmetros | Regra | Teto | Treino det / FP lat | Retirado det / FP lat / FP ESP |",
           "|---|---|---|---|---|---:|---|---|"]
    for x in s["folds"]:
        if not x["viavel"]:
            md.append(f"| {x['participante']} | {x['metodo']} | - | inviável | - | {x['teto']} | - | - |")
            continue
        par = x["parametros"]
        if "alfa" in par:
            ptxt = f"α={par['alfa']}"
        else:
            ptxt = f"π={par['pi'][1]:g}, A01={par['A'][0][1]:g}, A10={par['A'][1][0]:g}"
        md.append(f"| {x['participante']} | {x['metodo']} | {x['m']} | {ptxt} | {txt_regra(x['regra'])} | {x['teto']} | "
                  f"{x['c_treino']['det']} / {x['c_treino']['fp_lat']} | {x['h_det']} / {x['h_fp_lat']} / {x['h_fp_esp']} |")
    md += ["", "## Avisos do ajuste hierárquico (Presente, por fold e M)", "",
           "Folds (de 11) com dispersão relativa dos inícios > 1e-3, por M: "
           + ", ".join(f"M={m}: {v}" for m, v in s["folds_com_dispersao_acima_1e-3_por_m"].items()) + ".", "",
           "Folds com na_borda ou hiper_no_limite, por M: "
           + ", ".join(f"M={m}: {v}" for m, v in s["ajuste_hierarquico_na_borda_ou_limite"].items()) + ".", "",
           "| M | Fold | na_borda | hiper_no_limite | dispersão inícios | Ausente k, μ | a_k, b_k |", "|---|---|---|---|---:|---|---|"]
    for a in s["ajuste_hierarquico"]:
        md.append(f"| {a['m']} | {a['fold']} | {a['na_borda']} | {a['hiper_no_limite']} | {a['dispersao_inicios']:.1e} | "
                  f"{a['ausente_k']:.3f}, {a['ausente_mu']:.3f} | {a['a_k']:.3g}, {a['b_k']:.3g} |")
    md += ["", f"Tempo total: {total:.0f} s."]
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
