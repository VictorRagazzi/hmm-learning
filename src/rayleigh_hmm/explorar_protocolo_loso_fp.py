"""Exploratorio: LOSO do protocolo concatenado com varios tetos de FP lateral no treino.

Reutiliza explorar_protocolo_loso (pl): emissoes por fold, grades, regras, busca. As emissoes
(por fold e M) sao ajustadas uma vez e a busca de parametros/regra e feita uma vez por
(fold, M); a escolha da melhor configuracao e feita separadamente para cada teto.
Teto = fracao dos segmentos laterais de T (10 participantes, 480 segmentos), arredondada
para baixo; "atual" = FP lateral do bruto_referencia nos laterais de T (por fold).
O mesmo teto vale para rayleigh_loso e hmm_loso. Fold sem configuracao viavel = sem
deteccao (registrado em "folds" com viavel=False e contado no resumo).

Execucao: .venv/bin/python src/rayleigh_hmm/explorar_protocolo_loso_fp.py
Saidas: results/rayleigh_hmm/exploratorio_protocolo_loso_fp/{resultados.json,tabelas.md,por_segmento.csv}
"""

import csv
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import experimento as ex
import explorar_protocolo as ep
import explorar_protocolo_loso as pl
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI

# ---------------- configuracao ----------------
TETOS_FP = [0.05, 0.03, 0.02, 0.01]  # fracao dos segmentos laterais de treino
JANELAS_M = pl.JANELAS_M
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_protocolo_loso_fp")
# opcional: python explorar_protocolo_loso_fp.py 0.06,0.07,0.08 exploratorio_protocolo_loso_fp_6a8
if len(sys.argv) > 1:
    TETOS_FP = [float(x) for x in sys.argv[1].split(",")]
if len(sys.argv) > 2:
    PASTA = os.path.join(ex.PASTA_RESULTADOS, sys.argv[2])
N_PROCESSOS = pl.N_PROCESSOS
METODOS = pl.METODOS
N_SEG = pl.N_SEG
CABECALHO = "LOSO, sem herança; exploratório; tetos de FP lateral no treino variados; 11 participantes"
CHAVES = ["atual"] + [str(t) for t in TETOS_FP]


def rotulo(k):
    return "atual (FP do bruto_referencia em T)" if k == "atual" else f"{float(k) * 100:g}% dos laterais de T"


# ---------------- busca com varios tetos ----------------
def varrer_multi(det, tetos, treino_seg, melhores, parametros):
    pf, pc = ep.mascaras_regras(pl.EST, det)
    for regra in ex.REGRAS:
        ds, tempo = ep.decidir_com(pl.EST, pf, pc, regra)
        c = pl.contagens_treino(ds, tempo, treino_seg)
        for k, teto in tetos.items():
            if c[1] > teto:
                continue
            b = melhores[k]
            if b.get("c") is None or ep.chave(c) > ep.chave(b["c"]):
                b.update({"c": c, "regra": regra, "detectado": ds, "tempo_s": tempo, **parametros})


def tarefa(args):
    fold, m = args
    t0 = time.time()
    h = pl.PARTICIPANTES[fold]
    treino_seg = pl.PART_SEG != h
    no_treino = treino_seg[pl.EST["seg_id"]]
    z, n = pl.ZN[m]
    log_em, aviso = pl.ajustar_emissoes_treino(z, n, no_treino)
    log_pad = ep.padronizar(pl.EST, log_em)
    n_lat = int((pl.EST["m_lateral"] & treino_seg).sum())
    tetos = {"atual": pl.ZN["teto"][fold], **{str(t): int(np.floor(t * n_lat + 1e-9)) for t in TETOS_FP}}
    ray = {k: {} for k in tetos}
    for alfa in ex.GRADE_ALFA:
        varrer_multi(np.exp(-z) <= alfa, tetos, treino_seg, ray, {"alfa": alfa})
    hm = {k: {} for k in tetos}
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
        pi = np.array([1 - p1, p1])
        A = np.array([[1 - a01, a01], [a10, 1 - a10]])
        d = ep.viterbi_estado(log_pad, pi, A)[pl.EST["rows"], pl.EST["cols"]]
        varrer_multi(d, tetos, treino_seg, hm, {"pi": pi.tolist(), "A": A.tolist(), "pi_presente": p1,
                                                  "A_aus_pres": a01, "A_pres_aus": a10})
    return {"fold": fold, "m": m, "aviso": aviso, "rayleigh_loso": ray, "hmm_loso": hm, "tetos": tetos,
            "segundos": time.time() - t0}


# ---------------- principal ----------------
def main():
    t_ini = time.time()
    pl.ARQ = ep.carregar_fases()
    pl.EST = EST = ep.montar_estrutura(pl.ARQ)
    assert len(EST["segs"]) == N_SEG
    pl.PARTICIPANTES = PARTICIPANTES = sorted({p for p, _ in pl.ARQ})
    pl.PART_SEG = PART_SEG = np.array([s["participante"] for s in EST["segs"]])
    z_cum, _ = ep.z_plano(pl.ARQ, EST, None)
    ref = ep.finalizar(EST, {"alfa": ex.REFERENCIA_BRUTO["alfa"], "regra": ex.REFERENCIA_BRUTO["regra"]},
                       np.exp(-z_cum) <= ex.REFERENCIA_BRUTO["alfa"])
    ref["metricas"] = ep.metricas_metodo(EST, ref)
    pl.ZN = {m: ep.z_plano(pl.ARQ, EST, m) for m in JANELAS_M}
    pl.ZN["teto"] = [int(ref["detectado"][EST["m_lateral"] & (PART_SEG != h)].sum()) for h in PARTICIPANTES]
    print("tetos 'atual' por fold:", pl.ZN["teto"], flush=True)

    tarefas = [(f, m) for m in JANELAS_M for f in range(len(PARTICIPANTES))]
    saidas = []
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            print(f"[{k}/{len(tarefas)}] fold {PARTICIPANTES[r['fold']]} M={r['m']}: {r['segundos']:.0f}s "
                  f"(decorrido {time.time()-t_ini:.0f}s)", flush=True)

    por_teto, folds_all = {}, []
    for tk in CHAVES:
        c_met = {k: {"detectado": np.zeros(N_SEG, bool), "tempo_s": np.full(N_SEG, np.nan),
                     "m_seg": np.full(N_SEG, "", dtype=object), "vezes": np.zeros(N_SEG, int)} for k in METODOS}
        folds = []
        for f, h in enumerate(PARTICIPANTES):
            no_h = PART_SEG == h
            for nome in METODOS:
                melhor, m_esc = None, None
                for m in JANELAS_M:
                    sx = next(x for x in saidas if x["fold"] == f and x["m"] == m)
                    r = sx[nome][tk]
                    if r.get("c") is not None and (melhor is None or ep.chave(r["c"]) > ep.chave(melhor["c"])):
                        melhor, m_esc = r, m
                teto_int = next(x for x in saidas if x["fold"] == f)["tetos"][tk]
                c = c_met[nome]
                c["vezes"][no_h] += 1
                if melhor is None:
                    c["m_seg"][no_h] = "-"
                    folds.append({"teto_chave": tk, "fold": f, "participante": h, "metodo": nome, "m": None,
                                  "viavel": False, "teto_seg": teto_int})
                    continue
                c["detectado"][no_h] = melhor["detectado"][no_h]
                c["tempo_s"][no_h] = melhor["tempo_s"][no_h]
                c["m_seg"][no_h] = str(m_esc)
                dh = melhor["detectado"]
                folds.append({
                    "teto_chave": tk, "fold": f, "participante": h, "metodo": nome, "m": str(m_esc), "viavel": True,
                    "teto_seg": teto_int,
                    "parametros": {k: melhor[k] for k in ("alfa", "pi", "A") if k in melhor},
                    "regra": melhor["regra"], "c_treino": {"det": melhor["c"][0], "fp_lat": melhor["c"][1],
                                                           "fp_esp": melhor["c"][2], "tempo_medio_s": melhor["c"][3]},
                    "h_det": int(dh[EST["m_alvo_estim"] & no_h].sum()), "h_fp_lat": int(dh[EST["m_lateral"] & no_h].sum()),
                    "h_fp_esp": int(dh[EST["m_alvo_esp"] & no_h].sum())})
        for nome in METODOS:
            assert (c_met[nome]["vezes"] == 1).all()
            c_met[nome]["metricas"] = ep.metricas_metodo(EST, c_met[nome])
        todos = {"bruto_referencia": ref, **c_met}
        testes = {"rayleigh_loso vs hmm_loso": ep.testes_pareados(EST, todos["rayleigh_loso"], todos["hmm_loso"])}
        por_teto[tk] = {"c_met": c_met, "testes": testes, "folds": folds}
        folds_all += folds
        print(f"teto {tk}: inviaveis {sum(not x['viavel'] for x in folds)}; "
              + "; ".join(f"{n}: det {c_met[n]['metricas']['det']}/440 FP lat {c_met[n]['metricas']['fp_lateral']}/528"
                          for n in METODOS), flush=True)

    avisos = [{"fold": PARTICIPANTES[r["fold"]], "m": str(r["m"]), **r["aviso"]}
              for r in sorted(saidas, key=lambda r: (str(r["m"]), r["fold"]))]
    total = time.time() - t_ini
    saida = {
        "aviso": CABECALHO, "participantes": PARTICIPANTES,
        "configuracao": {"tetos_fp_fracao": TETOS_FP, "janelas_m": JANELAS_M, "grade_pi": GRADE_PI,
                         "grade_a_aus_pres": GRADE_A01, "grade_a_pres_aus": GRADE_A10, "grade_alfa": ex.GRADE_ALFA,
                         "n_regras": len(ex.REGRAS), "teto_atual_por_fold": dict(zip(PARTICIPANTES, pl.ZN["teto"])),
                         "tetos_seg_por_chave": {tk: sorted({x["tetos"][tk] for x in saidas}) for tk in CHAVES}},
        "bruto_referencia": ep.sem_arrays(ref)["metricas"],
        "por_teto": {tk: {"metricas": {n: por_teto[tk]["c_met"][n]["metricas"] for n in METODOS},
                          "testes_pareados": por_teto[tk]["testes"],
                          "folds": por_teto[tk]["folds"],
                          "folds_inviaveis": [{"participante": x["participante"], "metodo": x["metodo"]}
                                              for x in por_teto[tk]["folds"] if not x["viavel"]]} for tk in CHAVES},
        "ajuste_hierarquico": avisos,
        "folds_com_dispersao_acima_1e-3_por_m": {str(m): int(sum(a["dispersao_inicios"] > 1e-3 for a in avisos if a["m"] == str(m))) for m in JANELAS_M},
        "ajuste_hierarquico_na_borda_ou_limite": {str(m): int(sum(a["na_borda"] or a["hiper_no_limite"] for a in avisos if a["m"] == str(m))) for m in JANELAS_M},
        "tempo_total_s": total}
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(PASTA, "por_segmento.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["teto", "metodo", "fold_participante", "tipo", "freq_hz", "intensidade", "detectado", "tempo_s", "M_escolhido"])
        for tk in [None] + CHAVES:
            for nome in (["bruto_referencia"] if tk is None else METODOS):
                c = ref if tk is None else por_teto[tk]["c_met"][nome]
                for k, s in enumerate(EST["segs"]):
                    t = c["tempo_s"][k]
                    w.writerow(["fixo" if tk is None else tk, nome, s["participante"], s["tipo"], s["freq"], s["cond"],
                                int(c["detectado"][k]), "" if np.isnan(t) else f"{t:.1f}",
                                "None" if tk is None else c["m_seg"][k]])
    escrever_md(saida, total)
    print(f"tempo total {total:.0f}s; saidas em {PASTA}")


# ---------------- tabelas ----------------
def pc(x, n):
    return f"{pl.pct(x, n)} ({x})"


def escrever_md(s, total):
    cab = ("| Método | Det /440 | " + " | ".join(f"{c} /88" for c in ep.ORDEM_CONDICOES[1:])
           + " | FP lat /528 | FP ESP /88 | Proto alvo /88 | Proto lat /88 | Tempo médio/mediano (s) |")
    sep = "|---|" + "---:|" * 11

    def lin(nome, m):
        di = m["det_por_intensidade"]
        return (f"| {nome} | {pc(m['det'], 440)} | " + " | ".join(pc(di[c], 88) for c in ep.ORDEM_CONDICOES[1:])
                + f" | {pc(m['fp_lateral'], 528)} | {pc(m['fp_esp'], 88)} | {pc(m['proto_alvo_det_estimulo'], 88)} "
                f"| {pc(m['proto_lateral_det'], 88)} | {pl.f_t(m['tempo_medio_s'])} / {pl.f_t(m['tempo_mediano_s'])} |")

    md = [f"# {CABECALHO}", "",
          "Tetos de FP lateral no treino (T = 10 participantes, 480 segmentos laterais): fração × 480, arredondada para baixo "
          "(" + ", ".join(f"{t * 100:g}% = {int(np.floor(t * 480 + 1e-9))}" for t in TETOS_FP) + "); 'atual' = FP do bruto_referencia nos laterais de T (por fold). "
          "Mesmo teto para rayleigh_loso e hmm_loso. Emissões, M, parâmetros e regra escolhidos só em T. "
          "Fold sem configuração viável sob o teto = sem detecção (listado abaixo). Exploratório, 11 participantes.", "",
          "## Resumo (uma linha por teto × método)", "", "| Teto | " + cab[2:], "|---|---|" + "---:|" * 11]
    b = s["bruto_referencia"]
    md.append("| fixo | " + lin("bruto_referencia", b)[2:])
    for tk in CHAVES:
        for n in METODOS:
            md.append(f"| {rotulo(tk)} | " + lin(n, s["por_teto"][tk]["metricas"][n])[2:])
    md += ["", "Folds inviáveis (sem detecção) por teto:", ""]
    for tk in CHAVES:
        inv = s["por_teto"][tk]["folds_inviaveis"]
        md.append(f"- {rotulo(tk)}: " + ("nenhum" if not inv else ", ".join(f"{x['participante']} ({x['metodo']})" for x in inv)))
    for tk in CHAVES:
        d = s["por_teto"][tk]
        md += ["", f"## Teto: {rotulo(tk)}", "", cab, sep, lin("bruto_referencia (fixo)", b)]
        md += [lin(n, d["metricas"][n]) for n in METODOS]
        md += ["", "Primeira intensidade detectada por protocolo-alvo (% de 88):", "",
               "| Método | " + " | ".join(ep.ORDEM_CONDICOES + ["nenhuma"]) + " |", "|---|" + "---:|" * 7]
        for n, m in [("bruto_referencia", b)] + [(n, d["metricas"][n]) for n in METODOS]:
            p = m["primeira_intensidade"]
            md.append(f"| {n} | " + " | ".join(pl.pct(p[c], 88) for c in ep.ORDEM_CONDICOES + ["nenhuma"]) + " |")
        md += ["", "McNemar exato (só_A / só_B, p) e Wilcoxon de tempo nos pares detectados por ambos (n, mediana A/B, p).", "",
               "| A vs B | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |", "|---|---|---|---|---|---|---|"]
        for k, t in d["testes_pareados"].items():
            w = t["tempo"]
            pw = "-" if w["p_valor"] is None else f"{w['p_valor']:.3g}"
            md.append(f"| {k} | " + " | ".join(f"{t[q]['so_ref']}/{t[q]['so_metodo']}, p={t[q]['p_valor']:.3g}"
                      for q in ("alvos", "laterais", "esp", "alvos_proto", "laterais_proto"))
                      + f" | {w['n_pares']}, {pl.f_t(w['mediana_ref_s'])}/{pl.f_t(w['mediana_metodo_s'])}, p={pw} |")
        md += ["", "Por fold (retirado: det /40, FP lat /48, FP ESP /8):", "",
               "| Fold | Método | M | Parâmetros | Regra | Teto (seg) | Treino det / FP lat | Retirado det / FP lat / FP ESP |",
               "|---|---|---|---|---|---:|---|---|"]
        for x in d["folds"]:
            if not x["viavel"]:
                md.append(f"| {x['participante']} | {x['metodo']} | - | INVIÁVEL (sem detecção) | - | {x['teto_seg']} | - | - |")
                continue
            par = x["parametros"]
            ptxt = f"α={par['alfa']}" if "alfa" in par else f"π={par['pi'][1]:g}, A01={par['A'][0][1]:g}, A10={par['A'][1][0]:g}"
            md.append(f"| {x['participante']} | {x['metodo']} | {x['m']} | {ptxt} | {pl.txt_regra(x['regra'])} | {x['teto_seg']} | "
                      f"{x['c_treino']['det']} / {x['c_treino']['fp_lat']} | {x['h_det']} / {x['h_fp_lat']} / {x['h_fp_esp']} |")
    md += ["", "## Avisos do ajuste hierárquico (Presente, por fold e M; independem do teto)", "",
           "Folds (de 11) com dispersão relativa dos inícios > 1e-3, por M: "
           + ", ".join(f"M={m}: {v}" for m, v in s["folds_com_dispersao_acima_1e-3_por_m"].items()) + ".", "",
           "Folds com na_borda ou hiper_no_limite, por M: "
           + ", ".join(f"M={m}: {v}" for m, v in s["ajuste_hierarquico_na_borda_ou_limite"].items()) + ".", "",
           "| M | Fold | na_borda | hiper_no_limite | dispersão inícios |", "|---|---|---|---|---:|"]
    for a in s["ajuste_hierarquico"]:
        md.append(f"| {a['m']} | {a['fold']} | {a['na_borda']} | {a['hiper_no_limite']} | {a['dispersao_inicios']:.1e} |")
    md += ["", f"Tempo total: {total:.0f} s."]
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
