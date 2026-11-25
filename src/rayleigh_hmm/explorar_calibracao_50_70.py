"""Experimento LOSO: calibrar a emissao Presente apenas em 50, 60 e 70 dB vs 30-70 dB.

Testa diretamente a hipotese da conclusao de docs/relatorios/trabalho_final/main_pt.tex:
"Contar todo registro estimulado como resposta inclui exames nos quais a resposta e
fraca demais para ser vista, e estes puxam a densidade Presente em direcao a Ausente."

Compara em LOSO (sinal FC, teto = FP lateral do bruto_referencia no treino, igual a
explorar_loso_fc.py):
  1) Conjunto completo (30-70 dB + ESP):
     - HMM-REAL-30a70: Presente calibrado em 30, 40, 50, 60, 70 dB (baseline oficial)
     - HMM-REAL-cal50a70: Presente calibrado APENAS em 50, 60, 70 dB; busca e avaliacao em 30-70 dB
     - bruto_referencia: Rayleigh alfa=0.05, 3 consecutivas
  2) Subconjunto 50-70 dB + ESP (30 e 40 dB excluidos de calibracao, busca e avaliacao):
     - HMM-REAL-sub50a70
     - bruto_otimizado-sub50a70
     - bruto_referencia-sub50a70

Saidas: results/rayleigh_hmm/exploratorio_calibracao_50_70/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_calibracao_50_70.py
"""

import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import emissao
import experimento as ex
import explorar_loso as el
import hmm
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI
from explorar_loso_fc import teto_referencia

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_calibracao_50_70")
N_PROCESSOS = 16
INTENSIDADES_TODAS = ("30dB", "40dB", "50dB", "60dB", "70dB")
INTENSIDADES_ALTAS = ("50dB", "60dB", "70dB")

SEQ_TODAS = None


def ajustar_emissao_real(seq_treino, condicoes_presente):
    """Ajusta Ausente no ESP de treino e Presente marginal nas condicoes especificadas."""
    z_esp = np.concatenate([s["z"] for s in seq_treino if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    emi = {"ausente": emissao.ajustar_mle(z_esp)}
    alvos = [s for s in seq_treino if s["tipo"] == "alvo" and s["condicao"] in condicoes_presente]
    emi["z_sintetico"] = np.concatenate([s["z"] for s in alvos])
    emi["n_sintetico"] = np.concatenate([s["n_ep"] for s in alvos])
    emi["id_sintetico"] = np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])
    emi["marginal"] = ex.ajustar_marginal(emi)
    return emi


def _rodar_hmm_fold(seq_base, fold, rotulo, condicoes_presente):
    t0 = time.time()
    participantes = sorted({s["participante"] for s in seq_base})
    h = participantes[fold]
    i_tr = [i for i, s in enumerate(seq_base) if s["participante"] != h]
    i_h = [i for i, s in enumerate(seq_base) if s["participante"] == h]
    seq_tr, seq_h = [seq_base[i] for i in i_tr], [seq_base[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    teto = teto_referencia(seq_tr, masc_tr)
    emi = ajustar_emissao_real(seq_tr, condicoes_presente)

    grupos_tr = el.grupos_log_em(seq_tr, emi)
    melhores = {}
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
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
    masc_h = ex.mascaras(seq_h)
    hip = emi["marginal"]["hiper"]
    return {
        "tipo_tarefa": "hmm",
        "modo": rotulo,
        "fold": fold,
        "participante": h,
        "indices_h": i_h,
        "detectado_h": det_h,
        "tempo_h": tempo_h,
        "fp_lateral_h": float(det_h[masc_h["lateral"]].mean()),
        "teto_treino": teto,
        "escolha": escolha,
        "viavel": melhor is not None,
        "ausente": emi["ausente"],
        "hiper": {k: hip[k] for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media",
                                      "na_borda", "hiper_no_limite", "massa_fora_grade")},
        "n_sequencias_ajuste": hip["n_sequencias"],
        "n_pares_ajuste": int(len(emi["z_sintetico"])),
        "segundos": time.time() - t0,
    }


def _rodar_bruto_fold(seq_base, fold, rotulo):
    participantes = sorted({s["participante"] for s in seq_base})
    h = participantes[fold]
    i_tr = [i for i, s in enumerate(seq_base) if s["participante"] != h]
    i_h = [i for i, s in enumerate(seq_base) if s["participante"] == h]
    seq_tr, seq_h = [seq_base[i] for i in i_tr], [seq_base[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    teto = teto_referencia(seq_tr, masc_tr)
    melhores, _ = ex.buscar_bruto(seq_tr, masc_tr, teto)
    melhor = ex.melhor_geral(melhores)
    grupos = [(idx, dur, np.stack([seq_h[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_h)]
    gd = [(idx, *ex.indicadores(np.exp(-z) <= melhor["alfa"]), dur) for idx, dur, z in grupos]
    det, tempo = ex.decidir(gd, len(seq_h), melhor["regra"])
    masc_h = ex.mascaras(seq_h)
    return {
        "tipo_tarefa": "bruto",
        "modo": rotulo,
        "fold": fold,
        "participante": h,
        "indices_h": i_h,
        "detectado_h": det,
        "tempo_h": tempo,
        "alfa": melhor["alfa"],
        "regra": melhor["regra"],
        "metricas_treino": melhor["metricas"],
        "fp_lateral_h": float(det[masc_h["lateral"]].mean()),
        "teto_treino": teto,
    }


def tarefa(args):
    modo, fold = args
    if modo == "HMM-30a70":
        return _rodar_hmm_fold(SEQ_TODAS, fold, modo, INTENSIDADES_TODAS)
    if modo == "HMM-cal50a70":
        return _rodar_hmm_fold(SEQ_TODAS, fold, modo, INTENSIDADES_ALTAS)
    seq_sub = [s for s in SEQ_TODAS if s["condicao"] in ("ESP", *INTENSIDADES_ALTAS)]
    if modo == "HMM-sub50a70":
        return _rodar_hmm_fold(seq_sub, fold, modo, INTENSIDADES_ALTAS)
    if modo == "bruto-sub50a70":
        return _rodar_bruto_fold(seq_sub, fold, modo)
    raise ValueError(modo)


def _linha_geral(nome, seqs, masc, c):
    m = ex.metricas(masc, c["detectado"], c["tempo_s"])
    n_det = int(c["detectado"][masc["alvo_estimulo"]].sum())
    tot_det = int(masc["alvo_estimulo"].sum())
    n_lat = int(c["detectado"][masc["lateral"]].sum())
    tot_lat = int(masc["lateral"].sum())
    n_esp = int(c["detectado"][masc["alvo_esp"]].sum())
    tot_esp = int(masc["alvo_esp"].sum())
    return (
        f"| {nome} | {n_det}/{tot_det} ({100*m['deteccao']:.1f}%) | "
        f"{n_lat}/{tot_lat} ({100*m['fp_lateral']:.1f}%) | "
        f"{n_esp}/{tot_esp} ({100*m['fp_alvo_esp']:.1f}%) | "
        f"{el.fmt_t(m['tempo_medio_s'])} / {el.fmt_t(m['tempo_mediano_s'])} |"
    )


def _pareado(seqs, masc, c_a, c_b):
    sig = ex.significancia(seqs, masc, c_a, c_b)
    ambos = masc["alvo_estimulo"] & c_a["detectado"] & c_b["detectado"]
    d = c_a["tempo_s"][ambos] - c_b["tempo_s"][ambos]
    w = {
        "n_pares": int(ambos.sum()),
        "a_mais_rapido": int((d < 0).sum()),
        "a_mais_lento": int((d > 0).sum()),
        "iguais": int((d == 0).sum()),
        "dif_media_s": float(d.mean()) if ambos.sum() else 0.0,
        "p_valor": sig["tempo"]["p_valor"],
    }
    return {"mcnemar": {k: sig[k] for k in ("alvos", "laterais", "esp")}, "tempo": w}


def _fmt_pareado(nome_a, nome_b, p):
    mc = [f"{p['mcnemar'][k]['so_bruto']}-{p['mcnemar'][k]['so_hmm']}, p={p['mcnemar'][k]['p_valor']:.3g}"
          for k in ("alvos", "laterais", "esp")]
    w = p["tempo"]
    pw = "-" if w["p_valor"] is None else f"{w['p_valor']:.3g}"
    return (f"| {nome_a} vs {nome_b} | {mc[0]} | {mc[1]} | {mc[2]} | "
            f"n={w['n_pares']}, {w['a_mais_rapido']}/{w['a_mais_lento']}/{w['iguais']}, "
            f"{w['dif_media_s']:+.1f}, p={pw} |")


def razoes_verossimilhanca_exemplo(seq_todas, condicoes_presente):
    """Ajusta em todos os 11 participantes so para inspecionar a razao de densidades."""
    emi = ajustar_emissao_real(seq_todas, condicoes_presente)
    aus = emi["ausente"]
    pts = emi["marginal"]["pontos"]
    exemplos = []
    for n_ep, z_val in [(120, 0.5), (120, 1.0), (120, 3.0), (120, 6.0), (480, 0.5)]:
        lp = float(emissao.log_pdf_preditiva(np.array([z_val]), np.array([n_ep]), pts)[0])
        la = float(emissao.log_pdf_gamma(np.array([z_val]), aus["k"], aus["mu"])[0])
        exemplos.append({"N": n_ep, "Z": z_val, "razao_P_sobre_A": float(np.exp(lp - la))})
    hip = emi["marginal"]["hiper"]
    return {
        "hiper": {k: hip[k] for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media", "na_borda")},
        "exemplos": exemplos,
    }


def main():
    global SEQ_TODAS
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    t0 = time.time()
    SEQ_TODAS, _ = ex.carregar_sequencias()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    n_todas = len(SEQ_TODAS)
    masc_todas = ex.mascaras(SEQ_TODAS)

    seq_sub = [s for s in SEQ_TODAS if s["condicao"] in ("ESP", *INTENSIDADES_ALTAS)]
    n_sub = len(seq_sub)
    masc_sub = ex.mascaras(seq_sub)

    modos = ["HMM-30a70", "HMM-cal50a70", "HMM-sub50a70", "bruto-sub50a70"]
    tarefas = [(m, f) for m in modos for f in range(len(participantes))]
    print(f"Rodando {len(tarefas)} tarefas LOSO em {N_PROCESSOS} processos...", flush=True)

    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        saidas = []
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            if r["tipo_tarefa"] == "hmm":
                print(f"[{k}/{len(tarefas)}] {r['modo']} {r['participante']}: {r['segundos']:.0f}s "
                      f"(decorrido {time.time()-t0:.0f}s)", flush=True)

    por_modo = {m: [r for r in saidas if r["modo"] == m] for m in modos}

    c_todas = {
        "HMM-30a70": el.juntar(por_modo["HMM-30a70"], n_todas),
        "HMM-cal50a70": el.juntar(por_modo["HMM-cal50a70"], n_todas),
    }
    _, ref_todas = ex.buscar_bruto(SEQ_TODAS, masc_todas)
    c_todas["bruto_referencia"] = {"detectado": ref_todas["detectado"], "tempo_s": ref_todas["tempo_s"]}

    c_sub = {
        "HMM-sub50a70": el.juntar(por_modo["HMM-sub50a70"], n_sub),
        "bruto-sub50a70": el.juntar(por_modo["bruto-sub50a70"], n_sub),
    }
    _, ref_sub = ex.buscar_bruto(seq_sub, masc_sub)
    c_sub["bruto_referencia_sub"] = {"detectado": ref_sub["detectado"], "tempo_s": ref_sub["tempo_s"]}

    # Comparacao de hiperparametros medios entre folds
    resumo_hiper = {}
    for m in ("HMM-30a70", "HMM-cal50a70", "HMM-sub50a70"):
        rs = por_modo[m]
        resumo_hiper[m] = {
            k: float(np.mean([r["hiper"][k] for r in rs]))
            for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media")
        }
        resumo_hiper[m]["na_borda_folds"] = sum(1 for r in rs if r["hiper"]["na_borda"])
        resumo_hiper[m]["n_seq_ajuste"] = int(rs[0]["n_sequencias_ajuste"])

    razoes_30a70 = razoes_verossimilhanca_exemplo(SEQ_TODAS, INTENSIDADES_TODAS)
    razoes_50a70 = razoes_verossimilhanca_exemplo(SEQ_TODAS, INTENSIDADES_ALTAS)

    # Pareados no conjunto completo
    par_todas = {
        "HMM-cal50a70 vs HMM-30a70": _pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal50a70"], c_todas["HMM-30a70"]),
        "HMM-cal50a70 vs bruto_referencia": _pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal50a70"], c_todas["bruto_referencia"]),
        "HMM-30a70 vs bruto_referencia": _pareado(SEQ_TODAS, masc_todas, c_todas["HMM-30a70"], c_todas["bruto_referencia"]),
    }
    par_sub = {
        "HMM-sub50a70 vs bruto_referencia_sub": _pareado(seq_sub, masc_sub, c_sub["HMM-sub50a70"], c_sub["bruto_referencia_sub"]),
        "HMM-sub50a70 vs bruto-sub50a70": _pareado(seq_sub, masc_sub, c_sub["HMM-sub50a70"], c_sub["bruto-sub50a70"]),
    }

    # Por intensidade no conjunto completo
    met_todas = ["HMM-30a70", "HMM-cal50a70", "bruto_referencia"]
    linhas_inten = [
        "| Intensidade | " + " | ".join(f"{m} det / FP lat / tempo" for m in met_todas) + " |",
        "|---|" + "---|" * len(met_todas),
    ]
    dados_inten = {}
    for cond in INTENSIDADES_TODAS:
        cel = []
        for m in met_todas:
            r = ex.resumo(SEQ_TODAS, c_todas[m], lambda s, cond=cond: s["condicao"] == cond)
            dados_inten[f"{cond}/{m}"] = r
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} ({100*r['deteccao']:.1f}%) / "
                       f"{r['fp_lateral_detectados']}/{r['fp_lateral_total']} / {el.fmt_t(r['tempo_medio_s'])}")
        linhas_inten.append(f"| {cond} | " + " | ".join(cel) + " |")

    # Subtotal 50-70 dB no conjunto completo
    cel_5070 = []
    for m in met_todas:
        r = ex.resumo(SEQ_TODAS, c_todas[m], lambda s: s["condicao"] in INTENSIDADES_ALTAS)
        dados_inten[f"50-70dB/{m}"] = r
        cel_5070.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} ({100*r['deteccao']:.1f}%) / "
                        f"{r['fp_lateral_detectados']}/{r['fp_lateral_total']} / {el.fmt_t(r['tempo_medio_s'])}")
    linhas_inten.append(f"| **50-70dB (subtotal)** | " + " | ".join(cel_5070) + " |")

    tempo_total = time.time() - t0
    md = [
        "# Experimento LOSO: calibracao Presente em 50-70 dB vs 30-70 dB (sinal FC)\n",
        "Testa a hipotese da conclusao de `docs/relatorios/trabalho_final/main_pt.tex` de que incluir 30-40 dB "
        "na calibracao do estado Presente puxa a densidade Presente em direcao a Ausente.\n",
        "## 1. Conjunto completo (30-70 dB + ESP, 440 alvos / 528 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |",
        "|---|---|---|---|---|",
        _linha_geral("HMM-30a70 (calibracao 30-70 dB, baseline)", SEQ_TODAS, masc_todas, c_todas["HMM-30a70"]),
        _linha_geral("HMM-cal50a70 (calibracao so 50-70 dB)", SEQ_TODAS, masc_todas, c_todas["HMM-cal50a70"]),
        _linha_geral("bruto_referencia (Rayleigh a=0.05, 3 consec.)", SEQ_TODAS, masc_todas, c_todas["bruto_referencia"]),
        "\n### Testes pareados (conjunto completo)\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |",
        "|---|---|---|---|---|",
        *[_fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in par_todas.items()],
        "\n### Por intensidade (conjunto completo)\n",
        *linhas_inten,
        "\n## 2. Subconjunto restrito a 50-70 dB + ESP (30 e 40 dB excluidos de tudo: 264 alvos / 352 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |",
        "|---|---|---|---|---|",
        _linha_geral("HMM-sub50a70", seq_sub, masc_sub, c_sub["HMM-sub50a70"]),
        _linha_geral("bruto-sub50a70 (otimizado LOSO)", seq_sub, masc_sub, c_sub["bruto-sub50a70"]),
        _linha_geral("bruto_referencia (sub 50-70 dB)", seq_sub, masc_sub, c_sub["bruto_referencia_sub"]),
        "\n### Testes pareados (subconjunto 50-70 dB)\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |",
        "|---|---|---|---|---|",
        *[_fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in par_sub.items()],
        "\n## 3. Efeito na densidade Presente estimada (media dos 11 folds)\n",
        "| Calibracao | Seq./fold | a_k | b_k | a_rho | b_rho | k medio | rho medio | borda (folds) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {m} | {v['n_seq_ajuste']} | {v['a_k']:.3f} | {v['b_k']:.3f} | {v['a_rho']:.3f} | {v['b_rho']:.2f} | "
          f"{v['k_media']:.2f} | {v['rho_media']:.4f} | {v['na_borda_folds']}/11 |"
          for m, v in resumo_hiper.items()],
        "\n### Razao de verossimilhanca P(Z|N, Presente) / P(Z|N, Ausente) (11 participantes, ilustrativo)\n",
        "| (N, Z) | Calibracao 30-70 dB | Calibracao 50-70 dB |",
        "|---|---:|---:|",
        *[f"| N={a['N']}, Z={a['Z']} | {a['razao_P_sobre_A']:.2f}x | {b['razao_P_sobre_A']:.2f}x |"
          for a, b in zip(razoes_30a70["exemplos"], razoes_50a70["exemplos"])],
        f"\nTempo total de execucao: {tempo_total:.0f} s.",
    ]

    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    saida = {
        "pooled_todas": {m: ex.metricas(masc_todas, c["detectado"], c["tempo_s"]) for m, c in c_todas.items()},
        "pooled_sub50a70": {m: ex.metricas(masc_sub, c["detectado"], c["tempo_s"]) for m, c in c_sub.items()},
        "pareados_todas": par_todas,
        "pareados_sub50a70": par_sub,
        "por_intensidade_todas": dados_inten,
        "resumo_hiper_folds": resumo_hiper,
        "razoes_exemplo": {"30a70": razoes_30a70, "50a70": razoes_50a70},
        "folds": {m: [el.sem_arrays(r) for r in sorted(por_modo[m], key=lambda x: x["fold"])] for m in modos},
        "tempo_execucao_s": tempo_total,
    }
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
