"""Experimento LOSO: calibrar a emissao Presente apenas em 40, 50 e 60 dB vs 30-70 dB.

Reutiliza as funcoes de explorar_calibracao_50_70.py para testar a calibracao
restrita a ("40dB", "50dB", "60dB"):
  1) Conjunto completo (30-70 dB + ESP):
     - HMM-30a70: Presente calibrado em 30-70 dB (baseline oficial)
     - HMM-cal40a60: Presente calibrado APENAS em 40, 50, 60 dB; busca e avaliacao em 30-70 dB
     - bruto_referencia: Rayleigh alfa=0.05, 3 consecutivas
  2) Subconjunto 40-60 dB + ESP (30 e 70 dB excluidos de calibracao, busca e avaliacao):
     - HMM-sub40a60
     - bruto-sub40a60 (otimizado LOSO)
     - bruto_referencia (sub 40-60 dB)

Saidas: results/rayleigh_hmm/exploratorio_calibracao_40_60/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_calibracao_40_60.py
"""

import json
import multiprocessing as mp
import os
import time

import numpy as np

import experimento as ex
import explorar_loso as el
import explorar_calibracao_50_70 as base

PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_calibracao_40_60")
INTENSIDADES_40_60 = ("40dB", "50dB", "60dB")
SEQ_TODAS = None


def tarefa(args):
    modo, fold = args
    if modo == "HMM-30a70":
        return base._rodar_hmm_fold(SEQ_TODAS, fold, modo, base.INTENSIDADES_TODAS)
    if modo == "HMM-cal40a60":
        return base._rodar_hmm_fold(SEQ_TODAS, fold, modo, INTENSIDADES_40_60)
    seq_sub = [s for s in SEQ_TODAS if s["condicao"] in ("ESP", *INTENSIDADES_40_60)]
    if modo == "HMM-sub40a60":
        return base._rodar_hmm_fold(seq_sub, fold, modo, INTENSIDADES_40_60)
    if modo == "bruto-sub40a60":
        return base._rodar_bruto_fold(seq_sub, fold, modo)
    raise ValueError(modo)


def main():
    global SEQ_TODAS
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    t0 = time.time()
    SEQ_TODAS, _ = ex.carregar_sequencias()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    n_todas = len(SEQ_TODAS)
    masc_todas = ex.mascaras(SEQ_TODAS)

    seq_sub = [s for s in SEQ_TODAS if s["condicao"] in ("ESP", *INTENSIDADES_40_60)]
    n_sub = len(seq_sub)
    masc_sub = ex.mascaras(seq_sub)

    modos = ["HMM-30a70", "HMM-cal40a60", "HMM-sub40a60", "bruto-sub40a60"]
    tarefas = [(m, f) for m in modos for f in range(len(participantes))]
    print(f"Rodando {len(tarefas)} tarefas LOSO em {base.N_PROCESSOS} processos...", flush=True)

    with mp.get_context("fork").Pool(base.N_PROCESSOS) as pool:
        saidas = []
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            if r["tipo_tarefa"] == "hmm":
                print(f"[{k}/{len(tarefas)}] {r['modo']} {r['participante']}: {r['segundos']:.0f}s "
                      f"(decorrido {time.time()-t0:.0f}s)", flush=True)

    por_modo = {m: [r for r in saidas if r["modo"] == m] for m in modos}

    c_todas = {
        "HMM-30a70": el.juntar(por_modo["HMM-30a70"], n_todas),
        "HMM-cal40a60": el.juntar(por_modo["HMM-cal40a60"], n_todas),
    }
    _, ref_todas = ex.buscar_bruto(SEQ_TODAS, masc_todas)
    c_todas["bruto_referencia"] = {"detectado": ref_todas["detectado"], "tempo_s": ref_todas["tempo_s"]}

    c_sub = {
        "HMM-sub40a60": el.juntar(por_modo["HMM-sub40a60"], n_sub),
        "bruto-sub40a60": el.juntar(por_modo["bruto-sub40a60"], n_sub),
    }
    _, ref_sub = ex.buscar_bruto(seq_sub, masc_sub)
    c_sub["bruto_referencia_sub"] = {"detectado": ref_sub["detectado"], "tempo_s": ref_sub["tempo_s"]}

    resumo_hiper = {}
    for m in ("HMM-30a70", "HMM-cal40a60", "HMM-sub40a60"):
        rs = por_modo[m]
        resumo_hiper[m] = {
            k: float(np.mean([r["hiper"][k] for r in rs]))
            for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media")
        }
        resumo_hiper[m]["na_borda_folds"] = sum(1 for r in rs if r["hiper"]["na_borda"])
        resumo_hiper[m]["n_seq_ajuste"] = int(rs[0]["n_sequencias_ajuste"])

    razoes_30a70 = base.razoes_verossimilhanca_exemplo(SEQ_TODAS, base.INTENSIDADES_TODAS)
    razoes_40a60 = base.razoes_verossimilhanca_exemplo(SEQ_TODAS, INTENSIDADES_40_60)

    par_todas = {
        "HMM-cal40a60 vs HMM-30a70": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal40a60"], c_todas["HMM-30a70"]),
        "HMM-cal40a60 vs bruto_referencia": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal40a60"], c_todas["bruto_referencia"]),
        "HMM-30a70 vs bruto_referencia": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-30a70"], c_todas["bruto_referencia"]),
    }
    par_sub = {
        "HMM-sub40a60 vs bruto_referencia_sub": base._pareado(seq_sub, masc_sub, c_sub["HMM-sub40a60"], c_sub["bruto_referencia_sub"]),
        "HMM-sub40a60 vs bruto-sub40a60": base._pareado(seq_sub, masc_sub, c_sub["HMM-sub40a60"], c_sub["bruto-sub40a60"]),
    }

    met_todas = ["HMM-30a70", "HMM-cal40a60", "bruto_referencia"]
    linhas_inten = [
        "| Intensidade | " + " | ".join(f"{m} det / FP lat / tempo" for m in met_todas) + " |",
        "|---|" + "---|" * len(met_todas),
    ]
    dados_inten = {}
    for cond in base.INTENSIDADES_TODAS:
        cel = []
        for m in met_todas:
            r = ex.resumo(SEQ_TODAS, c_todas[m], lambda s, cond=cond: s["condicao"] == cond)
            dados_inten[f"{cond}/{m}"] = r
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} ({100*r['deteccao']:.1f}%) / "
                       f"{r['fp_lateral_detectados']}/{r['fp_lateral_total']} / {el.fmt_t(r['tempo_medio_s'])}")
        linhas_inten.append(f"| {cond} | " + " | ".join(cel) + " |")

    cel_4060 = []
    for m in met_todas:
        r = ex.resumo(SEQ_TODAS, c_todas[m], lambda s: s["condicao"] in INTENSIDADES_40_60)
        dados_inten[f"40-60dB/{m}"] = r
        cel_4060.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} ({100*r['deteccao']:.1f}%) / "
                        f"{r['fp_lateral_detectados']}/{r['fp_lateral_total']} / {el.fmt_t(r['tempo_medio_s'])}")
    linhas_inten.append(f"| **40-60dB (subtotal)** | " + " | ".join(cel_4060) + " |")

    tempo_total = time.time() - t0
    md = [
        "# Experimento LOSO: calibracao Presente em 40-60 dB vs 30-70 dB (sinal FC)\n",
        "## 1. Conjunto completo (30-70 dB + ESP, 440 alvos / 528 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |",
        "|---|---|---|---|---|",
        base._linha_geral("HMM-30a70 (calibracao 30-70 dB, baseline)", SEQ_TODAS, masc_todas, c_todas["HMM-30a70"]),
        base._linha_geral("HMM-cal40a60 (calibracao so 40-60 dB)", SEQ_TODAS, masc_todas, c_todas["HMM-cal40a60"]),
        base._linha_geral("bruto_referencia (Rayleigh a=0.05, 3 consec.)", SEQ_TODAS, masc_todas, c_todas["bruto_referencia"]),
        "\n### Testes pareados (conjunto completo)\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |",
        "|---|---|---|---|---|",
        *[base._fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in par_todas.items()],
        "\n### Por intensidade (conjunto completo)\n",
        *linhas_inten,
        "\n## 2. Subconjunto restrito a 40-60 dB + ESP (30 e 70 dB excluidos de tudo: 264 alvos / 352 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |",
        "|---|---|---|---|---|",
        base._linha_geral("HMM-sub40a60", seq_sub, masc_sub, c_sub["HMM-sub40a60"]),
        base._linha_geral("bruto-sub40a60 (otimizado LOSO)", seq_sub, masc_sub, c_sub["bruto-sub40a60"]),
        base._linha_geral("bruto_referencia (sub 40-60 dB)", seq_sub, masc_sub, c_sub["bruto_referencia_sub"]),
        "\n### Testes pareados (subconjunto 40-60 dB)\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |",
        "|---|---|---|---|---|",
        *[base._fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in par_sub.items()],
        "\n## 3. Efeito na densidade Presente estimada (media dos 11 folds)\n",
        "| Calibracao | Seq./fold | a_k | b_k | a_rho | b_rho | k medio | rho medio | borda (folds) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {m} | {v['n_seq_ajuste']} | {v['a_k']:.3f} | {v['b_k']:.3f} | {v['a_rho']:.3f} | {v['b_rho']:.2f} | "
          f"{v['k_media']:.2f} | {v['rho_media']:.4f} | {v['na_borda_folds']}/11 |"
          for m, v in resumo_hiper.items()],
        "\n### Razao de verossimilhanca P(Z|N, Presente) / P(Z|N, Ausente) (11 participantes, ilustrativo)\n",
        "| (N, Z) | Calibracao 30-70 dB | Calibracao 40-60 dB |",
        "|---|---:|---:|",
        *[f"| N={a['N']}, Z={a['Z']} | {a['razao_P_sobre_A']:.2f}x | {b['razao_P_sobre_A']:.2f}x |"
          for a, b in zip(razoes_30a70["exemplos"], razoes_40a60["exemplos"])],
        f"\nTempo total de execucao: {tempo_total:.0f} s.",
    ]

    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    saida = {
        "pooled_todas": {m: ex.metricas(masc_todas, c["detectado"], c["tempo_s"]) for m, c in c_todas.items()},
        "pooled_sub40a60": {m: ex.metricas(masc_sub, c["detectado"], c["tempo_s"]) for m, c in c_sub.items()},
        "pareados_todas": par_todas,
        "pareados_sub40a60": par_sub,
        "por_intensidade_todas": dados_inten,
        "resumo_hiper_folds": resumo_hiper,
        "razoes_exemplo": {"30a70": razoes_30a70, "40a60": razoes_40a60},
        "folds": {m: [el.sem_arrays(r) for r in sorted(por_modo[m], key=lambda x: x["fold"])] for m in modos},
        "tempo_execucao_s": tempo_total,
    }
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
