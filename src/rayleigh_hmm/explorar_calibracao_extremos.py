"""Experimento LOSO: calibrar Presente apenas em 60-70 dB e apenas em 50 dB vs 30-70 dB.

Reutiliza as funcoes de explorar_calibracao_50_70.py, mantendo busca e avaliacao
no conjunto completo (30-70 dB + ESP) para comparacao direta com o baseline oficial:
  - HMM-30a70: Presente calibrado em 30-70 dB (baseline)
  - HMM-cal60a70: Presente calibrado apenas em 60 e 70 dB (maiores SNRs)
  - HMM-cal50: Presente calibrado apenas em 50 dB (maior taxa de deteccao)
  - bruto_referencia: Rayleigh alfa=0.05, 3 consecutivas

Saidas: results/rayleigh_hmm/exploratorio_calibracao_extremos/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_calibracao_extremos.py
"""

import json
import multiprocessing as mp
import os
import time

import numpy as np

import experimento as ex
import explorar_loso as el
import explorar_calibracao_50_70 as base

PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_calibracao_extremos")
CONDICOES_POR_MODO = {
    "HMM-30a70": base.INTENSIDADES_TODAS,
    "HMM-cal60a70": ("60dB", "70dB"),
    "HMM-cal50": ("50dB",),
}
SEQ_TODAS = None


def tarefa(args):
    modo, fold = args
    return base._rodar_hmm_fold(SEQ_TODAS, fold, modo, CONDICOES_POR_MODO[modo])


def main():
    global SEQ_TODAS
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    t0 = time.time()
    SEQ_TODAS, _ = ex.carregar_sequencias()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    n_todas = len(SEQ_TODAS)
    masc_todas = ex.mascaras(SEQ_TODAS)

    modos = list(CONDICOES_POR_MODO.keys())
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

    resumo_hiper = {}
    for m in modos:
        rs = por_modo[m]
        resumo_hiper[m] = {
            k: float(np.mean([r["hiper"][k] for r in rs]))
            for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media")
        }
        resumo_hiper[m]["na_borda_folds"] = sum(1 for r in rs if r["hiper"]["na_borda"])
        resumo_hiper[m]["n_seq_ajuste"] = int(rs[0]["n_sequencias_ajuste"])

    razoes = {m: base.razoes_verossimilhanca_exemplo(SEQ_TODAS, conds) for m, conds in CONDICOES_POR_MODO.items()}

    par_todas = {
        "HMM-cal60a70 vs HMM-30a70": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal60a70"], c_todas["HMM-30a70"]),
        "HMM-cal50 vs HMM-30a70": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal50"], c_todas["HMM-30a70"]),
        "HMM-cal60a70 vs bruto_referencia": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal60a70"], c_todas["bruto_referencia"]),
        "HMM-cal50 vs bruto_referencia": base._pareado(SEQ_TODAS, masc_todas, c_todas["HMM-cal50"], c_todas["bruto_referencia"]),
    }

    met_todas = [*modos, "bruto_referencia"]
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

    tempo_total = time.time() - t0
    md = [
        "# Experimento LOSO: calibracao Presente em 60-70 dB e so 50 dB vs 30-70 dB (sinal FC)\n",
        "## 1. Conjunto completo (30-70 dB + ESP, 440 alvos / 528 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |",
        "|---|---|---|---|---|",
        base._linha_geral("HMM-30a70 (calibracao 30-70 dB, baseline)", SEQ_TODAS, masc_todas, c_todas["HMM-30a70"]),
        base._linha_geral("HMM-cal60a70 (calibracao so 60-70 dB)", SEQ_TODAS, masc_todas, c_todas["HMM-cal60a70"]),
        base._linha_geral("HMM-cal50 (calibracao so 50 dB)", SEQ_TODAS, masc_todas, c_todas["HMM-cal50"]),
        base._linha_geral("bruto_referencia (Rayleigh a=0.05, 3 consec.)", SEQ_TODAS, masc_todas, c_todas["bruto_referencia"]),
        "\n### Testes pareados\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |",
        "|---|---|---|---|---|",
        *[base._fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in par_todas.items()],
        "\n### Por intensidade\n",
        *linhas_inten,
        "\n## 2. Efeito na densidade Presente estimada (media dos 11 folds)\n",
        "| Calibracao | Seq./fold | a_k | b_k | a_rho | b_rho | k medio | rho medio | borda (folds) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {m} | {v['n_seq_ajuste']} | {v['a_k']:.3f} | {v['b_k']:.3f} | {v['a_rho']:.3f} | {v['b_rho']:.2f} | "
          f"{v['k_media']:.2f} | {v['rho_media']:.4f} | {v['na_borda_folds']}/11 |"
          for m, v in resumo_hiper.items()],
        "\n### Razao de verossimilhanca P(Z|N, Presente) / P(Z|N, Ausente) (11 participantes, ilustrativo)\n",
        "| (N, Z) | 30-70 dB | 60-70 dB | so 50 dB |",
        "|---|---:|---:|---:|",
        *[f"| N={a['N']}, Z={a['Z']} | {a['razao_P_sobre_A']:.2f}x | {b['razao_P_sobre_A']:.2f}x | {c['razao_P_sobre_A']:.2f}x |"
          for a, b, c in zip(razoes["HMM-30a70"]["exemplos"], razoes["HMM-cal60a70"]["exemplos"], razoes["HMM-cal50"]["exemplos"])],
        f"\nTempo total de execucao: {tempo_total:.0f} s.",
    ]

    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    saida = {
        "pooled": {m: ex.metricas(masc_todas, c["detectado"], c["tempo_s"]) for m, c in c_todas.items()},
        "pareados": par_todas,
        "por_intensidade": dados_inten,
        "resumo_hiper_folds": resumo_hiper,
        "razoes_exemplo": razoes,
        "folds": {m: [el.sem_arrays(r) for r in sorted(por_modo[m], key=lambda x: x["fold"])] for m in modos},
        "tempo_execucao_s": tempo_total,
    }
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
