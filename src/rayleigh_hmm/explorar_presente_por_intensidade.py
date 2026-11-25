"""Exploratorio (Parte 1): emissao Presente POR INTENSIDADE, registros isolados, LOSO, sinal FC.

Em vez de um unico modelo hierarquico ajustado com os alvos estimulados de 30-70 dB, ajusta 5 modelos
hierarquicos por fold (um por intensidade), cada um so com os alvos daquela intensidade dos participantes
de treino. Cada sequencia usa a densidade Presente preditiva da intensidade da sua gravacao
(laterais de X dB usam a curva de X dB). O ESP nao tem intensidade: usa a curva de 30 dB (regra principal);
como sensibilidade, o FP no alvo ESP tambem e medido com a curva conjunta 30-70 dB.
A emissao Ausente nao muda (Gamma MLE nos alvos ESP do treino). E uma curva por intensidade da populacao
de treino, nao por paciente.

Bracos: HMM-30a70 (baseline oficial), HMM-por-intensidade, bruto_referencia. Mesma grade de pi, A, regras e
mesmo teto de FP lateral (FP do bruto_referencia no treino); um unico (pi, A, regra) por fold, compartilhado
pelas intensidades. Reaproveita explorar_calibracao_50_70.py; nao altera arquivos existentes.

Saidas: results/rayleigh_hmm/exploratorio_presente_por_intensidade/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_presente_por_intensidade.py
"""

import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import emissao
import experimento as ex
import explorar_calibracao_50_70 as base
import explorar_loso as el
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI
from explorar_loso_fc import teto_referencia

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_presente_por_intensidade")
N_PROCESSOS = 16
INTENSIDADES = base.INTENSIDADES_TODAS
CURVA_ESP = "30dB"  # o ESP e tratado como o nivel mais baixo
CAMPOS_HIPER = ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media", "na_borda", "hiper_no_limite",
                "massa_fora_grade", "dispersao_relativa_inicios")
EXEMPLOS_Z = (0.5, 1.0, 3.0, 6.0)

SEQ_TODAS = None


def curva_da_sequencia(s):
    return CURVA_ESP if s["condicao"] == "ESP" else s["condicao"]


def grupos_log_em_por_intensidade(seqs, emis):
    """Como el.grupos_log_em, mas cada sequencia usa a emissao da sua curva."""
    return [(idx, dur, np.stack([ex.log_emissoes(seqs[i], emis[curva_da_sequencia(seqs[i])], "marginal") for i in idx]))
            for idx, dur in ex.indices_por_comprimento(seqs)]


def rodar_fold_por_intensidade(fold):
    t0 = time.time()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    h = participantes[fold]
    i_tr = [i for i, s in enumerate(SEQ_TODAS) if s["participante"] != h]
    i_h = [i for i, s in enumerate(SEQ_TODAS) if s["participante"] == h]
    seq_tr, seq_h = [SEQ_TODAS[i] for i in i_tr], [SEQ_TODAS[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    teto = teto_referencia(seq_tr, masc_tr)

    emis = {c: base.ajustar_emissao_real(seq_tr, (c,)) for c in INTENSIDADES}  # 5 curvas do fold
    emi_conj = base.ajustar_emissao_real(seq_tr, INTENSIDADES)  # so para a sensibilidade do ESP

    grupos_tr = grupos_log_em_por_intensidade(seq_tr, emis)
    melhores = {}
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        ex.avaliar_regras(seq_tr, masc_tr, el.detectores_hmm(grupos_tr, pi, A),
                          {"pi": pi.tolist(), "A": A.tolist()}, melhores, teto)
    melhor = ex.melhor_geral(melhores)

    masc_h = ex.mascaras(seq_h)
    det_esp_conj = np.zeros(len(seq_h), bool)
    if melhor is None:
        det_h, tempo_h = np.zeros(len(seq_h), bool), np.full(len(seq_h), np.nan)
        escolha = None
    else:
        pi, A = np.array(melhor["pi"]), np.array(melhor["A"])
        gd_h = el.detectores_hmm(grupos_log_em_por_intensidade(seq_h, emis), pi, A)
        det_h, tempo_h = ex.decidir(gd_h, len(seq_h), melhor["regra"])
        escolha = {"pi": melhor["pi"], "A": melhor["A"], "regra": melhor["regra"],
                   "metricas_treino": melhor["metricas"]}
        # sensibilidade: alvos ESP de h com a curva conjunta (mesmos pi, A, regra)
        i_esp = np.flatnonzero(masc_h["alvo_esp"])
        seq_esp = [seq_h[i] for i in i_esp]
        gd_esp = el.detectores_hmm(el.grupos_log_em(seq_esp, emi_conj), pi, A)
        d_esp, _ = ex.decidir(gd_esp, len(seq_esp), melhor["regra"])
        det_esp_conj[i_esp] = d_esp
    return {
        "fold": fold, "participante": h, "indices_h": i_h,
        "detectado_h": det_h, "tempo_h": tempo_h, "det_esp_conj_h": det_esp_conj,
        "fp_lateral_h": float(det_h[masc_h["lateral"]].mean()),
        "teto_treino": teto, "escolha": escolha, "viavel": melhor is not None,
        "hiper": {c: {k: emis[c]["marginal"]["hiper"][k] for k in CAMPOS_HIPER}
                     | {"n_sequencias": emis[c]["marginal"]["hiper"]["n_sequencias"]} for c in INTENSIDADES},
        "segundos": time.time() - t0,
    }


def tarefa(args):
    modo, fold = args
    if modo == "HMM-30a70":
        return base._rodar_hmm_fold(SEQ_TODAS, fold, modo, INTENSIDADES) | {"tipo_tarefa": "hmm", "modo": modo}
    return rodar_fold_por_intensidade(fold) | {"tipo_tarefa": "hmm", "modo": modo}


def razoes_por_intensidade(seq_todas):
    """Ajusta cada curva nos 11 participantes e mostra P/A em alguns (N, Z); N dentro do que a intensidade tem."""
    saida = {}
    for c in INTENSIDADES:
        emi = base.ajustar_emissao_real(seq_todas, (c,))
        aus, pts, hip = emi["ausente"], emi["marginal"]["pontos"], emi["marginal"]["hiper"]
        n_max = int(emi["n_sintetico"].max())
        exemplos = []
        for n_ep in sorted({60, n_max}):
            for z_val in EXEMPLOS_Z:
                lp = float(emissao.log_pdf_preditiva(np.array([z_val]), np.array([n_ep]), pts)[0])
                la = float(emissao.log_pdf_gamma(np.array([z_val]), aus["k"], aus["mu"])[0])
                exemplos.append({"N": n_ep, "Z": z_val, "razao_P_sobre_A": float(np.exp(lp - la))})
        saida[c] = {"hiper": {k: hip[k] for k in CAMPOS_HIPER}, "n_max": n_max, "exemplos": exemplos}
    return saida


def main():
    global SEQ_TODAS
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    t0 = time.time()
    SEQ_TODAS, _ = ex.carregar_sequencias()
    participantes = sorted({s["participante"] for s in SEQ_TODAS})
    n_seq = len(SEQ_TODAS)
    masc = ex.mascaras(SEQ_TODAS)

    modos = ["HMM-30a70", "HMM-por-intensidade"]
    tarefas = [(m, f) for m in modos for f in range(len(participantes))]
    print(f"Rodando {len(tarefas)} tarefas LOSO em {N_PROCESSOS} processos...", flush=True)
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        saidas = []
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            print(f"[{k}/{len(tarefas)}] {r['modo']} {r['participante']}: {r['segundos']:.0f}s "
                  f"(decorrido {time.time()-t0:.0f}s)", flush=True)
    por_modo = {m: sorted([r for r in saidas if r["modo"] == m], key=lambda r: r["fold"]) for m in modos}

    c_met = {m: el.juntar(por_modo[m], n_seq) for m in modos}
    _, ref = ex.buscar_bruto(SEQ_TODAS, masc)
    c_met["bruto_referencia"] = {"detectado": ref["detectado"], "tempo_s": ref["tempo_s"]}
    # sensibilidade: mesmo braco, mas alvos ESP avaliados com a curva conjunta
    det_conj = c_met["HMM-por-intensidade"]["detectado"].copy()
    for r in por_modo["HMM-por-intensidade"]:
        i_h = np.array(r["indices_h"])
        esp = masc["alvo_esp"][i_h]
        det_conj[i_h[esp]] = r["det_esp_conj_h"][esp]
    n_esp_conj = int(det_conj[masc["alvo_esp"]].sum())

    pareados = {
        "HMM-por-intensidade vs HMM-30a70": base._pareado(SEQ_TODAS, masc, c_met["HMM-por-intensidade"], c_met["HMM-30a70"]),
        "HMM-por-intensidade vs bruto_referencia": base._pareado(SEQ_TODAS, masc, c_met["HMM-por-intensidade"], c_met["bruto_referencia"]),
    }

    nomes = ["HMM-30a70", "HMM-por-intensidade", "bruto_referencia"]
    linhas_inten = ["| Intensidade | " + " | ".join(f"{m} det / FP lat / tempo medio" for m in nomes) + " |",
                    "|---|" + "---|" * len(nomes)]
    dados_inten = {}
    for cond in INTENSIDADES:
        cel = []
        for m in nomes:
            r = ex.resumo(SEQ_TODAS, c_met[m], lambda s, cond=cond: s["condicao"] == cond)
            dados_inten[f"{cond}/{m}"] = r
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} ({100*r['deteccao']:.1f}%) / "
                       f"{r['fp_lateral_detectados']}/{r['fp_lateral_total']} ({100*r['fp_lateral']:.1f}%) / {el.fmt_t(r['tempo_medio_s'])}")
        linhas_inten.append(f"| {cond} | " + " | ".join(cel) + " |")

    # hiperparametros e avisos por intensidade
    resumo_hiper = {}
    linhas_hip = ["| Intensidade | Seq./fold | k medio | rho medio | a_k | b_k | a_rho | b_rho | folds na_borda | hiper_no_limite | massa fora (max) | dispersao inicios > 1e-3 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in INTENSIDADES:
        hs = [r["hiper"][c] for r in por_modo["HMM-por-intensidade"]]
        v = {k: float(np.mean([x[k] for x in hs])) for k in ("k_media", "rho_media", "a_k", "b_k", "a_rho", "b_rho")}
        v |= {"na_borda_folds": sum(bool(x["na_borda"]) for x in hs), "no_limite_folds": sum(bool(x["hiper_no_limite"]) for x in hs),
              "massa_fora_max": float(max(max(x["massa_fora_grade"].values()) for x in hs)),
              "dispersao_folds": sum(x["dispersao_relativa_inicios"] > 1e-3 for x in hs), "n_seq": int(hs[0]["n_sequencias"])}
        resumo_hiper[c] = v
        linhas_hip.append(f"| {c} | {v['n_seq']} | {v['k_media']:.2f} | {v['rho_media']:.4f} | {v['a_k']:.3f} | {v['b_k']:.3f} | "
                          f"{v['a_rho']:.3f} | {v['b_rho']:.2f} | {v['na_borda_folds']}/11 | {v['no_limite_folds']}/11 | "
                          f"{v['massa_fora_max']:.2e} | {v['dispersao_folds']}/11 |")
    hip_base = [r["hiper"] for r in por_modo["HMM-30a70"]]
    resumo_hiper["30a70"] = {k: float(np.mean([x[k] for x in hip_base])) for k in ("k_media", "rho_media")}
    resumo_hiper["30a70"]["na_borda_folds"] = sum(bool(x["na_borda"]) for x in hip_base)

    razoes = razoes_por_intensidade(SEQ_TODAS)
    linhas_raz = ["| Intensidade | N | Z | P/A |", "|---|---:|---:|---:|"]
    for c, v in razoes.items():
        linhas_raz += [f"| {c} | {e['N']} | {e['Z']} | {e['razao_P_sobre_A']:.2f}x |" for e in v["exemplos"]]

    # escolhas de (pi, A, regra) por fold
    linhas_esc = ["| Fold | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Teto | Retirado det / FP lat / FP ESP |", "|---|---|---|---|---|---:|---|"]
    for r in por_modo["HMM-por-intensidade"]:
        if not r["escolha"]:
            linhas_esc.append(f"| {r['participante']} | inviavel | | | | {r['teto_treino']:.4f} | |")
            continue
        e, i_h = r["escolha"], np.array(r["indices_h"])
        d = r["detectado_h"]
        mh = {k: v[i_h] for k, v in masc.items()}
        linhas_esc.append(f"| {r['participante']} | {e['pi'][1]:g} | {e['A'][0][1]:g} | {e['A'][1][0]:g} | {el.txt_regra(e['regra'])} | "
                          f"{r['teto_treino']:.4f} | {int(d[mh['alvo_estimulo']].sum())} / {int(d[mh['lateral']].sum())} / {int(d[mh['alvo_esp']].sum())} |")

    tempo_total = time.time() - t0
    md = [
        "# Emissao Presente por intensidade (LOSO, sinal FC, registros isolados)\n",
        "5 modelos hierarquicos por fold (um por intensidade, so alvos daquela intensidade dos 10 participantes de treino). "
        "Cada sequencia usa a curva da propria gravacao; ESP usa a curva de 30 dB. Ausente inalterado. "
        "Teto de FP lateral por fold = FP do bruto_referencia no treino; um unico (pi, A, regra) por fold.\n",
        "## Geral (440 alvos / 528 laterais / 88 ESP)\n",
        "| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |", "|---|---|---|---|---|",
        *[base._linha_geral(n, SEQ_TODAS, masc, c_met[n]) for n in nomes],
        f"\nSensibilidade: HMM-por-intensidade com os alvos ESP avaliados pela curva conjunta 30-70 dB: FP ESP {n_esp_conj}/88 "
        f"({100*n_esp_conj/88:.1f}%) (regra principal, curva de 30 dB: {int(c_met['HMM-por-intensidade']['detectado'][masc['alvo_esp']].sum())}/88).\n",
        "## Testes pareados\n",
        "| A vs B | McNemar alvos | McNemar laterais | McNemar ESP | Wilcoxon tempo (n, A<B/A>B/A=B, dif media, p) |", "|---|---|---|---|---|",
        *[base._fmt_pareado(k.split(" vs ")[0], k.split(" vs ")[1], v) for k, v in pareados.items()],
        "\n## Por intensidade\n", *linhas_inten,
        "\n## Ajuste hierarquico por intensidade (media dos 11 folds; avisos)\n",
        *linhas_hip,
        f"\nBaseline conjunto 30-70 dB: k medio {resumo_hiper['30a70']['k_media']:.2f}, rho medio {resumo_hiper['30a70']['rho_media']:.4f}, "
        f"na_borda em {resumo_hiper['30a70']['na_borda_folds']}/11 folds.\n",
        "## Razao de verossimilhanca P(Z|N, Presente) / P(Z|N, Ausente) (curvas ajustadas nos 11 participantes, ilustrativo)\n",
        *linhas_raz,
        "\n## Escolhas por fold (HMM-por-intensidade)\n", *linhas_esc,
        f"\nTempo total de execucao: {tempo_total:.0f} s.",
    ]
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    saida = {
        "pooled": {m: ex.metricas(masc, c["detectado"], c["tempo_s"]) for m, c in c_met.items()},
        "sensibilidade_esp_curva_conjunta": {"fp_esp": n_esp_conj, "n": 88},
        "pareados": pareados, "por_intensidade": dados_inten, "resumo_hiper": resumo_hiper, "razoes": razoes,
        "folds": {m: [el.sem_arrays({k: v for k, v in r.items() if k != "det_esp_conj_h"}) for r in por_modo[m]] for m in modos},
        "tempo_execucao_s": tempo_total,
    }
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    print("\n".join(md))


if __name__ == "__main__":
    main()
