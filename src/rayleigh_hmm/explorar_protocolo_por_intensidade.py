"""Exploratorio (Parte 2): LOSO do protocolo concatenado com emissao Presente POR INTENSIDADE.

Mesmo LOSO de explorar_protocolo_loso.py (emissoes por fold so com T, M escolhido no treino, grades, regras,
teto de FP lateral = FP do bruto_referencia nos laterais de T, Viterbi causal por protocolo). Unica mudanca:
para cada (fold, M) ajustam-se 5 modelos hierarquicos Presente, um por intensidade (so os alvos daquela
intensidade de T); a janela de um segmento de X dB usa a curva de X dB e o segmento ESP usa a de 30 dB.
Ausente inalterado. Uma curva por intensidade da populacao de treino, nao por paciente.

Ordens: asc (ESP->30->...->70) e desc (70->...->30->ESP), como em explorar_protocolo_loso.py e
explorar_protocolo_descendente.py (reaproveita as funcoes de metricas/tabelas deste, por serem
independentes da ordem). Nao altera arquivos existentes: troca funcoes dos modulos em memoria.

Execucao: .venv/bin/python src/rayleigh_hmm/explorar_protocolo_por_intensidade.py asc|desc
Saidas: results/rayleigh_hmm/exploratorio_protocolo_por_intensidade/{asc,desc}/{resultados.json,tabelas.md,por_segmento.csv}
"""

import csv
import json
import os
import sys

import numpy as np

import emissao
import experimento as ex
import explorar_protocolo as ep
import explorar_protocolo_descendente as de
import explorar_protocolo_loso as lo

# ---------------- configuracao ----------------
ORDENS = {"asc": ["ESP", "30dB", "40dB", "50dB", "60dB", "70dB"], "desc": de.ORDEM}
EXISTENTES = {"asc": "exploratorio_protocolo_loso", "desc": "exploratorio_protocolo_descendente"}
PASTA_BASE = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_protocolo_por_intensidade")
INTENSIDADES = de.ESTIMULOS
CURVA_ESP = "30dB"
METODOS_TAB = ["bruto_referencia", "rayleigh_loso", "hmm_loso"]


# ---------------- emissao Presente por intensidade ----------------
def ajustar_emissoes_treino(z, n, no_treino):
    """Substitui lo.ajustar_emissoes_treino: Ausente igual; Presente com 5 curvas, uma por intensidade."""
    est = lo.EST
    seg_id = est["seg_id"]
    cond_janela = est["cond"][seg_id]
    no_esp = est["m_alvo_esp"][seg_id] & no_treino
    aus = emissao.ajustar_mle(z[no_esp])
    log_pres = np.empty(z.shape)
    por_int = {}
    for c in INTENSIDADES:
        no_c = est["m_alvo_estim"][seg_id] & (cond_janela == c) & no_treino
        matriz = emissao.matriz_verossimilhanca_sequencias(z[no_c], n[no_c], seg_id[no_c], ex.GRADE_HIER_K, ex.GRADE_HIER_RHO)
        hiper = emissao.ajustar_hierarquico(matriz, ex.GRADE_HIER_K, ex.GRADE_HIER_RHO, ex.INICIOS_HIER)
        n_seq = int(matriz.shape[0])
        del matriz
        usa = (cond_janela == c) | ((cond_janela == "ESP") & (c == CURVA_ESP))  # ESP usa a curva de 30 dB
        log_pres[usa] = emissao.log_pdf_preditiva(z[usa], n[usa], ex.pontos_marginal(hiper))
        por_int[c] = {"na_borda": bool(hiper["na_borda"]), "hiper_no_limite": bool(hiper["hiper_no_limite"]),
                      "dispersao_inicios": float(hiper["dispersao_relativa_inicios"]), "n_seq": n_seq,
                      "massa_fora_grade": float(max(hiper["massa_fora_grade"].values())), "k_media": float(hiper["k_media"]),
                      "rho_media": float(hiper["rho_media"]), "a_k": float(hiper["a_k"]), "b_k": float(hiper["b_k"])}
    log_em = np.stack([emissao.log_pdf_gamma(z, aus["k"], aus["mu"]), log_pres], axis=-1)
    aviso = {"na_borda": any(v["na_borda"] for v in por_int.values()),
             "hiper_no_limite": any(v["hiper_no_limite"] for v in por_int.values()),
             "dispersao_inicios": max(v["dispersao_inicios"] for v in por_int.values()),
             "a_k": por_int[CURVA_ESP]["a_k"], "b_k": por_int[CURVA_ESP]["b_k"],
             "ausente_k": float(aus["k"]), "ausente_mu": float(aus["mu"]), "por_intensidade": por_int}
    return log_em, aviso


# ---------------- comparacao com os resultados existentes (a partir dos por_segmento.csv) ----------------
def contagens_csv(caminho):
    """{metodo: {cond: [det_alvo, fp_lateral, fp_alvo_esp]}} lido de um por_segmento.csv."""
    out = {}
    for r in csv.DictReader(open(caminho)):
        v = out.setdefault(r["metodo"], {}).setdefault(r["intensidade"], [0, 0, 0])
        d = int(r["detectado"])
        if r["tipo"] == "lateral":
            v[1] += d
        elif r["intensidade"] == "ESP":
            v[2] += d
        else:
            v[0] += d
    return out


def comparar(ordem, pasta, existente):
    novo = json.load(open(os.path.join(pasta, "resultados.json")))["metricas"]
    velho = json.load(open(os.path.join(existente, "resultados.json")))["metricas"]
    cn = contagens_csv(os.path.join(pasta, "por_segmento.csv"))
    cv = contagens_csv(os.path.join(existente, "por_segmento.csv"))
    pct = lambda x, n: f"{100 * x / n:.1f}%"
    md = ["", "## Comparacao com os resultados existentes (emissao Presente conjunta 30-70 dB)", "",
          "| Metodo | Det | FP lat | FP ESP | Proto alvo | Proto lat | Tempo medio/mediano (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for rot, m in [("bruto_referencia (existente)", velho["bruto_referencia"]), ("bruto_referencia (esta rodada)", novo["bruto_referencia"]),
                   ("rayleigh_loso (existente)", velho["rayleigh_loso"]), ("rayleigh_loso (esta rodada)", novo["rayleigh_loso"]),
                   ("hmm_loso conjunta (existente)", velho["hmm_loso"]), ("**hmm_loso por intensidade**", novo["hmm_loso"])]:
        md.append(f"| {rot} | {pct(m['det'], 440)} ({m['det']}) | {pct(m['fp_lateral'], 528)} ({m['fp_lateral']}) | "
                  f"{pct(m['fp_esp'], 88)} ({m['fp_esp']}) | {pct(m['proto_alvo_det_estimulo'], 88)} | {pct(m['proto_lateral_det'], 88)} | "
                  f"{lo.f_t(m['tempo_medio_s'])} / {lo.f_t(m['tempo_mediano_s'])} |")
    md += ["", f"Quebra por condicao, ordem {' -> '.join(ordem)}: det alvo (/88) | FP lateral (/88) | FP alvo ESP (/88).", "",
           "| Metodo | Medida | " + " | ".join(ordem) + " |", "|---|---|" + "---:|" * len(ordem)]
    for rot, cc, nome in [("hmm_loso conjunta (existente)", cv, "hmm_loso"), ("hmm_loso por intensidade", cn, "hmm_loso"),
                          ("bruto_referencia", cn, "bruto_referencia")]:
        for k, medida in enumerate(("Det alvo", "FP lateral", "FP alvo ESP")):
            cel = []
            for c in ordem:
                if (k == 0 and c == "ESP") or (k == 2 and c != "ESP"):
                    cel.append("-")
                else:
                    v = cc[nome][c][k]
                    cel.append(f"{pct(v, 88)} ({v})")
            md.append(f"| {rot} | {medida} | " + " | ".join(cel) + " |")
    md += ["", "## Ajuste hierarquico por intensidade (folds x M com borda/limite, de 55)", "",
           "| Intensidade | na_borda | hiper_no_limite | dispersao inicios > 1e-3 | massa fora max | k medio | rho medio |", "|---|---:|---:|---:|---:|---:|---:|"]
    avisos = json.load(open(os.path.join(pasta, "resultados.json")))["ajuste_hierarquico"]
    for c in INTENSIDADES:
        v = [a["por_intensidade"][c] for a in avisos]
        md.append(f"| {c} | {sum(x['na_borda'] for x in v)} | {sum(x['hiper_no_limite'] for x in v)} | "
                  f"{sum(x['dispersao_inicios'] > 1e-3 for x in v)} | {max(x['massa_fora_grade'] for x in v):.2e} | "
                  f"{np.mean([x['k_media'] for x in v]):.2f} | {np.mean([x['rho_media'] for x in v]):.4f} |")
    with open(os.path.join(pasta, "tabelas.md"), "a") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    modo = sys.argv[1]
    ordem = ORDENS[modo]
    pasta = os.path.join(PASTA_BASE, modo)
    ep.ORDEM_CONDICOES = ordem  # montar_estrutura e tabelas leem este atributo na chamada
    assert len(ordem) == ep.N_COND and ordem[-1 if modo == "desc" else 0] == "ESP"
    ep.por_protocolo, ep.metricas_metodo = de.por_protocolo, de.metricas_metodo
    de.PASTA = pasta
    lo.PASTA = pasta
    lo.CABECALHO = (f"LOSO, sem herança; exploratório; protocolo {'DESCENDENTE' if modo == 'desc' else 'ASCENDENTE'} "
                    f"({' → '.join(ordem)}); emissão Presente POR INTENSIDADE (curva de X dB no segmento de X dB; ESP usa a de 30 dB)")
    de.CABECALHO = lo.CABECALHO
    lo.linha, lo.escrever_md = de.linha, de.escrever_md
    lo.ajustar_emissoes_treino = ajustar_emissoes_treino
    lo.main()
    comparar(ordem, pasta, os.path.join(ex.PASTA_RESULTADOS, EXISTENTES[modo]))


if __name__ == "__main__":
    main()
