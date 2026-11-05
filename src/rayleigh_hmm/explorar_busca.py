"""Busca exploratoria (hmm_viterbi): grade estendida de pi e A(Aus->Pres) + 3 criterios.

Nao altera experimento.py; so reaproveita suas funcoes. Resultados in-sample
(mesmos 11 participantes): servem para ver se a borda da grade importa e se
existe configuracao "nao mais lenta" que o Rayleigh bruto, nao para validacao.
Saidas: results/rayleigh_hmm/exploratorio_busca/{resultados.json,tabelas.md}
"""

import itertools
import json
import os
import time

import numpy as np

import experimento as ex
import hmm

# ---------------- configuracao ----------------
GRADE_PI = [1e-4, 5e-4, 1e-3, 5e-3] + ex.GRADE_PI_PRESENTE
GRADE_A01 = [1e-5, 1e-4, 5e-4] + ex.GRADE_A_AUS_PRES
GRADE_A10 = list(ex.GRADE_A_PRES_AUS)
GRADE_PI_ORIGINAL = list(ex.GRADE_PI_PRESENTE)
GRADE_A01_ORIGINAL = list(ex.GRADE_A_AUS_PRES)
MAX_PONTOS_FRONTEIRA = 15
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_busca")


# ---------------- avaliacao ----------------
def preparar():
    sequencias, esp = ex.carregar_sequencias()
    emissoes = ex.ajustar_emissoes(sequencias, esp)
    emissoes["marginal"] = ex.ajustar_marginal(emissoes)
    masc = ex.mascaras(sequencias)
    _, referencia = ex.buscar_bruto(sequencias, masc)
    grupos = [
        (idx, dur, np.stack([ex.log_emissoes(sequencias[i], emissoes, "marginal") for i in idx]))
        for idx, dur in ex.indices_por_comprimento(sequencias)
    ]
    return sequencias, masc, referencia, grupos


def avaliar_parametros(grupos, n_seq, p1, a01, a10):
    """Indicadores por grupo para um (pi, A); depois cada regra via ex.decidir."""
    pi = np.array([1.0 - p1, p1])
    A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
    return [(idx, *ex.indicadores(hmm.viterbi_alguma_presente(le, pi, A)), dur) for idx, dur, le in grupos]


def diferenca_pareada(masc, ref, detectado, tempo):
    ambos = masc["alvo_estimulo"] & ref["detectado"] & detectado
    return int(ambos.sum()), float(np.mean(tempo[ambos] - ref["tempo_s"][ambos]))


def varrer(grupos, sequencias, masc, ref):
    """Todos os candidatos (pi, A, regra) viaveis; guarda so metricas (sem arrays)."""
    viaveis, total = [], 0
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
        grupos_det = avaliar_parametros(grupos, len(sequencias), p1, a01, a10)
        for k, regra in enumerate(ex.REGRAS):
            total += 1
            det, tempo = ex.decidir(grupos_det, len(sequencias), regra)
            m = ex.metricas(masc, det, tempo)
            if m["fp_lateral"] > ex.FP_MAXIMO:
                continue
            n_pares, dif = diferenca_pareada(masc, ref, det, tempo)
            viaveis.append({"p1": p1, "a01": a01, "a10": a10, "k": k, "m": m,
                            "dif_pareada_s": dif, "n_pares": n_pares})
    return viaveis, total


# ---------------- criterios ----------------
def chave_a(c):
    return (c["m"]["deteccao"], -c["m"]["fp_lateral"], -(c["m"]["tempo_medio_s"] or np.inf))


def chave_b(c):
    return (c["m"]["deteccao"], -c["m"]["fp_lateral"], -(c["m"]["tempo_medio_s"] or np.inf))


def pareto(viaveis):
    """Fronteira deteccao (max) x tempo medio (min); remove metricas identicas."""
    unicos = {}
    for c in viaveis:
        if c["m"]["tempo_medio_s"] is None:
            continue
        chave = (c["m"]["deteccao"], c["m"]["tempo_medio_s"])
        if chave not in unicos or chave_a(c) > chave_a(unicos[chave]):
            unicos[chave] = c
    ordenados = sorted(unicos.values(), key=lambda c: (-c["m"]["deteccao"], c["m"]["tempo_medio_s"]))
    fronteira, melhor_tempo = [], np.inf
    for c in ordenados:  # deteccao decrescente: entra se o tempo melhora
        if c["m"]["tempo_medio_s"] < melhor_tempo - 1e-12:
            fronteira.append(c)
            melhor_tempo = c["m"]["tempo_medio_s"]
    return fronteira


def representativos(fronteira, det_minima):
    """<= MAX_PONTOS_FRONTEIRA pontos; sempre max deteccao e mais rapido com det >= det_minima."""
    obrigatorios = {0}
    elegiveis = [i for i, c in enumerate(fronteira) if c["m"]["deteccao"] >= det_minima - 1e-12]
    if elegiveis:
        obrigatorios.add(elegiveis[-1])
    if len(fronteira) <= MAX_PONTOS_FRONTEIRA:
        return list(fronteira)
    livres = MAX_PONTOS_FRONTEIRA - len(obrigatorios)
    espaco = set(np.linspace(0, len(fronteira) - 1, livres + 2).round().astype(int).tolist())
    escolha = sorted((obrigatorios | espaco))
    while len(escolha) > MAX_PONTOS_FRONTEIRA:
        escolha.remove(next(i for i in escolha[1:-1] if i not in obrigatorios))
    return [fronteira[i] for i in escolha]


# ---------------- relatorio ----------------
def completar(c, grupos, sequencias, masc, ref):
    """Recalcula arrays do candidato e adiciona comparacao pareada e testes."""
    gd = avaliar_parametros(grupos, len(sequencias), c["p1"], c["a01"], c["a10"])
    regra = ex.REGRAS[c["k"]]
    det, tempo = ex.decidir(gd, len(sequencias), regra)
    cand = {"detectado": det, "tempo_s": tempo}
    sig = ex.significancia(sequencias, masc, ref, cand)
    ambos = masc["alvo_estimulo"] & ref["detectado"] & det
    d = tempo[ambos] - ref["tempo_s"][ambos]
    m = ex.metricas(masc, det, tempo)
    return {
        "pi_presente": c["p1"], "A_aus_pres": c["a01"], "A_pres_aus": c["a10"], "regra": regra,
        "metricas": m,
        "n_det": int(det[masc["alvo_estimulo"]].sum()), "n_fp_lat": int(det[masc["lateral"]].sum()),
        "n_fp_esp": int(det[masc["alvo_esp"]].sum()),
        "pareado": {"n_pares": int(ambos.sum()), "hmm_mais_rapido": int((d < 0).sum()),
                    "hmm_mais_lento": int((d > 0).sum()), "iguais": int((d == 0).sum()),
                    "dif_media_s": float(d.mean()), "wilcoxon_p": sig["tempo"]["p_valor"]},
        "mcnemar": {k: sig[k] for k in ("alvos", "laterais", "esp")},
        "_cand": {"detectado": det, "tempo_s": tempo},
    }


def tabela_intensidade(sequencias, cand, ref):
    conds = sorted({s["condicao"] for s in sequencias if s["condicao"] != "ESP"}, key=ex.ordem_condicao)
    linhas = []
    for cond in conds:
        f = lambda s, cond=cond: s["condicao"] == cond
        a, b = ex.resumo(sequencias, ref, f), ex.resumo(sequencias, cand, f)
        linhas.append({"condicao": cond, "bruto": a, "hmm": b})
    return linhas


def fmt_t(x):
    return "-" if x is None else f"{x:.1f}"


def linha_md(rotulo, r):
    m, p, mc = r["metricas"], r["pareado"], r["mcnemar"]
    reg = ex.REGRAS[0] if False else r["regra"]
    texto_regra = ", ".join(f"{k}={v}" for k, v in reg.items() if k != "familia") + f" ({reg['familia']})"
    wp = "-" if p["wilcoxon_p"] is None else f"{p['wilcoxon_p']:.2g}"
    mc_txt = " / ".join(f"{mc[k]['so_bruto']}-{mc[k]['so_hmm']} p={mc[k]['p_valor']:.2g}" for k in ("alvos", "laterais", "esp"))
    return (f"| {rotulo} | {r['pi_presente']:g} | {r['A_aus_pres']:g} | {r['A_pres_aus']:g} | {texto_regra} | "
            f"{r['n_det']}/440 ({100*m['deteccao']:.1f}%) | {r['n_fp_lat']}/528 | {r['n_fp_esp']}/88 | "
            f"{fmt_t(m['tempo_medio_s'])} / {fmt_t(m['tempo_mediano_s'])} | {p['n_pares']} | "
            f"{p['hmm_mais_rapido']}/{p['hmm_mais_lento']}/{p['iguais']} | {p['dif_media_s']:+.1f} | {wp} | {mc_txt} |")


CABECALHO = ("| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | "
             "Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | "
             "McNemar so_bruto-so_hmm (alvos / laterais / ESP) |\n" + "|" + "---|" * 14)


def tabela_intensidade_md(linhas, nome):
    out = [f"\n### Por intensidade: {nome} vs bruto_referencia\n",
           "| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |", "|" + "---|" * 7]
    for l in linhas:
        b, h = l["bruto"], l["hmm"]
        out.append(f"| {l['condicao']} | {b['deteccao_detectados']}/{b['deteccao_total']} | {h['deteccao_detectados']}/{h['deteccao_total']} | "
                   f"{b['fp_lateral_detectados']}/{b['fp_lateral_total']} | {h['fp_lateral_detectados']}/{h['fp_lateral_total']} | "
                   f"{fmt_t(b['tempo_medio_s'])} | {fmt_t(h['tempo_medio_s'])} |")
    return "\n".join(out)


def sem_privado(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    t0 = time.time()
    sequencias, masc, ref, grupos = preparar()
    viaveis, total = varrer(grupos, sequencias, masc, ref)
    print(f"candidatos avaliados: {total}; viaveis: {len(viaveis)}")

    # a) criterio original: grade estendida e grade original (sanidade)
    a_ext = max(viaveis, key=chave_a)
    orig = [c for c in viaveis if c["p1"] in GRADE_PI_ORIGINAL and c["a01"] in GRADE_A01_ORIGINAL]
    a_orig = max(orig, key=chave_a)
    # b) nao mais lento que o bruto de referencia
    nao_lento = [c for c in viaveis if c["dif_pareada_s"] <= 0]
    b = max(nao_lento, key=chave_b) if nao_lento else None
    # c) Pareto
    fronteira = pareto(viaveis)
    det_ref = ref["detectado"][masc["alvo_estimulo"]].mean()
    pontos = representativos(fronteira, 252 / 440)

    escolhidos = {"a_grade_original": a_orig, "a_grade_estendida": a_ext}
    if b:
        escolhidos["b_nao_mais_lento"] = b
    else:  # criterio b vazio: reporta o mais proximo (menor dif. pareada) com deteccao >= bruto
        n_ref = int(ref["detectado"][masc["alvo_estimulo"]].sum())
        quase = [c for c in viaveis if round(c["m"]["deteccao"] * 440) >= n_ref]
        escolhidos["b_vazio_menor_dif_det_ge_bruto"] = min(quase, key=lambda c: (c["dif_pareada_s"], -c["m"]["deteccao"]))
    res = {n: completar(c, grupos, sequencias, masc, ref) for n, c in escolhidos.items()}
    res_front = [completar(c, grupos, sequencias, masc, ref) for c in pontos]
    ref_m = ex.metricas(masc, ref["detectado"], ref["tempo_s"])

    saida = {"aviso": "exploratorio, in-sample (mesmos 11 participantes); nao e validacao",
             "n_candidatos_avaliados": total, "n_viaveis": len(viaveis), "n_nao_mais_lentos": len(nao_lento),
             "n_pontos_fronteira_total": len(fronteira),
             "grades": {"pi": GRADE_PI, "A_aus_pres": GRADE_A01, "A_pres_aus": GRADE_A10},
             "bruto_referencia": ref_m,
             "escolhidos": {n: sem_privado(r) for n, r in res.items()},
             "fronteira_pareto": [sem_privado(r) for r in res_front]}

    md = ["# Busca exploratoria hmm_viterbi (in-sample, 11 participantes)\n",
          f"Candidatos avaliados: {total}; viaveis (FP lateral <= {ex.FP_MAXIMO}): {len(viaveis)}; "
          f"com dif. pareada <= 0: {len(nao_lento)}; pontos de Pareto (dedupe): {len(fronteira)}.\n",
          f"bruto_referencia: det {ref['detectado'][masc['alvo_estimulo']].sum()}/440, FP lat "
          f"{ref['detectado'][masc['lateral']].sum()}/528, FP ESP {ref['detectado'][masc['alvo_esp']].sum()}/88, "
          f"tempo medio/mediano {fmt_t(ref_m['tempo_medio_s'])}/{fmt_t(ref_m['tempo_mediano_s'])}.\n",
          "## Vencedores\n", CABECALHO]
    for n, r in res.items():
        md.append(linha_md(n, r))
    md += ["\n## Fronteira de Pareto (deteccao x tempo medio)\n", CABECALHO]
    for i, r in enumerate(res_front):
        md.append(linha_md(f"P{i+1}", r))
    for n in ("a_grade_estendida", "b_nao_mais_lento", "b_vazio_menor_dif_det_ge_bruto"):
        if n in res:
            md.append(tabela_intensidade_md(tabela_intensidade(sequencias, res[n]["_cand"], ref), n))
            saida["escolhidos"][n]["por_intensidade"] = [
                {"condicao": l["condicao"], "bruto": l["bruto"], "hmm": l["hmm"]}
                for l in tabela_intensidade(sequencias, res[n]["_cand"], ref)]
    tempo = time.time() - t0
    saida["tempo_execucao_s"] = tempo
    md.append(f"\nTempo de execucao: {tempo:.0f} s.")

    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
