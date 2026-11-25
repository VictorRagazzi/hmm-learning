"""Exploratorio: busca in-sample (explorar_busca, hmm_viterbi, grade estendida) com Rayleigh em
blocos INDEPENDENTES de M epocas, para M em VALORES_M.

Resultado in-sample (mesmos 11 participantes): limite otimista, nao e validacao. O teto de FP
lateral e o do projeto: FP do Rayleigh ACUMULADO de referencia (alfa 0,05, 3 consecutivas),
calculado antes de qualquer patch. Alem do HMM, busca o Rayleigh em blocos (alfa e regra) com
GRADE_ALFA_ESTENDIDA, sob o mesmo teto. Patches so em memoria (dados.rayleigh_z_acumulado e
parametros de experimento/explorar_busca), restaurados no final.
Saidas: results/rayleigh_hmm/exploratorio_independentes/{busca_M10,busca_M30}/,
        busca_resumo.md, busca_resumo.json
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_busca_independentes.py
"""
import json
import os

import dados
import experimento as ex
import explorar_busca
import explorar_independentes as ei

# ---------------- configuracao ----------------
VALORES_M = (10, 30)
TEMPO_MINIMO_S = 30  # nenhuma regra dispara antes de 30 s (como no oficial)
GRADE_ALFA_ESTENDIDA = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
PASTA_BASE = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_independentes")


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def seg(x):
    return "-" if x is None else f"{x:.1f}".replace(".", ",")


def texto_regra(regra):
    return ", ".join(f"{k}={v}" for k, v in regra.items() if k != "familia") + f" ({regra['familia']})"


def limpar(c):
    """Candidato do buscar_bruto sem arrays, serializavel."""
    return {k: v for k, v in c.items() if k not in ("detectado", "tempo_s")}


def linha(metodo, parametros, m):
    return (f"| {metodo} | {parametros} | {pct(m['deteccao'])} | {pct(m['fp_lateral'])} | "
            f"{pct(m['fp_alvo_esp'])} | {seg(m['tempo_medio_s'])} / {seg(m['tempo_mediano_s'])} |")


def params_hmm(r):
    return (f"pi={r['pi_presente']:g}, A01={r['A_aus_pres']:g}, A10={r['A_pres_aus']:g}, "
            f"{texto_regra(r['regra'])}")


def buscar_blocos(teto):
    """Rayleigh em blocos otimizado in-sample (com o patch ativo), sob o teto."""
    seq = ex.carregar_sequencias()[0]
    original = ex.GRADE_ALFA
    ex.GRADE_ALFA = GRADE_ALFA_ESTENDIDA
    try:
        melhores, _ = ex.buscar_bruto(seq, ex.mascaras(seq), teto)
    finally:
        ex.GRADE_ALFA = original
    melhor = ex.melhor_geral(melhores)
    return None if melhor is None else limpar(melhor)


def escrever_resumo(teto, ref_pooled, minimos, por_m):
    ref_txt = "alfa=0,05, 3 consecutivas"
    geral = ["| Metodo | Parametros | Deteccao | FP lateral | FP ESP | Tempo medio / mediano (s) |",
             "|---|---|---|---|---|---|",
             linha("Rayleigh acumulado (referencia oficial)", ref_txt, ref_pooled)]
    fronteiras = []
    for M in VALORES_M:
        h = por_m[M]["hmm_viterbi"]
        geral.append(linha(f"M={M}: HMM Viterbi melhor in-sample", params_hmm(h), h["metricas"]))
        b = por_m[M]["rayleigh_blocos"]
        if b is None:
            geral.append(f"| M={M}: Rayleigh blocos otimizado in-sample | nenhuma regra viavel | - | - | - | - |")
        else:
            geral.append(linha(f"M={M}: Rayleigh blocos otimizado in-sample",
                               f"alfa={b['alfa']:g}, {texto_regra(b['regra'])}", b["metricas"]))
        fronteiras += [f"\n### M={M}\n", "| Deteccao | FP lateral | Tempo medio (s) | Parametros |", "|---|---|---|---|"]
        for r in por_m[M]["fronteira_pareto_hmm"]:
            m = r["metricas"]
            fronteiras.append(f"| {pct(m['deteccao'])} | {pct(m['fp_lateral'])} | {seg(m['tempo_medio_s'])} | "
                              f"{params_hmm(r)} |")
    md = ["# Busca in-sample com Rayleigh em blocos independentes (exploratorio)\n",
          "Resultado in-sample (11 participantes): limite otimista, nao e validacao. Teto de FP lateral = "
          f"FP do Rayleigh acumulado de referencia ({pct(teto)}). MINIMO_AVALIACOES por M: "
          + ", ".join(f"M={M} -> {minimos[M]}" for M in VALORES_M) + ". Grade de alfa do Rayleigh em blocos: "
          + ", ".join(f"{a:g}" for a in GRADE_ALFA_ESTENDIDA) + ".\n",
          "## Geral\n", *geral, "\n## Fronteira de Pareto do HMM (deteccao x tempo medio)", *fronteiras]
    os.makedirs(PASTA_BASE, exist_ok=True)
    with open(os.path.join(PASTA_BASE, "busca_resumo.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    saida = {"aviso": "exploratorio, in-sample; nao e validacao", "teto_fp_lateral": teto,
             "referencia_acumulada_pooled": ref_pooled, "minimo_avaliacoes": minimos,
             "grade_alfa_estendida": GRADE_ALFA_ESTENDIDA,
             "por_M": {str(M): por_m[M] for M in VALORES_M}}
    with open(os.path.join(PASTA_BASE, "busca_resumo.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    print("\n".join(md))


def main():
    # 1) tudo que depende do acumulado, ANTES de qualquer patch
    seq_acum, _ = ex.carregar_sequencias()
    ref = ex.calcular_referencia(seq_acum, ex.mascaras(seq_acum))
    teto = ref["metricas"]["fp_lateral"]
    ref_pooled = ref["metricas"]

    originais = (dados.rayleigh_z_acumulado, ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES,
                 ex.FP_MAXIMO, explorar_busca.PASTA)
    minimos, por_m = {}, {}
    try:
        for M in VALORES_M:
            minimos[M] = max(1, TEMPO_MINIMO_S // M)
            dados.rayleigh_z_acumulado = ei.rayleigh_z_blocos
            ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES, ex.FP_MAXIMO = M, minimos[M], teto
            explorar_busca.PASTA = os.path.join(PASTA_BASE, f"busca_M{M}")
            explorar_busca.main()
            with open(os.path.join(explorar_busca.PASTA, "resultados.json")) as f:
                res = json.load(f)
            por_m[M] = {"hmm_viterbi": res["escolhidos"]["a_grade_estendida"],
                        "fronteira_pareto_hmm": res["fronteira_pareto"],
                        "rayleigh_blocos": buscar_blocos(teto)}
    finally:
        (dados.rayleigh_z_acumulado, ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES,
         ex.FP_MAXIMO, explorar_busca.PASTA) = originais
    escrever_resumo(teto, ref_pooled, minimos, por_m)


if __name__ == "__main__":
    main()
