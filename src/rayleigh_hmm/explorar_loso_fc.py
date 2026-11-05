"""Exploratorio: LOSO com o setup atual (sinal FC, teto = FP lateral do bruto_referencia no treino).

Reusa explorar_loso.py sobrescrevendo pasta, teto de FP por fold e numeros in-sample.
Teto de cada fold: FP lateral (contagem/total) do bruto_referencia nos participantes
de TREINO do fold (sem vazamento); comparacao com <= (folga 1e-12 em ex.avaliar_regras).
Saidas: results/rayleigh_hmm/exploratorio_loso_fc/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_loso_fc.py
"""
import json
import os

import experimento as ex
import explorar_loso as el

PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_loso_fc")


def teto_referencia(seq_tr, masc_tr):
    return ex.calcular_referencia(seq_tr, masc_tr)["metricas"]["fp_lateral"]


def main():
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    el.PASTA = PASTA
    el.teto_fp = teto_referencia
    el.IN_SAMPLE = {
        "HMM in-sample (hmm_viterbi)": {"det": 241, "fp_lat": 45, "fp_esp": 7, "tempo": (121.4, 70.0)},
        "bruto_ref in-sample": {"det": 246, "fp_lat": 45, "fp_esp": 3, "tempo": (132.9, 95.0)},
    }
    el.main()
    # pos-processamento: tetos por fold e rotulo correto
    seq, _ = ex.carregar_sequencias()
    parts = sorted({s["participante"] for s in seq})
    tetos = {}
    for h in parts:
        tr = [s for s in seq if s["participante"] != h]
        m = ex.mascaras(tr)
        tetos[h] = {"fp_lateral": teto_referencia(tr, m), "n_lateral": int(m["lateral"].sum())}
        tetos[h]["contagem"] = round(tetos[h]["fp_lateral"] * tetos[h]["n_lateral"])
    p = os.path.join(PASTA, "resultados.json")
    r = json.load(open(p))
    r["tetos_fp_por_fold"] = tetos
    r["aviso"] += "; sinal FC; teto de FP por fold = FP lateral do bruto_referencia no treino"
    json.dump(r, open(p, "w"), indent=1)
    p = os.path.join(PASTA, "tabelas.md")
    md = open(p).read().replace("FP lateral held-out acima de 7% (folds)", "FP lateral held-out acima de 7% (folds; informativo, o teto real e por fold)")
    md = md.replace("FP lateral de treino <= 0.07", "FP lateral de treino <= teto do fold (tabela abaixo)")
    md += "\n## Teto de FP lateral por fold (bruto_referencia no treino)\n\n| Held-out | Teto | Contagem/total |\n|---|---|---|\n"
    md += "\n".join(f"| {h} | {t['fp_lateral']:.4f} ({100*t['fp_lateral']:.1f}%) | {t['contagem']}/{t['n_lateral']} |" for h, t in tetos.items()) + "\n"
    open(p, "w").write(md)


if __name__ == "__main__":
    main()
