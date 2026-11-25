"""Exploratorio: LOSO do protocolo concatenado em ordem DESCENDENTE 70 -> 60 -> 50 -> 40 -> 30 -> ESP.

Mesmo LOSO de explorar_protocolo_loso.py (emissoes por fold so com T, grades, regras, teto de FP
lateral = FP do bruto_referencia nos laterais de T, Viterbi causal por protocolo com o estado
atravessando as gravacoes, Z reiniciando a cada gravacao). Unica mudanca: a ordem das gravacoes.
Objetivo: medir se a memoria do HMM "arrasta" a resposta para gravacoes sem resposta (FP nas
intensidades baixas e no ESP, que agora vem logo depois dos trechos com resposta).

Nao altera arquivos existentes: sobrescreve, em memoria, ep.ORDEM_CONDICOES e as funcoes de
metricas que assumiam o ESP na posicao 0 (ORDEM_CONDICOES[1:], v[1:], d2[:, 1:]), e a
saida (colunas/pasta) do modulo do LOSO ascendente. O resto (montar_estrutura, mascaras por nome
de condicao, Viterbi, regras por segmento) ja e independente da ordem.

Execucao: .venv/bin/python src/rayleigh_hmm/explorar_protocolo_descendente.py
Saidas: results/rayleigh_hmm/exploratorio_protocolo_descendente/{resultados.json,tabelas.md,por_segmento.csv}
"""

import os

import numpy as np

import experimento as ex
import explorar_protocolo as ep
import explorar_protocolo_loso as lo

# ---------------- configuracao ----------------
ORDEM = ["70dB", "60dB", "50dB", "40dB", "30dB", "ESP"]
ESTIMULOS = ["30dB", "40dB", "50dB", "60dB", "70dB"]  # ordem de exibicao das colunas
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_protocolo_descendente")
CABECALHO = ("LOSO, sem herança; exploratório; protocolo DESCENDENTE 70→60→50→40→30→ESP; cada intensidade decidida pela "
             "própria gravação; estado do HMM atravessa as gravações dentro do protocolo")


# ---------------- funcoes de ep que assumiam ESP na posicao 0 ----------------
def por_protocolo(est, ds):
    """(alvos: alguma deteccao nas intensidades com estimulo, laterais: algum FP em qualquer segmento)."""
    d2 = ds.reshape(-1, ep.N_COND)
    tp = est["tipo"].reshape(-1, ep.N_COND)[:, 0]
    j_est = [j for j, c in enumerate(ep.ORDEM_CONDICOES) if c != "ESP"]
    return d2[tp == "alvo"][:, j_est].any(axis=1), d2[tp == "lateral"].any(axis=1)


def metricas_metodo(est, c):
    ds, tempo = c["detectado"], c["tempo_s"]
    segs = est["segs"]
    ordem = ep.ORDEM_CONDICOES
    j_est = [j for j, x in enumerate(ordem) if x != "ESP"]
    por_int = {x: int(ds[est["m_alvo_estim"] & (est["cond"] == x)].sum()) for x in ESTIMULOS}
    por_cond = {}
    for x in ESTIMULOS + ["ESP"]:
        m = est["cond"] == x
        por_cond[x] = {"det_alvo": int(ds[m & (est["tipo"] == "alvo")].sum()) if x != "ESP" else None,
                       "fp_lateral": int(ds[m & est["m_lateral"]].sum()),
                       "fp_alvo_esp": int(ds[m & est["m_alvo_esp"]].sum()) if x == "ESP" else None,
                       "n": int((m & (est["tipo"] == "alvo")).sum())}
    t = tempo[est["m_alvo_estim"] & ds]
    protos = {}
    for k, s in enumerate(segs):
        if s["tipo"] == "alvo":
            protos.setdefault(s["proto"], []).append(k)
    primeira = {r: 0 for r in ordem + ["nenhuma"]}
    consistentes = com_det = 0
    tempos_prot = []
    for ks in protos.values():  # ks na ordem de ep.ORDEM_CONDICOES
        v = [bool(ds[k]) for k in ks]
        primeira[next((ordem[j] for j, x in enumerate(v) if x), "nenhuma")] += 1
        ks_e = [ks[j] for j in j_est]
        est_v = [v[j] for j in j_est]
        if any(est_v):
            com_det += 1
            f = est_v.index(True)
            consistentes += all(est_v[f:])
            acum = sum(segs[k]["dur_s"] for k in ks_e[:f])
            tempos_prot.append(acum + tempo[ks_e[f]])
    d2 = ds.reshape(-1, ep.N_COND)
    tipo_p = est["tipo"].reshape(-1, ep.N_COND)[:, 0]
    return {
        "proto_alvo_det_estimulo": int(d2[tipo_p == "alvo"][:, j_est].any(axis=1).sum()),
        "proto_lateral_det": int(d2[tipo_p == "lateral"].any(axis=1).sum()),
        "det": int(ds[est["m_alvo_estim"]].sum()), "det_por_intensidade": por_int, "por_condicao": por_cond,
        "fp_lateral": int(ds[est["m_lateral"]].sum()), "fp_esp": int(ds[est["m_alvo_esp"]].sum()),
        "tempo_medio_s": float(t.mean()) if t.size else None,
        "tempo_mediano_s": float(np.median(t)) if t.size else None,
        "primeira_intensidade": primeira, "protocolos_com_det_estimulo": com_det,
        "consistentes_acima": consistentes,
        "tempo_protocolo_medio_s": float(np.mean(tempos_prot)) if tempos_prot else None,
        "tempo_protocolo_mediano_s": float(np.median(tempos_prot)) if tempos_prot else None,
    }


# ---------------- tabelas ----------------
def linha(nome, m):
    di = m["det_por_intensidade"]
    return (f"| {nome} | {lo.pct(m['det'], 440)} ({m['det']}/440) | " + " | ".join(lo.pct(di[c], 88) for c in ESTIMULOS)
            + f" | {lo.pct(m['fp_lateral'], 528)} | {lo.pct(m['fp_esp'], 88)} | {lo.pct(m['proto_alvo_det_estimulo'], 88)} "
            f"| {lo.pct(m['proto_lateral_det'], 88)} | {lo.f_t(m['tempo_medio_s'])} / {lo.f_t(m['tempo_mediano_s'])} |")


def escrever_md(s, total):
    original(s, total)  # tabelas.md padrao, com colunas 30..70 e primeira intensidade na ordem 70..ESP
    ordem = ep.ORDEM_CONDICOES
    md = ["", "## Quebra por condição (ordem do protocolo: " + " → ".join(ordem) + ")", "",
          "Detecção nos alvos (de 88 por intensidade), FP nos laterais (de 88 por condição) e FP no alvo ESP (de 88). "
          "Contagens entre parênteses.", "",
          "| Método | Medida | " + " | ".join(ordem) + " |", "|---|---|" + "---:|" * len(ordem)]
    for n in ["bruto_referencia"] + lo.METODOS:
        pc = s["metricas"][n]["por_condicao"]
        for rot, ch in (("Det alvo", "det_alvo"), ("FP lateral", "fp_lateral"), ("FP alvo ESP", "fp_alvo_esp")):
            cel = []
            for c in ordem:
                v = pc[c][ch]
                cel.append("-" if v is None else f"{lo.pct(v, 88)} ({v})")
            md.append(f"| {n} | {rot} | " + " | ".join(cel) + " |")
    with open(os.path.join(PASTA, "tabelas.md"), "a") as f:
        f.write("\n".join(md) + "\n")


original = lo.escrever_md


def main():
    ep.ORDEM_CONDICOES = ORDEM  # montar_estrutura e tabelas leem este atributo na chamada
    assert len(ORDEM) == ep.N_COND and ORDEM[-1] == "ESP"
    ep.por_protocolo, ep.metricas_metodo = por_protocolo, metricas_metodo
    lo.PASTA, lo.CABECALHO, lo.linha, lo.escrever_md = PASTA, CABECALHO, linha, escrever_md
    lo.main()


if __name__ == "__main__":
    main()
