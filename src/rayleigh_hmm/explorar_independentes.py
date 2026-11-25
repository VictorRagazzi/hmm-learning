"""Exploratorio: LOSO com Rayleigh em blocos INDEPENDENTES de M epocas (sem acumular).

Mesma validacao oficial (explorar_loso.main, sinal FC, teto de FP por fold = FP lateral
do Rayleigh ACUMULADO de referencia nos participantes de treino), mas o Z de cada passo
usa so as M epocas do proprio bloco (sem sobreposicao). A emissao Presente recebe N = M
constante. Rodado para M em VALORES_M; nenhuma regra dispara antes de TEMPO_MINIMO_S.
Patches so em memoria (dados.rayleigh_z_acumulado, parametros de experimento e
explorar_loso), restaurados no final. Nao reexecuta o oficial: le o resultados.json dele.
Saidas: results/rayleigh_hmm/exploratorio_independentes/{M10,M30}/, resumo.md, resumo.json
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_independentes.py
"""
import json
import os

import numpy as np

import dados
import experimento as ex
import explorar_loso as el

# ---------------- configuracao ----------------
VALORES_M = (10, 30)
TEMPO_MINIMO_S = 30  # nenhuma regra dispara antes de 30 s (como no oficial)
PASTA_BASE = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_independentes")
JSON_OFICIAL = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_loso_fc", "resultados.json")
INTENSIDADES = el.INTENSIDADES
NOME_OFICIAL = "Rayleigh acumulado a=0,05, 3 consec. (referencia oficial)"
NOME_BLOCOS = "Rayleigh blocos a=0,05, 3 consec."


def rayleigh_z_blocos(janelas, valida_janelas=None):
    """Como dados.rayleigh_z_acumulado, mas sem acumular: cada bloco usa so as proprias epocas.

    Retorna z (n_janelas, n_freqs) e n (n_janelas,) = epocas validas do bloco.
    """
    n_janelas, m, _ = janelas.shape
    if valida_janelas is None:
        valida_janelas = np.ones((n_janelas, m), dtype=bool)
    fases = janelas / (np.abs(janelas) + dados.EPS) * valida_janelas[:, :, None]
    soma = fases.sum(axis=1)
    n_validas = valida_janelas.sum(axis=1)
    z = np.abs(soma) ** 2 / np.maximum(n_validas, 1)[:, None]
    return z, n_validas


def calcular_tetos(seq):
    """Teto por fold (FP lateral do Rayleigh acumulado de referencia no treino) e metricas pooled."""
    tetos = {}
    for h in sorted({s["participante"] for s in seq}):
        tr = [s for s in seq if s["participante"] != h]
        tetos[h] = ex.calcular_referencia(tr, ex.mascaras(tr))["metricas"]["fp_lateral"]
    masc = ex.mascaras(seq)
    ref = ex.calcular_referencia(seq, masc)
    pooled = ex.metricas(masc, ref["detectado"], ref["tempo_s"])
    por_inten = {c: ex.resumo(seq, ref, lambda s, c=c: s["condicao"] == c) for c in INTENSIDADES}
    return tetos, pooled, por_inten


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def seg(x):
    return "-" if x is None else f"{x:.1f}".replace(".", ",")


def linha(nome, m):
    return (f"| {nome} | {pct(m['deteccao'])} | {pct(m['fp_lateral'])} | {pct(m['fp_alvo_esp'])} | "
            f"{seg(m['tempo_medio_s'])} / {seg(m['tempo_mediano_s'])} |")


def frac(r):
    return None if not r["deteccao_total"] else r["deteccao_detectados"] / r["deteccao_total"]


def escrever_resumo(minimos, ref_acum_pooled):
    oficial = json.load(open(JSON_OFICIAL))
    por_m = {M: json.load(open(os.path.join(PASTA_BASE, f"M{M}", "resultados.json"))) for M in VALORES_M}
    geral = ["| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio / mediano (s) |", "|---|---|---|---|---|",
             linha("Oficial acumulado: HMM-REAL", oficial["pooled"]["HMM-REAL"]),
             linha(NOME_OFICIAL, oficial["pooled"]["bruto_referencia"])]
    nomes = [("HMM-REAL", "HMM-REAL"), ("HMM-SINTETICO", "HMM-SINTETICO"),
             ("Rayleigh otimizado LOSO (blocos)", "bruto_otimizado_LOSO"), (NOME_BLOCOS, "bruto_referencia")]
    for M in VALORES_M:
        for rotulo, chave in nomes:
            geral.append(linha(f"M={M}: {rotulo}", por_m[M]["pooled"][chave]))
    inten = ["| Intensidade | Oficial acumulado | " + " | ".join(f"M={M} blocos" for M in VALORES_M) + " |",
             "|---|---|" + "---|" * len(VALORES_M)]
    for c in INTENSIDADES:
        cel = [pct(frac(oficial["por_intensidade"][f"{c}/HMM-REAL"]))]
        cel += [pct(frac(por_m[M]["por_intensidade"][f"{c}/HMM-REAL"])) for M in VALORES_M]
        inten.append(f"| {c.replace('dB', ' dB')} | " + " | ".join(cel) + " |")
    md = ["# Rayleigh em blocos independentes x acumulado (LOSO, exploratorio)\n",
          "Exploratorio, validacao LOSO (fora da amostra); teto de FP lateral por fold = o oficial "
          "(Rayleigh acumulado de referencia no treino). Cada avaliacao usa so as M epocas do bloco. "
          "MINIMO_AVALIACOES por M: " + ", ".join(f"M={M} -> {minimos[M]}" for M in VALORES_M) +
          ". Em M=30 a gravacao de 70 dB tem so 2 avaliacoes.\n",
          "## Geral (pooled)\n", *geral, "\n## Deteccao do HMM-REAL por intensidade\n", *inten]
    os.makedirs(PASTA_BASE, exist_ok=True)
    open(os.path.join(PASTA_BASE, "resumo.md"), "w").write("\n".join(md) + "\n")
    saida = {"oficial_pooled": oficial["pooled"], "oficial_por_intensidade": oficial["por_intensidade"],
             "referencia_acumulada_recalculada": ref_acum_pooled, "minimo_avaliacoes": minimos,
             "por_M": {str(M): {"pooled": por_m[M]["pooled"], "por_intensidade": por_m[M]["por_intensidade"]}
                       for M in VALORES_M}}
    with open(os.path.join(PASTA_BASE, "resumo.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    print("\n".join(md))


def main():
    # 1) tudo que depende do acumulado, ANTES de qualquer patch
    seq_acum, _ = ex.carregar_sequencias()
    tetos, ref_pooled, _ = calcular_tetos(seq_acum)
    n_acum = len(seq_acum)

    def teto_oficial(seq_tr, masc_tr):
        h, = set(tetos) - {s["participante"] for s in seq_tr}
        return tetos[h]

    originais = (dados.rayleigh_z_acumulado, ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES,
                 el.PASTA, el.IN_SAMPLE, el.teto_fp)
    minimos = {}
    try:
        for M in VALORES_M:
            minimos[M] = max(1, TEMPO_MINIMO_S // M)
            dados.rayleigh_z_acumulado = rayleigh_z_blocos
            ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES = M, minimos[M]
            el.PASTA = os.path.join(PASTA_BASE, f"M{M}")
            el.IN_SAMPLE = {}
            el.teto_fp = teto_oficial
            el.main()
            assert len(el.SEQ) == n_acum, f"M={M}: {len(el.SEQ)} sequencias != {n_acum} do acumulado"
    finally:
        (dados.rayleigh_z_acumulado, ex.EPOCAS_POR_JANELA, ex.MINIMO_AVALIACOES,
         el.PASTA, el.IN_SAMPLE, el.teto_fp) = originais
    escrever_resumo(minimos, ref_pooled)


if __name__ == "__main__":
    main()
