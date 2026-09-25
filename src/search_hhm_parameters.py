"""Busca exploratoria de parametros para a decisao do HMM.

Alem da regra de decisao (MIN_CONSECUTIVE, MIN_PERCENT, MODO_REGRA_DECISAO),
esta versao tambem varia:

  - PI_INICIAL: distribuicao inicial usada pelo Viterbi.
  - K_SINTETICO: amplitude do tom sintetico usado para calibrar B(Presente)
    em get_obs_matrix.py.

CACHE EM 3 NIVEIS (do mais caro para o mais barato), para nao recalcular o
que nao precisa:

  1. Deteccao bruta (labels/p-values) nos arquivos *dB.mat: NAO depende de
     K_SINTETICO nem de PI_INICIAL -- a injecao sintetica so acontece na
     calibracao de B, sobre os arquivos ESP, nunca sobre os arquivos de
     estimulo. Por isso e calculada UMA UNICA VEZ para toda a busca.
  2. Matriz B: depende de K_SINTETICO (via
     get_obs_matrix.construir_matrizes_observacao, que reprocessa os
     arquivos ESP). Recalculada uma vez por valor de K_SINTETICO testado.
  3. Viterbi + regra de decisao: depende de (B, PI_INICIAL,
     MIN_CONSECUTIVE, MIN_PERCENT, MODO_REGRA_DECISAO). O Viterbi roda uma
     vez por combinacao (K_SINTETICO, PI_INICIAL); a regra de decisao e
     reavaliada sem custo extra, em cima do mesmo caminho de estados, para
     todas as combinacoes de MIN_CONSECUTIVE/MIN_PERCENT/MODO -- exatamente
     como no script original.

O "teto do detector bruto sozinho" nao depende de B nem de Viterbi (ele nao
usa nenhum dos dois), entao tambem e calculado uma unica vez, fora do loop
principal, em vez de ser recalculado a cada combinacao de K_SINTETICO/PI_INICIAL.

EXCLUSAO AUTOMATICA POR FP (novidade pedida): combinacoes do HMM com taxa de
falso positivo lateral acima de FP_MAXIMO_HMM sao descartadas da selecao do
melhor resultado, mesmo que teriam a melhor acuracia balanceada. Isso e
importante porque acuracia_balanceada = (deteccao + especificidade) / 2
pondera deteccao e FP igualmente, entao uma combinacao com deteccao muito
alta e FP tambem alto pode "ganhar" de uma combinacao mais segura na hora do
desempate por acuracia balanceada. O corte automatico evita isso, cortando
antes da comparacao. Ele NAO se aplica ao teto do detector bruto sozinho, que
continua sendo reportado sem filtro, como referencia do que da pra tirar do
detector sem HMM.

Nenhum resultado e salvo em arquivo (mesmo comportamento do script original).
"""

import glob
import os
from itertools import product

import numpy as np

import get_obs_matrix as gom

# ATENCAO: no script original o import era `from hhm_inference import ...`
# (h-h-m em vez de h-m-m). Mantive o mesmo nome de modulo aqui para nao
# quebrar o que ja funciona no seu projeto -- confira se e mesmo o nome do
# arquivo (parece um typo, mas so voce sabe se foi proposital).
import hhm_inference as hmi


# ============================================================
# ESPACO DE BUSCA
# ============================================================

MIN_CONSECUTIVE_VALUES = range(1, 13)
MIN_PERCENT_VALUES = (0.01, 0.025, 0.05, 0.075, 0.09,
                       0.10, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80, 0.90, 1.00)
MODO_REGRA_DECISAO_VALUES = ("OR", "AND")

# Cada tupla deve somar 1 e seguir a mesma ordem de gom.ESTADOS
# (["Ausente", "Presente"]).
PI_INICIAL_VALUES = (
    (0.99, 0.01),
    (0.95, 0.05),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.40, 0.60),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.30, 0.70),
    (0.20, 0.80),
    (0.10, 0.90),
    (0.05, 0.95),
    (0.01, 0.99),
)

# Amplitude do tom sintetico (em unidades de desvio-padrao da epoca) usada
# para calibrar B(Presente) em get_obs_matrix.py.
K_SINTETICO_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)
# K_SINTETICO_VALUES = (0.01,)

# Corte automatico de falso positivo, SO para a selecao do melhor resultado
# do HMM (nao afeta o teto do detector bruto sozinho).
FP_MAXIMO_HMM = 0.125  # 12.5%, meio do intervalo 10-15% pedido


# ============================================================
# VALIDACAO
# ============================================================

def validar_espaco_busca():
    for pi in PI_INICIAL_VALUES:
        if len(pi) != len(gom.ESTADOS):
            raise ValueError(
                f"PI_INICIAL {pi} tem tamanho diferente de ESTADOS {gom.ESTADOS}"
            )
        if not np.isclose(sum(pi), 1.0):
            raise ValueError(f"PI_INICIAL {pi} nao soma 1")
    if any(k <= 0 for k in K_SINTETICO_VALUES):
        raise ValueError("K_SINTETICO_VALUES deve conter apenas valores positivos")
    if not 0 < FP_MAXIMO_HMM < 1:
        raise ValueError("FP_MAXIMO_HMM deve estar entre 0 e 1")


# ============================================================
# NIVEL 1 - DETECCAO BRUTA (independente de K_SINTETICO e PI_INICIAL)
# ============================================================

def computar_deteccao_bruta(pasta_dados):
    """Roda o detector (sem Viterbi, sem B) uma unica vez, em todos os
    arquivos *dB.mat, nas frequencias de estimulo e nas laterais.

    Reusa exatamente as funcoes de leitura/deteccao de get_obs_matrix.py e
    hmm_inference.py -- nenhuma logica de deteccao e duplicada aqui.
    """
    thresholds = gom.calcular_thresholds_msc(
        gom.WINDOW_SIZE_EPOCHS, gom.P_VALUE_BOUNDARIES
    )

    caminhos = sorted(glob.glob(os.path.join(pasta_dados, hmi.PRESENT_PATTERN)))
    if not caminhos:
        raise RuntimeError(
            f"Nenhum arquivo encontrado em {pasta_dados} com padrao "
            f"{hmi.PRESENT_PATTERN}"
        )

    arquivos_processados = []
    for caminho in caminhos:
        participante, condicao, nivel_db = hmi.parse_nome_arquivo(caminho)
        dados_mat = hmi.carregar_mat_estimulo(caminho)

        por_frequencia_estimulo = {}
        for freq_alvo in dados_mat["freq_estim"]:
            _labels, _msc, p_values, _intervalos = hmi.construir_sequencia_labels(
                dados_mat["x"], gom.CHANNEL_INDEX, dados_mat["fs"], freq_alvo,
                thresholds,
            )
            por_frequencia_estimulo[float(freq_alvo)] = {"p_values": p_values}

        por_frequencia_lateral = {}
        for freq_lateral in dados_mat["bins_m"]:
            _labels, _msc, p_values, _intervalos = hmi.construir_sequencia_labels(
                dados_mat["x"], gom.CHANNEL_INDEX, dados_mat["fs"], freq_lateral,
                thresholds,
            )
            por_frequencia_lateral[float(freq_lateral)] = {"p_values": p_values}

        # As labels acima ja usam o detector (MSC) e os thresholds fixos;
        # guardamos so os p_values aqui porque, para o Viterbi, precisamos
        # recomputar as labels caso os thresholds mudem -- mas como
        # WINDOW_SIZE_EPOCHS/P_VALUE_BOUNDARIES sao fixos nesta busca,
        # guardamos tambem as labels para reusar no Viterbi sem refazer FFT.
        for freq_alvo in dados_mat["freq_estim"]:
            labels, _msc, _p, _int = hmi.construir_sequencia_labels(
                dados_mat["x"], gom.CHANNEL_INDEX, dados_mat["fs"], freq_alvo,
                thresholds,
            )
            por_frequencia_estimulo[float(freq_alvo)]["labels"] = labels
        for freq_lateral in dados_mat["bins_m"]:
            labels, _msc, _p, _int = hmi.construir_sequencia_labels(
                dados_mat["x"], gom.CHANNEL_INDEX, dados_mat["fs"], freq_lateral,
                thresholds,
            )
            por_frequencia_lateral[float(freq_lateral)]["labels"] = labels

        arquivos_processados.append({
            "arquivo": caminho,
            "participante": participante,
            "nivel_db": nivel_db,
            "por_frequencia_estimulo": por_frequencia_estimulo,
            "por_frequencia_lateral": por_frequencia_lateral,
        })

    return arquivos_processados


# ============================================================
# NIVEL 2 - MATRIZ B PARA UM K_SINTETICO ESPECIFICO
# ============================================================

def construir_matriz_b(pasta_dados, k_sintetico):
    """Recalcula B(Ausente)/B(Presente) com um K_SINTETICO especifico.

    K_SINTETICO e lido, dentro de get_obs_matrix.construir_matrizes_observacao
    (via injetar_tom_sintetico), como uma variavel do modulo get_obs_matrix.
    Por isso sobrescrevemos o atributo do modulo antes de chamar a funcao
    original, em vez de duplicar a logica de calibracao de B aqui.
    """
    valor_original = gom.K_SINTETICO
    try:
        gom.K_SINTETICO = k_sintetico
        resultados = gom.construir_matrizes_observacao(pasta_dados)
    finally:
        gom.K_SINTETICO = valor_original

    matriz_global = resultados["global"]
    matriz_b = np.asarray(
        [
            [matriz_global[estado][label] for label in gom.NIVEIS_OBSERVACAO]
            for estado in gom.ESTADOS
        ],
        dtype=float,
    )
    return matriz_b


# ============================================================
# NIVEL 3 - VITERBI (por combinacao K_SINTETICO x PI_INICIAL)
# ============================================================

def _stats_binaria(binaria):
    """Estatisticas de uma sequencia booleana, independentes de qualquer
    limiar de decisao (MIN_CONSECUTIVE/MIN_PERCENT sao aplicados depois,
    em cima destes numeros, sem custo de recomputo)."""
    if len(binaria) == 0:
        return {"n_janelas": 0, "max_consecutivas": 0, "percentual_presente": 0.0}
    return {
        "n_janelas": len(binaria),
        "max_consecutivas": hmi.maior_sequencia_consecutiva_binaria(binaria),
        "percentual_presente": sum(binaria) / len(binaria),
    }


def rodar_viterbi_para_combinacao(arquivos_deteccao, matriz_a, matriz_b, pi_inicial):
    """Roda Viterbi com um (B, PI_INICIAL) fixos, reusando as labels/p_values
    ja calculados na deteccao bruta (nivel 1) -- nenhum FFT e refeito aqui."""

    def _avaliar_grupo(grupo):
        saida = {}
        for freq, dados in grupo.items():
            caminho = hmi.viterbi(
                dados["labels"], matriz_a, matriz_b, pi_inicial,
                gom.ESTADOS, gom.NIVEIS_OBSERVACAO,
            )
            binaria_hmm = [estado == "Presente" for estado in caminho]
            binaria_bruto = [
                p <= gom.SIGNIFICANCE_LEVEL for p in dados["p_values"]
            ]
            saida[freq] = {
                **_stats_binaria(binaria_hmm),
                "deteccao_detector_bruto": _stats_binaria(binaria_bruto),
            }
        return saida

    resultado = []
    for arquivo in arquivos_deteccao:
        resultado.append({
            "por_frequencia_estimulo": _avaliar_grupo(arquivo["por_frequencia_estimulo"]),
            "por_frequencia_lateral": _avaliar_grupo(arquivo["por_frequencia_lateral"]),
        })
    return resultado


def _montar_resultado_apenas_bruto(arquivos_deteccao):
    """Mesmo formato de rodar_viterbi_para_combinacao, mas sem rodar Viterbi
    -- usado so para o teto do detector bruto sozinho (nao depende de B nem
    de PI_INICIAL)."""

    def _grupo(g):
        saida = {}
        for freq, dados in g.items():
            binaria_bruto = [
                p <= gom.SIGNIFICANCE_LEVEL for p in dados["p_values"]
            ]
            saida[freq] = {"deteccao_detector_bruto": _stats_binaria(binaria_bruto)}
        return saida

    return [
        {
            "por_frequencia_estimulo": _grupo(arquivo["por_frequencia_estimulo"]),
            "por_frequencia_lateral": _grupo(arquivo["por_frequencia_lateral"]),
        }
        for arquivo in arquivos_deteccao
    ]


# ============================================================
# REGRA DE DECISAO (pos-hoc, sem recomputo -- igual ao script original)
# ============================================================

def frequencia_detectada(resultado_frequencia, min_consecutive, min_percent, modo):
    c = resultado_frequencia["max_consecutivas"] >= min_consecutive
    p = resultado_frequencia["percentual_presente"] >= min_percent
    if modo == "AND":
        return c and p
    if modo == "OR":
        return c or p
    raise ValueError(f"Modo de regra de decisao invalido: {modo}")


def frequencia_detectada_bruto(resultado_frequencia, min_consecutive, min_percent, modo):
    bruto = resultado_frequencia["deteccao_detector_bruto"]
    c = bruto["max_consecutivas"] >= min_consecutive
    p = bruto["percentual_presente"] >= min_percent
    if modo == "AND":
        return c and p
    if modo == "OR":
        return c or p
    raise ValueError(f"Modo de regra de decisao invalido: {modo}")


def avaliar_parametros(resultados_arquivos, min_consecutive, min_percent, modo,
                        usar_bruto=False):
    """Deteccao, falso positivo lateral e acuracia balanceada, tratando cada
    frequencia (de estimulo ou lateral), em qualquer arquivo, como um
    experimento independente."""
    funcao = frequencia_detectada_bruto if usar_bruto else frequencia_detectada

    n_estimulo_total = 0
    n_estimulo_detectadas = 0
    n_lateral_total = 0
    n_lateral_detectadas = 0

    for resultado_arquivo in resultados_arquivos:
        for resultado_frequencia in resultado_arquivo["por_frequencia_estimulo"].values():
            n_estimulo_total += 1
            n_estimulo_detectadas += funcao(
                resultado_frequencia, min_consecutive, min_percent, modo
            )
        for resultado_frequencia in resultado_arquivo["por_frequencia_lateral"].values():
            n_lateral_total += 1
            n_lateral_detectadas += funcao(
                resultado_frequencia, min_consecutive, min_percent, modo
            )

    taxa_deteccao = n_estimulo_detectadas / n_estimulo_total
    taxa_falso_positivo = n_lateral_detectadas / n_lateral_total
    especificidade = 1.0 - taxa_falso_positivo
    acuracia_balanceada = (taxa_deteccao + especificidade) / 2.0

    return {
        "n_frequencias_estimulo": n_estimulo_total,
        "n_frequencias_estimulo_detectadas": n_estimulo_detectadas,
        "taxa_deteccao": taxa_deteccao,
        "n_frequencias_laterais": n_lateral_total,
        "n_frequencias_laterais_detectadas": n_lateral_detectadas,
        "taxa_falso_positivo": taxa_falso_positivo,
        "acuracia_balanceada": acuracia_balanceada,
    }


# ============================================================
# BUSCA PRINCIPAL
# ============================================================

def buscar_melhores_parametros(pasta_dados=gom.DATA_DIR):
    validar_espaco_busca()

    matriz_a = hmi.carregar_matriz_transicao()
    arquivos_deteccao = computar_deteccao_bruta(pasta_dados)

    combinacoes_regra = list(product(
        MIN_CONSECUTIVE_VALUES, MIN_PERCENT_VALUES, MODO_REGRA_DECISAO_VALUES,
    ))

    # --- Teto do detector bruto sozinho: nao depende de K_SINTETICO/PI_INICIAL,
    # calculado uma unica vez, fora do loop principal. ---
    resultado_apenas_bruto = _montar_resultado_apenas_bruto(arquivos_deteccao)
    melhor_bruto_isolado = None
    melhor_bruto_isolado_chave = None
    for min_consecutive, min_percent, modo in combinacoes_regra:
        metricas_bruto = avaliar_parametros(
            resultado_apenas_bruto, min_consecutive, min_percent, modo,
            usar_bruto=True,
        )
        chave_bruto = (
            metricas_bruto["acuracia_balanceada"],
            -metricas_bruto["taxa_falso_positivo"],
            metricas_bruto["taxa_deteccao"],
        )
        if melhor_bruto_isolado_chave is None or chave_bruto > melhor_bruto_isolado_chave:
            melhor_bruto_isolado_chave = chave_bruto
            melhor_bruto_isolado = {
                "min_consecutive": min_consecutive,
                "min_percent": min_percent,
                "modo_regra_decisao": modo,
                **metricas_bruto,
            }

    # --- Busca principal do HMM: K_SINTETICO (recalibra B) x PI_INICIAL
    # (Viterbi) x regra de decisao (pos-hoc), com corte automatico de FP. ---
    melhor = None
    melhor_chave = None
    total_topo = len(K_SINTETICO_VALUES) * len(PI_INICIAL_VALUES)
    combo_topo_atual = 0
    total_avaliadas = 0
    total_excluidas_por_fp = 0

    for k_sintetico in K_SINTETICO_VALUES:
        matriz_b = construir_matriz_b(pasta_dados, k_sintetico)

        for pi_inicial in PI_INICIAL_VALUES:
            combo_topo_atual += 1
            print(
                f"\rK_SINTETICO={k_sintetico:<6} PI_INICIAL={pi_inicial}  "
                f"({combo_topo_atual}/{total_topo})",
                end="", flush=True,
            )
            resultado_viterbi = rodar_viterbi_para_combinacao(
                arquivos_deteccao, matriz_a, matriz_b, pi_inicial,
            )

            for min_consecutive, min_percent, modo in combinacoes_regra:
                total_avaliadas += 1
                metricas = avaliar_parametros(
                    resultado_viterbi, min_consecutive, min_percent, modo,
                    usar_bruto=False,
                )

                if metricas["taxa_falso_positivo"] > FP_MAXIMO_HMM:
                    total_excluidas_por_fp += 1
                    continue

                chave = (
                    metricas["acuracia_balanceada"],
                    -metricas["taxa_falso_positivo"],
                    metricas["taxa_deteccao"],
                )
                if melhor_chave is None or chave > melhor_chave:
                    melhor_chave = chave
                    melhor = {
                        "k_sintetico": k_sintetico,
                        "pi_inicial": pi_inicial,
                        "min_consecutive": min_consecutive,
                        "min_percent": min_percent,
                        "modo_regra_decisao": modo,
                        **metricas,
                    }

    print()
    if melhor is None:
        raise RuntimeError(
            "Nenhuma combinacao ficou com taxa_falso_positivo <= "
            f"FP_MAXIMO_HMM ({FP_MAXIMO_HMM:.2%}); aumente o limite ou "
            "amplie o espaco de busca (K_SINTETICO_VALUES / PI_INICIAL_VALUES "
            "/ MIN_CONSECUTIVE_VALUES / MIN_PERCENT_VALUES)."
        )

    return {
        "melhor": melhor,
        "melhor_bruto_isolado": melhor_bruto_isolado,
        "total_avaliadas": total_avaliadas,
        "total_excluidas_por_fp": total_excluidas_por_fp,
        "total_combinacoes_regra": len(combinacoes_regra),
    }


# ============================================================
# IMPRESSAO
# ============================================================

def imprimir_melhor_resultado(resumo):
    melhor = resumo["melhor"]
    melhor_bruto_isolado = resumo["melhor_bruto_isolado"]

    print("\nMelhor combinacao (HMM)")
    print(f"K_SINTETICO: {melhor['k_sintetico']}")
    print(f"PI_INICIAL: {melhor['pi_inicial']}")
    print(f"MIN_CONSECUTIVE: {melhor['min_consecutive']}")
    print(f"MIN_PERCENT: {melhor['min_percent']:.3f}")
    print(f"MODO_REGRA_DECISAO: {melhor['modo_regra_decisao']}")
    print(f"Acuracia balanceada: {melhor['acuracia_balanceada']:.2%}")
    print(
        f"Taxa de deteccao: {melhor['taxa_deteccao']:.2%} "
        f"({melhor['n_frequencias_estimulo_detectadas']}/"
        f"{melhor['n_frequencias_estimulo']})"
    )
    print(
        "Taxa de falso positivo lateral: "
        f"{melhor['taxa_falso_positivo']:.2%} "
        f"({melhor['n_frequencias_laterais_detectadas']}/"
        f"{melhor['n_frequencias_laterais']})  "
        f"[limite: {FP_MAXIMO_HMM:.2%}]"
    )

    print("\nMelhor combinacao possivel usando SO o detector bruto (teto do detector sozinho)")
    print(f"MIN_CONSECUTIVE: {melhor_bruto_isolado['min_consecutive']}")
    print(f"MIN_PERCENT: {melhor_bruto_isolado['min_percent']:.3f}")
    print(f"MODO_REGRA_DECISAO: {melhor_bruto_isolado['modo_regra_decisao']}")
    print(f"Acuracia balanceada: {melhor_bruto_isolado['acuracia_balanceada']:.2%}")
    print(
        f"Taxa de deteccao: {melhor_bruto_isolado['taxa_deteccao']:.2%} "
        f"({melhor_bruto_isolado['n_frequencias_estimulo_detectadas']}/"
        f"{melhor_bruto_isolado['n_frequencias_estimulo']})"
    )
    print(
        "Taxa de falso positivo lateral: "
        f"{melhor_bruto_isolado['taxa_falso_positivo']:.2%} "
        f"({melhor_bruto_isolado['n_frequencias_laterais_detectadas']}/"
        f"{melhor_bruto_isolado['n_frequencias_laterais']})"
    )

    ganho = melhor["acuracia_balanceada"] - melhor_bruto_isolado["acuracia_balanceada"]
    print(
        f"\nGanho do HMM sobre o teto do detector bruto sozinho: "
        f"{ganho:+.2%} (acuracia balanceada)"
    )

    print(
        f"\nCombinacoes avaliadas para o HMM: {resumo['total_avaliadas']} "
        f"(regra de decisao x {resumo['total_combinacoes_regra']}); "
        f"{resumo['total_excluidas_por_fp']} descartadas por "
        f"FP > {FP_MAXIMO_HMM:.2%}."
    )


if __name__ == "__main__":
    resumo = buscar_melhores_parametros(gom.DATA_DIR)
    imprimir_melhor_resultado(resumo)