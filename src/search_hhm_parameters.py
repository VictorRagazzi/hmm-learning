"""Busca exploratoria de parametros para a decisao do HMM.

O detector e o Viterbi sao executados uma unica vez. Depois, as combinacoes
abaixo sao avaliadas sobre os caminhos de estados ja calculados. Nenhum
resultado e salvo em arquivo.
"""

from itertools import product

from hhm_inference import DATA_DIR, rodar_inferencia


# ============================================================
# ESPACO DE BUSCA
# ============================================================

MIN_CONSECUTIVE_VALUES = range(1, 11)
MIN_PERCENT_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50,
                      0.60, 0.70, 0.80, 0.90, 1.00)
MODO_REGRA_DECISAO_VALUES = ("OR", "AND")
K_DE_N_VALUES = range(1, 9)


def frequencia_detectada(
    resultado_frequencia,
    min_consecutive,
    min_percent,
    modo_regra_decisao,
):
    """Reavalia uma frequencia sem repetir detector ou Viterbi."""
    criterio_consecutivas = (
        resultado_frequencia["max_consecutivas"] >= min_consecutive
    )
    criterio_percentual = (
        resultado_frequencia["percentual_presente"] >= min_percent
    )

    if modo_regra_decisao == "AND":
        return criterio_consecutivas and criterio_percentual
    if modo_regra_decisao == "OR":
        return criterio_consecutivas or criterio_percentual
    raise ValueError(f"Modo de regra de decisao invalido: {modo_regra_decisao}")


def avaliar_parametros(
    resultados_arquivos,
    min_consecutive,
    min_percent,
    modo_regra_decisao,
    k_de_n,
):
    """Calcula deteccao, falso positivo lateral e acuracia balanceada."""
    deteccoes = 0
    falsos_positivos = 0

    for resultado_arquivo in resultados_arquivos:
        n_estimulo = sum(
            frequencia_detectada(
                resultado_frequencia,
                min_consecutive,
                min_percent,
                modo_regra_decisao,
            )
            for resultado_frequencia in (
                resultado_arquivo["por_frequencia_estimulo"].values()
            )
        )
        n_lateral = sum(
            frequencia_detectada(
                resultado_frequencia,
                min_consecutive,
                min_percent,
                modo_regra_decisao,
            )
            for resultado_frequencia in (
                resultado_arquivo["por_frequencia_lateral"].values()
            )
        )
        deteccoes += n_estimulo >= k_de_n
        falsos_positivos += n_lateral >= k_de_n

    n_arquivos = len(resultados_arquivos)
    taxa_deteccao = deteccoes / n_arquivos
    taxa_falso_positivo = falsos_positivos / n_arquivos
    especificidade = 1.0 - taxa_falso_positivo
    acuracia_balanceada = (taxa_deteccao + especificidade) / 2.0

    return {
        "taxa_deteccao": taxa_deteccao,
        "taxa_falso_positivo": taxa_falso_positivo,
        "acuracia_balanceada": acuracia_balanceada,
    }


def buscar_melhores_parametros(pasta_dados=DATA_DIR):
    """Executa a busca em grade e retorna somente a melhor combinacao."""
    resultados_arquivos = rodar_inferencia(pasta_dados)
    combinacoes = product(
        MIN_CONSECUTIVE_VALUES,
        MIN_PERCENT_VALUES,
        MODO_REGRA_DECISAO_VALUES,
        K_DE_N_VALUES,
    )
    total = (
        len(MIN_CONSECUTIVE_VALUES)
        * len(MIN_PERCENT_VALUES)
        * len(MODO_REGRA_DECISAO_VALUES)
        * len(K_DE_N_VALUES)
    )

    melhor = None
    melhor_chave = None

    for iteracao, parametros in enumerate(combinacoes, start=1):
        print(f"\rIteracao {iteracao}/{total}", end="", flush=True)
        min_consecutive, min_percent, modo, k_de_n = parametros
        metricas = avaliar_parametros(
            resultados_arquivos,
            min_consecutive,
            min_percent,
            modo,
            k_de_n,
        )

        # Prioriza a acuracia balanceada. Em empate, prefere menor falso
        # positivo lateral e, depois, maior taxa de deteccao.
        chave = (
            metricas["acuracia_balanceada"],
            -metricas["taxa_falso_positivo"],
            metricas["taxa_deteccao"],
        )
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor = {
                "iteracao": iteracao,
                "min_consecutive": min_consecutive,
                "min_percent": min_percent,
                "modo_regra_decisao": modo,
                "k_de_n": k_de_n,
                **metricas,
            }

    print()
    return melhor, total


def imprimir_melhor_resultado(melhor, total):
    print("\nMelhor iteracao")
    print(f"Iteracao: {melhor['iteracao']}/{total}")
    print(f"MIN_CONSECUTIVE: {melhor['min_consecutive']}")
    print(f"MIN_PERCENT: {melhor['min_percent']:.2f}")
    print(f"MODO_REGRA_DECISAO: {melhor['modo_regra_decisao']}")
    print(f"K_DE_N: {melhor['k_de_n']}")
    print(f"Acuracia balanceada: {melhor['acuracia_balanceada']:.2%}")
    print(f"Taxa de deteccao: {melhor['taxa_deteccao']:.2%}")
    print(
        "Taxa de falso positivo lateral: "
        f"{melhor['taxa_falso_positivo']:.2%}"
    )


if __name__ == "__main__":
    melhor_resultado, n_iteracoes = buscar_melhores_parametros(DATA_DIR)
    imprimir_melhor_resultado(melhor_resultado, n_iteracoes)
