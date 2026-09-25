"""Inferencia de estados (Viterbi) e regra de decisao para deteccao de ASSR.

Este modulo NAO reimplementa o detector espectral. Ele importa diretamente de
``get_obs_matrix.py`` as mesmas funcoes usadas para calibrar a matriz B
(``carregar_mat``, ``extrair_coeficientes_epocas``, ``calcular_msc_janelas``,
``calcular_thresholds_msc``, ``discretizar``).

Fluxo:

1. Carrega A (results/transition_matrix.json) e B (results/observation_matrix.json)
   ja calibradas.
2. Para cada arquivo com estimulo (*dB.mat), decodifica com Viterbi uma
   sequencia de estados POR FREQUENCIA de estimulacao (8 sequencias
   independentes, pois B e global mas as sequencias temporais de frequencias
   diferentes nao devem ser concatenadas).
3. Aplica uma regra de decisao (percentual minimo e/ou numero de janelas
   consecutivas em "Presente") sobre cada caminho de estados -> detectado_freq.
4. Em paralelo, aplica a MESMA regra de decisao diretamente sobre a
   significancia do detector bruto (p_value <= SIGNIFICANCE_LEVEL, janela a
   janela), sem passar pelo HMM/Viterbi. Isso serve como baseline: o detector
   configurado em get_obs_matrix.py (generico, hoje MSC, mas pode ser trocado)
   ja e um teste que funciona sozinho, entao o ganho do HMM so faz sentido se
   ficar acima dessa baseline.
5. Trata cada frequencia como um experimento independente ao agregar a taxa de
   deteccao (por exemplo, 1 frequencia detectada em 8 corresponde a 12,5%).
6. Repete o mesmo pipeline (HMM e baseline do detector bruto) diretamente nas
   frequencias laterais de ``binsM``, onde nao ha estimulo esperado, para
   estimar a taxa de falso positivo por frequencia, calibrada nos proprios
   arquivos com estimulo (usar so os ESP omitiria o efeito de qualquer
   artefato do estimulo/gravacao).
7. Salva um relatorio consolidado e imprime as analises por nivel de dB e por
   participante, comparando HMM vs. detector bruto.

Aviso metodologico: A e uma heuristica de persistencia (nao Baum-Welch) e
B(Presente) e uma resposta sintetica (Secao 4.5 do README). Os resultados
desta etapa continuam exploratorios, nao clinicos.
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from get_obs_matrix import (
    CHANNEL_INDEX,
    DATA_DIR,
    DETECTOR,
    ESTADOS,
    NIVEIS_OBSERVACAO,
    P_VALUE_BOUNDARIES,
    PI_INICIAL,
    SIGNIFICANCE_LEVEL,
    WINDOW_SIZE_EPOCHS,
    WINDOW_STEP_EPOCHS,
    calcular_msc_janelas,
    calcular_p_value_msc,
    calcular_thresholds_msc,
    carregar_mat,
    discretizar,
    extrair_coeficientes_epocas,
)

# ============================================================
# CONFIGURACOES
# ============================================================

RESULTS_DIR = "./results"
OBS_MATRIX_FILE = os.path.join(RESULTS_DIR, "observation_matrix.json")
TRAN_MATRIX_FILE = os.path.join(RESULTS_DIR, "transition_matrix.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "inference_analysis.json")

PRESENT_PATTERN = "*dB.mat"

# --- Regra de decisao sobre uma sequencia booleana (janela a janela) ---
# Detecta se a sequencia tiver ao menos MIN_CONSECUTIVE janelas seguidas com
# valor True e/ou se a fracao de janelas True for >= MIN_PERCENT.
# MODO_REGRA_DECISAO combina os dois criterios: "OR" (qualquer um basta) ou
# "AND" (os dois precisam valer). Use None em um dos limiares para
# desativa-lo. A MESMA regra e aplicada tanto ao caminho de estados do HMM
# (estado == "Presente") quanto a significancia bruta do detector
# (p_value <= SIGNIFICANCE_LEVEL), para que a comparacao entre os dois seja
# justa.
MIN_CONSECUTIVE = 3
MIN_PERCENT = 0.05
MODO_REGRA_DECISAO = "OR"  # "OR" ou "AND"

# ============================================================
# CARREGAMENTO DE A E B JA CALIBRADAS
# ============================================================

def carregar_matriz_transicao(caminho=TRAN_MATRIX_FILE):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    estados = dados["estados_linhas"]
    matriz = np.asarray(dados["matriz"], dtype=float)
    if list(estados) != list(ESTADOS):
        raise ValueError(
            f"Ordem de estados em {caminho} ({estados}) difere de ESTADOS "
            f"({ESTADOS}); ajuste a leitura antes de prosseguir"
        )
    if not np.allclose(matriz.sum(axis=1), 1.0):
        raise ValueError(f"{caminho}: linhas de A nao somam 1")
    return matriz


def carregar_matriz_observacao(caminho=OBS_MATRIX_FILE):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    estados = dados["estados_linhas"]
    labels = dados["labels_colunas"]
    matriz = np.asarray(dados["matriz"], dtype=float)
    configuracao = dados.get("configuracao", {})
    detector_arquivo = configuracao.get("detector")
    if detector_arquivo != DETECTOR:
        raise ValueError(
            f"{caminho}: detector {detector_arquivo!r} difere de {DETECTOR!r}; "
            "regenere a matriz B antes da inferencia"
        )
    if configuracao.get("tamanho_janela_epocas") != WINDOW_SIZE_EPOCHS:
        raise ValueError(
            f"{caminho}: tamanho de janela difere do codigo; regenere a matriz B"
        )
    if configuracao.get("passo_janela_epocas") != WINDOW_STEP_EPOCHS:
        raise ValueError(
            f"{caminho}: passo de janela difere do codigo; regenere a matriz B"
        )
    fronteiras_arquivo = configuracao.get("p_value_boundaries")
    if (
        fronteiras_arquivo is None
        or len(fronteiras_arquivo) != len(P_VALUE_BOUNDARIES)
        or not np.allclose(fronteiras_arquivo, P_VALUE_BOUNDARIES)
    ):
        raise ValueError(
            f"{caminho}: fronteiras de p-valor diferem do codigo; "
            "regenere a matriz B"
        )
    if list(estados) != list(ESTADOS):
        raise ValueError(
            f"Ordem de estados em {caminho} ({estados}) difere de ESTADOS "
            f"({ESTADOS}); ajuste a leitura antes de prosseguir"
        )
    if list(labels) != list(NIVEIS_OBSERVACAO):
        raise ValueError(
            f"Ordem de labels em {caminho} ({labels}) difere de "
            f"NIVEIS_OBSERVACAO ({NIVEIS_OBSERVACAO}); "
            "regenere a matriz B ou ajuste a leitura"
        )
    if not np.allclose(matriz.sum(axis=1), 1.0):
        raise ValueError(f"{caminho}: linhas de B nao somam 1")
    return matriz


# ============================================================
# VITERBI
# ============================================================

def viterbi(sequencia_labels, matriz_a, matriz_b, pi_inicial, estados, labels):
    """Decodificacao Viterbi padrao em log-espaco. Retorna lista de estados."""
    n_estados = len(estados)
    n_obs = len(sequencia_labels)
    if n_obs == 0:
        return []

    indices_obs = [labels.index(l) for l in sequencia_labels]
    log_a = np.log(matriz_a + 1e-300)
    log_b = np.log(matriz_b + 1e-300)
    log_pi = np.log(np.asarray(pi_inicial, dtype=float) + 1e-300)

    delta = np.zeros((n_obs, n_estados))
    psi = np.zeros((n_obs, n_estados), dtype=int)

    delta[0] = log_pi + log_b[:, indices_obs[0]]
    for t in range(1, n_obs):
        for j in range(n_estados):
            transicoes = delta[t - 1] + log_a[:, j]
            psi[t, j] = int(np.argmax(transicoes))
            delta[t, j] = transicoes[psi[t, j]] + log_b[j, indices_obs[t]]

    caminho_idx = np.zeros(n_obs, dtype=int)
    caminho_idx[-1] = int(np.argmax(delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        caminho_idx[t] = psi[t + 1, caminho_idx[t + 1]]

    return [estados[i] for i in caminho_idx]


# ============================================================
# SEQUENCIA DE OBSERVACOES PARA UMA FREQUENCIA
# ============================================================

def construir_sequencia_labels(x, canal, fs, frequencia, thresholds):
    """Reusa exatamente o detector de get_obs_matrix.py para gerar labels."""
    coeficientes, _ = extrair_coeficientes_epocas(x, canal, fs, frequencia)
    valores_msc, intervalos = calcular_msc_janelas(coeficientes)
    labels = [
        discretizar(valor, thresholds, NIVEIS_OBSERVACAO)
        for valor in valores_msc
    ]
    p_values = [
        calcular_p_value_msc(valor, WINDOW_SIZE_EPOCHS)
        for valor in valores_msc
    ]
    return labels, valores_msc, p_values, intervalos


# ============================================================
# REGRA DE DECISAO (compartilhada entre HMM e detector bruto)
# ============================================================

def maior_sequencia_consecutiva_binaria(binaria):
    maior = 0
    atual = 0
    for valor in binaria:
        if valor:
            atual += 1
            maior = max(maior, atual)
        else:
            atual = 0
    return maior


def avaliar_binaria(binaria):
    """Aplica a regra de consecutivos e/ou percentual sobre uma sequencia
    booleana janela-a-janela. E' o nucleo comum usado tanto para o caminho de
    estados do HMM (apos comparar com o estado alvo) quanto para a
    significancia bruta do detector (p_value <= SIGNIFICANCE_LEVEL)."""
    if len(binaria) == 0:
        return {
            "n_janelas": 0,
            "max_consecutivas": 0,
            "percentual_presente": 0.0,
            "detectado": False,
        }

    max_consec = maior_sequencia_consecutiva_binaria(binaria)
    percentual = sum(binaria) / len(binaria)

    criterio_consec = (
        MIN_CONSECUTIVE is not None and max_consec >= MIN_CONSECUTIVE
    )
    criterio_percent = (
        MIN_PERCENT is not None and percentual >= MIN_PERCENT
    )

    if MIN_CONSECUTIVE is None and MIN_PERCENT is None:
        raise ValueError("Configure ao menos MIN_CONSECUTIVE ou MIN_PERCENT")
    elif MIN_CONSECUTIVE is None:
        detectado = criterio_percent
    elif MIN_PERCENT is None:
        detectado = criterio_consec
    elif MODO_REGRA_DECISAO == "AND":
        detectado = criterio_consec and criterio_percent
    else:
        detectado = criterio_consec or criterio_percent

    return {
        "n_janelas": len(binaria),
        "max_consecutivas": max_consec,
        "percentual_presente": percentual,
        "criterio_consecutivas_ok": criterio_consec,
        "criterio_percentual_ok": criterio_percent,
        "detectado": detectado,
    }


def avaliar_caminho(caminho, estado_alvo="Presente"):
    """Aplica a regra de decisao sobre um caminho de estados do HMM."""
    binaria = [estado == estado_alvo for estado in caminho]
    return avaliar_binaria(binaria)


def avaliar_detector_bruto(p_values, alpha=SIGNIFICANCE_LEVEL):
    """Aplica a MESMA regra de decisao diretamente sobre a significancia do
    detector bruto (sem HMM/Viterbi): cada janela conta como "positiva" se
    p_value <= alpha. Serve de baseline generica, valida para qualquer
    detector configurado em get_obs_matrix.py (hoje MSC)."""
    binaria = [p <= alpha for p in p_values]
    return avaliar_binaria(binaria)


# ============================================================
# PARSING DE NOME DE ARQUIVO
# ============================================================

def parse_nome_arquivo(caminho):
    """Extrai participante e condicao de nomes tipo 'Ab30dB.mat' / 'AbESP.mat'."""
    nome = os.path.basename(caminho)
    m = re.match(r"^(.*?)(\d+dB|ESP)\.mat$", nome)
    if not m:
        raise ValueError(f"Nao foi possivel interpretar o nome do arquivo: {nome}")
    participante, condicao = m.group(1), m.group(2)
    nivel_db = None
    if condicao != "ESP":
        nivel_db = int(condicao.replace("dB", ""))
    return participante, condicao, nivel_db


# ============================================================
# PROCESSAMENTO DE UM ARQUIVO
# ============================================================

def processar_arquivo(arquivo, matriz_a, matriz_b, thresholds):
    """Roda Viterbi + detector bruto + regra de decisao nas frequencias de
    estimulo e laterais."""
    freq_estim = arquivo["freq_estim"]
    bins_m = arquivo["bins_m"]

    resultado_por_frequencia = {}
    for freq_alvo, freq_controle in zip(freq_estim, bins_m):
        labels, valores_msc, p_values, intervalos = construir_sequencia_labels(
            arquivo["x"], CHANNEL_INDEX, arquivo["fs"], freq_alvo, thresholds
        )
        caminho = viterbi(labels, matriz_a, matriz_b, PI_INICIAL, ESTADOS,
                           NIVEIS_OBSERVACAO)
        avaliacao = avaliar_caminho(caminho)
        avaliacao_bruta = avaliar_detector_bruto(p_values)
        resultado_por_frequencia[float(freq_alvo)] = {
            "frequencia_controle_associada": float(freq_controle),
            "labels": labels,
            "valores_msc": valores_msc.tolist(),
            "p_values": p_values,
            "intervalos_epocas": [list(intervalo) for intervalo in intervalos],
            "caminho_estados": caminho,
            **avaliacao,
            "deteccao_detector_bruto": avaliacao_bruta,
        }

    n_detectadas = sum(
        r["detectado"] for r in resultado_por_frequencia.values()
    )
    n_detectadas_bruto = sum(
        r["deteccao_detector_bruto"]["detectado"]
        for r in resultado_por_frequencia.values()
    )

    resultado_lateral = {}
    for frequencia_lateral in bins_m:
        labels, valores_msc, p_values, intervalos = construir_sequencia_labels(
            arquivo["x"], CHANNEL_INDEX, arquivo["fs"], frequencia_lateral,
            thresholds,
        )
        caminho = viterbi(labels, matriz_a, matriz_b, PI_INICIAL, ESTADOS,
                           NIVEIS_OBSERVACAO)
        avaliacao = avaliar_caminho(caminho)
        avaliacao_bruta = avaliar_detector_bruto(p_values)
        resultado_lateral[float(frequencia_lateral)] = {
            "labels": labels,
            "valores_msc": valores_msc.tolist(),
            "p_values": p_values,
            "intervalos_epocas": [list(intervalo) for intervalo in intervalos],
            "caminho_estados": caminho,
            **avaliacao,
            "deteccao_detector_bruto": avaliacao_bruta,
        }

    n_detectadas_lateral = sum(
        r["detectado"] for r in resultado_lateral.values()
    )
    n_detectadas_lateral_bruto = sum(
        r["deteccao_detector_bruto"]["detectado"]
        for r in resultado_lateral.values()
    )
    return {
        "por_frequencia_estimulo": resultado_por_frequencia,
        "n_frequencias_detectadas": n_detectadas,
        "n_frequencias_detectadas_bruto": n_detectadas_bruto,
        "n_frequencias_total": len(resultado_por_frequencia),
        "por_frequencia_lateral": resultado_lateral,
        "n_lateral_detectadas": n_detectadas_lateral,
        "n_lateral_detectadas_bruto": n_detectadas_lateral_bruto,
        "n_lateral_total": len(resultado_lateral),
    }


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def rodar_inferencia(pasta_dados=DATA_DIR):
    matriz_a = carregar_matriz_transicao()
    matriz_b = carregar_matriz_observacao()
    thresholds = calcular_thresholds_msc(WINDOW_SIZE_EPOCHS, P_VALUE_BOUNDARIES)

    caminhos = sorted(glob.glob(os.path.join(pasta_dados, PRESENT_PATTERN)))
    if not caminhos:
        raise RuntimeError(
            f"Nenhum arquivo encontrado em {pasta_dados} com padrao "
            f"{PRESENT_PATTERN}"
        )

    resultados_arquivos = []
    for caminho in caminhos:
        participante, condicao, nivel_db = parse_nome_arquivo(caminho)
        dados_mat = carregar_mat_estimulo(caminho)
        resultado = processar_arquivo(dados_mat, matriz_a, matriz_b, thresholds)
        resultados_arquivos.append(
            {
                "arquivo": caminho,
                "participante": participante,
                "nivel_db": nivel_db,
                **resultado,
            }
        )

    return resultados_arquivos


def carregar_mat_estimulo(caminho):
    """Como carregar_mat, mas sem exigir sufixo 'ESP' no nome do participante."""
    dados = carregar_mat_generico(caminho)
    return dados


def carregar_mat_generico(caminho):
    """Reaproveita a validacao de carregar_mat, ajustando so o campo 'participante'.

    carregar_mat (de get_obs_matrix.py) assume sufixo 'ESP.mat' para extrair o
    nome do participante. Para arquivos com estimulo (*dB.mat) isso resultaria
    em um nome de participante errado, entao aqui repetimos a leitura e so
    corrigimos esse campo com parse_nome_arquivo.
    """
    dados = carregar_mat(caminho)  # valida x, Fs, freqEstim, binsM, etc.
    participante, condicao, nivel_db = parse_nome_arquivo(caminho)
    dados["participante"] = participante
    dados["condicao"] = condicao
    return dados


# ============================================================
# ANALISES AGREGADAS
# ============================================================

def analisar_por_nivel_db(resultados_arquivos):
    grupos = defaultdict(list)
    for r in resultados_arquivos:
        grupos[r["nivel_db"]].append(r)

    analise = {}
    for nivel_db in sorted(grupos):
        registros = grupos[nivel_db]
        n_frequencias = sum(r["n_frequencias_total"] for r in registros)
        n_detectados = sum(r["n_frequencias_detectadas"] for r in registros)
        n_detectados_bruto = sum(
            r["n_frequencias_detectadas_bruto"] for r in registros
        )
        n_laterais = sum(r["n_lateral_total"] for r in registros)
        n_falsos_positivos = sum(r["n_lateral_detectadas"] for r in registros)
        n_falsos_positivos_bruto = sum(
            r["n_lateral_detectadas_bruto"] for r in registros
        )
        analise[nivel_db] = {
            "n_arquivos": len(registros),
            "n_frequencias_estimulo": n_frequencias,
            "n_frequencias_estimulo_detectadas": n_detectados,
            "taxa_deteccao": n_detectados / n_frequencias,
            "n_frequencias_estimulo_detectadas_detector_bruto": n_detectados_bruto,
            "taxa_deteccao_detector_bruto": n_detectados_bruto / n_frequencias,
            "n_frequencias_laterais": n_laterais,
            "n_frequencias_laterais_detectadas": n_falsos_positivos,
            "taxa_falso_positivo_lateral": n_falsos_positivos / n_laterais,
            "n_frequencias_laterais_detectadas_detector_bruto": n_falsos_positivos_bruto,
            "taxa_falso_positivo_lateral_detector_bruto": (
                n_falsos_positivos_bruto / n_laterais
            ),
        }
    return analise


def analisar_por_participante(resultados_arquivos):
    grupos = defaultdict(list)
    for r in resultados_arquivos:
        grupos[r["participante"]].append(r)

    analise = {}
    for participante in sorted(grupos):
        registros = grupos[participante]
        analise[participante] = {
            "n_arquivos": len(registros),
            "detalhe_por_arquivo": [
                {
                    "arquivo": os.path.basename(r["arquivo"]),
                    "nivel_db": r["nivel_db"],
                    "n_frequencias_estimulo_detectadas": (
                        f"{r['n_frequencias_detectadas']}/"
                        f"{r['n_frequencias_total']}"
                    ),
                    "n_frequencias_estimulo_detectadas_detector_bruto": (
                        f"{r['n_frequencias_detectadas_bruto']}/"
                        f"{r['n_frequencias_total']}"
                    ),
                    "n_frequencias_laterais_detectadas": (
                        f"{r['n_lateral_detectadas']}/{r['n_lateral_total']}"
                    ),
                    "n_frequencias_laterais_detectadas_detector_bruto": (
                        f"{r['n_lateral_detectadas_bruto']}/{r['n_lateral_total']}"
                    ),
                }
                for r in registros
            ],
            "taxa_deteccao_estimulo": sum(
                r["n_frequencias_detectadas"] for r in registros
            ) / sum(r["n_frequencias_total"] for r in registros),
            "taxa_deteccao_estimulo_detector_bruto": sum(
                r["n_frequencias_detectadas_bruto"] for r in registros
            ) / sum(r["n_frequencias_total"] for r in registros),
            "taxa_falso_positivo_lateral": sum(
                r["n_lateral_detectadas"] for r in registros
            ) / sum(r["n_lateral_total"] for r in registros),
            "taxa_falso_positivo_lateral_detector_bruto": sum(
                r["n_lateral_detectadas_bruto"] for r in registros
            ) / sum(r["n_lateral_total"] for r in registros),
        }
    return analise


def construir_relatorio(resultados_arquivos):
    por_nivel = analisar_por_nivel_db(resultados_arquivos)
    por_participante = analisar_por_participante(resultados_arquivos)

    total_frequencias = sum(r["n_frequencias_total"] for r in resultados_arquivos)
    total_detectadas = sum(r["n_frequencias_detectadas"] for r in resultados_arquivos)
    total_detectadas_bruto = sum(
        r["n_frequencias_detectadas_bruto"] for r in resultados_arquivos
    )
    total_laterais = sum(r["n_lateral_total"] for r in resultados_arquivos)
    total_falsos_positivos = sum(r["n_lateral_detectadas"] for r in resultados_arquivos)
    total_falsos_positivos_bruto = sum(
        r["n_lateral_detectadas_bruto"] for r in resultados_arquivos
    )

    return {
        "nome": "relatorio_inferencia_hmm",
        "configuracao": {
            "detector": DETECTOR,
            "tamanho_janela_epocas": WINDOW_SIZE_EPOCHS,
            "passo_janela_epocas": WINDOW_STEP_EPOCHS,
            "min_consecutive": MIN_CONSECUTIVE,
            "min_percent": MIN_PERCENT,
            "modo_regra_decisao": MODO_REGRA_DECISAO,
            "alpha_detector_bruto": SIGNIFICANCE_LEVEL,
        },
        "resumo_global": {
            "n_arquivos": len(resultados_arquivos),
            "n_frequencias_estimulo": total_frequencias,
            "n_frequencias_estimulo_detectadas": total_detectadas,
            "taxa_deteccao_estimulo": total_detectadas / total_frequencias,
            "n_frequencias_estimulo_detectadas_detector_bruto": total_detectadas_bruto,
            "taxa_deteccao_estimulo_detector_bruto": (
                total_detectadas_bruto / total_frequencias
            ),
            "n_frequencias_laterais": total_laterais,
            "n_frequencias_laterais_detectadas": total_falsos_positivos,
            "taxa_falso_positivo_lateral": total_falsos_positivos / total_laterais,
            "n_frequencias_laterais_detectadas_detector_bruto": (
                total_falsos_positivos_bruto
            ),
            "taxa_falso_positivo_lateral_detector_bruto": (
                total_falsos_positivos_bruto / total_laterais
            ),
        },
        "por_nivel_db": {
            str(nivel): dados for nivel, dados in por_nivel.items()
        },
        "por_participante": por_participante,
        "detalhe_completo_por_arquivo": [
            {
                "arquivo": os.path.basename(r["arquivo"]),
                "participante": r["participante"],
                "nivel_db": r["nivel_db"],
                "n_frequencias_detectadas": r["n_frequencias_detectadas"],
                "n_frequencias_detectadas_detector_bruto": r[
                    "n_frequencias_detectadas_bruto"
                ],
                "n_frequencias_total": r["n_frequencias_total"],
                "n_lateral_detectadas": r["n_lateral_detectadas"],
                "n_lateral_detectadas_detector_bruto": r[
                    "n_lateral_detectadas_bruto"
                ],
                "n_lateral_total": r["n_lateral_total"],
                "por_frequencia_estimulo": {
                    str(freq): {
                        "max_consecutivas": dados["max_consecutivas"],
                        "percentual_presente": dados["percentual_presente"],
                        "detectado": dados["detectado"],
                        "caminho_estados": dados["caminho_estados"],
                        "detector_bruto": dados["deteccao_detector_bruto"],
                    }
                    for freq, dados in r["por_frequencia_estimulo"].items()
                },
                "por_frequencia_lateral": {
                    str(freq): {
                        "max_consecutivas": dados["max_consecutivas"],
                        "percentual_presente": dados["percentual_presente"],
                        "detectado": dados["detectado"],
                        "caminho_estados": dados["caminho_estados"],
                        "detector_bruto": dados["deteccao_detector_bruto"],
                    }
                    for freq, dados in r["por_frequencia_lateral"].items()
                },
            }
            for r in resultados_arquivos
        ],
        "aviso": (
            "Analise exploratoria. A e heuristica de persistencia, "
            "B(Presente) usa injecao sintetica, e os limiares por frequencia "
            "ainda nao foram validados clinicamente. A linha 'detector_bruto' "
            "aplica a mesma regra de decisao diretamente sobre a "
            "significancia (p_value <= alpha) do detector configurado em "
            "get_obs_matrix.py, sem HMM, como baseline de comparacao."
        ),
    }


def salvar_relatorio(relatorio, caminho=OUTPUT_FILE):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return caminho


# ============================================================
# IMPRESSAO
# ============================================================

def imprimir_relatorio(relatorio):
    print("=== CONFIGURACAO DA REGRA DE DECISAO ===")
    cfg = relatorio["configuracao"]
    print(
        f"MIN_CONSECUTIVE={cfg['min_consecutive']}  "
        f"MIN_PERCENT={cfg['min_percent']}  "
        f"MODO={cfg['modo_regra_decisao']}  "
        f"(detector bruto: {cfg['detector']}, alpha={cfg['alpha_detector_bruto']})"
    )

    resumo = relatorio["resumo_global"]
    print(
        f"\nHMM         -> deteccao (estimulo): "
        f"{resumo['taxa_deteccao_estimulo']:.2%}  |  "
        f"falso positivo (lateral): "
        f"{resumo['taxa_falso_positivo_lateral']:.2%}"
    )
    print(
        f"Detector bruto -> deteccao (estimulo): "
        f"{resumo['taxa_deteccao_estimulo_detector_bruto']:.2%}  |  "
        f"falso positivo (lateral): "
        f"{resumo['taxa_falso_positivo_lateral_detector_bruto']:.2%}"
    )

    print("\n=== DETECCAO POR NIVEL DE ESTIMULO (dB) — HMM vs. detector bruto ===")
    for nivel, dados in relatorio["por_nivel_db"].items():
        print(
            f"  {nivel:>4} dB:"
            f"  HMM {dados['taxa_deteccao']:.2%} "
            f"({dados['n_frequencias_estimulo_detectadas']}/"
            f"{dados['n_frequencias_estimulo']})"
            f"  |  bruto {dados['taxa_deteccao_detector_bruto']:.2%} "
            f"({dados['n_frequencias_estimulo_detectadas_detector_bruto']}/"
            f"{dados['n_frequencias_estimulo']})"
        )
        print(
            f"          FP lateral:"
            f"  HMM {dados['taxa_falso_positivo_lateral']:.2%} "
            f"({dados['n_frequencias_laterais_detectadas']}/"
            f"{dados['n_frequencias_laterais']})"
            f"  |  bruto {dados['taxa_falso_positivo_lateral_detector_bruto']:.2%} "
            f"({dados['n_frequencias_laterais_detectadas_detector_bruto']}/"
            f"{dados['n_frequencias_laterais']})"
        )

    print("\n=== DETECCAO POR PARTICIPANTE (estimulo vs. lateral/ruido) ===")
    for participante, dados in relatorio["por_participante"].items():
        print(
            f"\n  Participante {participante}: "
            f"HMM estimulo={dados['taxa_deteccao_estimulo']:.2%} "
            f"(bruto={dados['taxa_deteccao_estimulo_detector_bruto']:.2%})  |  "
            f"HMM FP lateral={dados['taxa_falso_positivo_lateral']:.2%} "
            f"(bruto={dados['taxa_falso_positivo_lateral_detector_bruto']:.2%})"
        )
        for item in dados["detalhe_por_arquivo"]:
            nivel = item["nivel_db"]
            print(
                f"    {item['arquivo']:20} nivel={nivel:>4} dB  "
                f"estimulo: HMM {item['n_frequencias_estimulo_detectadas']:>5}  "
                f"bruto {item['n_frequencias_estimulo_detectadas_detector_bruto']:>5}   "
                f"lateral: HMM {item['n_frequencias_laterais_detectadas']:>5}  "
                f"bruto {item['n_frequencias_laterais_detectadas_detector_bruto']:>5}"
            )


if __name__ == "__main__":
    resultados_arquivos = rodar_inferencia(DATA_DIR)
    relatorio = construir_relatorio(resultados_arquivos)
    caminho_saida = salvar_relatorio(relatorio)
    imprimir_relatorio(relatorio)
    print(f"\nRelatorio completo salvo em: {caminho_saida}")