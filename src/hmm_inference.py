"""Inferencia de estados (Viterbi) e regra de decisao para deteccao de ASSR.

Este modulo NAO reimplementa o detector espectral. Ele importa diretamente de
``get_obs_matrix.py`` as mesmas funcoes usadas para calibrar a matriz B
(``carregar_mat``, ``extrair_coeficientes_epocas``, ``calcular_msc_janelas``,
``calcular_thresholds_msc``, ``discretizar``). Essas funcoes delegam ao
DETECTOR ATIVO de get_obs_matrix.py (MSC, MMSC, Rayleigh ou CSM - ver
detectors.py), entao escolher um detector diferente la tambem muda o
comportamento aqui, sem duplicar logica.

IMPORTANTE: o detector usado para calibrar B (observation_matrix.json) tem
que ser o MESMO usado aqui na inferencia. Este modulo escolhe o detector
(via ``gom.selecionar_detector()``) ANTES de carregar a matriz B, e
``carregar_matriz_observacao`` valida que o detector salvo no JSON bate com
o detector ativo, abortando com uma mensagem clara se nao bater.

Fluxo:

1. Escolhe o detector ativo (mesmo mecanismo de get_obs_matrix.py:
   ASSR_DETECTOR no ambiente, ou menu interativo).
2. Carrega A (results/transition_matrix.json) e B (results/observation_matrix.json)
   ja calibradas, validando que B foi calibrada com o MESMO detector.
3. Para cada arquivo com estimulo (*dB.mat), decodifica com Viterbi uma
   sequencia de estados POR FREQUENCIA de estimulacao (8 sequencias
   independentes, pois B e global mas as sequencias temporais de frequencias
   diferentes nao devem ser concatenadas).
4. Aplica uma regra de decisao (percentual minimo e/ou numero de janelas
   consecutivas em "Presente") sobre cada caminho de estados -> detectado_freq.
5. Em paralelo, aplica a MESMA regra de decisao diretamente sobre a
   significancia do detector bruto (p_value <= SIGNIFICANCE_LEVEL, janela a
   janela), sem passar pelo HMM/Viterbi. Isso serve como baseline: o detector
   configurado em get_obs_matrix.py (generico: MSC, MMSC, Rayleigh ou CSM)
   ja e um teste que funciona sozinho, entao o ganho do HMM so faz sentido se
   ficar acima dessa baseline.
6. Trata cada frequencia como um experimento independente ao agregar a taxa de
   deteccao (por exemplo, 1 frequencia detectada em 8 corresponde a 12,5%).
7. Repete o mesmo pipeline (HMM e baseline do detector bruto) diretamente nas
   frequencias laterais de ``binsM``, onde nao ha estimulo esperado, para
   estimar a taxa de falso positivo por frequencia, calibrada nos proprios
   arquivos com estimulo (usar so os ESP omitiria o efeito de qualquer
   artefato do estimulo/gravacao).
8. Salva um relatorio consolidado e imprime as analises por nivel de dB e por
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

import get_obs_matrix as gom
import detectors as det_registry
from get_obs_matrix import (
    CHANNEL_INDEX,
    DATA_DIR,
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
MIN_CONSECUTIVE = 4
MIN_PERCENT = 0.10
MODO_REGRA_DECISAO = "OR"  # "OR" ou "AND"

# Cada tupla representa um experimento HMM. Um elemento usa um unico detector;
# dois ou mais elementos fundem evidencias sincronizadas no mesmo Viterbi.
# Exemplos: [("rayleigh",), ("csm",), ("rayleigh", "csm")].
DETECTOR_COMBINATIONS = [("rayleigh", "mmsc", "csm")]

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


def carregar_matriz_observacao(caminho=OBS_MATRIX_FILE, detector_nome=None):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    estados = dados["estados_linhas"]
    labels = dados["labels_colunas"]
    matriz = np.asarray(dados["matriz"], dtype=float)
    configuracao = dados.get("configuracao", {})
    detector_arquivo = configuracao.get("detector")
    detector_esperado = detector_nome or gom.DETECTOR
    if detector_arquivo != detector_esperado:
        raise ValueError(
            f"{caminho}: detector {detector_arquivo!r} difere do detector "
            f"esperado {detector_esperado!r}; regenere a matriz B com o mesmo "
            "detector (rode get_obs_matrix.py escolhendo o mesmo detector, "
            "ou defina a variavel de ambiente ASSR_DETECTOR igual nos dois "
            "scripts) antes da inferencia"
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


def viterbi_multidetector(sequencias_por_detector, matriz_a,
                          matrizes_b_por_detector, pi_inicial, estados, labels):
    """Viterbi com varias evidencias por janela.

    As emissoes sao combinadas por produto, equivalente a soma em log-espaco,
    assumindo independencia condicional dos detectores dado o estado.
    """
    nomes = list(sequencias_por_detector)
    if not nomes:
        raise ValueError("Informe ao menos um detector")
    tamanhos = {len(sequencias_por_detector[nome]) for nome in nomes}
    if len(tamanhos) != 1:
        raise ValueError("As sequencias dos detectores precisam estar sincronizadas")
    n_obs = tamanhos.pop()
    if n_obs == 0:
        return []

    n_estados = len(estados)
    log_a = np.log(np.asarray(matriz_a) + 1e-300)
    log_pi = np.log(np.asarray(pi_inicial, dtype=float) + 1e-300)
    log_bs = {
        nome: np.log(np.asarray(matrizes_b_por_detector[nome]) + 1e-300)
        for nome in nomes
    }
    indices = {
        nome: [labels.index(label) for label in sequencias_por_detector[nome]]
        for nome in nomes
    }

    def emissao(t):
        return sum(log_bs[nome][:, indices[nome][t]] for nome in nomes)

    delta = np.zeros((n_obs, n_estados))
    psi = np.zeros((n_obs, n_estados), dtype=int)
    delta[0] = log_pi + emissao(0)
    for t in range(1, n_obs):
        log_emissao = emissao(t)
        for j in range(n_estados):
            candidatos = delta[t - 1] + log_a[:, j]
            psi[t, j] = int(np.argmax(candidatos))
            delta[t, j] = candidatos[psi[t, j]] + log_emissao[j]
    caminho_idx = np.zeros(n_obs, dtype=int)
    caminho_idx[-1] = int(np.argmax(delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        caminho_idx[t] = psi[t + 1, caminho_idx[t + 1]]
    return [estados[i] for i in caminho_idx]


# ============================================================
# SEQUENCIA DE OBSERVACOES PARA UMA FREQUENCIA
# ============================================================

def construir_sequencia_labels(x, canal, fs, frequencia, thresholds):
    """Reusa exatamente o detector ativo de get_obs_matrix.py para gerar labels."""
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


def construir_sequencia_detector(x, canal, fs, frequencia, detector):
    """Constroi observacoes explicitamente para um detector, sem estado global."""
    coeficientes, _ = extrair_coeficientes_epocas(x, canal, fs, frequencia)
    valores, intervalos = det_registry.calcular_estatistica_janelas(
        coeficientes, detector, WINDOW_SIZE_EPOCHS, WINDOW_STEP_EPOCHS
    )
    thresholds = detector.thresholds(WINDOW_SIZE_EPOCHS, P_VALUE_BOUNDARIES)
    labels = [discretizar(v, thresholds, NIVEIS_OBSERVACAO) for v in valores]
    p_values = [detector.p_value(v, WINDOW_SIZE_EPOCHS) for v in valores]
    return labels, valores, p_values, intervalos


def validar_combinacoes(combinacoes):
    normalizadas = []
    for combinacao in combinacoes:
        nomes = (combinacao,) if isinstance(combinacao, str) else tuple(combinacao)
        if not nomes:
            raise ValueError("Combinacao de detectores vazia")
        if len(set(nomes)) != len(nomes):
            raise ValueError(f"Detector repetido na combinacao {nomes}")
        for nome in nomes:
            det_registry.obter_detector(nome)
        normalizadas.append(nomes)
    if len(set(normalizadas)) != len(normalizadas):
        raise ValueError("DETECTOR_COMBINATIONS contem combinacoes repetidas")
    return normalizadas


def nome_combinacao(combinacao):
    return "+".join(combinacao)


def construir_matriz_b_detector(pasta_dados, detector_nome, k_sintetico=None):
    """Calibra B para um detector preservando a configuracao global anterior."""
    anterior = gom.DETECTOR
    k_anterior = gom.K_SINTETICO
    try:
        gom.selecionar_detector(detector_nome, interativo=False)
        if k_sintetico is not None:
            gom.K_SINTETICO = k_sintetico
        resultados = gom.construir_matrizes_observacao(pasta_dados)
    finally:
        gom.K_SINTETICO = k_anterior
        gom.selecionar_detector(anterior, interativo=False)
    global_ = resultados["global"]
    return np.asarray([
        [global_[estado][label] for label in NIVEIS_OBSERVACAO]
        for estado in ESTADOS
    ], dtype=float)


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
    detector configurado em get_obs_matrix.py (MSC, MMSC, Rayleigh ou CSM)."""
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
    # Escolhe o detector ativo ANTES de carregar B, para que a validacao de
    # consistencia abaixo compare com o detector correto.
    gom.selecionar_detector()

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
            "detector": gom.DETECTOR,
            "detector_nome_exibicao": gom.DETECTOR_ATUAL.nome_exibicao,
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
            "significancia (p_value <= alpha) do detector ativo (ver "
            "'detector' acima), sem HMM, como baseline de comparacao."
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
    cfg = relatorio["configuracao"]
    print("=" * 64)
    print("RELATORIO DE INFERENCIA HMM")
    print("=" * 64)
    print(f"Detector: {cfg['detector_nome_exibicao']}  (chave: {cfg['detector']!r})")
    print(
        f"Regra de decisao: MIN_CONSECUTIVE={cfg['min_consecutive']}  "
        f"MIN_PERCENT={cfg['min_percent']}  MODO={cfg['modo_regra_decisao']}  "
        f"(alpha do detector bruto={cfg['alpha_detector_bruto']})"
    )

    resumo = relatorio["resumo_global"]
    print(
        f"\nHMM            -> deteccao (estimulo): "
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


# ============================================================
# PIPELINE COM MULTIPLOS DETECTORES
# ============================================================

def _processar_grupo_multidetector(x, fs, frequencias, combinacao,
                                   matriz_a, matrizes_b):
    saida = {}
    for frequencia in frequencias:
        evidencias = {}
        bruto = {}
        intervalos_referencia = None
        for nome in combinacao:
            detector = det_registry.obter_detector(nome)
            labels, valores, p_values, intervalos = construir_sequencia_detector(
                x, CHANNEL_INDEX, fs, frequencia, detector
            )
            if intervalos_referencia is not None and intervalos != intervalos_referencia:
                raise ValueError("Detectores produziram janelas nao sincronizadas")
            intervalos_referencia = intervalos
            evidencias[nome] = labels
            bruto[nome] = {
                **avaliar_detector_bruto(p_values),
                "valores": valores.tolist(),
                "p_values": p_values,
            }
        caminho = viterbi_multidetector(
            evidencias, matriz_a, matrizes_b, PI_INICIAL, ESTADOS,
            NIVEIS_OBSERVACAO,
        )
        saida[float(frequencia)] = {
            **avaliar_caminho(caminho),
            "caminho_estados": caminho,
            "labels_por_detector": evidencias,
            "detector_bruto_por_detector": bruto,
            "intervalos_epocas": [list(i) for i in (intervalos_referencia or [])],
        }
    return saida


def processar_arquivo_multidetector(arquivo, matriz_a, matrizes_b, combinacao):
    estimulo = _processar_grupo_multidetector(
        arquivo["x"], arquivo["fs"], arquivo["freq_estim"], combinacao,
        matriz_a, matrizes_b,
    )
    lateral = _processar_grupo_multidetector(
        arquivo["x"], arquivo["fs"], arquivo["bins_m"], combinacao,
        matriz_a, matrizes_b,
    )
    return {
        "por_frequencia_estimulo": estimulo,
        "por_frequencia_lateral": lateral,
    }


def rodar_inferencia_multidetector(pasta_dados=DATA_DIR,
                                   combinacoes=DETECTOR_COMBINATIONS):
    combinacoes = validar_combinacoes(combinacoes)
    nomes = sorted({nome for combinacao in combinacoes for nome in combinacao})
    matriz_a = carregar_matriz_transicao()
    matrizes_b = {
        nome: construir_matriz_b_detector(pasta_dados, nome) for nome in nomes
    }
    caminhos = sorted(glob.glob(os.path.join(pasta_dados, PRESENT_PATTERN)))
    if not caminhos:
        raise RuntimeError(f"Nenhum arquivo encontrado com {PRESENT_PATTERN}")
    arquivos = []
    for caminho in caminhos:
        participante, _, nivel_db = parse_nome_arquivo(caminho)
        arquivos.append((caminho, participante, nivel_db,
                         carregar_mat_estimulo(caminho)))

    resultados = {}
    for combinacao in combinacoes:
        chave = nome_combinacao(combinacao)
        resultados[chave] = []
        b_combinacao = {nome: matrizes_b[nome] for nome in combinacao}
        for caminho, participante, nivel_db, arquivo in arquivos:
            resultados[chave].append({
                "arquivo": caminho,
                "participante": participante,
                "nivel_db": nivel_db,
                **processar_arquivo_multidetector(
                    arquivo, matriz_a, b_combinacao, combinacao
                ),
            })
    return resultados


def _metricas_resultados(resultados, detector_bruto=None):
    estimulo = [d for arq in resultados
                for d in arq["por_frequencia_estimulo"].values()]
    lateral = [d for arq in resultados
               for d in arq["por_frequencia_lateral"].values()]
    if detector_bruto is None:
        positivos_estimulo = sum(d["detectado"] for d in estimulo)
        positivos_lateral = sum(d["detectado"] for d in lateral)
    else:
        positivos_estimulo = sum(
            d["detector_bruto_por_detector"][detector_bruto]["detectado"]
            for d in estimulo
        )
        positivos_lateral = sum(
            d["detector_bruto_por_detector"][detector_bruto]["detectado"]
            for d in lateral
        )
    taxa_deteccao = positivos_estimulo / len(estimulo)
    taxa_fp = positivos_lateral / len(lateral)
    return {
        "n_frequencias_estimulo": len(estimulo),
        "n_frequencias_estimulo_detectadas": positivos_estimulo,
        "taxa_deteccao": taxa_deteccao,
        "n_frequencias_laterais": len(lateral),
        "n_frequencias_laterais_detectadas": positivos_lateral,
        "taxa_falso_positivo": taxa_fp,
        "acuracia_balanceada": (taxa_deteccao + 1.0 - taxa_fp) / 2.0,
    }


def construir_relatorio_multidetector(resultados_por_combinacao):
    detectores = sorted({
        nome for chave in resultados_por_combinacao for nome in chave.split("+")
    })
    # O bruto de um detector independe da combinacao; usa a primeira que o contem.
    bruto = {}
    for nome in detectores:
        chave = next(c for c in resultados_por_combinacao
                     if nome in c.split("+"))
        bruto[nome] = _metricas_resultados(
            resultados_por_combinacao[chave], detector_bruto=nome
        )
    hmm = {
        chave: _metricas_resultados(resultados)
        for chave, resultados in resultados_por_combinacao.items()
    }
    return {
        "nome": "relatorio_inferencia_hmm_multidetector",
        "configuracao": {
            "combinacoes_detectores": [c.split("+") for c in hmm],
            "fusao_emissoes": "produto_condicional (soma de log-probabilidades)",
            "hipotese_fusao": "independencia condicional dado o estado",
            "tamanho_janela_epocas": WINDOW_SIZE_EPOCHS,
            "passo_janela_epocas": WINDOW_STEP_EPOCHS,
            "min_consecutive": MIN_CONSECUTIVE,
            "min_percent": MIN_PERCENT,
            "modo_regra_decisao": MODO_REGRA_DECISAO,
            "alpha_detector_bruto": SIGNIFICANCE_LEVEL,
        },
        "hmm_por_combinacao": hmm,
        "detector_bruto_por_teste": bruto,
        "detalhe_por_combinacao": resultados_por_combinacao,
        "aviso": (
            "Resultados exploratorios, nao clinicamente validados. A fusao "
            "assume independencia condicional, embora detectores calculados "
            "sobre o mesmo EEG possam ser correlacionados."
        ),
    }


def imprimir_relatorio_multidetector(relatorio):
    cfg = relatorio["configuracao"]
    print("=" * 72)
    print("INFERENCIA HMM COM MULTIPLOS DETECTORES")
    print("=" * 72)
    print(
        f"Regra de decisao: MIN_CONSECUTIVE={cfg['min_consecutive']}  "
        f"MIN_PERCENT={cfg['min_percent']}  MODO={cfg['modo_regra_decisao']}  "
        f"(alpha dos detectores brutos={cfg['alpha_detector_bruto']})"
    )
    print("\nHMM por detector/combinacao")
    for nome, m in relatorio["hmm_por_combinacao"].items():
        print(f"  {nome:24s} acuracia={m['acuracia_balanceada']:.2%}  "
              f"deteccao={m['taxa_deteccao']:.2%}  "
              f"FP={m['taxa_falso_positivo']:.2%}")
    print("\nDetector bruto por teste")
    for nome, m in relatorio["detector_bruto_por_teste"].items():
        exibicao = det_registry.obter_detector(nome).nome_exibicao
        print(f"  {nome:10s} ({exibicao})\n"
              f"    acuracia={m['acuracia_balanceada']:.2%}  "
              f"deteccao={m['taxa_deteccao']:.2%}  "
              f"FP={m['taxa_falso_positivo']:.2%}")

    for combinacao, resultados in relatorio["detalhe_por_combinacao"].items():
        detectores = combinacao.split("+")
        print(f"\n=== DETECCAO POR NIVEL DE ESTIMULO (dB) — {combinacao} ===")
        niveis = sorted({r["nivel_db"] for r in resultados})
        for nivel in niveis:
            grupo = [r for r in resultados if r["nivel_db"] == nivel]
            hmm = _metricas_resultados(grupo)
            brutos = [
                f"{nome} {(_metricas_resultados(grupo, nome)['taxa_deteccao']):.2%}"
                for nome in detectores
            ]
            brutos_fp = [
                f"{nome} {(_metricas_resultados(grupo, nome)['taxa_falso_positivo']):.2%}"
                for nome in detectores
            ]
            print(
                f"  {nivel:>4} dB: HMM {hmm['taxa_deteccao']:.2%} "
                f"({hmm['n_frequencias_estimulo_detectadas']}/"
                f"{hmm['n_frequencias_estimulo']}) | bruto: {', '.join(brutos)}"
            )
            print(
                f"          FP lateral: HMM {hmm['taxa_falso_positivo']:.2%} "
                f"({hmm['n_frequencias_laterais_detectadas']}/"
                f"{hmm['n_frequencias_laterais']}) | bruto: {', '.join(brutos_fp)}"
            )

        print(f"\n=== DETECCAO POR PARTICIPANTE — {combinacao} ===")
        participantes = sorted({r["participante"] for r in resultados})
        for participante in participantes:
            grupo = [r for r in resultados if r["participante"] == participante]
            hmm = _metricas_resultados(grupo)
            bruto_resumo = []
            for nome in detectores:
                metrica = _metricas_resultados(grupo, nome)
                bruto_resumo.append(
                    f"{nome}: estimulo={metrica['taxa_deteccao']:.2%}, "
                    f"FP={metrica['taxa_falso_positivo']:.2%}"
                )
            print(
                f"\n  Participante {participante}: "
                f"HMM estimulo={hmm['taxa_deteccao']:.2%}  "
                f"HMM FP lateral={hmm['taxa_falso_positivo']:.2%}  |  "
                f"bruto: {'; '.join(bruto_resumo)}"
            )
            for item in grupo:
                estimulo = item["por_frequencia_estimulo"].values()
                lateral = item["por_frequencia_lateral"].values()
                n_estimulo = sum(d["detectado"] for d in estimulo)
                n_lateral = sum(d["detectado"] for d in lateral)
                contagens_brutas = []
                for nome in detectores:
                    bruto_estimulo = sum(
                        d["detector_bruto_por_detector"][nome]["detectado"]
                        for d in item["por_frequencia_estimulo"].values()
                    )
                    bruto_lateral = sum(
                        d["detector_bruto_por_detector"][nome]["detectado"]
                        for d in item["por_frequencia_lateral"].values()
                    )
                    contagens_brutas.append(
                        f"{nome} {bruto_estimulo/8}/{bruto_lateral/8}"
                    )
                print(
                    f"    {os.path.basename(item['arquivo']):20} "
                    f"nivel={item['nivel_db']:>4} dB  "
                    f"HMM estimulo/lateral={n_estimulo/8}/{n_lateral/8}  |  "
                    f"bruto estimulo/lateral: {', '.join(contagens_brutas)}\n"
                )


if __name__ == "__main__":
    resultados_arquivos = rodar_inferencia_multidetector(DATA_DIR)
    relatorio = construir_relatorio_multidetector(resultados_arquivos)
    caminho_saida = salvar_relatorio(relatorio)
    imprimir_relatorio_multidetector(relatorio)
    print(f"\nRelatorio completo salvo em: {caminho_saida}")
