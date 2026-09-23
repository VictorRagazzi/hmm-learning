"""Inferencia de estados (Viterbi) e regra de decisao para deteccao de ASSR.

Este modulo NAO reimplementa o detector espectral. Ele importa diretamente de
``get_obs_matrix.py`` as mesmas funcoes usadas para calibrar a matriz B
(``carregar_mat``, ``extrair_potencias_epocas``, ``calcular_f_janelas``,
``calcular_thresholds_f``, ``discretizar``). Se o teste estatistico mudar la
(por exemplo, trocar o teste F por outro detector), este script herda a
mudanca automaticamente, sem precisar ser editado.

Fluxo:

1. Carrega A (results/transition_matrix.json) e B (results/observation_matrix.json)
   ja calibradas.
2. Para cada arquivo com estimulo (*dB.mat), decodifica com Viterbi uma
   sequencia de estados POR FREQUENCIA de estimulacao (8 sequencias
   independentes, pois B e global mas as sequencias temporais de frequencias
   diferentes nao devem ser concatenadas).
3. Aplica uma regra de decisao (percentual minimo e/ou numero de janelas
   consecutivas em "Presente") sobre cada caminho de estados -> detectado_freq.
4. Combina as 8 frequencias com uma regra k-de-N -> detectado_arquivo.
5. Repete o mesmo pipeline sobre pares de frequencias "laterais" (bins de
   binsM, sem estimulo esperado) para estimar a taxa de falso positivo
   composta da mesma regra k-de-N, calibrada nos proprios arquivos com
   estimulo (usar so os ESP omitiria o efeito de qualquer artefato do
   estimulo/gravacao).
6. Salva um relatorio consolidado e imprime as analises por nivel de dB e por
   participante.

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
    ESTADOS,
    NIVEIS_OBSERVACAO,
    P_VALUE_BOUNDARIES,
    PI_INICIAL,
    WINDOW_SIZE_EPOCHS,
    calcular_f_janelas,
    calcular_p_value,
    calcular_thresholds_f,
    carregar_mat,
    discretizar,
    extrair_potencias_epocas,
)

# ============================================================
# CONFIGURACOES
# ============================================================

RESULTS_DIR = "./results"
OBS_MATRIX_FILE = os.path.join(RESULTS_DIR, "observation_matrix.json")
TRAN_MATRIX_FILE = os.path.join(RESULTS_DIR, "transition_matrix.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "inference_analysis.json")

PRESENT_PATTERN = "*dB.mat"

# --- Regra de decisao sobre o caminho de estados de UMA frequencia ---
# Detecta se o caminho tiver ao menos MIN_CONSECUTIVE janelas seguidas em
# "Presente" e/ou se a fracao de janelas em "Presente" for >= MIN_PERCENT.
# MODO_REGRA_DECISAO combina os dois criterios: "OR" (qualquer um basta) ou
# "AND" (os dois precisam valer). Use None em um dos limiares para
# desativa-lo.
MIN_CONSECUTIVE = 3
MIN_PERCENT = 0.30
MODO_REGRA_DECISAO = "OR"  # "OR" ou "AND"

# --- Regra k-de-N entre as frequencias de estimulacao ---
# Um arquivo e considerado "detectado" se pelo menos K_DE_N das frequencias
# de estimulacao (de um total de N, tipicamente 8) forem detectadas pela
# regra acima.
K_DE_N = 2

# Mesma regra k-de-N, aplicada as frequencias "laterais" (controle), para
# estimar a taxa de falso positivo composta do metodo completo.
K_DE_N_LATERAL = K_DE_N


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
# SEQUENCIA DE OBSERVACOES PARA UM PAR (ALVO, RUIDO)
# ============================================================

def construir_sequencia_labels(x, canal, fs, freq_alvo, freq_ruido, thresholds):
    """Reusa exatamente o detector de get_obs_matrix.py para gerar labels."""
    pot_alvo, pot_ruido, _, _ = extrair_potencias_epocas(
        x, canal, fs, freq_alvo, freq_ruido
    )
    valores_f, _ = calcular_f_janelas(pot_alvo, pot_ruido)
    labels = [discretizar(v, thresholds, NIVEIS_OBSERVACAO) for v in valores_f]
    return labels, valores_f


# ============================================================
# REGRA DE DECISAO SOBRE UM CAMINHO DE ESTADOS
# ============================================================

def maior_sequencia_consecutiva(caminho, estado_alvo="Presente"):
    maior = 0
    atual = 0
    for estado in caminho:
        if estado == estado_alvo:
            atual += 1
            maior = max(maior, atual)
        else:
            atual = 0
    return maior


def avaliar_caminho(caminho, estado_alvo="Presente"):
    """Aplica a regra de consecutivos e/ou percentual sobre um caminho."""
    if len(caminho) == 0:
        return {
            "n_janelas": 0,
            "max_consecutivas": 0,
            "percentual_presente": 0.0,
            "detectado": False,
        }

    max_consec = maior_sequencia_consecutiva(caminho, estado_alvo)
    percentual = caminho.count(estado_alvo) / len(caminho)

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
        "n_janelas": len(caminho),
        "max_consecutivas": max_consec,
        "percentual_presente": percentual,
        "criterio_consecutivas_ok": criterio_consec,
        "criterio_percentual_ok": criterio_percent,
        "detectado": detectado,
    }


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
# PARES LATERAIS (CONTROLE) PARA ESTIMAR FALSO POSITIVO COMPOSTO
# ============================================================

def construir_pares_laterais(bins_m):
    """Cria pares (pseudo-alvo, pseudo-ruido) a partir de binsM.

    binsM[i] ja e usado como denominador (ruido) do teste em freqEstim[i], e
    portanto nao pode ser reaproveitado como par (binsM[i], binsM[i]). Aqui
    cada binsM[i] vira um pseudo-alvo testado contra o proximo bin de
    controle (circular), evitando reusar o mesmo par numerador/denominador
    ja consumido na calibracao de B. Isso NAO tem estimulo esperado em
    nenhum dos dois lados, servindo como controle de falso positivo.
    """
    n = len(bins_m)
    return [(bins_m[i], bins_m[(i + 1) % n]) for i in range(n)]


# ============================================================
# PROCESSAMENTO DE UM ARQUIVO
# ============================================================

def processar_arquivo(arquivo, matriz_a, matriz_b, thresholds):
    """Roda Viterbi + regra de decisao nas frequencias de estimulo e laterais."""
    freq_estim = arquivo["freq_estim"]
    bins_m = arquivo["bins_m"]
    pares_laterais = construir_pares_laterais(bins_m)

    resultado_por_frequencia = {}
    for freq_alvo, freq_ruido in zip(freq_estim, bins_m):
        labels, valores_f = construir_sequencia_labels(
            arquivo["x"], CHANNEL_INDEX, arquivo["fs"], freq_alvo, freq_ruido,
            thresholds,
        )
        caminho = viterbi(labels, matriz_a, matriz_b, PI_INICIAL, ESTADOS,
                           NIVEIS_OBSERVACAO)
        avaliacao = avaliar_caminho(caminho)
        resultado_por_frequencia[float(freq_alvo)] = {
            "freq_ruido": float(freq_ruido),
            "labels": labels,
            "caminho_estados": caminho,
            **avaliacao,
        }

    n_detectadas = sum(
        r["detectado"] for r in resultado_por_frequencia.values()
    )
    detectado_arquivo = n_detectadas >= K_DE_N

    resultado_lateral = {}
    for pseudo_alvo, pseudo_ruido in pares_laterais:
        labels, valores_f = construir_sequencia_labels(
            arquivo["x"], CHANNEL_INDEX, arquivo["fs"], pseudo_alvo,
            pseudo_ruido, thresholds,
        )
        caminho = viterbi(labels, matriz_a, matriz_b, PI_INICIAL, ESTADOS,
                           NIVEIS_OBSERVACAO)
        avaliacao = avaliar_caminho(caminho)
        resultado_lateral[float(pseudo_alvo)] = {
            "pseudo_ruido": float(pseudo_ruido),
            "labels": labels,
            "caminho_estados": caminho,
            **avaliacao,
        }

    n_detectadas_lateral = sum(
        r["detectado"] for r in resultado_lateral.values()
    )
    falso_positivo_arquivo = n_detectadas_lateral >= K_DE_N_LATERAL

    return {
        "por_frequencia_estimulo": resultado_por_frequencia,
        "n_frequencias_detectadas": n_detectadas,
        "n_frequencias_total": len(resultado_por_frequencia),
        "detectado_arquivo": detectado_arquivo,
        "por_frequencia_lateral": resultado_lateral,
        "n_lateral_detectadas": n_detectadas_lateral,
        "n_lateral_total": len(resultado_lateral),
        "falso_positivo_arquivo": falso_positivo_arquivo,
    }


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def rodar_inferencia(pasta_dados=DATA_DIR):
    matriz_a = carregar_matriz_transicao()
    matriz_b = carregar_matriz_observacao()
    thresholds = calcular_thresholds_f(WINDOW_SIZE_EPOCHS, P_VALUE_BOUNDARIES)

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
        grupos[r["nivel_db"]].append(r["detectado_arquivo"])

    analise = {}
    for nivel_db in sorted(grupos):
        deteccoes = grupos[nivel_db]
        analise[nivel_db] = {
            "n_arquivos": len(deteccoes),
            "n_detectados": sum(deteccoes),
            "taxa_deteccao": sum(deteccoes) / len(deteccoes),
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
                    "detectado_estimulo": r["detectado_arquivo"],
                    "n_frequencias_estimulo_detectadas": (
                        f"{r['n_frequencias_detectadas']}/"
                        f"{r['n_frequencias_total']}"
                    ),
                    "falso_positivo_lateral": r["falso_positivo_arquivo"],
                    "n_frequencias_laterais_detectadas": (
                        f"{r['n_lateral_detectadas']}/{r['n_lateral_total']}"
                    ),
                }
                for r in registros
            ],
            "taxa_deteccao_estimulo": sum(
                r["detectado_arquivo"] for r in registros
            ) / len(registros),
            "taxa_falso_positivo_lateral": sum(
                r["falso_positivo_arquivo"] for r in registros
            ) / len(registros),
        }
    return analise


def construir_relatorio(resultados_arquivos):
    por_nivel = analisar_por_nivel_db(resultados_arquivos)
    por_participante = analisar_por_participante(resultados_arquivos)

    taxa_deteccao_global = sum(
        r["detectado_arquivo"] for r in resultados_arquivos
    ) / len(resultados_arquivos)
    taxa_falso_positivo_global = sum(
        r["falso_positivo_arquivo"] for r in resultados_arquivos
    ) / len(resultados_arquivos)

    return {
        "nome": "relatorio_inferencia_hmm",
        "configuracao": {
            "min_consecutive": MIN_CONSECUTIVE,
            "min_percent": MIN_PERCENT,
            "modo_regra_decisao": MODO_REGRA_DECISAO,
            "k_de_n": K_DE_N,
            "k_de_n_lateral": K_DE_N_LATERAL,
        },
        "resumo_global": {
            "n_arquivos": len(resultados_arquivos),
            "taxa_deteccao_estimulo": taxa_deteccao_global,
            "taxa_falso_positivo_lateral": taxa_falso_positivo_global,
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
                "detectado_arquivo": r["detectado_arquivo"],
                "n_frequencias_detectadas": r["n_frequencias_detectadas"],
                "n_frequencias_total": r["n_frequencias_total"],
                "falso_positivo_arquivo": r["falso_positivo_arquivo"],
                "n_lateral_detectadas": r["n_lateral_detectadas"],
                "n_lateral_total": r["n_lateral_total"],
                "por_frequencia_estimulo": {
                    str(freq): {
                        "max_consecutivas": dados["max_consecutivas"],
                        "percentual_presente": dados["percentual_presente"],
                        "detectado": dados["detectado"],
                        "caminho_estados": dados["caminho_estados"],
                    }
                    for freq, dados in r["por_frequencia_estimulo"].items()
                },
                "por_frequencia_lateral": {
                    str(freq): {
                        "max_consecutivas": dados["max_consecutivas"],
                        "percentual_presente": dados["percentual_presente"],
                        "detectado": dados["detectado"],
                        "caminho_estados": dados["caminho_estados"],
                    }
                    for freq, dados in r["por_frequencia_lateral"].items()
                },
            }
            for r in resultados_arquivos
        ],
        "aviso": (
            "Analise exploratoria. A e heuristica de persistencia, "
            "B(Presente) usa injecao sintetica, e os limiares k-de-N / "
            "consecutivos ainda nao foram validados clinicamente."
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
        f"K_DE_N={cfg['k_de_n']}  K_DE_N_LATERAL={cfg['k_de_n_lateral']}"
    )

    resumo = relatorio["resumo_global"]
    print(
        f"\nTaxa de deteccao global (estimulo): "
        f"{resumo['taxa_deteccao_estimulo']:.2%}  |  "
        f"Taxa de falso positivo global (lateral): "
        f"{resumo['taxa_falso_positivo_lateral']:.2%}"
    )

    print("\n=== DETECCAO POR NIVEL DE ESTIMULO (dB) ===")
    for nivel, dados in relatorio["por_nivel_db"].items():
        print(
            f"  {nivel:>4} dB: {dados['taxa_deteccao']:.2%} "
            f"({dados['n_detectados']}/{dados['n_arquivos']} arquivos)"
        )

    print("\n=== DETECCAO POR PARTICIPANTE (estimulo vs. lateral/ruido) ===")
    for participante, dados in relatorio["por_participante"].items():
        print(
            f"\n  Participante {participante}: "
            f"deteccao estimulo = {dados['taxa_deteccao_estimulo']:.2%}, "
            f"falso positivo lateral = {dados['taxa_falso_positivo_lateral']:.2%}"
        )
        for item in dados["detalhe_por_arquivo"]:
            nivel = item["nivel_db"]
            print(
                f"    {item['arquivo']:20} nivel={nivel:>4} dB  "
                f"estimulo: {'DETECTADO' if item['detectado_estimulo'] else '-- '} "
                f"({item['n_frequencias_estimulo_detectadas']})   "
                f"lateral: {'FALSO-POS' if item['falso_positivo_lateral'] else '-- '} "
                f"({item['n_frequencias_laterais_detectadas']})"
            )


if __name__ == "__main__":
    resultados_arquivos = rodar_inferencia(DATA_DIR)
    relatorio = construir_relatorio(resultados_arquivos)
    caminho_saida = salvar_relatorio(relatorio)
    imprimir_relatorio(relatorio)
    print(f"\nRelatorio completo salvo em: {caminho_saida}")