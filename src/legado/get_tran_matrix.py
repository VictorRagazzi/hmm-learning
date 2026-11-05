"""Estima e salva uma matriz de transicao heuristica para o HMM de ASSR.

Cada arquivo e tratado como uma sequencia que permanece no estado associado a
sua condicao: arquivos ESP representam ``Ausente`` e arquivos dB representam
``Presente``. Para um arquivo com N janelas, contam-se N-1 autopassagens e uma
saida implicita. Portanto, esta matriz descreve persistencia baseada no tamanho
das gravacoes; ela nao foi estimada a partir de estados ocultos observados.
"""

import glob
import json
import os

import h5py
import numpy as np


# ============================================================
# CONFIGURACOES
# ============================================================

DATA_DIR = "./data"
RESULTS_DIR = "./results"
OUTPUT_FILE = "transition_matrix.json"

ESP_PATTERN = "*ESP.mat"
PRESENT_PATTERN = "*dB.mat"
ESTADOS = ["Ausente", "Presente"]

WINDOW_SIZE_EPOCHS = 10
WINDOW_STEP_EPOCHS = 5


def n_janelas_de(n_epocas):
    """Retorna quantas janelas completas cabem em uma gravacao."""
    n_janelas = (n_epocas - WINDOW_SIZE_EPOCHS) // WINDOW_STEP_EPOCHS + 1
    return max(n_janelas, 0)


def carregar_n_epocas(caminho):
    """Valida a geometria basica do .mat e retorna seu numero de epocas."""
    with h5py.File(caminho, "r") as arquivo:
        faltantes = {"x", "Fs"} - set(arquivo.keys())
        if faltantes:
            raise ValueError(f"{caminho}: variaveis ausentes: {sorted(faltantes)}")
        x_shape = arquivo["x"].shape
        fs = float(np.asarray(arquivo["Fs"]).ravel()[0])

    if len(x_shape) != 3:
        raise ValueError(f"{caminho}: x deve ter 3 dimensoes; recebido {x_shape}")
    if x_shape[0] != 16:
        raise ValueError(f"{caminho}: esperado 16 canais no eixo 0; recebido {x_shape}")
    if x_shape[2] != int(fs):
        raise ValueError(
            f"{caminho}: eixo de amostras ({x_shape[2]}) difere de Fs ({fs:g})"
        )

    return x_shape[1]


def calcular_p_self(padrao_arquivos, label):
    """Calcula persistencia usando N-1 autopassagens e uma saida por arquivo."""
    arquivos = sorted(glob.glob(padrao_arquivos))
    total_janelas = 0
    total_self = 0
    arquivos_validos = []

    print(f"\n--- {label} ---")
    for caminho in arquivos:
        n_epocas = carregar_n_epocas(caminho)
        n_janelas = n_janelas_de(n_epocas)
        if n_janelas < 1:
            print(
                f"{caminho}: {n_epocas} epocas -> "
                "descartado (janelas insuficientes)"
            )
            continue

        n_self = n_janelas - 1
        total_janelas += n_janelas
        total_self += n_self
        arquivos_validos.append(
            {
                "arquivo": caminho,
                "n_epocas": n_epocas,
                "n_janelas": n_janelas,
                "n_auto_transicoes": n_self,
            }
        )
        print(f"{caminho}: {n_epocas} epocas -> {n_janelas} janelas")

    if total_janelas == 0:
        raise RuntimeError(f"Nenhuma janela valida encontrada para {label}")

    p_self = total_self / total_janelas
    print(
        f"TOTAL: {total_janelas} janelas, {total_self} auto-transicoes "
        f"-> p_self = {p_self:.6f}"
    )
    return {
        "p_self": p_self,
        "n_arquivos_encontrados": len(arquivos),
        "n_arquivos_validos": len(arquivos_validos),
        "total_janelas": total_janelas,
        "total_auto_transicoes": total_self,
        "arquivos": arquivos_validos,
    }


def construir_matriz_transicao(pasta_dados=DATA_DIR):
    """Constroi A a partir das duracoes dos arquivos ESP e com estimulo."""
    ausente = calcular_p_self(
        os.path.join(pasta_dados, ESP_PATTERN),
        "Ausente (arquivos *ESP.mat)",
    )
    presente = calcular_p_self(
        os.path.join(pasta_dados, PRESENT_PATTERN),
        "Presente (arquivos *dB.mat)",
    )

    p_ausente = ausente["p_self"]
    p_presente = presente["p_self"]
    probabilidades = {
        "Ausente": {
            "Ausente": p_ausente,
            "Presente": 1.0 - p_ausente,
        },
        "Presente": {
            "Ausente": 1.0 - p_presente,
            "Presente": p_presente,
        },
    }

    return {
        "nome": "matriz_transicao_A",
        "descricao": "P(proximo estado | estado atual)",
        "estados_linhas": list(ESTADOS),
        "estados_colunas": list(ESTADOS),
        "matriz": [
            [probabilidades[origem][destino] for destino in ESTADOS]
            for origem in ESTADOS
        ],
        "probabilidades": probabilidades,
        "configuracao": {
            "tamanho_janela_epocas": WINDOW_SIZE_EPOCHS,
            "passo_janela_epocas": WINDOW_STEP_EPOCHS,
            "padrao_ausente": ESP_PATTERN,
            "padrao_presente": PRESENT_PATTERN,
        },
        "resumo": {
            "Ausente": ausente,
            "Presente": presente,
        },
        "metodo": (
            "Para cada arquivo com N janelas, contam-se N-1 auto-transicoes "
            "e uma saida implicita para o outro estado."
        ),
        "aviso": (
            "Heuristica baseada na duracao dos arquivos; nao e uma estimativa "
            "de transicoes ocultas observadas nem uma validacao clinica."
        ),
    }


def salvar_tabela_transicao(resultado, pasta_resultados=RESULTS_DIR):
    """Grava a matriz A e metadados em results/transition_matrix.json."""
    os.makedirs(pasta_resultados, exist_ok=True)
    caminho_saida = os.path.join(pasta_resultados, OUTPUT_FILE)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return caminho_saida


def imprimir_matriz(resultado):
    probabilidades = resultado["probabilidades"]
    print("\n=== MATRIZ DE TRANSICAO A ===")
    print(f"{'':12}{'Ausente':>12}{'Presente':>12}")
    for origem in ESTADOS:
        print(
            f"{origem:12}"
            f"{probabilidades[origem]['Ausente']:12.6f}"
            f"{probabilidades[origem]['Presente']:12.6f}"
        )


if __name__ == "__main__":
    resultado = construir_matriz_transicao(DATA_DIR)
    caminho_saida = salvar_tabela_transicao(resultado)
    imprimir_matriz(resultado)
    print(f"\nTabela salva em: {caminho_saida}")
