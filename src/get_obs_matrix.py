"""Estima as linhas Ausente e Presente da matriz de emissao B.

A observacao e a magnitude quadratica da coerencia (MSC), calculada
separadamente para cada frequencia de estimulacao e em janelas de epocas::

    MSC = |sum_m X_m(f_alvo)|**2 / (M * sum_m |X_m(f_alvo)|**2)

Sob H0, coeficientes complexos gaussianos circulares e epocas independentes,
``MSC ~ Beta(1, M - 1)``, em que M e o numero de epocas da janela. A MSC mede
fase e magnitude; ela nao e CSM nem o teste F espectral local.

``binsM`` continua sendo lido e preservado como metadado/controle, mas nao e
usado como denominador da MSC.

Os limites dos labels sao quantis teoricos dessa distribuicao, definidos por
faixas de p-valor. Assim, B(Ausente) nao e forcada a ser uniforme como ocorria
quando os cortes eram quantis dos proprios dados ESP. As contagens das
frequencias sao agrupadas em uma unica matriz B compartilhada. Os resultados
por frequencia sao preservados apenas como diagnostico, pois as sequencias
temporais de frequencias diferentes nao devem ser concatenadas.

A taxa de falsos positivos observada nos arquivos ESP e impressa para verificar
as hipoteses do teste; os valores ainda nao constituem calibracao clinica.
"""

import glob
import json
import os

import h5py
import numpy as np
from scipy.stats import beta as distribuicao_msc


# ============================================================
# CONFIGURACOES
# ============================================================

DATA_DIR = "./data"
ESP_PATTERN = "*ESP.mat"
RESULTS_DIR = "./results"
OUTPUT_FILE = "observation_matrix.json"

ESTADOS = ["Ausente", "Presente"]
PI_INICIAL = [0.99, 0.01]

CHANNEL_INDEX = 0
DETECTOR = "magnitude_quadratica_coerencia_msc"

# Cada janela gera uma observacao. So entram janelas completas.
WINDOW_SIZE_EPOCHS = 10
WINDOW_STEP_EPOCHS = 5

# Labels em ordem crescente de evidencia contra H0 (Ausente).
NIVEIS_OBSERVACAO = ["muito_baixo", "baixo", "medio", "alto", "muito_alto"]

# Fronteiras superiores de p-valor, em ordem decrescente. Para os cinco
# labels acima, as faixas sao:
#   muito_baixo: p > 0.50
#   baixo:       0.10 < p <= 0.50
#   medio:       0.05 < p <= 0.10
#   alto:        0.01 < p <= 0.05
#   muito_alto:  p <= 0.01
# Deve haver exatamente len(NIVEIS_OBSERVACAO) - 1 valores.
P_VALUE_BOUNDARIES = [0.50, 0.10, 0.05, 0.01]
SIGNIFICANCE_LEVEL = 0.05

# amplitude_injetada = K_SINTETICO * desvio_padrao(epoca)
K_SINTETICO = 0.01

# Pseudocontagem de Jeffreys por celula, para evitar emissoes exatamente zero.
# Use 0.0 para obter apenas as frequencias relativas observadas.
SMOOTHING = 0.5
EPS = 1e-12


# ============================================================
# LEITURA E VALIDACAO
# ============================================================

def carregar_mat(caminho):
    """Abre um .mat v7.3 e devolve seus dados e metadados relevantes."""
    with h5py.File(caminho, "r") as arquivo:
        faltantes = {"Fs", "freqEstim", "binsM", "x"} - set(arquivo.keys())
        if faltantes:
            raise ValueError(f"{caminho}: variaveis ausentes: {sorted(faltantes)}")

        fs = float(np.asarray(arquivo["Fs"]).ravel()[0])
        freq_estim = np.asarray(arquivo["freqEstim"], dtype=float).ravel()
        bins_m = np.asarray(arquivo["binsM"], dtype=float).ravel()
        x = np.asarray(arquivo["x"])

    if x.ndim != 3:
        raise ValueError(f"{caminho}: x deve ter 3 dimensoes; recebido {x.shape}")
    if not 0 <= CHANNEL_INDEX < x.shape[0]:
        raise ValueError(
            f"{caminho}: canal {CHANNEL_INDEX} fora de x com {x.shape[0]} canais"
        )
    if len(freq_estim) != len(bins_m):
        raise ValueError(
            f"{caminho}: freqEstim ({len(freq_estim)}) e binsM "
            f"({len(bins_m)}) tem tamanhos diferentes"
        )
    if not np.all(np.isfinite(freq_estim)) or not np.all(np.isfinite(bins_m)):
        raise ValueError(f"{caminho}: freqEstim/binsM contem valor nao finito")
    if np.any(freq_estim <= 0) or np.any(bins_m <= 0):
        raise ValueError(f"{caminho}: frequencias devem ser positivas")
    if np.any(freq_estim >= fs / 2) or np.any(bins_m >= fs / 2):
        raise ValueError(f"{caminho}: frequencia igual ou acima de Nyquist ({fs / 2})")

    return {
        "caminho": caminho,
        "participante": os.path.basename(caminho).removesuffix("ESP.mat"),
        "condicao": "ESP",
        "fs": fs,
        "freq_estim": freq_estim,
        "bins_m": bins_m,
        "x": x,
    }


def validar_configuracao():
    if len(ESTADOS) != 2:
        raise ValueError("Esta etapa requer exatamente os estados Ausente e Presente")
    if not np.isclose(sum(PI_INICIAL), 1.0):
        raise ValueError("PI_INICIAL deve somar 1")
    if WINDOW_SIZE_EPOCHS < 1 or WINDOW_STEP_EPOCHS < 1:
        raise ValueError("Tamanho e passo da janela devem ser inteiros positivos")
    if len(NIVEIS_OBSERVACAO) < 2:
        raise ValueError("Defina pelo menos dois niveis de observacao")
    if len(set(NIVEIS_OBSERVACAO)) != len(NIVEIS_OBSERVACAO):
        raise ValueError("NIVEIS_OBSERVACAO nao pode conter nomes repetidos")
    if len(P_VALUE_BOUNDARIES) != len(NIVEIS_OBSERVACAO) - 1:
        raise ValueError(
            "P_VALUE_BOUNDARIES deve ter len(NIVEIS_OBSERVACAO) - 1 valores"
        )
    limites_p = np.asarray(P_VALUE_BOUNDARIES, dtype=float)
    if np.any(limites_p <= 0) or np.any(limites_p >= 1):
        raise ValueError("As fronteiras de p-valor devem estar entre 0 e 1")
    if np.any(np.diff(limites_p) >= 0):
        raise ValueError("P_VALUE_BOUNDARIES deve estar em ordem decrescente")
    if not 0 < SIGNIFICANCE_LEVEL < 1:
        raise ValueError("SIGNIFICANCE_LEVEL deve estar entre 0 e 1")
    if SMOOTHING < 0:
        raise ValueError("SMOOTHING nao pode ser negativo")


def validar_correspondencia_frequencias(arquivos):
    """Garante que alvos e controles mantem a mesma ordem entre arquivos."""
    freq_ref = arquivos[0]["freq_estim"]
    bins_ref = arquivos[0]["bins_m"]
    if len(np.unique(freq_ref)) != len(freq_ref):
        raise ValueError("freqEstim contem frequencias repetidas")
    if len(np.unique(bins_ref)) != len(bins_ref):
        raise ValueError("binsM contem frequencias repetidas")
    if np.any(np.isclose(freq_ref[:, None], bins_ref[None, :])):
        raise ValueError("Uma frequencia de controle coincide com um alvo")

    for arquivo in arquivos[1:]:
        if not np.array_equal(arquivo["freq_estim"], freq_ref):
            raise ValueError(
                f"{arquivo['caminho']}: freqEstim difere do arquivo de referencia"
            )
        if not np.array_equal(arquivo["bins_m"], bins_ref):
            raise ValueError(
                f"{arquivo['caminho']}: binsM difere do arquivo de referencia"
            )


# ============================================================
# DETECTOR E DISCRETIZACAO
# ============================================================

def localizar_bin_fft(n_amostras, fs, frequencia):
    """Retorna o indice e a frequencia real do bin FFT mais proximo."""
    frequencias_fft = np.fft.rfftfreq(n_amostras, d=1.0 / fs)
    indice = int(np.argmin(np.abs(frequencias_fft - frequencia)))
    return indice, float(frequencias_fft[indice])


def extrair_coeficientes_epocas(x, canal, fs, freq_alvo):
    """Extrai o coeficiente FFT complexo do alvo em cada epoca."""
    epocas = np.asarray(x[canal, :, :], dtype=float)
    n_amostras = epocas.shape[1]
    idx_alvo, bin_alvo = localizar_bin_fft(n_amostras, fs, freq_alvo)
    espectro = np.fft.rfft(epocas, axis=1)
    return espectro[:, idx_alvo], bin_alvo


def injetar_tom_sintetico(x, canal, fs, freq_alvo, k):
    """Copia x e soma, em cada epoca do canal escolhido, k*std(epoca)*seno."""
    x_sintetico = np.asarray(x, dtype=float).copy()
    epocas = x_sintetico[canal]
    tempo = np.arange(epocas.shape[1]) / fs
    seno = np.sin(2 * np.pi * freq_alvo * tempo)
    amplitudes = k * np.std(epocas, axis=1)
    x_sintetico[canal] = epocas + amplitudes[:, None] * seno[None, :]
    return x_sintetico


def calcular_msc_janelas(coeficientes):
    """Calcula uma MSC entre epocas para cada janela completa."""
    n_epocas = len(coeficientes)
    valores = []
    intervalos = []
    for inicio in range(0, n_epocas - WINDOW_SIZE_EPOCHS + 1, WINDOW_STEP_EPOCHS):
        fim = inicio + WINDOW_SIZE_EPOCHS
        janela = coeficientes[inicio:fim]
        numerador = np.abs(np.sum(janela)) ** 2
        denominador = WINDOW_SIZE_EPOCHS * np.sum(np.abs(janela) ** 2)
        valor_msc = numerador / (denominador + EPS)
        valores.append(float(np.clip(valor_msc, 0.0, 1.0)))
        intervalos.append((inicio, fim))
    return np.asarray(valores), intervalos


def calcular_thresholds_msc(n_epocas_janela, p_value_boundaries):
    """Converte fronteiras de p-valor em quantis da Beta(1, M-1)."""
    if n_epocas_janela < 2:
        raise ValueError("A MSC requer pelo menos duas epocas por janela")
    thresholds = distribuicao_msc.isf(
        np.asarray(p_value_boundaries, dtype=float),
        1,
        n_epocas_janela - 1,
    )
    if not np.all(np.isfinite(thresholds)) or np.any(np.diff(thresholds) <= 0):
        raise ValueError("As fronteiras de p-valor geraram thresholds degenerados")
    return thresholds


def calcular_p_value_msc(valor_msc, n_epocas_janela):
    """Calcula P(MSC >= valor | H0) pela distribuicao nula teorica."""
    return float(
        distribuicao_msc.sf(valor_msc, 1, n_epocas_janela - 1)
    )


def discretizar(valor, thresholds, labels):
    indice = int(np.searchsorted(thresholds, valor, side="right"))
    return labels[indice]


def montar_histograma(valores, thresholds, labels, smoothing=0.0):
    """Retorna probabilidades e contagens, com pseudocontagem configuravel."""
    contagens = np.zeros(len(labels), dtype=int)
    for valor in valores:
        indice = int(np.searchsorted(thresholds, valor, side="right"))
        contagens[indice] += 1

    total = int(contagens.sum())
    if total == 0:
        raise ValueError("Nenhuma janela completa disponivel para o histograma")
    probabilidades = (contagens + smoothing) / (total + len(labels) * smoothing)
    return (
        dict(zip(labels, probabilidades.tolist())),
        dict(zip(labels, contagens.tolist())),
    )


# ============================================================
# MATRIZES DE OBSERVACAO
# ============================================================

def construir_matrizes_observacao(pasta_dados):
    """Constroi uma B global e preserva diagnosticos separados por frequencia.

    A matriz global e obtida somando todas as contagens antes da normalizacao.
    Isso equivale a estimar uma distribuicao marginal sobre frequencias e evita
    dar o mesmo peso a grupos com numeros diferentes de observacoes.
    """
    validar_configuracao()
    caminhos = sorted(glob.glob(os.path.join(pasta_dados, ESP_PATTERN)))
    if not caminhos:
        raise RuntimeError(
            f"Nenhum arquivo encontrado em {pasta_dados} com padrao {ESP_PATTERN}"
        )

    arquivos = [carregar_mat(caminho) for caminho in caminhos]
    validar_correspondencia_frequencias(arquivos)

    thresholds = calcular_thresholds_msc(WINDOW_SIZE_EPOCHS, P_VALUE_BOUNDARIES)
    msc_critica = float(
        calcular_thresholds_msc(WINDOW_SIZE_EPOCHS, [SIGNIFICANCE_LEVEL])[0]
    )
    matrizes_por_frequencia = {}
    valores_globais_ausente = []
    valores_globais_presente = []
    registros_globais = []

    for indice_freq, freq_alvo in enumerate(arquivos[0]["freq_estim"]):
        valores_ausente = []
        valores_presente = []
        registros = []

        for arquivo in arquivos:
            freq_controle = arquivo["bins_m"][indice_freq]
            coeficientes, bin_alvo = extrair_coeficientes_epocas(
                arquivo["x"], CHANNEL_INDEX, arquivo["fs"], freq_alvo
            )
            _, bin_controle = localizar_bin_fft(
                arquivo["x"].shape[2], arquivo["fs"], freq_controle
            )
            msc_ausente, intervalos = calcular_msc_janelas(coeficientes)

            x_sintetico = injetar_tom_sintetico(
                arquivo["x"], CHANNEL_INDEX, arquivo["fs"], freq_alvo, K_SINTETICO
            )
            coeficientes_s, _ = extrair_coeficientes_epocas(
                x_sintetico, CHANNEL_INDEX, arquivo["fs"], freq_alvo
            )
            msc_presente, _ = calcular_msc_janelas(coeficientes_s)

            valores_ausente.extend(msc_ausente)
            valores_presente.extend(msc_presente)

            for estado, valores in (
                ("Ausente", msc_ausente),
                ("Presente", msc_presente),
            ):
                for valor, (inicio, fim) in zip(valores, intervalos):
                    registros.append(
                        {
                            "participante": arquivo["participante"],
                            "condicao": arquivo["condicao"],
                            "canal": CHANNEL_INDEX,
                            "frequencia": float(freq_alvo),
                            "frequencia_controle": float(freq_controle),
                            "bin_fft_alvo": float(bin_alvo),
                            "bin_fft_controle": float(bin_controle),
                            "epoca_inicio": inicio,
                            "epoca_fim_exclusivo": fim,
                            "fs": arquivo["fs"],
                            "estado_calibracao": estado,
                            "valor_msc": float(valor),
                            "p_value": calcular_p_value_msc(
                                valor, WINDOW_SIZE_EPOCHS
                            ),
                            "label": discretizar(valor, thresholds, NIVEIS_OBSERVACAO),
                        }
                    )

        valores_ausente = np.asarray(valores_ausente)
        valores_presente = np.asarray(valores_presente)
        b_ausente, contagens_ausente = montar_histograma(
            valores_ausente, thresholds, NIVEIS_OBSERVACAO, SMOOTHING
        )
        b_presente, contagens_presente = montar_histograma(
            valores_presente, thresholds, NIVEIS_OBSERVACAO, SMOOTHING
        )

        resultado_frequencia = {
            "Ausente": b_ausente,
            "Presente": b_presente,
            "contagens": {
                "Ausente": contagens_ausente,
                "Presente": contagens_presente,
            },
            "thresholds_msc": thresholds.copy(),
            "p_value_boundaries": list(P_VALUE_BOUNDARIES),
            "msc_critica": msc_critica,
            "alpha": SIGNIFICANCE_LEVEL,
            "n_arquivos": len(arquivos),
            "n_janelas": len(valores_ausente),
            "registros": registros,
        }
        matrizes_por_frequencia[float(freq_alvo)] = resultado_frequencia
        valores_globais_ausente.extend(valores_ausente)
        valores_globais_presente.extend(valores_presente)
        registros_globais.extend(registros)

    b_global_ausente, contagens_globais_ausente = montar_histograma(
        valores_globais_ausente, thresholds, NIVEIS_OBSERVACAO, SMOOTHING
    )
    b_global_presente, contagens_globais_presente = montar_histograma(
        valores_globais_presente, thresholds, NIVEIS_OBSERVACAO, SMOOTHING
    )
    matriz_global = {
        "Ausente": b_global_ausente,
        "Presente": b_global_presente,
        "contagens": {
            "Ausente": contagens_globais_ausente,
            "Presente": contagens_globais_presente,
        },
        "thresholds_msc": thresholds.copy(),
        "p_value_boundaries": list(P_VALUE_BOUNDARIES),
        "msc_critica": msc_critica,
        "alpha": SIGNIFICANCE_LEVEL,
        "n_arquivos": len(arquivos),
        "n_frequencias": len(matrizes_por_frequencia),
        "n_janelas_por_estado": len(valores_globais_ausente),
        "registros": registros_globais,
    }

    return {
        "global": matriz_global,
        "por_frequencia": matrizes_por_frequencia,
    }


def calcular_taxa_falso_positivo(dados, chave_n_janelas):
    """Calcula a fracao de registros ESP acima da MSC critica configurada."""
    falsos_positivos = sum(
        registro["estado_calibracao"] == "Ausente"
        and registro["valor_msc"] >= dados["msc_critica"]
        for registro in dados["registros"]
    )
    return falsos_positivos / dados[chave_n_janelas]


def construir_tabela_para_salvar(resultados):
    """Cria uma representacao JSON enxuta da matriz B e seus metadados."""
    matriz_global = resultados["global"]
    diagnosticos = {}
    for frequencia, dados in resultados["por_frequencia"].items():
        diagnosticos[str(frequencia)] = {
            "n_janelas_por_estado": dados["n_janelas"],
            "taxa_falso_positivo_esp": calcular_taxa_falso_positivo(
                dados, "n_janelas"
            ),
            "contagens": dados["contagens"],
        }

    probabilidades = {
        estado: matriz_global[estado]
        for estado in ESTADOS
    }
    return {
        "nome": "matriz_emissao_B",
        "descricao": (
            "P(label de observacao | estado), com frequencias agrupadas"
        ),
        "estados_linhas": list(ESTADOS),
        "labels_colunas": list(NIVEIS_OBSERVACAO),
        "matriz": [
            [probabilidades[estado][label] for label in NIVEIS_OBSERVACAO]
            for estado in ESTADOS
        ],
        "probabilidades": probabilidades,
        "contagens": matriz_global["contagens"],
        "configuracao": {
            "canal": CHANNEL_INDEX,
            "detector": DETECTOR,
            "tamanho_janela_epocas": WINDOW_SIZE_EPOCHS,
            "passo_janela_epocas": WINDOW_STEP_EPOCHS,
            "p_value_boundaries": list(P_VALUE_BOUNDARIES),
            "alpha": SIGNIFICANCE_LEVEL,
            "distribuicao_nula": {
                "nome": "Beta",
                "parametro_a": 1,
                "parametro_b": WINDOW_SIZE_EPOCHS - 1,
            },
            "uso_binsM": "controle_lateral; nao entra no calculo da MSC",
            "k_sintetico": K_SINTETICO,
            "smoothing": SMOOTHING,
        },
        "resumo": {
            "n_arquivos": matriz_global["n_arquivos"],
            "n_frequencias": matriz_global["n_frequencias"],
            "n_janelas_por_estado": matriz_global["n_janelas_por_estado"],
            "thresholds_msc": matriz_global["thresholds_msc"].tolist(),
            "msc_critica": matriz_global["msc_critica"],
            "taxa_falso_positivo_global_esp": calcular_taxa_falso_positivo(
                matriz_global, "n_janelas_por_estado"
            ),
        },
        "diagnostico_por_frequencia": diagnosticos,
        "aviso": (
            "A linha Presente usa uma senoide sintetica coerente em fase, "
            "sem jitter fisiologico, e nao representa calibracao clinica "
            "validada."
        ),
    }


def salvar_tabela_observacao(resultados, pasta_resultados=RESULTS_DIR):
    """Grava a matriz B e metadados em results/observation_matrix.json."""
    os.makedirs(pasta_resultados, exist_ok=True)
    caminho_saida = os.path.join(pasta_resultados, OUTPUT_FILE)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(
            construir_tabela_para_salvar(resultados),
            arquivo,
            ensure_ascii=False,
            indent=2,
        )
        arquivo.write("\n")
    return caminho_saida


if __name__ == "__main__":
    resultados = construir_matrizes_observacao(DATA_DIR)
    matriz_global = resultados["global"]
    caminho_saida = salvar_tabela_observacao(resultados)

    print(f"Detector: {DETECTOR}")
    print(
        f"Janela: {WINDOW_SIZE_EPOCHS} epocas; passo: {WINDOW_STEP_EPOCHS}; "
        f"canal: {CHANNEL_INDEX}; k sintetico: {K_SINTETICO}"
    )
    taxa_fp_global = calcular_taxa_falso_positivo(
        matriz_global, "n_janelas_por_estado"
    )

    print("\n=== Matriz B global (frequencias agrupadas) ===")
    print("Thresholds MSC:", matriz_global["thresholds_msc"])
    print(
        f"MSC critica (alpha={matriz_global['alpha']}): "
        f"{matriz_global['msc_critica']:.6f}; "
        f"FP global observado em ESP: {taxa_fp_global:.2%}"
    )
    print(
        f"Registros: {matriz_global['n_arquivos']} arquivos, "
        f"{matriz_global['n_frequencias']} frequencias, "
        f"{matriz_global['n_janelas_por_estado']} janelas por estado"
    )
    print("Contagens(Ausente):", matriz_global["contagens"]["Ausente"])
    print("B(Ausente):", matriz_global["Ausente"])
    print("Contagens(Presente):", matriz_global["contagens"]["Presente"])
    print("B(Presente):", matriz_global["Presente"])

    print("\nDiagnostico de falso positivo por frequencia:")
    for freq, dados in resultados["por_frequencia"].items():
        taxa_fp = calcular_taxa_falso_positivo(dados, "n_janelas")
        print(f"  {freq:4.1f} Hz: {taxa_fp:.2%} ({dados['n_janelas']} janelas)")

    print(f"\nTabela salva em: {caminho_saida}")
