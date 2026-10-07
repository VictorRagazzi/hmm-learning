"""Experimento HMM com emissoes continuas do teste de Rayleigh.

Este experimento e deliberadamente exploratorio e in-sample: usa todas as
janelas das frequencias estimuladas para ajustar a distribuicao do estado
Presente e avalia o HMM nos mesmos participantes. Ele existe para testar se a
matriz B sintetica e a discretizacao em cinco labels limitam o HMM atual.

Para cada tamanho de janela, os p-valores de Rayleigh sao modelados por duas
distribuicoes Beta continuas:

    Ausente: p-valores nas frequencias freqEstim dos arquivos ESP
    Presente: p-valores nas frequencias freqEstim dos arquivos com estimulo

O Viterbi recebe diretamente log f(p | estado), sem discretizar o p-valor.
As frequencias binsM dos arquivos estimulados ficam reservadas para medir FP.
"""

import argparse
import glob
import json
import os
import tempfile
from itertools import product

import numpy as np
from scipy.stats import beta as beta_dist

import detectors
import get_obs_matrix as gom
import hmm_inference as hmi


DATA_DIR = "./data"
OUTPUT_FILE = "./results/continuous_rayleigh_hmm.json"
WINDOW_SIZES = (10, 20, 30, 60)
P_SELF_VALUES = (0.50, 0.70, 0.85, 0.95, 0.99)
PI_ABSENT_VALUES = (0.50, 0.80, 0.95)
MIN_CONSECUTIVE_VALUES = tuple(range(1, 9))
MIN_PERCENT_VALUES = (0.05, 0.10, 0.20, 0.30, 0.50)
RULE_MODES = ("OR", "AND")
MAX_FALSE_POSITIVE = 0.05
ALPHA_RAW = 0.05
P_EPS = 1e-9


def _fit_beta(values):
    values = np.clip(np.asarray(values, dtype=float), P_EPS, 1.0 - P_EPS)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Ajuste Beta requer pelo menos dois p-valores finitos")
    a, b, _, _ = beta_dist.fit(values, floc=0.0, fscale=1.0)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        raise ValueError(f"Parametros Beta invalidos: a={a}, b={b}")
    return {
        "a": float(a),
        "b": float(b),
        "n": int(len(values)),
        "media_p": float(np.mean(values)),
        "mediana_p": float(np.median(values)),
    }


def _p_values_janelas(coeficientes, window_size, step, detector_name="rayleigh"):
    detector = detectors.obter_detector(detector_name)
    valores, intervalos = detectors.calcular_estatistica_janelas(
        coeficientes, detector, window_size, step
    )
    p_values = np.asarray(
        [detector.p_value(valor, window_size) for valor in valores], dtype=float
    )
    return p_values, intervalos


def _carregar_coeficientes(pasta_dados, detector_name="rayleigh"):
    def extrair_coeficiente(arquivo, frequencia):
        if detector_name == "spectral_f":
            return gom.extrair_coeficientes_espectrais_locais(
                arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], frequencia,
                arquivo["freq_estim"], detectors.SPECTRAL_F_NOISE_BINS,
            )[0]
        return gom.extrair_coeficientes_epocas(
            arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], frequencia
        )[0]

    esp = []
    for caminho in sorted(glob.glob(os.path.join(pasta_dados, gom.ESP_PATTERN))):
        arquivo = gom.carregar_mat(caminho)
        sequencias = []
        for frequencia in arquivo["freq_estim"]:
            coeficientes = extrair_coeficiente(arquivo, frequencia)
            sequencias.append((float(frequencia), coeficientes))
        esp.append({"arquivo": caminho, "sequencias": sequencias})
    if not esp:
        raise RuntimeError(f"Nenhum arquivo ESP encontrado em {pasta_dados}")

    estimulo = []
    padrao = os.path.join(pasta_dados, hmi.PRESENT_PATTERN)
    for caminho in sorted(glob.glob(padrao)):
        participante, _, nivel_db = hmi.parse_nome_arquivo(caminho)
        arquivo = hmi.carregar_mat_estimulo(caminho)

        def extrair(frequencias):
            saida = []
            for frequencia in frequencias:
                coeficientes = extrair_coeficiente(arquivo, frequencia)
                saida.append((float(frequencia), coeficientes))
            return saida

        estimulo.append({
            "arquivo": caminho,
            "participante": participante,
            "nivel_db": nivel_db,
            "estimulo": extrair(arquivo["freq_estim"]),
            "lateral": extrair(arquivo["bins_m"]),
        })
    if not estimulo:
        raise RuntimeError(f"Nenhum arquivo estimulado encontrado em {pasta_dados}")
    return esp, estimulo


def _preparar_janela(esp, estimulo, window_size, step=None,
                     exigir_janela_estimulo=True, detector_name="rayleigh"):
    step = step or max(1, window_size // 2)
    p_ausente = []
    arquivos_esp_usados = set()
    for arquivo in esp:
        for _, coeficientes in arquivo["sequencias"]:
            p_values, _ = _p_values_janelas(
                coeficientes, window_size, step, detector_name
            )
            if len(p_values):
                arquivos_esp_usados.add(arquivo["arquivo"])
                p_ausente.extend(p_values)

    evidencias = []
    p_presente = []
    for arquivo in estimulo:
        item = {
            "arquivo": arquivo["arquivo"],
            "participante": arquivo["participante"],
            "nivel_db": arquivo["nivel_db"],
            "estimulo": [],
            "lateral": [],
        }
        for grupo in ("estimulo", "lateral"):
            for frequencia, coeficientes in arquivo[grupo]:
                p_values, intervalos = _p_values_janelas(
                    coeficientes, window_size, step, detector_name
                )
                if not len(p_values) and exigir_janela_estimulo:
                    raise ValueError(
                        f"{arquivo['arquivo']} nao possui uma janela completa "
                        f"de {window_size} epocas"
                    )
                item[grupo].append({
                    "frequencia": frequencia,
                    "p_values": p_values,
                    "intervalos": intervalos,
                    "n_epocas_total": int(len(coeficientes)),
                })
                if grupo == "estimulo":
                    p_presente.extend(p_values)
        evidencias.append(item)

    modelos = {
        "Ausente": _fit_beta(p_ausente),
        "Presente": _fit_beta(p_presente),
    }
    calibracao = {
        "n_arquivos_esp_encontrados": len(esp),
        "n_arquivos_esp_usados": len(arquivos_esp_usados),
        "n_arquivos_estimulo": len(estimulo),
        "n_p_values_ausente": len(p_ausente),
        "n_p_values_presente": len(p_presente),
    }
    return evidencias, modelos, calibracao


def _viterbi_continuo(p_values, matriz_a, pi_inicial, modelos):
    p = np.clip(np.asarray(p_values, dtype=float), P_EPS, 1.0 - P_EPS)
    log_emissao = np.column_stack([
        beta_dist.logpdf(p, modelos[estado]["a"], modelos[estado]["b"])
        for estado in ("Ausente", "Presente")
    ])
    log_emissao = np.nan_to_num(log_emissao, neginf=-1e300, posinf=1e300)
    log_a = np.log(np.asarray(matriz_a, dtype=float) + 1e-300)
    delta = np.log(np.asarray(pi_inicial, dtype=float) + 1e-300) + log_emissao[0]
    psi = np.zeros((len(p), 2), dtype=np.int8)
    for t in range(1, len(p)):
        candidatos = delta[:, None] + log_a
        psi[t] = np.argmax(candidatos, axis=0)
        delta = np.max(candidatos, axis=0) + log_emissao[t]
    estados = np.zeros(len(p), dtype=np.int8)
    estados[-1] = np.argmax(delta)
    for t in range(len(p) - 2, -1, -1):
        estados[t] = psi[t + 1, estados[t + 1]]
    return estados == 1


def _stats_binaria(binaria):
    binaria = np.asarray(binaria, dtype=bool)
    maximo = atual = 0
    for valor in binaria:
        atual = atual + 1 if valor else 0
        maximo = max(maximo, atual)
    return {
        "n_janelas": int(len(binaria)),
        "max_consecutivas": int(maximo),
        "percentual_positivo": float(np.mean(binaria)) if len(binaria) else 0.0,
    }


def _rodar_caminhos(evidencias, matriz_a, pi_inicial, modelos):
    saida = []
    for arquivo in evidencias:
        item = {k: arquivo[k] for k in ("arquivo", "participante", "nivel_db")}
        for grupo in ("estimulo", "lateral"):
            item[grupo] = []
            for frequencia in arquivo[grupo]:
                caminho = _viterbi_continuo(
                    frequencia["p_values"], matriz_a, pi_inicial, modelos
                )
                item[grupo].append({
                    "frequencia": frequencia["frequencia"],
                    **_stats_binaria(caminho),
                })
        saida.append(item)
    return saida


def _resultado_bruto(evidencias):
    saida = []
    for arquivo in evidencias:
        item = {k: arquivo[k] for k in ("arquivo", "participante", "nivel_db")}
        for grupo in ("estimulo", "lateral"):
            item[grupo] = [{
                "frequencia": frequencia["frequencia"],
                **_stats_binaria(frequencia["p_values"] <= ALPHA_RAW),
            } for frequencia in arquivo[grupo]]
        saida.append(item)
    return saida


def _detectou(stats, consecutivas, percentual, modo):
    c_ok = stats["max_consecutivas"] >= consecutivas
    p_ok = stats["percentual_positivo"] >= percentual
    return c_ok and p_ok if modo == "AND" else c_ok or p_ok


def _metricas(resultados, regra, nivel_db=None):
    consecutivas, percentual, modo = regra
    totais = {"estimulo": 0, "lateral": 0}
    positivos = {"estimulo": 0, "lateral": 0}
    for arquivo in resultados:
        if nivel_db is not None and arquivo["nivel_db"] != nivel_db:
            continue
        for grupo in ("estimulo", "lateral"):
            for frequencia in arquivo[grupo]:
                totais[grupo] += 1
                positivos[grupo] += _detectou(
                    frequencia, consecutivas, percentual, modo
                )
    taxa_deteccao = positivos["estimulo"] / totais["estimulo"]
    taxa_fp = positivos["lateral"] / totais["lateral"]
    return {
        "n_frequencias_estimulo": totais["estimulo"],
        "n_frequencias_estimulo_detectadas": positivos["estimulo"],
        "taxa_deteccao": taxa_deteccao,
        "n_frequencias_laterais": totais["lateral"],
        "n_frequencias_laterais_detectadas": positivos["lateral"],
        "taxa_falso_positivo": taxa_fp,
        "acuracia_balanceada": (taxa_deteccao + 1.0 - taxa_fp) / 2.0,
    }


def _melhor_regra(resultados, regras):
    melhor = None
    melhor_chave = None
    for regra in regras:
        metricas = _metricas(resultados, regra)
        if metricas["taxa_falso_positivo"] > MAX_FALSE_POSITIVE:
            continue
        chave = (
            metricas["acuracia_balanceada"],
            metricas["taxa_deteccao"],
            -metricas["taxa_falso_positivo"],
        )
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor = {
                "min_consecutive": regra[0],
                "min_percent": regra[1],
                "modo_regra_decisao": regra[2],
                **metricas,
            }
    return melhor


def _completar_resultado(melhor, resultados):
    regra = (
        melhor["min_consecutive"], melhor["min_percent"],
        melhor["modo_regra_decisao"],
    )
    melhor["por_nivel_db"] = {
        str(nivel): _metricas(resultados, regra, nivel)
        for nivel in sorted({arquivo["nivel_db"] for arquivo in resultados})
    }
    return melhor


def _baseline_gravacao_completa(estimulo):
    resultados = []
    for arquivo in estimulo:
        item = {k: arquivo[k] for k in ("arquivo", "participante", "nivel_db")}
        for grupo in ("estimulo", "lateral"):
            item[grupo] = []
            for frequencia, coeficientes in arquivo[grupo]:
                p_value = detectors.RAYLEIGH.p_value(
                    detectors.RAYLEIGH.estatistica(coeficientes), len(coeficientes)
                )
                item[grupo].append({
                    "frequencia": frequencia,
                    **_stats_binaria([p_value <= ALPHA_RAW]),
                })
        resultados.append(item)
    regra = (1, 1.0, "AND")
    global_ = _metricas(resultados, regra)
    global_["por_nivel_db"] = {
        str(nivel): _metricas(resultados, regra, nivel)
        for nivel in sorted({arquivo["nivel_db"] for arquivo in resultados})
    }
    return global_


def executar(pasta_dados=DATA_DIR):
    esp, estimulo = _carregar_coeficientes(pasta_dados)
    regras = list(product(
        MIN_CONSECUTIVE_VALUES, MIN_PERCENT_VALUES, RULE_MODES
    ))
    resultados_janela = {}
    for window_size in WINDOW_SIZES:
        print(f"Janela {window_size} epocas...", flush=True)
        evidencias, modelos, calibracao = _preparar_janela(
            esp, estimulo, window_size
        )
        bruto_resultados = _resultado_bruto(evidencias)
        melhor_bruto = _melhor_regra(bruto_resultados, regras)
        if melhor_bruto is not None:
            _completar_resultado(melhor_bruto, bruto_resultados)

        melhor_hmm = None
        melhor_chave = None
        for p_ausente, p_presente, pi_ausente in product(
            P_SELF_VALUES, P_SELF_VALUES, PI_ABSENT_VALUES
        ):
            matriz_a = np.asarray([
                [p_ausente, 1.0 - p_ausente],
                [1.0 - p_presente, p_presente],
            ])
            pi_inicial = [pi_ausente, 1.0 - pi_ausente]
            caminhos = _rodar_caminhos(
                evidencias, matriz_a, pi_inicial, modelos
            )
            candidato = _melhor_regra(caminhos, regras)
            if candidato is None:
                continue
            chave = (
                candidato["acuracia_balanceada"],
                candidato["taxa_deteccao"],
                -candidato["taxa_falso_positivo"],
            )
            if melhor_chave is None or chave > melhor_chave:
                melhor_chave = chave
                melhor_hmm = {
                    "matriz_a": matriz_a.tolist(),
                    "pi_inicial": pi_inicial,
                    **candidato,
                }
                melhor_caminhos = caminhos
        if melhor_hmm is not None:
            _completar_resultado(melhor_hmm, melhor_caminhos)
        resultados_janela[str(window_size)] = {
            "window_size_epochs": window_size,
            "window_step_epochs": max(1, window_size // 2),
            "modelos_emissao_beta": modelos,
            "calibracao": calibracao,
            "melhor_hmm": melhor_hmm,
            "melhor_rayleigh_bruto_janelado": melhor_bruto,
        }

    validos = [
        resultado["melhor_hmm"] for resultado in resultados_janela.values()
        if resultado["melhor_hmm"] is not None
    ]
    melhor_global = max(validos, key=lambda x: (
        x["acuracia_balanceada"], x["taxa_deteccao"],
        -x["taxa_falso_positivo"],
    )) if validos else None
    melhor_tamanho = next((
        int(tamanho) for tamanho, resultado in resultados_janela.items()
        if resultado["melhor_hmm"] is melhor_global
    ), None)
    return {
        "nome": "hmm_rayleigh_emissao_continua_real",
        "status": "complete",
        "aviso_vazamento": (
            "B(Presente) foi ajustada com todas as frequencias estimuladas e "
            "os mesmos participantes usados na avaliacao. Resultado in-sample, "
            "deliberadamente com vazamento, sem estimativa de generalizacao."
        ),
        "configuracao": {
            "detector": "rayleigh",
            "emissao": "Beta continua ajustada aos p-valores por estado",
            "window_sizes_epochs": list(WINDOW_SIZES),
            "window_step": "metade do tamanho da janela",
            "fp_maximo_global": MAX_FALSE_POSITIVE,
            "alpha_rayleigh_bruto": ALPHA_RAW,
            "canal": gom.CHANNEL_INDEX,
            "p_self_values": list(P_SELF_VALUES),
            "pi_absent_values": list(PI_ABSENT_VALUES),
            "min_consecutive_values": list(MIN_CONSECUTIVE_VALUES),
            "min_percent_values": list(MIN_PERCENT_VALUES),
            "rule_modes": list(RULE_MODES),
        },
        "n_arquivos_esp": len(esp),
        "n_arquivos_estimulo": len(estimulo),
        "melhor_window_size_epochs": melhor_tamanho,
        "melhor_hmm_global": melhor_global,
        "por_tamanho_janela": resultados_janela,
        "baseline_rayleigh_gravacao_completa": _baseline_gravacao_completa(
            estimulo
        ),
    }


def salvar(resultado, caminho=OUTPUT_FILE):
    pasta = os.path.dirname(caminho) or "."
    os.makedirs(pasta, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=pasta, delete=False, suffix=".tmp"
    ) as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)
        temporario = arquivo.name
    os.replace(temporario, caminho)


def imprimir(resultado):
    print("\nRayleigh + emissoes continuas reais (resultado in-sample)")
    for tamanho, item in resultado["por_tamanho_janela"].items():
        hmm = item["melhor_hmm"]
        bruto = item["melhor_rayleigh_bruto_janelado"]
        if hmm is None:
            print(f"  janela {tamanho}: nenhum HMM com FP <= 5%")
            continue
        print(
            f"  janela {tamanho:>2}: HMM det={hmm['taxa_deteccao']:.2%}, "
            f"FP={hmm['taxa_falso_positivo']:.2%}, "
            f"BA={hmm['acuracia_balanceada']:.2%}"
        )
        if bruto is not None:
            print(
                f"             bruto janelado det={bruto['taxa_deteccao']:.2%}, "
                f"FP={bruto['taxa_falso_positivo']:.2%}, "
                f"BA={bruto['acuracia_balanceada']:.2%}"
            )
    base = resultado["baseline_rayleigh_gravacao_completa"]
    print(
        f"  gravacao completa: det={base['taxa_deteccao']:.2%}, "
        f"FP={base['taxa_falso_positivo']:.2%}, "
        f"BA={base['acuracia_balanceada']:.2%}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    resultado = executar(args.data_dir)
    salvar(resultado, args.output)
    imprimir(resultado)
