"""Compara tempo de deteccao causal do HMM continuo e Rayleigh acumulado.

As emissoes continuas sao ajustadas com os mesmos participantes avaliados,
deliberadamente in-sample. O HMM e atualizado somente com observacoes ja
disponiveis; nao usa o backtracking de uma gravacao completa para atribuir um
tempo de deteccao. O Rayleigh e recalculado cumulativamente nos mesmos pontos
de decisao. Ambos selecionam configuracoes com FP lateral global <= 5%.
"""

import argparse
import json
import os
import tempfile
from itertools import product

import numpy as np
from scipy.stats import beta as beta_dist

import continuous_rayleigh_hmm as crh
import detectors


OUTPUT_FILE = "./results/early_detection_rayleigh_hmm.json"
WINDOW_SIZES = (10, 20, 30, 60, 90, 120, 180)
STEP_MODES = ("half_overlap", "no_overlap")
P_SELF_VALUES = (0.50, 0.70, 0.85, 0.95, 0.99)
PI_ABSENT_VALUES = (0.50, 0.80, 0.95)
MIN_CONSECUTIVE_VALUES = tuple(range(1, 9))
MIN_PERCENT_VALUES = (0.05, 0.10, 0.20, 0.30, 0.50)
RULE_MODES = ("OR", "AND")
RAYLEIGH_ALPHA_VALUES = (
    0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001,
)
MAX_FALSE_POSITIVE = 0.05
P_EPS = 1e-9


def _step(window_size, mode):
    if mode == "half_overlap":
        return max(1, window_size // 2)
    if mode == "no_overlap":
        return window_size
    raise ValueError(f"Modo de passo desconhecido: {mode}")


def _online_hmm_states(p_values, matriz_a, pi_inicial, modelos):
    """Estado terminal de Viterbi a cada instante, sem informacao futura."""
    p = np.clip(np.asarray(p_values, dtype=float), P_EPS, 1.0 - P_EPS)
    if not len(p):
        return np.asarray([], dtype=bool)
    log_emissao = np.column_stack([
        beta_dist.logpdf(p, modelos[estado]["a"], modelos[estado]["b"])
        for estado in ("Ausente", "Presente")
    ])
    log_emissao = np.nan_to_num(log_emissao, neginf=-1e300, posinf=1e300)
    log_a = np.log(np.asarray(matriz_a, dtype=float) + 1e-300)
    delta = np.log(np.asarray(pi_inicial, dtype=float) + 1e-300) + log_emissao[0]
    estados = [int(np.argmax(delta))]
    for t in range(1, len(p)):
        candidatos = delta[:, None] + log_a
        delta = np.max(candidatos, axis=0) + log_emissao[t]
        estados.append(int(np.argmax(delta)))
    return np.asarray(estados, dtype=bool)


def _primeiro_disparo(binaria, finais_epocas, regra):
    consecutivas, percentual, modo = regra
    atual = 0
    positivos = 0
    for indice, valor in enumerate(np.asarray(binaria, dtype=bool)):
        atual = atual + 1 if valor else 0
        positivos += int(valor)
        c_ok = atual >= consecutivas
        p_ok = positivos / (indice + 1) >= percentual
        detectou = c_ok and p_ok if modo == "AND" else c_ok or p_ok
        if detectou:
            return int(finais_epocas[indice])
    return None


def _resultados_hmm(evidencias, matriz_a, pi_inicial, modelos):
    saida = []
    for arquivo in evidencias:
        item = {k: arquivo[k] for k in ("arquivo", "participante", "nivel_db")}
        for grupo in ("estimulo", "lateral"):
            item[grupo] = []
            for frequencia in arquivo[grupo]:
                estados = _online_hmm_states(
                    frequencia["p_values"], matriz_a, pi_inicial, modelos
                )
                item[grupo].append({
                    "frequencia": frequencia["frequencia"],
                    "binaria": estados,
                    "finais_epocas": np.asarray(
                        [fim for _, fim in frequencia["intervalos"]], dtype=int
                    ),
                    "n_epocas_total": frequencia["n_epocas_total"],
                })
        saida.append(item)
    return saida


def _resultados_detector_acumulado(evidencias, estimulo_coeficientes,
                                   detector_name):
    detector = detectors.obter_detector(detector_name)
    por_arquivo = {arquivo["arquivo"]: arquivo for arquivo in estimulo_coeficientes}
    saida = []
    for arquivo in evidencias:
        fonte = por_arquivo[arquivo["arquivo"]]
        item = {k: arquivo[k] for k in ("arquivo", "participante", "nivel_db")}
        for grupo in ("estimulo", "lateral"):
            item[grupo] = []
            for evidencia, (frequencia, coeficientes) in zip(
                arquivo[grupo], fonte[grupo], strict=True
            ):
                finais = np.asarray(
                    [fim for _, fim in evidencia["intervalos"]], dtype=int
                )
                p_values = np.asarray([
                    detector.p_value(detector.estatistica(coeficientes[:fim]), fim)
                    for fim in finais
                ])
                item[grupo].append({
                    "frequencia": frequencia,
                    "p_values_acumulados": p_values,
                    "finais_epocas": finais,
                    "n_epocas_total": int(len(coeficientes)),
                })
        saida.append(item)
    return saida


def _metricas_temporais(resultados, detectar, nivel_db=None):
    totais = {"estimulo": 0, "lateral": 0}
    positivos = {"estimulo": 0, "lateral": 0}
    tempos_detectados = {"estimulo": [], "lateral": []}
    tempos_exame = {"estimulo": [], "lateral": []}
    for arquivo in resultados:
        if nivel_db is not None and arquivo["nivel_db"] != nivel_db:
            continue
        for grupo in ("estimulo", "lateral"):
            for frequencia in arquivo[grupo]:
                tempo = detectar(frequencia)
                total = frequencia["n_epocas_total"]
                totais[grupo] += 1
                if tempo is not None:
                    positivos[grupo] += 1
                    tempos_detectados[grupo].append(tempo)
                    tempos_exame[grupo].append(tempo)
                else:
                    tempos_exame[grupo].append(total)
    taxa_deteccao = positivos["estimulo"] / totais["estimulo"]
    taxa_fp = positivos["lateral"] / totais["lateral"]

    def resumo_tempos(valores):
        return {
            "media_epocas": float(np.mean(valores)) if valores else None,
            "mediana_epocas": float(np.median(valores)) if valores else None,
        }

    return {
        "n_frequencias_estimulo": totais["estimulo"],
        "n_frequencias_estimulo_detectadas": positivos["estimulo"],
        "taxa_deteccao": taxa_deteccao,
        "n_frequencias_laterais": totais["lateral"],
        "n_frequencias_laterais_detectadas": positivos["lateral"],
        "taxa_falso_positivo": taxa_fp,
        "acuracia_balanceada": (taxa_deteccao + 1.0 - taxa_fp) / 2.0,
        "tempo_ate_deteccao_estimulo": resumo_tempos(
            tempos_detectados["estimulo"]
        ),
        "tempo_exame_estimulo_inclui_nao_detectados": resumo_tempos(
            tempos_exame["estimulo"]
        ),
    }


def _avaliar_regra_hmm(resultados, regra, nivel_db=None):
    return _metricas_temporais(
        resultados,
        lambda f: _primeiro_disparo(f["binaria"], f["finais_epocas"], regra),
        nivel_db,
    )


def _avaliar_alpha_rayleigh(resultados, alpha, nivel_db=None):
    def detectar(frequencia):
        indices = np.flatnonzero(frequencia["p_values_acumulados"] <= alpha)
        return int(frequencia["finais_epocas"][indices[0]]) if len(indices) else None

    return _metricas_temporais(resultados, detectar, nivel_db)


def _chave(metricas):
    tempo = metricas["tempo_exame_estimulo_inclui_nao_detectados"]["media_epocas"]
    return (
        metricas["acuracia_balanceada"],
        metricas["taxa_deteccao"],
        -metricas["taxa_falso_positivo"],
        -tempo,
    )


def _melhor_regra_hmm(resultados, regras):
    candidatos = []
    for regra in regras:
        metricas = _avaliar_regra_hmm(resultados, regra)
        if metricas["taxa_falso_positivo"] <= MAX_FALSE_POSITIVE:
            candidatos.append((_chave(metricas), regra, metricas))
    return max(candidatos, key=lambda x: x[0]) if candidatos else None


def _melhor_alpha_rayleigh(resultados):
    candidatos = []
    for alpha in RAYLEIGH_ALPHA_VALUES:
        metricas = _avaliar_alpha_rayleigh(resultados, alpha)
        if metricas["taxa_falso_positivo"] <= MAX_FALSE_POSITIVE:
            candidatos.append((_chave(metricas), alpha, metricas))
    return max(candidatos, key=lambda x: x[0]) if candidatos else None


def _adicionar_niveis(resultado, resultados, avaliar):
    resultado["por_nivel_db"] = {
        str(nivel): avaliar(resultados, nivel)
        for nivel in sorted({arquivo["nivel_db"] for arquivo in resultados})
    }


def _comparacao_pareada(hmm_resultados, rayleigh_resultados, regra, alpha):
    diferencas = []
    hmm_antes = iguais = rayleigh_antes = 0
    apenas_hmm = apenas_rayleigh = nenhum = 0
    for arquivo_hmm, arquivo_rayleigh in zip(
        hmm_resultados, rayleigh_resultados, strict=True
    ):
        for freq_hmm, freq_rayleigh in zip(
            arquivo_hmm["estimulo"], arquivo_rayleigh["estimulo"], strict=True
        ):
            tempo_hmm = _primeiro_disparo(
                freq_hmm["binaria"], freq_hmm["finais_epocas"], regra
            )
            indices = np.flatnonzero(
                freq_rayleigh["p_values_acumulados"] <= alpha
            )
            tempo_rayleigh = (
                int(freq_rayleigh["finais_epocas"][indices[0]])
                if len(indices) else None
            )
            if tempo_hmm is not None and tempo_rayleigh is not None:
                diferenca = tempo_hmm - tempo_rayleigh
                diferencas.append(diferenca)
                if diferenca < 0:
                    hmm_antes += 1
                elif diferenca > 0:
                    rayleigh_antes += 1
                else:
                    iguais += 1
            elif tempo_hmm is not None:
                apenas_hmm += 1
            elif tempo_rayleigh is not None:
                apenas_rayleigh += 1
            else:
                nenhum += 1
    return {
        "n_ambos_detectaram": len(diferencas),
        "hmm_detectou_antes": hmm_antes,
        "mesmo_tempo": iguais,
        "rayleigh_detectou_antes": rayleigh_antes,
        "apenas_hmm_detectou": apenas_hmm,
        "apenas_rayleigh_detectou": apenas_rayleigh,
        "nenhum_detectou": nenhum,
        "diferenca_hmm_menos_rayleigh_media_epocas": (
            float(np.mean(diferencas)) if diferencas else None
        ),
        "diferenca_hmm_menos_rayleigh_mediana_epocas": (
            float(np.median(diferencas)) if diferencas else None
        ),
    }


def executar_detector(esp, estimulo, detector_name="rayleigh"):
    regras = list(product(
        MIN_CONSECUTIVE_VALUES, MIN_PERCENT_VALUES, RULE_MODES
    ))
    por_configuracao = {}
    for window_size, step_mode in product(WINDOW_SIZES, STEP_MODES):
        step = _step(window_size, step_mode)
        chave_config = f"window_{window_size}_step_{step}"
        print(f"{chave_config}...", flush=True)
        evidencias, modelos, calibracao = crh._preparar_janela(
            esp, estimulo, window_size, step=step,
            exigir_janela_estimulo=False,
            detector_name=detector_name,
        )

        detector_resultados = _resultados_detector_acumulado(
            evidencias, estimulo, detector_name
        )
        melhor_rayleigh_item = _melhor_alpha_rayleigh(detector_resultados)
        melhor_rayleigh = None
        if melhor_rayleigh_item:
            _, alpha, metricas = melhor_rayleigh_item
            melhor_rayleigh = {"alpha_sequencial": alpha, **metricas}
            _adicionar_niveis(
                melhor_rayleigh, detector_resultados,
                lambda resultados, nivel: _avaliar_alpha_rayleigh(
                    resultados, alpha, nivel
                ),
            )

        melhor_hmm = None
        melhor_hmm_chave = None
        melhor_hmm_resultados = None
        melhor_hmm_regra = None
        for p_ausente, p_presente, pi_ausente in product(
            P_SELF_VALUES, P_SELF_VALUES, PI_ABSENT_VALUES
        ):
            matriz_a = np.asarray([
                [p_ausente, 1.0 - p_ausente],
                [1.0 - p_presente, p_presente],
            ])
            pi_inicial = [pi_ausente, 1.0 - pi_ausente]
            resultados_hmm = _resultados_hmm(
                evidencias, matriz_a, pi_inicial, modelos
            )
            candidato = _melhor_regra_hmm(resultados_hmm, regras)
            if candidato is None:
                continue
            chave, regra, metricas = candidato
            if melhor_hmm_chave is None or chave > melhor_hmm_chave:
                melhor_hmm_chave = chave
                melhor_hmm_regra = regra
                melhor_hmm_resultados = resultados_hmm
                melhor_hmm = {
                    "matriz_a": matriz_a.tolist(),
                    "pi_inicial": pi_inicial,
                    "min_consecutive": regra[0],
                    "min_percent": regra[1],
                    "modo_regra_decisao": regra[2],
                    **metricas,
                }
        if melhor_hmm is not None:
            _adicionar_niveis(
                melhor_hmm, melhor_hmm_resultados,
                lambda resultados, nivel: _avaliar_regra_hmm(
                    resultados, melhor_hmm_regra, nivel
                ),
            )

        comparacao_pareada = None
        if melhor_hmm is not None and melhor_rayleigh is not None:
            comparacao_pareada = _comparacao_pareada(
                melhor_hmm_resultados, detector_resultados,
                melhor_hmm_regra, melhor_rayleigh["alpha_sequencial"],
            )

        por_configuracao[chave_config] = {
            "window_size_epochs": window_size,
            "window_step_epochs": step,
            "step_mode": step_mode,
            "modelos_emissao_beta": modelos,
            "calibracao": calibracao,
            "melhor_hmm_causal": melhor_hmm,
            "melhor_detector_acumulado": melhor_rayleigh,
            # Compatibilidade com o artefato Rayleigh original.
            "melhor_rayleigh_acumulado": (
                melhor_rayleigh if detector_name == "rayleigh" else None
            ),
            "comparacao_pareada_estimulo": comparacao_pareada,
        }

    hmm_validos = [
        (nome, item["melhor_hmm_causal"])
        for nome, item in por_configuracao.items()
        if item["melhor_hmm_causal"] is not None
    ]
    rayleigh_validos = [
        (nome, item["melhor_detector_acumulado"])
        for nome, item in por_configuracao.items()
        if item["melhor_detector_acumulado"] is not None
    ]
    melhor_hmm_nome, melhor_hmm = max(
        hmm_validos, key=lambda item: _chave(item[1])
    )
    melhor_rayleigh_nome, melhor_rayleigh = max(
        rayleigh_validos, key=lambda item: _chave(item[1])
    )
    return {
        "nome": f"deteccao_precoce_hmm_continuo_vs_{detector_name}_acumulado",
        "status": "complete",
        "aviso_vazamento": (
            "Emissoes e parametros foram ajustados e avaliados nos mesmos "
            "participantes. Resultado deliberadamente in-sample."
        ),
        "definicao_tempo": (
            "Uma epoca tem um segundo nos dados atuais; o tempo e o indice "
            "final da primeira janela/prefixo que satisfaz a decisao causal."
        ),
        "configuracao": {
            "detector": detector_name,
            "window_sizes_epochs": list(WINDOW_SIZES),
            "step_modes": list(STEP_MODES),
            "fp_maximo_global": MAX_FALSE_POSITIVE,
            "rayleigh_alpha_values": list(RAYLEIGH_ALPHA_VALUES),
            "p_self_values": list(P_SELF_VALUES),
            "pi_absent_values": list(PI_ABSENT_VALUES),
            "min_consecutive_values": list(MIN_CONSECUTIVE_VALUES),
            "min_percent_values": list(MIN_PERCENT_VALUES),
            "rule_modes": list(RULE_MODES),
        },
        "melhor_hmm_configuracao": melhor_hmm_nome,
        "melhor_hmm_global": melhor_hmm,
        "melhor_detector_configuracao": melhor_rayleigh_nome,
        "melhor_detector_global": melhor_rayleigh,
        "melhor_rayleigh_configuracao": melhor_rayleigh_nome,
        "melhor_rayleigh_global": melhor_rayleigh,
        "por_configuracao": por_configuracao,
    }


def executar(pasta_dados=crh.DATA_DIR):
    esp, estimulo = crh._carregar_coeficientes(pasta_dados)
    return executar_detector(esp, estimulo, "rayleigh")


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
    print("\nDeteccao precoce causal (epocas = segundos nos dados atuais)")
    for nome, item in resultado["por_configuracao"].items():
        hmm = item["melhor_hmm_causal"]
        rayleigh = item["melhor_rayleigh_acumulado"]
        print(f"  {nome}")
        if hmm:
            tempo = hmm["tempo_ate_deteccao_estimulo"]["mediana_epocas"]
            print(f"    HMM: det={hmm['taxa_deteccao']:.2%}, "
                  f"FP={hmm['taxa_falso_positivo']:.2%}, mediana={tempo}s")
        if rayleigh:
            tempo = rayleigh["tempo_ate_deteccao_estimulo"]["mediana_epocas"]
            print(f"    Rayleigh: det={rayleigh['taxa_deteccao']:.2%}, "
                  f"FP={rayleigh['taxa_falso_positivo']:.2%}, "
                  f"mediana={tempo}s, alpha={rayleigh['alpha_sequencial']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=crh.DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    resultado = executar(args.data_dir)
    salvar(resultado, args.output)
    imprimir(resultado)
