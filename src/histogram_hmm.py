"""Emissão por histograma da estatística ORD: ESP original e ESP + senoide.

Observações são valores do detector, não alturas do histograma nem p-valores.
As densidades são constantes dentro de cada intervalo comum aos dois estados.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import detectors
import get_obs_matrix as gom
import hmm_inference as hmi


N_BINS = 20
SMOOTHING = 0.5
OUTPUT_FILE = "results/histogram_hmm.json"
ESTADOS = ("Ausente", "Presente")


def ajustar_histogramas(registros, n_bins=N_BINS):
    """Normaliza contagens globais; integral da densidade de cada estado = 1."""
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins deve ser inteiro >= 2")
    # O registro atual reporta estatísticas normalizadas em [0, 1], inclusive
    # F/(1+F) para Hotelling e F espectral local. Não é a estatística F bruta.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    modelos = {"edges": edges.tolist(), "estados": {}}
    for estado in ESTADOS:
        values = np.asarray([
            r["valor_msc"] for r in registros if r["estado_calibracao"] == estado
        ])
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Calibração {estado} vazia ou não finita")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Estatística fora do suporte [0, 1]")
        counts, _ = np.histogram(values, bins=edges)
        masses = (counts + SMOOTHING) / (len(values) + n_bins * SMOOTHING)
        modelos["estados"][estado] = {
            "contagens": counts.tolist(), "n": len(values),
            "probabilidades": masses.tolist(),
            "densidades": (masses / np.diff(edges)).tolist(),
        }
    return modelos


def log_emissoes(valores, modelos):
    """Avalia log f_j(x); x=1 pertence ao último bin; fora do suporte é erro."""
    values = np.asarray(valores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Observações devem ser um vetor finito")
    edges = np.asarray(modelos["edges"])
    if np.any((values < edges[0]) | (values > edges[-1])):
        raise ValueError("Observação fora do suporte do histograma")
    ids = np.minimum(np.searchsorted(edges, values, side="right") - 1,
                     len(edges) - 2)
    return np.column_stack([
        np.log(np.asarray(modelos["estados"][s]["densidades"])[ids])
        for s in ESTADOS
    ])


def viterbi(valores, matriz_a, pi, modelos):
    """Viterbi completo com backtracking, usando densidades da estatística."""
    emissions = log_emissoes(valores, modelos)
    if not len(emissions):
        return []
    a = np.asarray(matriz_a, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if (a.shape != (2, 2) or pi.shape != (2,) or np.any(a < 0)
            or np.any(pi < 0) or not np.allclose(a.sum(axis=1), 1)
            or not np.isclose(pi.sum(), 1)):
        raise ValueError("A e pi devem ser distribuições válidas de dois estados")
    with np.errstate(divide="ignore"):
        log_a = np.log(a)
        delta = np.log(pi) + emissions[0]
    psi = np.zeros((len(emissions), 2), dtype=int)
    for t in range(1, len(emissions)):
        candidates = delta[:, None] + log_a
        psi[t] = np.argmax(candidates, axis=0)
        delta = np.max(candidates, axis=0) + emissions[t]
    path = np.zeros(len(emissions), dtype=int)
    path[-1] = np.argmax(delta)
    for t in range(len(path) - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return [ESTADOS[i] for i in path]


def executar(pasta_dados="data", detector="rayleigh", n_bins=N_BINS):
    gom.selecionar_detector(detector, interativo=False)
    calibration = gom.construir_matrizes_observacao(pasta_dados)
    records = calibration["global"]["registros"]
    modelos = ajustar_histogramas(records, n_bins)
    with open("results/transition_matrix.json", encoding="utf-8") as f:
        transitions = json.load(f)
    cfg_a = transitions["configuracao"]
    if (cfg_a["tamanho_janela_epocas"] != gom.WINDOW_SIZE_EPOCHS
            or cfg_a["passo_janela_epocas"] != gom.WINDOW_STEP_EPOCHS):
        raise ValueError("Regenere A: janelamento incompatível com a emissão")
    a = transitions["matriz"]
    sequences = []
    for path in sorted(Path(pasta_dados).glob("*dB.mat")):
        participant, _, db = hmi.parse_nome_arquivo(str(path))
        arquivo = hmi.carregar_mat_estimulo(str(path))
        for group, frequencies in (("alvo", arquivo["freq_estim"]),
                                   ("lateral", arquivo["bins_m"])):
            for freq_index, freq in enumerate(frequencies):
                if detector == "spectral_f":
                    coefficients, _ = gom.extrair_coeficientes_espectrais_locais(
                        arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], freq,
                        arquivo["freq_estim"], detectors.SPECTRAL_F_NOISE_BINS)
                else:
                    coefficients, _ = gom.extrair_coeficientes_epocas(
                        arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], freq)
                values, intervals = detectors.calcular_estatistica_janelas(
                    coefficients, gom.DETECTOR_ATUAL, gom.WINDOW_SIZE_EPOCHS,
                    gom.WINDOW_STEP_EPOCHS)
                states = viterbi(values, a, gom.PI_INICIAL, modelos)
                p_values = [gom.DETECTOR_ATUAL.p_value(float(v), gom.WINDOW_SIZE_EPOCHS)
                            for v in values]
                sequences.append({
                    "arquivo": str(path), "participante": participant,
                    "nivel_db": db, "grupo": group, "frequencia": float(freq),
                    "canal": gom.CHANNEL_INDEX, "fs": arquivo["fs"],
                    "frequencia_alvo": float(arquivo["freq_estim"][freq_index]),
                    "frequencia_controle": float(arquivo["bins_m"][freq_index]),
                    "intervalos": [list(i) for i in intervals],
                    "valores_ord": np.asarray(values).tolist(), "estados": states,
                    "p_values_diagnostico": p_values,
                    "decisao": hmi.avaliar_caminho(states),
                })
    diagnostics = gom.construir_tabela_para_salvar(calibration)
    summary = {}
    for group in ("alvo", "lateral"):
        items = [s for s in sequences if s["grupo"] == group]
        summary[group] = {
            "n_sequencias": len(items),
            "n_janelas": sum(len(s["valores_ord"]) for s in items),
            "taxa_detectada": (sum(s["decisao"]["detectado"] for s in items)
                               / len(items)) if items else None,
        }
    return {
        "nome": "hmm_histograma_estatistica_ord",
        "configuracao": {"detector": detector, "n_bins": n_bins,
            "window_size_epochs": gom.WINDOW_SIZE_EPOCHS,
            "window_step_epochs": gom.WINDOW_STEP_EPOCHS,
            "k_sintetico": gom.K_SINTETICO, "pseudocontagem": SMOOTHING,
            "canal": gom.CHANNEL_INDEX, "observacao": "estatistica_ord",
            "min_consecutive": hmi.MIN_CONSECUTIVE,
            "min_percent": hmi.MIN_PERCENT,
            "modo_regra_decisao": hmi.MODO_REGRA_DECISAO},
        "aviso": "Presente sintético coerente em fase; sem validação clínica. "
                 "Histograma é uma densidade constante por intervalo. "
                 "Viterbi completo usa observações futuras no backtracking.",
        "modelos_emissao_histograma": modelos, "matriz_a": a,
        "pi_inicial": gom.PI_INICIAL,
        "matriz_b_massas": [modelos["estados"][s]["probabilidades"] for s in ESTADOS],
        "calibracao": {"resumo": diagnostics["resumo"],
            "diagnostico_por_frequencia": diagnostics["diagnostico_por_frequencia"],
            "nota": "p-valores e labels são diagnósticos; emissões usam valor_ord. "
                    "valor_msc nos registros legados significa estatística do detector selecionado."},
        "registros_calibracao": records, "sequencias": sequences,
        "resumo_inferencia": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default="rayleigh",
                        choices=("rayleigh", "msc", "mmsc", "csm", "hotelling", "spectral_f"))
    parser.add_argument("--bins", type=int, default=N_BINS)
    parser.add_argument("--k", type=float, default=gom.K_SINTETICO)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    if not np.isfinite(args.k) or args.k < 0:
        parser.error("--k deve ser finito e não negativo")
    gom.K_SINTETICO = args.k
    result = executar(args.data, args.detector, args.bins)
    from continuous_rayleigh_hmm import salvar
    salvar(result, args.output)
    print(f"Histograma e inferência gravados em {args.output}")
