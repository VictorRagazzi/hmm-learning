"""Compara histogramas com Rayleigh bruto usando somente parâmetros já salvos.

Não executa busca ou seleção. Recalibra apenas a emissão ESP/sintética por
janela e K fixados. Usa Viterbi completo, como o experimento contínuo original.
"""

import json
from pathlib import Path

import numpy as np

import continuous_rayleigh_hmm as crh
import detectors
import get_obs_matrix as gom
import histogram_hmm as hh


OUTPUT_FILE = "results/compare_histogram_rayleigh.json"
CONTINUOUS_SOURCE = "results/continuous_rayleigh_hmm.json"
SEARCH_SOURCE = "results/search_hmm_parameters.json"
RULE_KEYS = ("min_consecutive", "min_percent", "modo_regra_decisao")


def configuracoes_salvas():
    continuous = json.loads(Path(CONTINUOUS_SOURCE).read_text())
    configs = []
    for size, item in continuous["por_tamanho_janela"].items():
        configs.append({
            "nome": f"continuo_{size}", "origem": CONTINUOUS_SOURCE,
            "janela": int(size), "passo": item["window_step_epochs"],
            "k": gom.K_SINTETICO, "hmm": item["melhor_hmm"],
            "bruto": item["melhor_rayleigh_bruto_janelado"],
        })
    search = json.loads(Path(SEARCH_SOURCE).read_text())
    winner = search["melhor_hmm_por_combinacao"]["rayleigh"]
    configs.append({
        "nome": "search_anterior_rayleigh", "origem": SEARCH_SOURCE,
        "janela": search["configuracao_busca"]["window_size_epochs"],
        "passo": search["configuracao_busca"]["window_step_epochs"],
        "k": winner["k_sintetico"], "hmm": winner,
        "bruto": search["melhor_bruto_por_detector"]["rayleigh"],
    })
    return configs


def decidir(binaria, regra):
    if not len(binaria):
        return False
    run = longest = 0
    for value in binaria:
        run = run + 1 if value else 0
        longest = max(longest, run)
    consecutive = longest >= regra["min_consecutive"]
    fraction = np.mean(binaria) >= regra["min_percent"]
    return bool(consecutive and fraction if regra["modo_regra_decisao"] == "AND"
                else consecutive or fraction)


def resumir(registros):
    result = {}
    for method in ("hmm_histograma", "bruto_mesma_regra", "bruto_regra_anterior"):
        metrics = {}
        for group in ("estimulo", "lateral"):
            items = [r for r in registros if r["grupo"] == group]
            positives = sum(r[method] for r in items)
            metrics[group] = {"n": len(items), "positivos": positives,
                              "taxa": positives / len(items) if items else None}
        metrics["acuracia_balanceada"] = (
            metrics["estimulo"]["taxa"] + 1 - metrics["lateral"]["taxa"]) / 2
        result[method] = metrics
    return result


def executar():
    configs = configuracoes_salvas()
    esp, estimulo = crh._carregar_coeficientes("data", "rayleigh")
    # Extrai os coeficientes sintéticos uma vez por K, sem ajustar com estímulos.
    synthetic = {}
    for k in sorted({c["k"] for c in configs}):
        synthetic[k] = []
        for item in esp:
            arquivo = gom.carregar_mat(item["arquivo"])
            for freq in arquivo["freq_estim"]:
                injected = gom.injetar_tom_sintetico(
                    arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], freq, k)
                coeff, _ = gom.extrair_coeficientes_epocas(
                    injected, gom.CHANNEL_INDEX, arquivo["fs"], freq)
                synthetic[k].append(coeff)
    detector = detectors.obter_detector("rayleigh")
    results = []
    for config in configs:
        size, step = config["janela"], config["passo"]
        def statistics(coeff):
            return detectors.calcular_estatistica_janelas(coeff, detector, size, step)
        calibration = []
        for state, coefficients in (
            ("Ausente", [c for f in esp for _, c in f["sequencias"]]),
            ("Presente", synthetic[config["k"]]),
        ):
            for coeff in coefficients:
                values, _ = statistics(coeff)
                calibration.extend({"estado_calibracao": state, "valor_msc": float(v)}
                                   for v in values)
        models = hh.ajustar_histogramas(calibration)
        rows = []
        for file in estimulo:
            for group in ("estimulo", "lateral"):
                for freq, coeff in file[group]:
                    values, intervals = statistics(coeff)
                    states = hh.viterbi(values, config["hmm"]["matriz_a"],
                                        config["hmm"]["pi_inicial"], models)
                    p = np.asarray([detector.p_value(float(v), size) for v in values])
                    rows.append({
                        "participante": file["participante"], "arquivo": file["arquivo"],
                        "nivel_db": file["nivel_db"], "grupo": group, "frequencia": freq,
                        "intervalos": intervals, "valores_ord": values.tolist(),
                        "p_values": p.tolist(), "estados_hmm": states,
                        "hmm_histograma": decidir([s == "Presente" for s in states], config["hmm"]),
                        "bruto_mesma_regra": decidir(p <= .05, config["hmm"]),
                        "bruto_regra_anterior": decidir(p <= .05, config["bruto"]),
                    })
        summary = resumir(rows)
        # A baseline histórica precisa reproduzir as contagens já armazenadas.
        raw = summary["bruto_regra_anterior"]
        assert raw["estimulo"]["positivos"] == config["bruto"]["n_frequencias_estimulo_detectadas"]
        assert raw["lateral"]["positivos"] == config["bruto"]["n_frequencias_laterais_detectadas"]
        for state in hh.ESTADOS:
            assert np.isclose(sum(models["estados"][state]["probabilidades"]), 1)
        results.append({
            "configuracao": {"nome": config["nome"], "origem": config["origem"],
                "janela": size, "passo": step, "k": config["k"],
                "matriz_a": config["hmm"]["matriz_a"],
                "pi_inicial": config["hmm"]["pi_inicial"],
                "regra_hmm": {k: config["hmm"][k] for k in RULE_KEYS},
                "regra_bruto_anterior": {k: config["bruto"][k] for k in RULE_KEYS}},
            "modelos_emissao_histograma": models, "resumo": summary,
            "hmm_anterior": {k: config["hmm"][k] for k in
                             ("taxa_deteccao", "taxa_falso_positivo")},
            "por_nivel_db": {str(db): resumir([r for r in rows if r["nivel_db"] == db])
                             for db in (30, 40, 50, 60, 70)},
            "por_frequencia": {str(freq): resumir([r for r in rows if r["frequencia"] in (freq, freq+1)])
                               for freq in (81, 83, 85, 87, 89, 91, 93, 95)},
            "registros": rows,
        })
        print(config["nome"], {m: (round(v["estimulo"]["taxa"]*100, 2),
                                   round(v["lateral"]["taxa"]*100, 2))
                                for m, v in summary.items()}, flush=True)
    return {"nome": "comparacao_histograma_rayleigh_parametros_fixos",
        "busca_executada": False, "n_arquivos_esp": len(esp),
        "n_arquivos_estimulados": len(estimulo), "n_bins": hh.N_BINS,
        "pseudocontagem": hh.SMOOTHING, "alpha_bruto": .05,
        "aviso": "Parâmetros históricos selecionados neste conjunto. Teste exploratório, "
                 "sem nova seleção nem validação externa. Viterbi completo por arquivo/frequência.",
        "configuracoes": results}


if __name__ == "__main__":
    crh.salvar(executar(), OUTPUT_FILE)
    print(f"Comparação salva em {OUTPUT_FILE}")
