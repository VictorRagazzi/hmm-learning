"""Teste causal por paciente x frequência ESP -> 30 -> ... -> 70 dB.

Reutiliza configurações históricas fixas; não executa busca. Emissão ajustada
somente com ESP original e ESP + senoide, na estatística Rayleigh PLV².
"""

import json
from pathlib import Path

import h5py
import numpy as np

import continuous_rayleigh_hmm as crh
import detectors
import get_obs_matrix as gom
import histogram_hmm as hh
import phase_transition_hmm as phase
import phase_transition_discrete_hmm as discrete_phase


OUTPUT_FILE = "results/compare_histogram_phase_rayleigh.json"
REPORT_FILE = "results/compare_histogram_phase_rayleigh.md"
CONTINUOUS_SOURCE = "results/phase_transition_hmm.json"
DISCRETE_SOURCE = "results/phase_transition_discrete_hmm.json"
K_SINTETICO = 0.01


def estados_causais(valores, matriz_a, pi, modelos):
    emissions = hh.log_emissoes(valores, modelos)
    if not len(emissions):
        return np.zeros(0, dtype=bool)
    with np.errstate(divide="ignore"):
        log_a = np.log(np.asarray(matriz_a))
        delta = np.log(np.asarray(pi)) + emissions[0]
    states = [bool(np.argmax(delta))]
    for emission in emissions[1:]:
        delta = np.max(delta[:, None] + log_a, axis=0) + emission
        states.append(bool(np.argmax(delta)))
    return np.asarray(states)


def preparar_protocolos(esp, estimulo, size, step):
    index = phase._indexar_arquivos(esp, estimulo)
    detector = detectors.obter_detector("rayleigh")
    protocols = []
    for participant, files in sorted(index.items()):
        for group in ("estimulo", "lateral"):
            for freq_index in range(len(files[30][group])):
                phases, offset = [], 0
                for level in phase.LEVELS:
                    file = files[level]
                    key = "sequencias" if level == "ESP" and group == "estimulo" else group
                    freq, coeff = file[key][freq_index]
                    expected_key = "sequencias" if group == "estimulo" else group
                    if freq != files["ESP"][expected_key][freq_index][0]:
                        raise ValueError("Frequências não correspondem entre as fases")
                    values, intervals = detectors.calcular_estatistica_janelas(
                        coeff, detector, size, step)
                    phases.append({
                        "nivel": level, "arquivo": file["arquivo"],
                        "canal": gom.CHANNEL_INDEX, "fs": file["fs"],
                        "frequencia": freq, "inicio_global": offset,
                        "n_epocas": len(coeff), "valores_ord": values.tolist(),
                        "p_values": [detector.p_value(float(v), size) for v in values],
                        "intervalos_locais": intervals,
                        "finais_locais": [end for _, end in intervals],
                    })
                    offset += len(coeff)
                protocols.append({"participante": participant, "grupo": group,
                    "frequencia_indice": freq_index, "fases": phases,
                    "n_epocas_total": offset})
    return protocols


def avaliar(protocolos, binarias, consecutive):
    prepared = [(
        np.concatenate([f["p_values"] for f in p["fases"]]),
        np.concatenate([f["inicio_global"] + np.asarray(f["finais_locais"], dtype=int)
                        for f in p["fases"]]),
        np.concatenate([np.full(len(f["p_values"]), i, dtype=int)
                        for i, f in enumerate(p["fases"])]),
    ) for p in protocolos]
    details = phase._avaliar_preparado(protocolos, prepared, binarias, consecutive)
    for protocol, (_, ends, phases), binary, detail in zip(
            protocolos, prepared, binarias, details, strict=True):
        first = phase._primeiro_disparo(binary, ends, phases, consecutive)
        detail["disparo_em_qualquer_fase"] = first is not None
        detail["primeiro_disparo_total_s"] = first[0] if first else None
        detail["tempo_deteccao_desde_30_s"] = (
            detail["tempo_desde_inicio_estimulo_epocas"] if detail["detectou"] else None)
        detail["tempo_deteccao_total_s"] = (
            detail["tempo_total_epocas"] if detail["detectou"] else None)
        if detail["detectou"]:
            level_index = phase.LEVELS.index(detail["primeiro_nivel_db"])
            detail["tempo_na_intensidade_s"] = (
                detail["tempo_total_epocas"] - protocol["fases"][level_index]["inicio_global"])
        else:
            detail["tempo_na_intensidade_s"] = None
    return details


def resumir(details):
    def metricas(items):
        targets = [d for d in items if d["grupo"] == "estimulo"]
        controls = [d for d in items if d["grupo"] == "lateral"]
        detected = [d for d in targets if d["detectou"]]
        # FP solicitado: qualquer disparo da lateral no protocolo completo.
        fp = sum(d["disparo_em_qualquer_fase"] for d in controls)
        det_rate = len(detected) / len(targets)
        fp_rate = fp / len(controls)
        times = [d["tempo_deteccao_desde_30_s"] for d in detected]
        return {
            "n_alvos": len(targets), "n_laterais": len(controls),
            "detectados": len(detected), "taxa_deteccao": det_rate,
            "fp_lateral": fp, "taxa_fp_lateral": fp_rate,
            "acuracia_balanceada_fp_lateral": (det_rate + 1 - fp_rate) / 2,
            "fp_alvo_esp": sum(d["fp_esp"] for d in targets),
            "fp_lateral_apos_30": sum(d["detectou"] for d in controls),
            "tempo_desde_30_medio_detectados_s": float(np.mean(times)) if times else None,
            "tempo_desde_30_mediano_detectados_s": float(np.median(times)) if times else None,
            "duracao_media_desde_30_inclui_nao_detectados_s": float(np.mean([
                d["tempo_desde_inicio_estimulo_epocas"] for d in targets])),
            "primeira_deteccao_por_db": {str(db): sum(d["primeiro_nivel_db"] == db for d in targets)
                                        for db in phase.LEVELS[1:]},
        }
    return {"global": metricas(details),
        "por_paciente": {p: metricas([d for d in details if d["participante"] == p])
                         for p in sorted({d["participante"] for d in details})},
        "por_frequencia": {str(i): metricas([d for d in details if d["frequencia_indice"] == i])
                           for i in sorted({d["frequencia_indice"] for d in details})},
        "diagnostico_fp_combinado_historico": phase._resumir(details)}


def comparar_pareado(hmm, raw):
    key = lambda d: (d["participante"], d["frequencia_indice"])
    controls = {key(d): d for d in raw if d["grupo"] == "estimulo"}
    pairs = []
    for h in hmm:
        if h["grupo"] != "estimulo":
            continue
        r = controls[key(h)]
        pairs.append({"paciente": h["participante"], "frequencia": h["frequencia"],
            "hmm_detectou": h["detectou"], "rayleigh_detectou": r["detectou"],
            "hmm_tempo_s": h["tempo_deteccao_desde_30_s"],
            "rayleigh_tempo_s": r["tempo_deteccao_desde_30_s"]})
    common = [p for p in pairs if p["hmm_detectou"] and p["rayleigh_detectou"]]
    return {"ambos_detectaram": len(common),
        "hmm_antes": sum(p["hmm_tempo_s"] < p["rayleigh_tempo_s"] for p in common),
        "rayleigh_antes": sum(p["hmm_tempo_s"] > p["rayleigh_tempo_s"] for p in common),
        "empates": sum(p["hmm_tempo_s"] == p["rayleigh_tempo_s"] for p in common),
        "so_hmm": sum(p["hmm_detectou"] and not p["rayleigh_detectou"] for p in pairs),
        "so_rayleigh": sum(p["rayleigh_detectou"] and not p["hmm_detectou"] for p in pairs),
        "pares": pairs}


def executar():
    historical = json.loads(Path(CONTINUOUS_SOURCE).read_text())["por_detector"]["rayleigh"]
    discrete = json.loads(Path(DISCRETE_SOURCE).read_text())["por_detector"]["rayleigh"]
    configs = [(f"continuo_{name}", item, CONTINUOUS_SOURCE)
               for name, item in historical["por_configuracao"].items()]
    dname = discrete["melhor_hmm_configuracao"]
    ditem = dict(discrete["por_configuracao"][dname])
    ditem["detector_bruto"] = ditem["detector_bruto_janelado"]
    configs.append((f"discreto_{dname}", ditem, DISCRETE_SOURCE))
    esp, estimulo = crh._carregar_coeficientes("data", "rayleigh")
    phase._adicionar_laterais_esp(esp, "rayleigh")
    # Tempos em épocas só podem ser apresentados como segundos após esta checagem.
    for item in esp + estimulo:
        with h5py.File(item["arquivo"], "r") as f:
            fs = float(np.asarray(f["Fs"]).ravel()[0])
            item["fs"] = fs
            if not np.isclose(f["x"].shape[2] / fs, 1):
                raise ValueError("Este protocolo exige época de um segundo para o relógio histórico")
    synthetic = []
    for item in esp:
        file = gom.carregar_mat(item["arquivo"])
        for freq in file["freq_estim"]:
            injected = gom.injetar_tom_sintetico(
                file["x"], gom.CHANNEL_INDEX, file["fs"], freq, K_SINTETICO)
            synthetic.append(gom.extrair_coeficientes_epocas(
                injected, gom.CHANNEL_INDEX, file["fs"], freq)[0])
    detector = detectors.obter_detector("rayleigh")
    results = {}
    for name, config, source in configs:
        size, step = config["window_size_epochs"], config["window_step_epochs"]
        protocols = preparar_protocolos(esp, estimulo, size, step)
        records = []
        for state, coefficients in (("Ausente", [c for f in esp for _, c in f["sequencias"]]),
                                    ("Presente", synthetic)):
            for coeff in coefficients:
                values, _ = detectors.calcular_estatistica_janelas(coeff, detector, size, step)
                records.extend({"estado_calibracao": state, "valor_msc": float(v)} for v in values)
        models = hh.ajustar_histogramas(records)
        h, r = config["hmm"], config["detector_bruto"]
        # Controle de implementação: reaplica exatamente as emissões antigas.
        # Controle metodológico: histograma da estatística com Presente real,
        # mantendo a origem dos dados do modelo antigo. Só diagnóstico in-sample.
        real_records = [
            {"estado_calibracao": "Ausente" if f["nivel"] == "ESP" else "Presente",
             "valor_msc": float(v)}
            for p in protocols if p["grupo"] == "estimulo"
            for f in p["fases"] for v in f["valores_ord"]
        ]
        real_models = hh.ajustar_histogramas(real_records)
        hmm_states, raw_states, old_states, real_hist_states = [], [], [], []
        for protocol in protocols:
            values = np.concatenate([p["valores_ord"] for p in protocol["fases"]])
            p_values = np.concatenate([p["p_values"] for p in protocol["fases"]])
            hmm_states.append(estados_causais(values, h["matriz_a"], h["pi_inicial"], models))
            raw_states.append(p_values <= r["alpha"])
            if "modelos_emissao_beta" in config:
                old_states.append(phase._estados_hmm(
                    p_values, h["matriz_a"], h["pi_inicial"], config["modelos_emissao_beta"]))
            else:
                labels = [discrete_phase._label(float(p)) for p in p_values]
                old_states.append(discrete_phase._online_states(
                    labels, h["matriz_a"], h["pi_inicial"], config["B"]))
            real_hist_states.append(estados_causais(values, h["matriz_a"], h["pi_inicial"], real_models))
        hmm_details = avaliar(protocols, hmm_states, h["min_consecutive"])
        raw_details = avaliar(protocols, raw_states, r["min_consecutive"])
        old_details = avaliar(protocols, old_states, h["min_consecutive"])
        real_hist_details = avaliar(protocols, real_hist_states, h["min_consecutive"])
        # Reproduzir a baseline prova que sequência e relógio são os históricos.
        old_metrics = phase._resumir(raw_details)
        for metric in ("taxa_deteccao", "taxa_fp_combinada"):
            assert np.isclose(old_metrics[metric], r[metric]), (name, metric)
            assert np.isclose(phase._resumir(old_details)[metric], h[metric]), (name, "hmm antigo", metric)
        results[name] = {
            "origem": source, "janela": size, "passo": step,
            "matriz_a": h["matriz_a"], "pi": h["pi_inicial"],
            "min_consecutive_hmm": h["min_consecutive"],
            "alpha_rayleigh": r["alpha"], "min_consecutive_rayleigh": r["min_consecutive"],
            "modelos_emissao_histograma": models,
            "hmm": resumir(hmm_details), "rayleigh": resumir(raw_details),
            "hmm_antigo_recalculado": resumir(old_details),
            "histograma_presente_real_diagnostico": resumir(real_hist_details),
            "modelos_histograma_real_diagnostico": real_models,
            "pareamento": comparar_pareado(hmm_details, raw_details),
            "detalhes_hmm": hmm_details, "detalhes_rayleigh": raw_details,
            "detalhes_hmm_antigo": old_details,
            "detalhes_histograma_real_diagnostico": real_hist_details,
            "protocolos": protocols,
        }
        print(name, {method: (round(results[name][method]["global"]["taxa_deteccao"]*100, 2),
                              round(results[name][method]["global"]["taxa_fp_lateral"]*100, 2))
                     for method in ("hmm", "rayleigh", "hmm_antigo_recalculado",
                                    "histograma_presente_real_diagnostico")}, flush=True)
    # Estes destaques são escolhidos nos artefatos antigos, antes do novo teste.
    hkey = "continuo_" + historical["melhor_hmm_configuracao"]
    rkey = "continuo_" + historical["melhor_bruto_configuracao"]
    return {"nome": "histograma_rayleigh_protocolo_concatenado_parametros_fixos",
        "busca_executada": False, "levels": list(phase.LEVELS),
        "k_sintetico": K_SINTETICO, "n_bins": hh.N_BINS, "canal": gom.CHANNEL_INDEX,
        "n_pacientes": len(esp),
        "n_trajetorias_alvo": sum(p["grupo"] == "estimulo" for p in protocols),
        "n_trajetorias_laterais": sum(p["grupo"] == "lateral" for p in protocols),
        "metodo": "Janelas dentro de cada arquivo; observações concatenadas por paciente/frequência. "
                  "Estado terminal Viterbi causal, sem reinicializar entre intensidades. "
                  "FP lateral conta qualquer disparo inclusive ESP. Detecção alvo após início de 30 dB.",
        "aviso": "Teste com parâmetros históricos in-sample, sem nova seleção. "
                 "Emissão Presente sintética; não é validação clínica ou externa.",
        "destaque_hmm_predefinido": hkey, "destaque_rayleigh_predefinido": rkey,
        "destaque_discreto_predefinido": "discreto_" + dname,
        "pareamento_destaques": comparar_pareado(results[hkey]["detalhes_hmm"],
                                                results[rkey]["detalhes_rayleigh"]),
        "configuracoes": results}


def relatorio(result):
    h = result["configuracoes"][result["destaque_hmm_predefinido"]]
    r = result["configuracoes"][result["destaque_rayleigh_predefinido"]]
    lines = ["# Comparação causal por paciente × frequência", "",
        "ESP → 30 → 40 → 50 → 60 → 70 dB. Parâmetros históricos fixos; sem search.", "",
        f"HMM: janela/passo {h['janela']}/{h['passo']}; Rayleigh: {r['janela']}/{r['passo']}.", "",
        "FP = laterais com qualquer disparo / laterais medidas (inclui ESP). "
        "Tempo = média desde início de 30 dB, somente entre detectados. "
        "Não detectados têm tempo nulo; duração censurada é reportada separadamente no JSON.", "",
        "| Paciente | Detecção HMM | FP HMM | Tempo HMM (s) | Detecção Rayleigh | FP Rayleigh | Tempo Rayleigh (s) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def time(v):
        return "—" if v is None else f"{v:.0f}"
    for patient, hm in h["hmm"]["por_paciente"].items():
        rm = r["rayleigh"]["por_paciente"][patient]
        lines.append(f"| {patient} | {hm['taxa_deteccao']:.2%} | {hm['taxa_fp_lateral']:.2%} | "
            f"{time(hm['tempo_desde_30_medio_detectados_s'])} | {rm['taxa_deteccao']:.2%} | "
            f"{rm['taxa_fp_lateral']:.2%} | {time(rm['tempo_desde_30_medio_detectados_s'])} |")
    lines += ["", "## Todas as configurações fixas", "",
        "| Origem / janela / passo | Detecção HMM | FP HMM | Detecção Rayleigh | FP Rayleigh |",
        "|---|---:|---:|---:|---:|"]
    for name, item in result["configuracoes"].items():
        hm, rm = item["hmm"]["global"], item["rayleigh"]["global"]
        lines.append(f"| {name} | {hm['taxa_deteccao']:.2%} | {hm['taxa_fp_lateral']:.2%} | "
                     f"{rm['taxa_deteccao']:.2%} | {rm['taxa_fp_lateral']:.2%} |")
    lines += ["", "## Controle da forma antiga e da origem de Presente", "",
        "Todos usam A, pi, janela e regra do HMM histórico, sem busca. "
        "Presente real é somente diagnóstico in-sample com os mesmos dados da versão antiga.", "",
        "| Configuração | HMM antigo: det./FP lateral | Histograma Presente real: det./FP lateral | Histograma Presente sintético: det./FP lateral |",
        "|---|---:|---:|---:|"]
    for name, item in result["configuracoes"].items():
        cells = []
        for method in ("hmm_antigo_recalculado", "histograma_presente_real_diagnostico", "hmm"):
            metrics = item[method]["global"]
            cells.append(f"{metrics['taxa_deteccao']:.2%} / {metrics['taxa_fp_lateral']:.2%}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    result = executar()
    crh.salvar(result, OUTPUT_FILE)
    Path(REPORT_FILE).write_text(relatorio(result), encoding="utf-8")
    print(f"Salvos {OUTPUT_FILE} e {REPORT_FILE}")
