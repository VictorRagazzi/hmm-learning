"""Compara o ganho temporal do HMM para Rayleigh, MSC e MMSC isoladamente."""

import argparse
import json
import os
import tempfile

import continuous_rayleigh_hmm as crh
import early_detection_rayleigh_hmm as early


DETECTOR_NAMES = ("rayleigh", "msc", "mmsc", "hotelling", "spectral_f")
OUTPUT_FILE = "./results/early_detection_detectors_hmm.json"
RAYLEIGH_RESULT_FILE = "./results/early_detection_rayleigh_hmm.json"


def executar(pasta_dados=crh.DATA_DIR, caminho_saida=OUTPUT_FILE):
    por_detector = {}
    if os.path.exists(caminho_saida):
        with open(caminho_saida, "r", encoding="utf-8") as arquivo:
            anterior = json.load(arquivo)
        por_detector.update(anterior.get("por_detector", {}))
    if "rayleigh" not in por_detector and os.path.exists(RAYLEIGH_RESULT_FILE):
        with open(RAYLEIGH_RESULT_FILE, "r", encoding="utf-8") as arquivo:
            rayleigh = json.load(arquivo)
        if rayleigh.get("status") == "complete":
            por_detector["rayleigh"] = rayleigh
    for resultado in por_detector.values():
        if "melhor_detector_global" not in resultado:
            resultado["melhor_detector_global"] = resultado.get(
                "melhor_rayleigh_global"
            )
        if "melhor_detector_configuracao" not in resultado:
            resultado["melhor_detector_configuracao"] = resultado.get(
                "melhor_rayleigh_configuracao"
            )
        for item in resultado.get("por_configuracao", {}).values():
            if "melhor_detector_acumulado" not in item:
                item["melhor_detector_acumulado"] = item.get(
                    "melhor_rayleigh_acumulado"
                )
    for detector_name in DETECTOR_NAMES:
        if detector_name in por_detector:
            print(f"\n=== Detector {detector_name}: reutilizado ===", flush=True)
            continue
        print(f"\n=== Detector {detector_name} ===", flush=True)
        esp, estimulo = crh._carregar_coeficientes(pasta_dados, detector_name)
        por_detector[detector_name] = early.executar_detector(
            esp, estimulo, detector_name
        )
        salvar({
            "nome": "comparacao_hmm_continuo_vs_detector_acumulado",
            "status": "running",
            "detectores": list(DETECTOR_NAMES),
            "por_detector": por_detector,
        }, caminho_saida)

    resumo = {}
    for nome, resultado in por_detector.items():
        hmm = resultado["melhor_hmm_global"]
        bruto = resultado.get(
            "melhor_detector_global", resultado["melhor_rayleigh_global"]
        )
        resumo[nome] = {
            "melhor_hmm_configuracao": resultado["melhor_hmm_configuracao"],
            "hmm_taxa_deteccao": hmm["taxa_deteccao"],
            "hmm_taxa_falso_positivo": hmm["taxa_falso_positivo"],
            "hmm_acuracia_balanceada": hmm["acuracia_balanceada"],
            "hmm_tempo_mediano_deteccao_epocas": (
                hmm["tempo_ate_deteccao_estimulo"]["mediana_epocas"]
            ),
            "melhor_bruto_configuracao": resultado.get(
                "melhor_detector_configuracao",
                resultado["melhor_rayleigh_configuracao"],
            ),
            "bruto_taxa_deteccao": bruto["taxa_deteccao"],
            "bruto_taxa_falso_positivo": bruto["taxa_falso_positivo"],
            "bruto_acuracia_balanceada": bruto["acuracia_balanceada"],
            "bruto_tempo_mediano_deteccao_epocas": (
                bruto["tempo_ate_deteccao_estimulo"]["mediana_epocas"]
            ),
            "ganho_hmm_acuracia_balanceada": (
                hmm["acuracia_balanceada"] - bruto["acuracia_balanceada"]
            ),
            "ganho_hmm_deteccao": (
                hmm["taxa_deteccao"] - bruto["taxa_deteccao"]
            ),
        }
    return {
        "nome": "comparacao_hmm_continuo_vs_detector_acumulado",
        "status": "complete",
        "detectores": list(DETECTOR_NAMES),
        "aviso_vazamento": (
            "Emissoes e parametros foram ajustados e avaliados nos mesmos "
            "participantes. Resultado deliberadamente in-sample."
        ),
        "resumo_por_detector": resumo,
        "por_detector": por_detector,
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
    print("\nResumo HMM versus o mesmo detector acumulado")
    for nome, r in resultado["resumo_por_detector"].items():
        print(
            f"  {nome:8s} HMM det={r['hmm_taxa_deteccao']:.2%}, "
            f"FP={r['hmm_taxa_falso_positivo']:.2%}, "
            f"BA={r['hmm_acuracia_balanceada']:.2%} | "
            f"bruto det={r['bruto_taxa_deteccao']:.2%}, "
            f"FP={r['bruto_taxa_falso_positivo']:.2%}, "
            f"BA={r['bruto_acuracia_balanceada']:.2%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=crh.DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    resultado = executar(args.data_dir, args.output)
    salvar(resultado, args.output)
    imprimir(resultado)
