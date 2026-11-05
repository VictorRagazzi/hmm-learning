"""Valida o protocolo concatenado com leave-one-subject-out completo.

Em cada dobra, emissoes Beta e todos os hiperparametros sao escolhidos nos
dez participantes de treino. O participante restante e usado somente uma vez
para produzir as predicoes externas daquela dobra.
"""

import argparse
import json
import os
import tempfile

import numpy as np

import continuous_rayleigh_hmm as crh
import early_detection_rayleigh_hmm as early
import phase_transition_hmm as phase


OUTPUT_FILE = "./results/phase_transition_hmm_loso.json"


def _ajustar_modelos(protocolos):
    ausente, presente = [], []
    for protocolo in protocolos:
        if protocolo["grupo"] != "estimulo":
            continue
        for fase in protocolo["fases"]:
            destino = ausente if fase["nivel"] == "ESP" else presente
            destino.extend(fase["p_values"])
    return {
        "Ausente": crh._fit_beta(ausente),
        "Presente": crh._fit_beta(presente),
    }


def _sem_detalhes(resultado):
    return {k: v for k, v in resultado.items() if k != "detalhes"}


def _avaliar_hmm(protocolos, candidato, modelos):
    matriz_a = np.asarray(candidato[1], dtype=float)
    pi = candidato[2]
    consecutivas = candidato[3]
    detalhes = phase._avaliar(
        protocolos,
        lambda p: phase._estados_hmm(p, matriz_a, pi, modelos),
        consecutivas,
    )
    return phase._resumir(detalhes), detalhes


def _avaliar_bruto(protocolos, candidato):
    alpha, consecutivas = candidato[1], candidato[2]
    detalhes = phase._avaliar(
        protocolos, lambda p: p <= alpha, consecutivas
    )
    return phase._resumir(detalhes), detalhes


def executar_detector(pasta_dados, detector_nome):
    esp, estimulo = crh._carregar_coeficientes(pasta_dados, detector_nome)
    # Reutiliza a inclusao correta das laterais ESP feita pelo experimento
    # in-sample, sem duplicar as regras especiais do F espectral.
    phase._adicionar_laterais_esp(esp, detector_nome)
    cache = {}
    for tamanho in phase.WINDOW_SIZES:
        for modo in phase.STEP_MODES:
            passo = early._step(tamanho, modo)
            nome = f"window_{tamanho}_step_{passo}"
            protocolos, _ = phase._montar_protocolos(
                esp, estimulo, detector_nome, tamanho, passo
            )
            cache[nome] = protocolos

    participantes = sorted({
        p["participante"] for protocolos in cache.values() for p in protocolos
    })
    dobras = {}
    detalhes_hmm = []
    detalhes_bruto = []
    for indice, teste in enumerate(participantes, 1):
        print(
            f"{detector_nome}: dobra {indice}/{len(participantes)} ({teste})",
            flush=True,
        )
        melhor_hmm = melhor_bruto = None
        for configuracao, protocolos in cache.items():
            treino = [p for p in protocolos if p["participante"] != teste]
            modelos = _ajustar_modelos(treino)
            candidato_hmm = phase._melhor_hmm(treino, modelos)
            candidato_bruto = phase._melhor_bruto(treino)
            if candidato_hmm is not None:
                item = (candidato_hmm[0], configuracao, candidato_hmm, modelos)
                if melhor_hmm is None or item[0] > melhor_hmm[0]:
                    melhor_hmm = item
            if candidato_bruto is not None:
                item = (candidato_bruto[0], configuracao, candidato_bruto)
                if melhor_bruto is None or item[0] > melhor_bruto[0]:
                    melhor_bruto = item
        if melhor_hmm is None or melhor_bruto is None:
            raise RuntimeError(f"{teste}: nenhum candidato respeitou FP de treino")

        _, config_hmm, candidato_hmm, modelos = melhor_hmm
        teste_hmm = [
            p for p in cache[config_hmm] if p["participante"] == teste
        ]
        resumo_hmm, detalhe_hmm = _avaliar_hmm(
            teste_hmm, candidato_hmm, modelos
        )
        detalhes_hmm.extend(detalhe_hmm)

        _, config_bruto, candidato_bruto = melhor_bruto
        teste_bruto = [
            p for p in cache[config_bruto] if p["participante"] == teste
        ]
        resumo_bruto, detalhe_bruto = _avaliar_bruto(
            teste_bruto, candidato_bruto
        )
        detalhes_bruto.extend(detalhe_bruto)

        dobras[teste] = {
            "hmm": {
                "configuracao": config_hmm,
                "matriz_a": candidato_hmm[1],
                "pi_inicial": candidato_hmm[2],
                "min_consecutive": candidato_hmm[3],
                "modelos_emissao_beta": modelos,
                "metricas_treino": _sem_detalhes(candidato_hmm[4]),
                "metricas_teste": resumo_hmm,
            },
            "detector_bruto": {
                "configuracao": config_bruto,
                "alpha": candidato_bruto[1],
                "min_consecutive": candidato_bruto[2],
                "metricas_treino": _sem_detalhes(candidato_bruto[3]),
                "metricas_teste": resumo_bruto,
            },
        }

    return {
        "participantes": participantes,
        "hmm": {**phase._resumir(detalhes_hmm), "detalhes": detalhes_hmm},
        "detector_bruto": {
            **phase._resumir(detalhes_bruto), "detalhes": detalhes_bruto
        },
        "dobras": dobras,
    }


def executar(pasta_dados=crh.DATA_DIR, detectores=phase.DETECTOR_NAMES):
    return {
        "nome": "hmm_transicao_fases_leave_one_subject_out",
        "status": "complete",
        "metodo": (
            "Em cada dobra, emissoes, janela, passo, A, pi e regra sao "
            "selecionados nos outros dez participantes. O participante "
            "retido e usado somente na avaliacao."
        ),
        "configuracao": {
            "detectores": list(detectores),
            "window_sizes_epochs": list(phase.WINDOW_SIZES),
            "step_modes": list(phase.STEP_MODES),
            "fp_maximo_treino": phase.MAX_FALSE_POSITIVE,
        },
        "por_detector": {
            nome: executar_detector(pasta_dados, nome) for nome in detectores
        },
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


def consolidar(caminhos):
    """Combina artefatos por detector produzidos em processos separados."""
    resultados = []
    for caminho in caminhos:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            resultados.append(json.load(arquivo))
    base = resultados[0]
    por_detector = {}
    for resultado in resultados:
        por_detector.update(resultado["por_detector"])
    base["configuracao"]["detectores"] = list(por_detector)
    base["por_detector"] = por_detector
    return base


def imprimir(resultado):
    print("\nLeave-one-subject-out")
    for nome, item in resultado["por_detector"].items():
        hmm, bruto = item["hmm"], item["detector_bruto"]
        print(
            f"{nome:10s} HMM det={hmm['taxa_deteccao']:.2%}, "
            f"FP={hmm['taxa_fp_combinada']:.2%}, "
            f"tempo={hmm['tempo_total_medio_epocas_inclui_nao_detectados']:.1f}s | "
            f"bruto det={bruto['taxa_deteccao']:.2%}, "
            f"FP={bruto['taxa_fp_combinada']:.2%}, "
            f"tempo={bruto['tempo_total_medio_epocas_inclui_nao_detectados']:.1f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=crh.DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--detector", choices=phase.DETECTOR_NAMES)
    parser.add_argument("--merge-inputs", nargs="+")
    args = parser.parse_args()
    if args.merge_inputs:
        resultado = consolidar(args.merge_inputs)
    else:
        nomes = (args.detector,) if args.detector else phase.DETECTOR_NAMES
        resultado = executar(args.data_dir, nomes)
    salvar(resultado, args.output)
    imprimir(resultado)
