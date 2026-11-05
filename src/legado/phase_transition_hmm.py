"""Testa o HMM como detector de transicao no protocolo ESP -> 30 -> ... -> 70 dB.

Este experimento e deliberadamente in-sample. As emissoes e os parametros sao
ajustados nos mesmos participantes avaliados. Janelas nunca atravessam a
fronteira entre dois arquivos; somente as sequencias de evidencias resultantes
sao concatenadas por participante e frequencia.
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
import early_detection_rayleigh_hmm as early
import get_obs_matrix as gom


OUTPUT_FILE = "./results/phase_transition_hmm.json"
DETECTOR_NAMES = ("rayleigh", "msc", "mmsc", "hotelling", "spectral_f")
LEVELS = ("ESP", 30, 40, 50, 60, 70)
WINDOW_SIZES = early.WINDOW_SIZES
STEP_MODES = early.STEP_MODES
P_SELF_VALUES = early.P_SELF_VALUES
PI_ABSENT_VALUES = early.PI_ABSENT_VALUES
MIN_CONSECUTIVE_VALUES = early.MIN_CONSECUTIVE_VALUES
ALPHA_VALUES = early.RAYLEIGH_ALPHA_VALUES
MAX_FALSE_POSITIVE = 0.05
P_EPS = 1e-9


def _participante_esp(caminho):
    nome = os.path.basename(caminho)
    if not nome.endswith("ESP.mat"):
        raise ValueError(f"Nome ESP inesperado: {nome}")
    return nome[:-7]


def _indexar_arquivos(esp, estimulo):
    indice = {}
    for arquivo in esp:
        participante = _participante_esp(arquivo["arquivo"])
        indice.setdefault(participante, {})["ESP"] = arquivo
    for arquivo in estimulo:
        indice.setdefault(arquivo["participante"], {})[
            arquivo["nivel_db"]
        ] = arquivo
    incompletos = {
        participante: [nivel for nivel in LEVELS if nivel not in arquivos]
        for participante, arquivos in indice.items()
        if any(nivel not in arquivos for nivel in LEVELS)
    }
    if incompletos:
        raise ValueError(f"Protocolos incompletos: {incompletos}")
    return indice


def _janelas(coeficientes, tamanho, passo, detector_nome):
    p_values, intervalos = crh._p_values_janelas(
        coeficientes, tamanho, passo, detector_nome
    )
    finais = np.asarray([fim for _, fim in intervalos], dtype=int)
    return p_values, finais, int(len(coeficientes))


def _montar_protocolos(esp, estimulo, detector_nome, tamanho, passo):
    indice = _indexar_arquivos(esp, estimulo)
    protocolos = []
    p_ausente = []
    p_presente = []
    for participante in sorted(indice):
        arquivos = indice[participante]
        for grupo in ("estimulo", "lateral"):
            n_freq = len(arquivos[30][grupo])
            for freq_idx in range(n_freq):
                fases = []
                offset = 0
                for nivel in LEVELS:
                    arquivo = arquivos[nivel]
                    if nivel == "ESP":
                        chave = "sequencias" if grupo == "estimulo" else "lateral"
                        frequencia, coeficientes = arquivo[chave][freq_idx]
                    else:
                        frequencia, coeficientes = arquivo[grupo][freq_idx]
                    p_values, finais, n_epocas = _janelas(
                        coeficientes, tamanho, passo, detector_nome
                    )
                    if grupo == "estimulo":
                        (p_ausente if nivel == "ESP" else p_presente).extend(p_values)
                    fases.append({
                        "nivel": nivel,
                        "frequencia": float(frequencia),
                        "p_values": p_values,
                        "finais_locais": finais,
                        "inicio_global": offset,
                        "n_epocas": n_epocas,
                    })
                    offset += n_epocas
                protocolos.append({
                    "participante": participante,
                    "grupo": grupo,
                    "frequencia_indice": freq_idx,
                    "fases": fases,
                    "n_epocas_total": offset,
                })
    modelos = {
        "Ausente": crh._fit_beta(p_ausente),
        "Presente": crh._fit_beta(p_presente),
    }
    return protocolos, modelos


def _concatenar(protocolo):
    p_values = np.concatenate([fase["p_values"] for fase in protocolo["fases"]])
    finais = np.concatenate([
        fase["inicio_global"] + fase["finais_locais"]
        for fase in protocolo["fases"]
    ])
    fase_indices = np.concatenate([
        np.full(len(fase["p_values"]), indice, dtype=int)
        for indice, fase in enumerate(protocolo["fases"])
    ])
    return p_values, finais, fase_indices


def _estados_hmm(p_values, matriz_a, pi_inicial, modelos):
    p = np.clip(np.asarray(p_values), P_EPS, 1.0 - P_EPS)
    log_emissao = np.column_stack([
        beta_dist.logpdf(p, modelos[estado]["a"], modelos[estado]["b"])
        for estado in ("Ausente", "Presente")
    ])
    log_emissao = np.nan_to_num(log_emissao, neginf=-1e300, posinf=1e300)
    log_a = np.log(np.asarray(matriz_a) + 1e-300)
    delta = np.log(np.asarray(pi_inicial) + 1e-300) + log_emissao[0]
    estados = [int(np.argmax(delta))]
    for t in range(1, len(p)):
        delta = np.max(delta[:, None] + log_a, axis=0) + log_emissao[t]
        estados.append(int(np.argmax(delta)))
    return np.asarray(estados, dtype=bool)


def _primeiro_disparo(binaria, finais, fase_indices, min_consecutive, fase_min=0):
    consecutivas = 0
    for valor, fim, fase in zip(binaria, finais, fase_indices, strict=True):
        consecutivas = consecutivas + 1 if valor else 0
        if fase >= fase_min and consecutivas >= min_consecutive:
            return int(fim), int(fase)
    return None


def _avaliar_preparado(protocolos, preparados, binarias, min_consecutive):
    detalhes = []
    for protocolo, preparado, binaria in zip(
        protocolos, preparados, binarias, strict=True
    ):
        _, finais, fases = preparado
        fp_esp = _primeiro_disparo(
            binaria[fases == 0], finais[fases == 0], fases[fases == 0],
            min_consecutive,
        ) is not None
        disparo = _primeiro_disparo(
            binaria, finais, fases, min_consecutive, fase_min=1
        )
        detectou = disparo is not None
        nivel = LEVELS[disparo[1]] if detectou else None
        inicio_estimulo = protocolo["fases"][1]["inicio_global"]
        detalhe = {
            "participante": protocolo["participante"],
            "grupo": protocolo["grupo"],
            "frequencia_indice": protocolo["frequencia_indice"],
            "frequencia": protocolo["fases"][0]["frequencia"],
            "fp_esp": fp_esp,
            "detectou": detectou,
            "primeiro_nivel_db": nivel,
            "tempo_total_epocas": disparo[0] if detectou else protocolo["n_epocas_total"],
            "tempo_desde_inicio_estimulo_epocas": (
                disparo[0] - inicio_estimulo if detectou
                else protocolo["n_epocas_total"] - inicio_estimulo
            ),
        }
        detalhes.append(detalhe)
    return detalhes


def _avaliar(protocolos, gerar_binaria, min_consecutive):
    preparados = [_concatenar(protocolo) for protocolo in protocolos]
    binarias = [gerar_binaria(item[0]) for item in preparados]
    return _avaliar_preparado(
        protocolos, preparados, binarias, min_consecutive
    )


def _resumir(detalhes):
    alvos = [d for d in detalhes if d["grupo"] == "estimulo"]
    laterais = [d for d in detalhes if d["grupo"] == "lateral"]
    detectados = [d for d in alvos if d["detectou"]]
    fp_esp = sum(d["fp_esp"] for d in alvos)
    fp_lateral = sum(d["detectou"] for d in laterais)
    total_negativos = len(alvos) + len(laterais)
    fp_combinado = fp_esp + fp_lateral

    def grupo(chave):
        saida = {}
        for valor in sorted({d[chave] for d in alvos}):
            itens = [d for d in alvos if d[chave] == valor]
            if chave == "frequencia_indice":
                controles = [d for d in laterais if d[chave] == valor]
                rotulo = str(itens[0]["frequencia"])
                frequencia_lateral = controles[0]["frequencia"]
            else:
                controles = [d for d in laterais if d[chave] == valor]
                rotulo = str(valor)
                frequencia_lateral = None
            n_negativos = len(itens) + len(controles)
            falsos = sum(d["fp_esp"] for d in itens) + sum(
                d["detectou"] for d in controles
            )
            saida[rotulo] = {
                "n": len(itens),
                "detectados": sum(d["detectou"] for d in itens),
                "taxa_deteccao": float(np.mean([d["detectou"] for d in itens])),
                "fp_esp": sum(d["fp_esp"] for d in itens),
                "fp_lateral": sum(d["detectou"] for d in controles),
                "taxa_fp_combinada": falsos / n_negativos,
            }
            if frequencia_lateral is not None:
                saida[rotulo]["frequencia_lateral_hz"] = frequencia_lateral
        return saida

    return {
        "n_participante_frequencia": len(alvos),
        "detectados_ate_70db": len(detectados),
        "taxa_deteccao": len(detectados) / len(alvos),
        "fp_esp": fp_esp,
        "taxa_fp_esp": fp_esp / len(alvos),
        "fp_lateral": fp_lateral,
        "taxa_fp_lateral": fp_lateral / len(laterais),
        "taxa_fp_combinada": fp_combinado / total_negativos,
        "acuracia_balanceada_com_fp_combinado": (
            len(detectados) / len(alvos) + 1.0 - fp_combinado / total_negativos
        ) / 2.0,
        "tempo_total_medio_epocas_inclui_nao_detectados": float(np.mean([
            d["tempo_total_epocas"] for d in alvos
        ])),
        "tempo_desde_estimulo_medio_epocas_inclui_nao_detectados": float(np.mean([
            d["tempo_desde_inicio_estimulo_epocas"] for d in alvos
        ])),
        "primeira_deteccao_por_nivel_db": {
            str(nivel): sum(d["primeiro_nivel_db"] == nivel for d in alvos)
            for nivel in LEVELS[1:]
        },
        "por_participante": grupo("participante"),
        "por_frequencia": grupo("frequencia_indice"),
    }


def _chave(resumo):
    return (
        resumo["acuracia_balanceada_com_fp_combinado"],
        resumo["taxa_deteccao"], -resumo["taxa_fp_combinada"],
        -resumo["tempo_total_medio_epocas_inclui_nao_detectados"],
    )


def _melhor_bruto(protocolos):
    candidatos = []
    for alpha, consecutivas in product(ALPHA_VALUES, MIN_CONSECUTIVE_VALUES):
        detalhes = _avaliar(
            protocolos, lambda p, a=alpha: p <= a, consecutivas
        )
        resumo = _resumir(detalhes)
        if resumo["taxa_fp_combinada"] <= MAX_FALSE_POSITIVE:
            candidatos.append((_chave(resumo), alpha, consecutivas, resumo, detalhes))
    return max(candidatos, key=lambda item: item[0]) if candidatos else None


def _melhor_hmm(protocolos, modelos):
    candidatos = []
    preparados = [_concatenar(protocolo) for protocolo in protocolos]
    for p_ausente, p_presente, pi_ausente in product(
        P_SELF_VALUES, P_SELF_VALUES, PI_ABSENT_VALUES
    ):
        matriz_a = np.asarray([
            [p_ausente, 1.0 - p_ausente],
            [1.0 - p_presente, p_presente],
        ])
        pi = [pi_ausente, 1.0 - pi_ausente]
        binarias = []
        for preparado in preparados:
            binarias.append(_estados_hmm(
                preparado[0], matriz_a, pi, modelos
            ))
        for consecutivas in MIN_CONSECUTIVE_VALUES:
            detalhes = _avaliar_preparado(
                protocolos, preparados, binarias, consecutivas
            )
            resumo = _resumir(detalhes)
            if resumo["taxa_fp_combinada"] <= MAX_FALSE_POSITIVE:
                candidatos.append((
                    _chave(resumo), matriz_a.tolist(), pi, consecutivas,
                    resumo, detalhes,
                ))
    return max(candidatos, key=lambda item: item[0]) if candidatos else None


def _adicionar_laterais_esp(esp, detector_nome):
    """Acrescenta binsM aos registros ESP produzidos pelo carregador comum."""
    for item in esp:
        arquivo = gom.carregar_mat(item["arquivo"])
        laterais = []
        for frequencia in arquivo["bins_m"]:
            if detector_nome == "spectral_f":
                coeficientes = gom.extrair_coeficientes_espectrais_locais(
                    arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], frequencia,
                    arquivo["freq_estim"], detectors.SPECTRAL_F_NOISE_BINS,
                )[0]
            else:
                coeficientes = gom.extrair_coeficientes_epocas(
                    arquivo["x"], gom.CHANNEL_INDEX, arquivo["fs"], frequencia
                )[0]
            laterais.append((float(frequencia), coeficientes))
        item["lateral"] = laterais
    return esp


def executar_detector(pasta_dados, detector_nome):
    esp, estimulo = crh._carregar_coeficientes(pasta_dados, detector_nome)
    _adicionar_laterais_esp(esp, detector_nome)
    configuracoes = {}
    for tamanho, modo_passo in product(WINDOW_SIZES, STEP_MODES):
        passo = early._step(tamanho, modo_passo)
        nome = f"window_{tamanho}_step_{passo}"
        print(f"{detector_nome}: {nome}", flush=True)
        protocolos, modelos = _montar_protocolos(
            esp, estimulo, detector_nome, tamanho, passo
        )
        bruto = _melhor_bruto(protocolos)
        hmm = _melhor_hmm(protocolos, modelos)
        configuracoes[nome] = {
            "window_size_epochs": tamanho,
            "window_step_epochs": passo,
            "modelos_emissao_beta": modelos,
            "hmm": None if hmm is None else {
                "matriz_a": hmm[1], "pi_inicial": hmm[2],
                "min_consecutive": hmm[3], **hmm[4], "detalhes": hmm[5],
            },
            "detector_bruto": None if bruto is None else {
                "alpha": bruto[1], "min_consecutive": bruto[2],
                **bruto[3], "detalhes": bruto[4],
            },
        }
    melhores_hmm = [(n, v["hmm"]) for n, v in configuracoes.items() if v["hmm"]]
    melhores_bruto = [
        (n, v["detector_bruto"]) for n, v in configuracoes.items()
        if v["detector_bruto"]
    ]
    nome_hmm, hmm = max(melhores_hmm, key=lambda x: _chave(x[1]))
    nome_bruto, bruto = max(melhores_bruto, key=lambda x: _chave(x[1]))
    return {
        "melhor_hmm_configuracao": nome_hmm,
        "melhor_hmm": hmm,
        "melhor_bruto_configuracao": nome_bruto,
        "melhor_bruto": bruto,
        "por_configuracao": configuracoes,
    }


def executar(pasta_dados=crh.DATA_DIR):
    por_detector = {}
    for nome in DETECTOR_NAMES:
        por_detector[nome] = executar_detector(pasta_dados, nome)
    return {
        "nome": "hmm_transicao_fases_protocolo_concatenado",
        "status": "complete",
        "aviso_vazamento": (
            "Emissoes, A, pi e regras foram ajustadas e avaliadas nos mesmos "
            "11 participantes. Resultado in-sample; nao estima generalizacao."
        ),
        "metodo": (
            "Sequencias por participante e frequencia na ordem ESP, 30, 40, "
            "50, 60 e 70 dB. Janelas nao cruzam arquivos."
        ),
        "configuracao": {
            "detectores": list(DETECTOR_NAMES),
            "window_sizes_epochs": list(WINDOW_SIZES),
            "step_modes": list(STEP_MODES),
            "fp_maximo_combinado": MAX_FALSE_POSITIVE,
            "levels": list(LEVELS),
        },
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
    print("\nProtocolo concatenado ESP -> 30 -> 40 -> 50 -> 60 -> 70 dB")
    for nome, item in resultado["por_detector"].items():
        hmm, bruto = item["melhor_hmm"], item["melhor_bruto"]
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
    args = parser.parse_args()
    resultado = executar(args.data_dir)
    salvar(resultado, args.output)
    imprimir(resultado)
