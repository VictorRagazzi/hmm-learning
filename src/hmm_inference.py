"""Aplica o HMM escolhido às intensidades concatenadas de todos os pacientes."""

import argparse
import json

import numpy as np

import config
from dados import carregar_dados, preparar_protocolos, salvar_json
from get_obs_matrix import construir_matrizes_observacao, log_emissoes, salvar_distribuicoes
from get_tran_matrix import construir_matriz_transicao
from viterbi import viterbi


def validar_regra(regra):
    c, p, modo = regra["consecutivas"], regra["percentual"], regra["modo"]
    if (c is None and p is None or modo not in ("AND", "OR")
            or c is not None and (isinstance(c, bool) or not isinstance(c, int) or c < 1)
            or p is not None and (not np.isfinite(p) or not 0 < p <= 1)):
        raise ValueError("Regra: consecutivas >=1, percentual em (0,1], ao menos um ativo; AND/OR")


def primeiro_disparo(estados, observacoes, regra, somente_estimulo=False):
    """Consecutivas/fração usam apenas o prefixo e continuam entre intensidades."""
    validar_regra(regra)
    consecutivas, positivos = 0, 0
    for i, (estado, obs) in enumerate(zip(estados, observacoes, strict=True), 1):
        consecutivas = consecutivas + 1 if estado else 0
        positivos += int(estado)
        criterios = []
        if regra["consecutivas"] is not None:
            criterios.append(consecutivas >= regra["consecutivas"])
        if regra["percentual"] is not None:
            criterios.append(positivos / i >= regra["percentual"])
        passou = all(criterios) if regra["modo"] == "AND" else any(criterios)
        if passou and (not somente_estimulo or obs["nivel"] != "ESP"):
            return obs
    return None


def avaliar(protocolos, modelos, a, pi, regra, guardar_caminhos=False):
    resultados = []
    for protocolo in protocolos:
        obs = protocolo["observacoes"]
        p = [o["p_value"] for o in obs]
        caminho, estados = viterbi(log_emissoes(p, modelos), a, pi)
        primeiro = primeiro_disparo(estados, obs, regra)
        deteccao = primeiro_disparo(estados, obs, regra, somente_estimulo=True)
        inicio_estimulo = protocolo["fases"][1]["inicio_s"]
        resultado = {"paciente": protocolo["paciente"], "grupo": protocolo["grupo"],
            "frequencia": protocolo["frequencia"], "alvo_hz": protocolo["alvo_hz"],
            "controle_hz": protocolo["controle_hz"],
            "detectou": deteccao is not None, "disparo_total": primeiro is not None,
            "fp_esp": primeiro is not None and primeiro["nivel"] == "ESP",
            "primeiro_nivel_db": deteccao["nivel"] if deteccao else None,
            "tempo_desde_30_s": deteccao["fim_s"] - inicio_estimulo if deteccao else None,
            "duracao_desde_30_s": (deteccao["fim_s"] if deteccao else protocolo["duracao_s"]) - inicio_estimulo}
        if guardar_caminhos:
            resultado.update({"observacoes": obs, "caminho_viterbi": caminho.tolist(),
                              "estados_causais": estados.tolist()})
        resultados.append(resultado)
    return resultados


def metricas(resultados):
    alvos = [r for r in resultados if r["grupo"] == "alvo"]
    laterais = [r for r in resultados if r["grupo"] == "lateral"]
    detectados = [r for r in alvos if r["detectou"]]
    fp = sum(r["disparo_total"] for r in laterais)
    det_rate, fp_rate = len(detectados) / len(alvos), fp / len(laterais)
    tempos = [r["tempo_desde_30_s"] for r in detectados]
    return {"n_alvos": len(alvos), "n_laterais": len(laterais), "detectados": len(detectados),
            "fp_lateral": fp, "taxa_deteccao": det_rate, "taxa_fp_lateral": fp_rate,
            "acuracia_balanceada": (det_rate + 1 - fp_rate) / 2,
            "fp_alvo_esp": sum(r["fp_esp"] for r in alvos),
            "tempo_medio_detectados_s": float(np.mean(tempos)) if tempos else None,
            "tempo_mediano_detectados_s": float(np.median(tempos)) if tempos else None,
            "duracao_media_inclui_nao_detectados_s": float(np.mean([r["duracao_desde_30_s"] for r in alvos]))}


def resumir(resultados):
    return {"global": metricas(resultados),
            "por_paciente": {p: metricas([r for r in resultados if r["paciente"] == p])
                             for p in sorted({r["paciente"] for r in resultados})},
            "por_frequencia": {str(f): metricas([r for r in resultados if r["alvo_hz"] == f])
                               for f in sorted({r["frequencia"] for r in resultados if r["grupo"] == "alvo"})}}


def executar(usar_busca=False):
    cfg = config.configuracao()
    if usar_busca:
        busca = json.loads((config.RESULTS_DIR / "search_hmm_parameters.json").read_text())
        if busca["configuracao"].get("detector", "rayleigh") != cfg["detector"]:
            raise ValueError("Detector diferente da busca salva; refaça a busca para o detector escolhido")
        escolhido = busca["melhores"][cfg["metodo"]]
        cfg.update({k: escolhido[k] for k in ("janela", "passo", "A", "pi", "regra")})
        # Usa também a priori/corte/canal salvos pela busca.
        cfg.update({k: busca["configuracao"][k] for k in ("priori", "presente_min_db", "p_eps")})
        if cfg["p_eps"] != config.P_EPS or busca["configuracao"]["canal"] != config.CHANNEL_INDEX:
            raise ValueError("Canal ou clipping diferentes da busca; ajuste config.py ou refaça a busca")
    dados = carregar_dados()
    protocolos = preparar_protocolos(dados, cfg["janela"], cfg["passo"], cfg["detector"])
    a = construir_matriz_transicao(dados, cfg["janela"], cfg["passo"])
    b = construir_matrizes_observacao(protocolos, cfg["presente_min_db"], cfg["priori"])
    b["configuracao"] = cfg
    resultados = avaliar(protocolos, b[cfg["metodo"]], cfg["A"], cfg["pi"], cfg["regra"], True)
    analise = {"configuracao": cfg, "metricas": resumir(resultados), "resultados": resultados,
               "n_arquivos": sum(len(f) for f in dados.values()), "n_pacientes": len(dados),
               "n_protocolos": len(protocolos), "n_janelas": sum(len(p["observacoes"]) for p in protocolos)}
    salvar_json(a, config.RESULTS_DIR / "transition_matrix.json")
    salvar_json(b, config.RESULTS_DIR / "observation_matrix.json")
    salvar_json(analise, config.RESULTS_DIR / "inference_analysis.json")
    salvar_distribuicoes(b)
    print("A calculada por duração:", a["matriz"])
    print("A aplicada no Viterbi:", cfg["A"])
    print("B contínua:", b[cfg["metodo"]])
    print("Métricas in-sample:", analise["metricas"]["global"])
    print("Resultados em", config.RESULTS_DIR)
    return analise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usar-busca", action="store_true", help="aplica o vencedor salvo, sem editar config.py")
    executar(parser.parse_args().usar_busca)
