"""Busca em grade de janela, passo, A, pi e regra; todos os pacientes no ajuste."""

import argparse
from itertools import product

import numpy as np

import config
from dados import carregar_dados, preparar_protocolos, salvar_json
from get_obs_matrix import construir_matrizes_observacao, log_emissoes
from get_tran_matrix import construir_matriz_transicao
from hmm_inference import avaliar, resumir
from viterbi import viterbi_lote


def construir_grade(rapida=False):
    """Listas explícitas de possibilidades; --rapida serve só para verificar execução."""
    janelas = (90,) if rapida else config.WINDOW_SIZE_VALUES
    probabilidades = (.8, .55) if rapida else config.PERSISTENCE_VALUES
    pi_values = (.9,) if rapida else config.PI_ABSENT_VALUES
    consecutivas = (1, 3) if rapida else config.MIN_CONSECUTIVE_VALUES
    percentuais = (.1, .6) if rapida else config.MIN_PERCENT_VALUES
    regras = [{"consecutivas": c, "percentual": p, "modo": modo}
              for c, p, modo in product(consecutivas, percentuais, ("OR", "AND"))]
    regras += [{"consecutivas": c, "percentual": None, "modo": "OR"} for c in consecutivas]
    regras += [{"consecutivas": None, "percentual": p, "modo": "OR"} for p in percentuais]
    return {"janelamentos": [(m, passo) for m in janelas for passo in (m // 2, m)],
            "persistencias": probabilidades, "pi_ausente": pi_values, "regras": regras}


def candidatos_temporais(grade, a_heuristica):
    matrizes = [[[a, 1 - a], [1 - b, b]]
                for a, b in product(grade["persistencias"], repeat=2)]
    for a in (a_heuristica, config.MATRIX_A):
        if not any(np.allclose(a, outra) for outra in matrizes):
            matrizes.append(a)
    return [{"A": a, "pi": [p, 1 - p]} for a, p in product(matrizes, grade["pi_ausente"])]


def preparar_lote(protocolos):
    """Completa sequências curtas com padding; a máscara impede que ele conte."""
    tamanho = max(len(p["observacoes"]) for p in protocolos)
    p_values = np.full((len(protocolos), tamanho), .5)
    validos = np.zeros_like(p_values, dtype=bool)
    escopo = np.zeros_like(p_values, dtype=bool)
    alvos = np.array([p["grupo"] == "alvo" for p in protocolos])
    for i, protocolo in enumerate(protocolos):
        obs = protocolo["observacoes"]
        p_values[i, :len(obs)] = [o["p_value"] for o in obs]
        validos[i, :len(obs)] = True
        escopo[i, :len(obs)] = [o["nivel"] != "ESP" or not alvos[i] for o in obs]
    return p_values, validos, escopo, alvos


def contar_regras(estados, validos, escopo, alvos, regras):
    """Conta detecções/FP sem repetir Viterbi para cada regra.

    Para AND, consecutivas e percentual precisam passar no mesmo instante.
    Eixos: candidato A/pi, protocolo, tempo.
    """
    estados = estados & validos[None]
    corridas = np.zeros(estados.shape, dtype=int)
    for t in range(estados.shape[2]):
        anterior = corridas[:, :, t - 1] if t else 0
        corridas[:, :, t] = (anterior + 1) * estados[:, :, t]
    fracoes = estados.cumsum(axis=2) / np.arange(1, estados.shape[2] + 1)
    # initial trata protocolos sem nenhuma janela, mantendo-os no denominador.
    max_corrida = np.max(np.where(escopo[None], corridas, 0), axis=2, initial=0)
    max_fracao = np.max(np.where(escopo[None], fracoes, -1), axis=2, initial=-1)
    fracoes_and = {}
    contagens = np.zeros((len(estados), len(regras), 2), dtype=int)
    for j, regra in enumerate(regras):
        c, p, modo = regra["consecutivas"], regra["percentual"], regra["modo"]
        if c is None:
            detectou = max_fracao >= p
        elif p is None:
            detectou = max_corrida >= c
        elif modo == "OR":
            detectou = (max_corrida >= c) | (max_fracao >= p)
        else:
            if c not in fracoes_and:
                fracoes_and[c] = np.max(np.where(escopo[None] & (corridas >= c), fracoes, -1),
                                         axis=2, initial=-1)
            detectou = fracoes_and[c] >= p
        contagens[:, j, 0] = detectou[:, alvos].sum(axis=1)
        contagens[:, j, 1] = detectou[:, ~alvos].sum(axis=1)
    return contagens


def escolher_parametros(protocolos, modelos, candidatos, regras):
    p, validos, escopo, alvos = preparar_lote(protocolos)
    emissoes = log_emissoes(p, modelos)
    melhor = None
    # Lotes de 32 limitam memória sem mudar o resultado matemático da busca.
    for inicio in range(0, len(candidatos), 32):
        lote = candidatos[inicio:inicio + 32]
        estados = viterbi_lote(emissoes, lote)
        contagens = contar_regras(estados, validos, escopo, alvos, regras)
        det, fp = contagens[:, :, 0], contagens[:, :, 1]
        n_alvos, n_laterais = int(alvos.sum()), int((~alvos).sum())
        elegivel = fp / n_laterais <= config.FP_MAXIMO
        ba = (det / n_alvos + 1 - fp / n_laterais) / 2
        # Maximiza BA; empata por detecção e depois FP. Empate final: primeiro da grade.
        indices = np.flatnonzero(elegivel)
        if len(indices) == 0:
            continue
        scores = [(float(ba.flat[i]), int(det.flat[i]), -int(fp.flat[i])) for i in indices]
        indice = int(indices[max(range(len(scores)), key=lambda i: scores[i])])
        i, j = np.unravel_index(indice, ba.shape)
        score = (float(ba[i, j]), int(det[i, j]), -int(fp[i, j]))
        if melhor is None or score > tuple(melhor["score"]):
            melhor = {**lote[i], "regra": regras[j], "score": list(score)}
    return melhor


def buscar(rapida=False):
    cfg, grade = config.configuracao(), construir_grade(rapida)
    dados = carregar_dados()
    melhores = {"mle": None, "map": None}
    avaliacoes = 0
    for tamanho, passo in grade["janelamentos"]:
        protocolos = preparar_protocolos(dados, tamanho, passo)
        b = construir_matrizes_observacao(protocolos, cfg["presente_min_db"], cfg["priori"])
        a = construir_matriz_transicao(dados, tamanho, passo)
        candidatos = candidatos_temporais(grade, a["matriz"])
        for metodo in melhores:
            escolhido = escolher_parametros(protocolos, b[metodo], candidatos, grade["regras"])
            avaliacoes += len(candidatos) * len(grade["regras"])
            if escolhido is not None and (melhores[metodo] is None
                    or tuple(escolhido["score"]) > tuple(melhores[metodo]["score"])):
                eventos = avaliar(protocolos, b[metodo], escolhido["A"], escolhido["pi"], escolhido["regra"])
                resumo = resumir(eventos)
                if (resumo["global"]["detectados"] != escolhido["score"][1]
                        or resumo["global"]["fp_lateral"] != -escolhido["score"][2]):
                    raise RuntimeError("Busca em lote divergiu da inferência individual")
                melhores[metodo] = {**escolhido, "janela": tamanho, "passo": passo,
                                    "B": b[metodo], "metricas": resumo, "eventos": eventos}
        print(f"Janela/passo {tamanho}/{passo} concluídos; {avaliacoes:,} avaliações", flush=True)
    if any(m is None for m in melhores.values()):
        raise RuntimeError("Nenhum candidato sob o teto FP; revise a grade")
    resultado = {"configuracao": cfg, "grade": grade, "rapida": rapida, "avaliacoes": avaliacoes,
                 "melhores": melhores, "n_pacientes": len(dados),
                 "descricao": "B, seleção e avaliação nos mesmos pacientes (in-sample)."}
    nome = "search_hmm_parameters_rapida.json" if rapida else "search_hmm_parameters.json"
    salvar_json(resultado, config.RESULTS_DIR / nome)
    for metodo, escolhido in melhores.items():
        print(metodo.upper(), {k: escolhido[k] for k in ("janela", "passo", "A", "pi", "regra")})
        print(escolhido["metricas"]["global"])
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapida", action="store_true", help="grade pequena para verificar o funcionamento")
    buscar(parser.parse_args().rapida)
