"""Viterbi em log-espaço: caminho completo e melhores estados a cada prefixo."""

import numpy as np


def validar_probabilidades(a, pi):
    a, pi = np.asarray(a, dtype=float), np.asarray(pi, dtype=float)
    if (a.shape != (2, 2) or pi.shape != (2,) or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(pi)) or np.any(a < 0) or np.any(pi < 0)
            or not np.allclose(a.sum(axis=1), 1) or not np.isclose(pi.sum(), 1)):
        raise ValueError("A deve ser 2x2, pi deve ter 2 entradas, com probabilidades normalizadas")
    return a, pi


def viterbi(log_b, a, pi):
    """log_b[t,j] = log da densidade da observação t no estado j.

    O caminho completo usa backtracking. Para primeira detecção, use somente
    os estados causais: o melhor estado terminal de cada prefixo observado.
    """
    a, pi = validar_probabilidades(a, pi)
    log_b = np.asarray(log_b)
    if log_b.ndim != 2 or log_b.shape[1] != 2 or not np.all(np.isfinite(log_b)):
        raise ValueError("Emissões devem ser uma matriz finita de tamanho T x 2")
    if len(log_b) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    with np.errstate(divide="ignore"):
        log_a, log_pi = np.log(a), np.log(pi)
    delta = log_pi + log_b[0]
    anteriores = np.zeros((len(log_b), 2), dtype=int)
    causais = [int(np.argmax(delta))]
    for t in range(1, len(log_b)):
        candidatos = delta[:, None] + log_a
        anteriores[t] = np.argmax(candidatos, axis=0)
        delta = np.max(candidatos, axis=0) + log_b[t]
        causais.append(int(np.argmax(delta)))
    caminho = np.zeros(len(log_b), dtype=int)
    caminho[-1] = np.argmax(delta)
    for t in range(len(log_b) - 1, 0, -1):
        caminho[t - 1] = anteriores[t, caminho[t]]
    return caminho, np.asarray(causais)


def viterbi_lote(log_b, candidatos):
    """Mesma recorrência causal para várias A/pi e sequências, usada na busca.

    Eixos de log_b: sequência, tempo, estado. Eixos de saída: candidato,
    sequência, tempo. NumPy calcula as combinações simultaneamente.
    """
    estados = np.zeros((len(candidatos), *log_b.shape[:2]), dtype=bool)
    if log_b.shape[1] == 0:
        return estados
    with np.errstate(divide="ignore"):
        log_a = np.log([c["A"] for c in candidatos])
        log_pi = np.log([c["pi"] for c in candidatos])
    delta = log_pi[:, None, :] + log_b[None, :, 0, :]
    estados[:, :, 0] = delta[:, :, 1] > delta[:, :, 0]
    for t in range(1, log_b.shape[1]):
        transicoes = delta[:, :, :, None] + log_a[:, None, :, :]
        delta = np.max(transicoes, axis=2) + log_b[None, :, t, :]
        estados[:, :, t] = delta[:, :, 1] > delta[:, :, 0]
    return estados
