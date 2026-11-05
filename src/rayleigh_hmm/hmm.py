"""Forward-backward de 2 estados (0 = Ausente, 1 = Presente), vetorizado.

Processa um lote de sequencias de MESMO comprimento de uma vez. As emissoes
entram como log-densidades (continuas, podem ser > 0) e sao convertidas
para verossimilhancas relativas por passo, o que nao altera a posteriori.
"""

import numpy as np


def posteriori_filtrada(log_emissao, pi, A):
    """P(estado_t = Presente | observacoes 1..t): causal, usada no exame online.

    log_emissao: (n_seq, T, 2); pi: (2,); A: (2, 2). Retorna (n_seq, T).
    """
    n_seq, T, _ = log_emissao.shape
    emissao = np.exp(log_emissao - log_emissao.max(axis=2, keepdims=True))
    filtrada = np.empty((n_seq, T))
    a = pi[None, :] * emissao[:, 0]
    a /= a.sum(axis=1, keepdims=True)
    filtrada[:, 0] = a[:, 1]
    for t in range(1, T):
        a = (a @ A) * emissao[:, t]
        a /= a.sum(axis=1, keepdims=True)
        filtrada[:, t] = a[:, 1]
    return filtrada


def posteriori_estados(log_emissao, pi, A):
    """P(estado_t = Presente | todas as observacoes da sequencia).

    log_emissao: (n_seq, T, 2); pi: (2,); A: (2, 2) com linhas somando 1.
    Retorna (n_seq, T).
    """
    n_seq, T, _ = log_emissao.shape
    emissao = np.exp(log_emissao - log_emissao.max(axis=2, keepdims=True))

    alfa = np.empty((n_seq, T, 2))
    a = pi[None, :] * emissao[:, 0]
    alfa[:, 0] = a / a.sum(axis=1, keepdims=True)
    for t in range(1, T):
        a = (alfa[:, t - 1] @ A) * emissao[:, t]
        alfa[:, t] = a / a.sum(axis=1, keepdims=True)

    beta = np.ones((n_seq, 2))
    gama = np.empty((n_seq, T))
    for t in range(T - 1, -1, -1):
        g = alfa[:, t] * beta
        gama[:, t] = g[:, 1] / g.sum(axis=1)
        if t > 0:
            b = (emissao[:, t] * beta) @ A.T
            beta = b / b.sum(axis=1, keepdims=True)
    return gama


def viterbi_alguma_presente(log_emissao, pi, A):
    """Para cada t: o caminho mais provavel de 1..t passa por Presente?

    Causal: det[:, t] so usa observacoes ate t. Sem backtrack: tem[s] guarda se
    o melhor caminho que termina em s passou por Presente, de modo que cada t
    reaproveita o passo anterior.

    Empates (predecessor ou estado final) ficam com Ausente, o menor indice do
    argmax: a deteccao so ocorre se Presente for estritamente mais provavel.

    log_emissao: (n_seq, T, 2); pi: (2,); A: (2, 2). Retorna booleano (n_seq, T).
    """
    n_seq, T, _ = log_emissao.shape
    with np.errstate(divide="ignore"):
        log_pi, log_A = np.log(pi), np.log(A)
    linhas = np.arange(n_seq)
    delta = log_pi[None, :] + log_emissao[:, 0]
    tem = np.tile(np.array([False, True]), (n_seq, 1))
    det = np.empty((n_seq, T), dtype=bool)
    det[:, 0] = tem[linhas, delta.argmax(axis=1)]
    for t in range(1, T):
        candidatos = delta[:, :, None] + log_A[None, :, :]  # (n_seq, de, para)
        psi = candidatos.argmax(axis=1)
        delta = candidatos.max(axis=1) + log_emissao[:, t]
        tem = tem[linhas[:, None], psi]
        tem[:, 1] = True
        det[:, t] = tem[linhas, delta.argmax(axis=1)]
    return det
