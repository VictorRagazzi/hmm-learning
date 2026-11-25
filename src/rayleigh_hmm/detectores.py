"""Detectores de ASSR convertidos em Z = -ln p (p-valor do teste no acumulado).

Todo detector e calculado por SOMAS ACUMULADAS ao longo das epocas: o Z de
qualquer intervalo [ini, fim) de epocas consecutivas sai pela diferenca de duas
somas. Serve ao protocolo normal (ini = 0) e ao concatenado (ultimas M epocas).
Canal FC = indice 0, sem rejeicao de epocas. Usa a forma fechada do logsf
(sem underflow, Z finito mesmo com resposta forte).

Execucao (so para conferir): importado por explorar_detectores.py e pelos testes.
"""

import numpy as np
from scipy.special import gammaln, logsumexp

import dados

# ---------------- configuracao ----------------
DETECTORES = ["rayleigh", "msc", "hotelling", "mmsc", "f_espectral"]
CANAL_FC = 0
CANAIS_MMSC = (0, 13, 15, 1)  # FC, Fz, Cz, F4 (L = 4)
FAIXAS_VIZINHOS_HZ = ((70, 79), (98, 107))  # fora de 80-97 Hz, iguais para todos os alvos
TETO_COERENCIA = 1.0 - 1e-15  # evita log(0) quando a coerencia chega a 1


# ---------------- somas acumuladas ----------------
def _acumular(a):
    """Soma acumulada no eixo das epocas (eixo 0), com uma linha de zeros no inicio."""
    return np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), np.cumsum(a, axis=0)])


def _somas_do_tipo(espectro, bins, bins_viz):
    """Somas acumuladas para os bins `bins` (alvo ou lateral). espectro: (canais, epocas, bins_fft)."""
    x = espectro[CANAL_FC][:, bins]  # (epocas, n_freqs)
    y = np.moveaxis(espectro[list(CANAIS_MMSC)][:, :, bins], 0, 1)  # (epocas, L, n_freqs)
    viz = (np.abs(espectro[CANAL_FC][:, bins_viz]) ** 2).sum(axis=1)  # (epocas,)
    return {
        "u": _acumular(x / (np.abs(x) + dados.EPS)),  # fasores unitarios (Rayleigh)
        "x": _acumular(x),
        "p": _acumular(np.abs(x) ** 2),
        "rr": _acumular(x.real ** 2),
        "ii": _acumular(x.imag ** 2),
        "ri": _acumular(x.real * x.imag),
        "v": _acumular(y),  # soma dos vetores Y_i
        "s": _acumular(np.einsum("elf,emf->elmf", y, y.conj())),  # soma de Y Y^H
        "viz": _acumular(viz),  # soma de |X_viz|^2 sobre os bins vizinhos
    }


def _indices_bins(freqs_fft, frequencias):
    idx = np.array([int(np.argmin(np.abs(freqs_fft - f))) for f in frequencias])
    if np.max(np.abs(freqs_fft[idx] - frequencias)) > 1e-6:
        raise ValueError("frequencia fora da grade da FFT")
    return idx


def carregar_somas(caminho):
    """Le o .mat e devolve as somas acumuladas (n_epocas + 1 linhas) por tipo ("alvo", "lateral")."""
    d = dados.carregar_x(caminho)
    n_amostras = d["x"].shape[2]
    freqs_fft = np.fft.rfftfreq(n_amostras, d=1.0 / d["fs"])
    espectro = np.fft.rfft(d["x"], axis=2)  # (canais, epocas, bins_fft)
    bins_viz = np.concatenate([np.flatnonzero((freqs_fft >= lo - 1e-9) & (freqs_fft <= hi + 1e-9))
                               for lo, hi in FAIXAS_VIZINHOS_HZ])
    return {
        "n_epocas": int(d["x"].shape[1]),
        "duracao_epoca_s": n_amostras / d["fs"],
        "n_viz": int(bins_viz.size),
        "freqs": {"alvo": d["freq_estim"], "lateral": d["bins_m"]},
        "alvo": _somas_do_tipo(espectro, _indices_bins(freqs_fft, d["freq_estim"]), bins_viz),
        "lateral": _somas_do_tipo(espectro, _indices_bins(freqs_fft, d["bins_m"]), bins_viz),
    }


# ---------------- um Z por detector (dif = soma no intervalo; n = epocas, (n_jan, 1)) ----------------
def z_rayleigh(dif, n, n_viz):
    """Rayleigh: p = exp(-Z), Z = |soma dos fasores unitarios|^2 / n."""
    return np.abs(dif("u")) ** 2 / n


def z_msc(dif, n, n_viz):
    """MSC do canal FC; sob H0 ~ Beta(1, n-1), p = (1-MSC)^(n-1)."""
    msc = np.abs(dif("x")) ** 2 / (n * dif("p"))
    return -(n - 1) * np.log1p(-np.minimum(msc, TETO_COERENCIA))


def z_hotelling(dif, n, n_viz):
    """T^2 de Hotelling sobre (Re, Im) de X; F = (n-2)/(2(n-1)) T^2 ~ F(2, n-2); sf = (1 + 2F/d2)^(-d2/2)."""
    soma = dif("x")
    mr, mi = soma.real / n, soma.imag / n
    crr = (dif("rr") - n * mr * mr) / (n - 1)
    cii = (dif("ii") - n * mi * mi) / (n - 1)
    cri = (dif("ri") - n * mr * mi) / (n - 1)
    t2 = n * (cii * mr * mr - 2 * cri * mr * mi + crr * mi * mi) / (crr * cii - cri * cri)
    d2 = n - 2
    f = d2 / (2 * (n - 1)) * t2
    return (d2 / 2) * np.log1p(2 * f / d2)


def z_mmsc(dif, n, n_viz):
    """MSC multivariada (L canais): k2 = V^H S^-1 V / n ~ Beta(L, n-L) sob H0.

    sf de Beta(L, m) com L inteiro, y = 1 - k2: y^m * sum_{j<L} C(m+j-1, j) (1-y)^j.
    """
    v = np.moveaxis(dif("v"), -1, 1)  # (n_jan, n_freqs, L)
    s = np.moveaxis(dif("s"), -1, 1)  # (n_jan, n_freqs, L, L)
    sol = np.linalg.solve(s, v[..., None])[..., 0]
    k2 = np.clip(np.einsum("...l,...l->...", v.conj(), sol).real / n, 0.0, TETO_COERENCIA)
    n_canais = v.shape[-1]
    m = n - n_canais
    j = np.arange(n_canais)
    mj = m[..., None] + j - 1  # (n_jan, 1, L)
    log_c = gammaln(mj + 1) - gammaln(j + 1) - gammaln(mj - j + 1)
    termos = log_c + j * np.log(np.maximum(k2, 1e-300))[..., None]
    return -(m * np.log1p(-k2) + logsumexp(termos, axis=-1))


def z_f_espectral(dif, n, n_viz):
    """F = n |media X_alvo|^2 / media |X_viz|^2 ~ F(2, 2 n L_viz) sob H0; Z = n L_viz log1p(F / (n L_viz))."""
    potencia_viz = dif("viz")[:, None] / (n * n_viz)
    f = np.abs(dif("x")) ** 2 / (n * potencia_viz)
    return n * n_viz * np.log1p(f / (n * n_viz))


FUNCOES_Z = {"rayleigh": z_rayleigh, "msc": z_msc, "hotelling": z_hotelling,
             "mmsc": z_mmsc, "f_espectral": z_f_espectral}


# ---------------- interface ----------------
def z_intervalo(somas, detector, tipo, ini, fim):
    """Z (n_janelas, n_freqs) e n (n_janelas,) dos intervalos de epocas [ini, fim) de cada janela."""
    ini, fim = np.asarray(ini), np.asarray(fim)
    c = somas[tipo]
    n = (fim - ini).astype(float)[:, None]

    def dif(nome):
        return c[nome][fim] - c[nome][ini]

    return FUNCOES_Z[detector](dif, n, somas["n_viz"]), (fim - ini)
