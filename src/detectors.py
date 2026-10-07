"""Detectores históricos: estatística e p-valor; B ajusta os p-valores de qualquer um.

MSC/CSM: coerência complexa; Rayleigh: fase; MMSC: split-half experimental;
Hotelling: covariância real/imaginária; F espectral: alvo e dois bins de ruído.
As fórmulas e distribuições nulas são as da implementação anterior ao MVP.
"""

from functools import lru_cache
import numpy as np
from scipy.stats import beta as beta_dist, f as f_dist

EPS = 1e-12
SPECTRAL_F_NOISE_BINS = 2

def _msc_stat(janela):
    m = len(janela)
    numerador = np.abs(np.sum(janela)) ** 2
    denominador = m * np.sum(np.abs(janela) ** 2)
    return float(np.clip(numerador / (denominador + EPS), 0.0, 1.0))


def _msc_pvalue(valor, m):
    return float(beta_dist.sf(valor, 1, m - 1))


def _rayleigh_stat(janela):
    m = len(janela)
    fases_unitarias = janela / (np.abs(janela) + EPS)
    plv = np.abs(np.sum(fases_unitarias)) / m
    return float(np.clip(plv ** 2, 0.0, 1.0))


def _rayleigh_pvalue(valor, m):
    z = m * valor
    return float(np.clip(np.exp(-z), 0.0, 1.0))


def _csm_estatistica_f(janela):
    m = len(janela)
    media = np.mean(janela)
    residuos = janela - media
    s2 = float(np.sum(np.abs(residuos) ** 2)) / (m - 1)
    f_stat = m * (np.abs(media) ** 2) / (s2 + EPS)
    return f_stat, m


def _csm_stat(janela):
    f_stat, m = _csm_estatistica_f(janela)
    equivalente = f_stat / (f_stat + (m - 1) + EPS)
    return float(np.clip(equivalente, 0.0, 1.0))


def _csm_pvalue(valor, m):
    f_stat = valor * (m - 1) / (1 - valor + EPS)
    return float(f_dist.sf(f_stat, 2, 2 * (m - 1)))


def _mmsc_m_efetivo(m):
    return max(m // 2, 2)


def _mmsc_stat(janela):
    m = len(janela)
    metade = m // 2
    if metade < 2:
        return _msc_stat(janela)
    msc_1 = _msc_stat(janela[:metade])
    msc_2 = _msc_stat(janela[metade:2 * metade])
    return float(np.clip((msc_1 + msc_2) / 2.0, 0.0, 1.0))


@lru_cache(maxsize=None)
def _mmsc_null_table(m):
    """Tabela da cauda nula da media de duas MSCs independentes."""
    b = _mmsc_m_efetivo(m) - 1
    grid = np.linspace(0.0, 1.0, 8193)
    t = 2.0 * grid
    lower = np.maximum(0.0, t - 1.0)
    upper = np.minimum(1.0, t)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x = lower[:, None] + (nodes[None, :] + 1.0) * (
        upper - lower
    )[:, None] / 2.0
    density = b * np.power(np.maximum(1.0 - x, 0.0), b - 1)
    y = np.clip(t[:, None] - x, 0.0, 1.0)
    cdf_y = 1.0 - np.power(1.0 - y, b)
    integral = (upper - lower) / 2.0 * np.sum(
        weights[None, :] * density * cdf_y, axis=1
    )
    cdf = 1.0 - np.power(1.0 - lower, b) + integral
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    return grid, np.clip(1.0 - cdf, 0.0, 1.0)


def _mmsc_pvalue(valor, m):
    grid, tail = _mmsc_null_table(m)
    return float(np.interp(np.clip(valor, 0.0, 1.0), grid, tail))


def _hotelling_f(janela):
    valores = np.asarray(janela)
    m = len(valores)
    if m <= 2:
        return 0.0, m
    xy = np.column_stack((valores.real, valores.imag))
    media = np.mean(xy, axis=0)
    cov = np.cov(xy, rowvar=False, ddof=1)
    inversa = np.linalg.pinv(cov, hermitian=True)
    t2 = m * float(media @ inversa @ media)
    f_stat = (m - 2) * t2 / (2 * (m - 1))
    return max(float(f_stat), 0.0), m


def _hotelling_stat(janela):
    f_stat, _ = _hotelling_f(janela)
    return float(np.clip(f_stat / (1.0 + f_stat), 0.0, 1.0))


def _hotelling_pvalue(valor, m):
    if m <= 2:
        return 1.0
    f_stat = valor / (1.0 - valor + EPS)
    return float(f_dist.sf(f_stat, 2, m - 2))


def _spectral_f_value(janela):
    valores = np.asarray(janela)
    if valores.ndim != 2 or valores.shape[1] != 1 + SPECTRAL_F_NOISE_BINS:
        raise ValueError(
            "spectral_f requer matriz (epocas, 1 alvo + "
            f"{SPECTRAL_F_NOISE_BINS} bins de ruido)"
        )
    m = len(valores)
    alvo = valores[:, 0]
    ruido = valores[:, 1:]
    potencia_coerente = m * np.abs(np.mean(alvo)) ** 2
    potencia_ruido = float(np.mean(np.abs(ruido) ** 2))
    return max(float(potencia_coerente / (potencia_ruido + EPS)), 0.0), m


def _spectral_f_stat(janela):
    f_stat, _ = _spectral_f_value(janela)
    return float(np.clip(f_stat / (1.0 + f_stat), 0.0, 1.0))


def _spectral_f_pvalue(valor, m):
    f_stat = valor / (1.0 - valor + EPS)
    return float(f_dist.sf(f_stat, 2, 2 * m * SPECTRAL_F_NOISE_BINS))

DETECTORES = {
    "msc": (_msc_stat, _msc_pvalue),
    "mmsc": (_mmsc_stat, _mmsc_pvalue),
    "rayleigh": (_rayleigh_stat, _rayleigh_pvalue),
    "csm": (_csm_stat, _csm_pvalue),
    "hotelling": (_hotelling_stat, _hotelling_pvalue),
    "spectral_f": (_spectral_f_stat, _spectral_f_pvalue),
}


def calcular_janelas(coeficientes, tamanho, passo, detector):
    """Cada detector fornece a estatística e sua cauda nula específica."""
    if detector not in DETECTORES:
        raise ValueError(f"Detector inválido: {detector}. Opções: {', '.join(DETECTORES)}")
    if tamanho < 10 or passo < 1 or int(tamanho) != tamanho or int(passo) != passo:
        raise ValueError("Janela inteira >=10 e passo inteiro >=1")
    estatistica, p_value = DETECTORES[detector]
    valores, p_values, intervalos = [], [], []
    for inicio in range(0, len(coeficientes) - tamanho + 1, passo):
        valor = estatistica(coeficientes[inicio:inicio + tamanho])
        valores.append(valor)
        p_values.append(p_value(valor, tamanho))
        intervalos.append([inicio, inicio + tamanho])
    return valores, p_values, intervalos
