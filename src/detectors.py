"""Registro plugavel de detectores para deteccao de resposta evocada (ASSR).

Cada detector define, a partir dos coeficientes FFT complexos de uma janela
de M epocas (um coeficiente por epoca, no bin alvo):

  - ``estatistica(janela)``  -> um numero em [0, 1], onde valores maiores
    indicam mais evidencia contra H0 (ausencia de resposta). Normalizar tudo
    em [0, 1] mantem get_obs_matrix.py / hmm_inference.py / busca de
    parametros identicos, independente do detector escolhido.
  - ``p_value(valor, M)``    -> P(estatistica >= valor | H0), pela
    distribuicao nula teorica de cada teste.
  - ``thresholds(M, p_boundaries)`` -> converte uma lista de p-valores
    (fronteiras dos labels discretos) nos thresholds correspondentes da
    estatistica, em ORDEM CRESCENTE (mesma convencao de
    get_obs_matrix.P_VALUE_BOUNDARIES, que esta em ordem decrescente de
    p-valor = ordem crescente de evidencia).

Para adicionar um detector novo: escreva as 3 funcoes acima, embrulhe em um
``Detector`` e registre em DETECTOR_REGISTRY. Nenhum outro arquivo do projeto
precisa ser alterado.

NOTA DE NOMENCLATURA: "MMSC" e "CSM" nao tem uma definicao unica e universal
na literatura de ASSR. As definicoes abaixo sao escolhas razoaveis e estao
documentadas em cada Detector; ajuste-as se voce tinha uma formula especifica
em mente.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import f as f_dist

EPS = 1e-12
SPECTRAL_F_NOISE_BINS = 2


@dataclass(frozen=True)
class Detector:
    nome: str  # chave curta, usada em DETECTOR_NAME / ASSR_DETECTOR / JSON
    nome_exibicao: str
    descricao: str
    estatistica: Callable[[np.ndarray], float]
    p_value: Callable[[float, int], float]
    thresholds: Callable[[int, np.ndarray], np.ndarray]


# ============================================================
# 1) MSC - Magnitude-Squared Coherence
# ============================================================

def _msc_stat(janela):
    m = len(janela)
    numerador = np.abs(np.sum(janela)) ** 2
    denominador = m * np.sum(np.abs(janela) ** 2)
    return float(np.clip(numerador / (denominador + EPS), 0.0, 1.0))


def _msc_pvalue(valor, m):
    return float(beta_dist.sf(valor, 1, m - 1))


def _msc_thresholds(m, p_boundaries):
    return beta_dist.isf(np.asarray(p_boundaries, dtype=float), 1, m - 1)


MSC = Detector(
    nome="msc",
    nome_exibicao="MSC (Coerencia Quadratica de Magnitude)",
    descricao=(
        "Usa fase e magnitude conjuntamente. H0: coeficientes complexos "
        "gaussianos circulares, epocas independentes. MSC ~ Beta(1, M-1)."
    ),
    estatistica=_msc_stat,
    p_value=_msc_pvalue,
    thresholds=_msc_thresholds,
)


# ============================================================
# 2) Rayleigh test - coerencia de fase (PLV^2)
# ============================================================
# Ignora a magnitude dos coeficientes e testa so se a FASE se concentra em
# torno de um valor entre epocas. Estatistica reportada: PLV^2 (0 a 1).
# P-valor pela aproximacao classica de Rayleigh (Zar, 1999; tambem usada em
# softwares de ASSR como Dobie & Wilson, 1996): com z = M * PLV^2,
# P(PLV^2 >= v | H0) = exp(-z). Essa aproximacao e boa para M nao muito
# pequeno (tipicamente M >= 10, que e o WINDOW_SIZE_EPOCHS padrao do projeto).

def _rayleigh_stat(janela):
    m = len(janela)
    fases_unitarias = janela / (np.abs(janela) + EPS)
    plv = np.abs(np.sum(fases_unitarias)) / m
    return float(np.clip(plv ** 2, 0.0, 1.0))


def _rayleigh_pvalue(valor, m):
    z = m * valor
    return float(np.clip(np.exp(-z), 0.0, 1.0))


def _rayleigh_thresholds(m, p_boundaries):
    p = np.asarray(p_boundaries, dtype=float)
    z = -np.log(p)
    return z / m


RAYLEIGH = Detector(
    nome="rayleigh",
    nome_exibicao="Teste de Rayleigh (coerencia de fase, PLV^2)",
    descricao=(
        "Usa APENAS a fase dos coeficientes (ignora magnitude). H0: fases "
        "uniformemente distribuidas entre epocas. Estatistica = PLV^2; "
        "p-valor pela aproximacao classica P(PLV^2 >= v) = exp(-M*v) "
        "(Zar, 1999). Mais sensivel a respostas consistentes em fase mas "
        "fracas em amplitude."
    ),
    estatistica=_rayleigh_stat,
    p_value=_rayleigh_pvalue,
    thresholds=_rayleigh_thresholds,
)


# ============================================================
# 3) CSM - teste T^2 circular de Hotelling sobre a media complexa. Sob as
#    mesmas hipoteses gaussianas e variancia circular da MSC, e algebricamente
#    equivalente a MSC; permanece como uma forma classica alternativa de
#    expressar o mesmo teste, nao como evidencia independente.
# ============================================================
# F = M * |media|^2 / variancia_residual  ~  F(2, 2(M-1)) sob H0.
# Para reaproveitar a mesma discretizacao/threshold em escala [0,1] dos
# outros detectores, a estatistica reportada e a transformacao monotonica
# msc_equivalente = F / (F + (M-1)), que preserva a ordenacao de evidencia
# sua transformacao F/(F+M-1) e exatamente a MSC sob esse estimador.

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


def _csm_thresholds(m, p_boundaries):
    f_thresh = f_dist.isf(np.asarray(p_boundaries, dtype=float), 2, 2 * (m - 1))
    return f_thresh / (f_thresh + (m - 1))


CSM = Detector(
    nome="csm",
    nome_exibicao="CSM (T^2 de Hotelling sobre a media complexa)",
    descricao=(
        "Teste T2 circular de Hotelling sobre a media complexa. Com o "
        "estimador circular de variancia, F = M*|media|^2/s2 ~ "
        "F(2, 2(M-1)); a transformacao para [0,1] e algebricamente igual "
        "a MSC sob as mesmas hipoteses. E uma formulacao alternativa, nao "
        "um detector independente da MSC."
    ),
    estatistica=_csm_stat,
    p_value=_csm_pvalue,
    thresholds=_csm_thresholds,
)


# ============================================================
# 4) MMSC - MSC media entre duas metades da janela (split-half)
# ============================================================
# Divide a janela em duas metades. Sob H0, cada MSC tem distribuicao
# Beta(1, h-1); a media de duas dessas variaveis tem uma distribuicao de
# convolucao, calculada numericamente em uma tabela interpolada.

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


def _mmsc_thresholds(m, p_boundaries):
    grid, tail = _mmsc_null_table(m)
    return np.interp(
        np.asarray(p_boundaries, dtype=float), tail[::-1], grid[::-1]
    )


MMSC = Detector(
    nome="mmsc",
    nome_exibicao="MMSC (MSC media entre sub-janelas / split-half)",
    descricao=(
        "Estatistica experimental split-half: divide a janela em duas "
        "metades, calcula MSC em cada uma e usa a media. O p-valor usa a "
        "convolucao numerica das duas distribuicoes Beta independentes sob "
        "H0. Isso calibra a estatistica implementada, mas nao implica que o "
        "método seja superior a MSC."
    ),
    estatistica=_mmsc_stat,
    p_value=_mmsc_pvalue,
    thresholds=_mmsc_thresholds,
)


# ============================================================
# 5) Hotelling T2 geral sobre partes real e imaginaria
# ============================================================

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


def _hotelling_thresholds(m, p_boundaries):
    if m <= 2:
        return np.ones_like(np.asarray(p_boundaries, dtype=float))
    f_thresh = f_dist.isf(np.asarray(p_boundaries, dtype=float), 2, m - 2)
    return f_thresh / (1.0 + f_thresh)


HOTELLING = Detector(
    nome="hotelling",
    nome_exibicao="Hotelling T2 geral (real/imaginario)",
    descricao=(
        "Testa media complexa zero estimando a covariancia 2x2 completa das "
        "partes real e imaginaria. Sob normalidade multivariada, a forma "
        "escalada segue F(2, M-2). Nao impoe variancia circular."
    ),
    estatistica=_hotelling_stat,
    p_value=_hotelling_pvalue,
    thresholds=_hotelling_thresholds,
)


# ============================================================
# 6) F espectral local: alvo coerente / potencia de bins de ruido
# ============================================================

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


def _spectral_f_thresholds(m, p_boundaries):
    f_thresh = f_dist.isf(
        np.asarray(p_boundaries, dtype=float),
        2, 2 * m * SPECTRAL_F_NOISE_BINS,
    )
    return f_thresh / (1.0 + f_thresh)


SPECTRAL_F = Detector(
    nome="spectral_f",
    nome_exibicao="F espectral local (alvo coerente / ruido lateral)",
    descricao=(
        "Compara M*|media complexa no alvo|^2 com a potencia media em dois "
        "bins locais sem estimulo. Sob bins complexos gaussianos, circulares "
        "e independentes com mesma variancia, segue F(2, 4M)."
    ),
    estatistica=_spectral_f_stat,
    p_value=_spectral_f_pvalue,
    thresholds=_spectral_f_thresholds,
)


DETECTOR_REGISTRY = {
    d.nome: d for d in (MSC, MMSC, RAYLEIGH, CSM, HOTELLING, SPECTRAL_F)
}


def obter_detector(nome):
    try:
        return DETECTOR_REGISTRY[nome]
    except KeyError:
        disponiveis = ", ".join(sorted(DETECTOR_REGISTRY))
        raise ValueError(
            f"Detector {nome!r} desconhecido. Disponiveis: {disponiveis}"
        )


def listar_detectores():
    return list(DETECTOR_REGISTRY.values())


# ============================================================
# Estatistica por janelas (generico para qualquer detector)
# ============================================================

def calcular_estatistica_janelas(coeficientes, detector, tamanho_janela, passo_janela):
    """Aplica ``detector.estatistica`` a cada janela completa de epocas.

    Substitui a antiga ``calcular_msc_janelas`` (especifica de MSC) de forma
    generica: qualquer detector do registro pode ser usado aqui.
    """
    n_epocas = len(coeficientes)
    valores = []
    intervalos = []
    for inicio in range(0, n_epocas - tamanho_janela + 1, passo_janela):
        fim = inicio + tamanho_janela
        janela = coeficientes[inicio:fim]
        valores.append(detector.estatistica(janela))
        intervalos.append((inicio, fim))
    return np.asarray(valores), intervalos


# ============================================================
# Impressao "bonita" para selecao/relato do detector
# ============================================================

def imprimir_menu_detectores():
    print("\n" + "=" * 64)
    print("DETECTORES DISPONIVEIS")
    print("=" * 64)
    for i, d in enumerate(listar_detectores(), start=1):
        print(f"  [{i}] {d.nome:10s} {d.nome_exibicao}")
        print(f"        {d.descricao}")
    print("=" * 64)


def imprimir_banner_detector(detector):
    print()
    # print("\n" + "-" * 64)
    # print(f"Detector ativo: {detector.nome_exibicao}  (chave: {detector.nome!r})")
    # print(f"  {detector.descricao}")
    # print("-" * 64)


def escolher_detector_interativo(padrao="msc"):
    """Mostra o menu e le a escolha do usuario (nome ou numero)."""
    imprimir_menu_detectores()
    nomes = [d.nome for d in listar_detectores()]
    try:
        escolha = input(
            f"Escolha um detector pelo nome ou numero [{padrao}]: "
        ).strip().lower()
    except EOFError:
        escolha = ""
    if not escolha:
        detector = obter_detector(padrao)
    elif escolha.isdigit():
        idx = int(escolha) - 1
        if not 0 <= idx < len(nomes):
            raise ValueError(f"Numero invalido: {escolha}")
        detector = obter_detector(nomes[idx])
    else:
        detector = obter_detector(escolha)
    imprimir_banner_detector(detector)
    return detector
