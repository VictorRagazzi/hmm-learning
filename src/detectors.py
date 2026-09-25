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
from typing import Callable

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import f as f_dist

EPS = 1e-12


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
# 3) CSM - aqui implementado como o teste T^2 de Hotelling sobre a media
#    complexa (as vezes chamado de "T2circ" na literatura de ASSR: John &
#    Picton, 2000). Testa se a media complexa dos coeficientes e
#    significativamente diferente de zero, usando um estimador de variancia
#    residual (diferente do estimador usado na MSC).
# ============================================================
# F = (M-1) * |media|^2 / variancia_residual  ~  F(2, 2(M-1)) sob H0.
# Para reaproveitar a mesma discretizacao/threshold em escala [0,1] dos
# outros detectores, a estatistica reportada e a transformacao monotonica
# msc_equivalente = F / (F + (M-1)), que preserva a ordenacao de evidencia
# mas usa o estimador de variancia proprio do teste T^2/F (nao o da MSC).

def _csm_estatistica_f(janela):
    m = len(janela)
    media = np.mean(janela)
    residuos = janela - media
    s2 = float(np.sum(np.abs(residuos) ** 2)) / (m - 1)
    f_stat = (m - 1) * (np.abs(media) ** 2) / (s2 + EPS)
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
        "Testa se a media complexa dos coeficientes difere de zero, usando "
        "um estimador de variancia residual proprio (T2circ, John & Picton, "
        "2000). F = (M-1)*|media|^2/variancia ~ F(2, 2(M-1)). "
        "Matematicamente relacionado a MSC por uma transformacao monotonica, "
        "mas reportado como teste separado por usar outro estimador de "
        "variancia e ser uma referencia classica distinta na literatura."
    ),
    estatistica=_csm_stat,
    p_value=_csm_pvalue,
    thresholds=_csm_thresholds,
)


# ============================================================
# 4) MMSC - MSC media entre duas metades da janela (split-half)
# ============================================================
# Divide a janela de M epocas em duas metades de M//2 epocas, calcula a MSC
# em cada metade separadamente e usa a media das duas. Isso da uma
# estatistica menos sensivel a uma unica meia-janela ruidosa. O p-valor e
# aproximado tratando a MSC media como Beta(1, M//2 - 1) (aproximacao
# conservadora: a soma/media de duas Betas independentes nao e exatamente
# Beta, mas essa aproximacao e simples e funciona bem na pratica para
# ordenar evidencia).

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


def _mmsc_pvalue(valor, m):
    m_ef = _mmsc_m_efetivo(m)
    return float(beta_dist.sf(valor, 1, m_ef - 1))


def _mmsc_thresholds(m, p_boundaries):
    m_ef = _mmsc_m_efetivo(m)
    return beta_dist.isf(np.asarray(p_boundaries, dtype=float), 1, m_ef - 1)


MMSC = Detector(
    nome="mmsc",
    nome_exibicao="MMSC (MSC media entre sub-janelas / split-half)",
    descricao=(
        "Divide a janela em duas metades, calcula a MSC em cada uma e usa a "
        "media das duas. Reduz a chance de uma metade ruidosa dominar a "
        "decisao, ao custo de M/2 epocas por metade. P-valor aproximado por "
        "Beta(1, M//2 - 1) (conservador, nao exato)."
    ),
    estatistica=_mmsc_stat,
    p_value=_mmsc_pvalue,
    thresholds=_mmsc_thresholds,
)


DETECTOR_REGISTRY = {d.nome: d for d in (MSC, MMSC, RAYLEIGH, CSM)}


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
    print("\n" + "-" * 64)
    print(f"Detector ativo: {detector.nome_exibicao}  (chave: {detector.nome!r})")
    print(f"  {detector.descricao}")
    print("-" * 64)


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