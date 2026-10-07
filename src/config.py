"""Configurações do MVP. Edite aqui antes de executar os scripts."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
RESULTS_DIR = RAIZ / "results"
OUTPUTS_DIR = RAIZ / "outputs"

ESTADOS = ("Ausente", "Presente")
NIVEIS = ("ESP", 30, 40, 50, 60, 70)
CHANNEL_INDEX = 0
# Detector aplicado ao ajuste de B, à inferência e à busca (um por execução).
# Opções: rayleigh, msc, mmsc, csm, hotelling, spectral_f.
DETECTOR_NAME = "rayleigh"
WINDOW_SIZE_EPOCHS = 90
WINDOW_STEP_EPOCHS = 45
PRESENT_MIN_DB = 50
P_EPS = 1e-9  # Evita log(0) no ajuste das distribuições Beta.

# Versão escolhida: Beta MAP; MLE permanece para comparação.
METODO = "map"
MATRIX_A = [[0.80, 0.20], [0.45, 0.55]]
PI_INICIAL = [0.90, 0.10]
MIN_CONSECUTIVE = 3
MIN_PERCENT = 0.60  # Fração de janelas Presente desde o começo de ESP.
MODO_REGRA_DECISAO = "OR"

# Priori de Presente: mu ~ Beta(u,v); kappa ~ Gamma(shape,rate).
MU_PRIOR_ALPHA = 2.0
MU_PRIOR_BETA = 4.0
KAPPA_PRIOR_SHAPE = 2.0
KAPPA_PRIOR_RATE = 1.0

# Busca completa, somente in-sample (todos os pacientes).
WINDOW_SIZE_VALUES = (10, 20, 30, 60, 90, 120, 180)
PERSISTENCE_VALUES = (.99, .95, .90, .85, .80, .75, .70, .65, .60, .55, .50)
PI_ABSENT_VALUES = PERSISTENCE_VALUES
MIN_CONSECUTIVE_VALUES = tuple(range(1, 13))
MIN_PERCENT_VALUES = (.01, .025, .05, .075, .09, .10, .20, .30, .40, .50,
                      .60, .70, .80, .90, 1.)
FP_MAXIMO = 0.05


def configuracao():
    """Parâmetros que acompanham cada resultado salvo."""
    return {
        "canal": CHANNEL_INDEX, "detector": DETECTOR_NAME, "janela": WINDOW_SIZE_EPOCHS,
        "passo": WINDOW_STEP_EPOCHS, "presente_min_db": PRESENT_MIN_DB,
        "priori": [MU_PRIOR_ALPHA, MU_PRIOR_BETA,
                   KAPPA_PRIOR_SHAPE, KAPPA_PRIOR_RATE],
        "metodo": METODO, "A": MATRIX_A, "pi": PI_INICIAL,
        "regra": {"consecutivas": MIN_CONSECUTIVE, "percentual": MIN_PERCENT,
                  "modo": MODO_REGRA_DECISAO},
        "p_eps": P_EPS, "avaliacao": "in-sample",
    }
