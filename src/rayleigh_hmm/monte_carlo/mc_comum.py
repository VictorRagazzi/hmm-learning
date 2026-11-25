"""Nucleo compartilhado (EXPLORATORIO): emissao Presente por Monte Carlo e por MCMC.

Nao altera nenhum modulo existente; importa os de src/rayleigh_hmm.

- pontos_mc: M amostras independentes de (k, rho) das Gammas ajustadas por ML-II.
- mcmc_hiper: Metropolis-Hastings em theta = log(a_k, b_k, a_rho, b_rho) com 4
  cadeias avancando em bloco (uma unica multiplicacao de matriz por iteracao).
- ajustar_fixo: baseline sem priori nem hierarquia, um unico (k, rho) por maxima verossimilhanca.
- pontos_mcmc: sorteia hiperparametros da posteriori e (k, rho) de cada um.
"""
import os
import sys
import time

import numpy as np
from scipy import optimize, special

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emissao  # noqa: E402
import experimento as ex  # noqa: E402

# ---------------- configuracao ----------------
SEMENTE = 20240601
M_AMOSTRAS = 2000
N_ITER = 6000           # iteracoes por cadeia (inclui o burn-in); tau ~ 13 => ~400 amostras indep. por cadeia
BURN_IN = 2000
THIN = 10
LIMITES_LOG = (-5.0, 11.0)   # hiperpriori uniforme em theta dentro destes limites
ACEITACAO_ALVO = 0.25
JANELA_ADAPTACAO = 250       # so durante o burn-in
INICIO_DISPERSO = (1.0, 0.5, 5.0, 10.0)
INICIOS_CADEIAS = list(ex.INICIOS_HIER) + [INICIO_DISPERSO]
NOMES_PARAM = ["a_k", "b_k", "a_rho", "b_rho"]
USAR_GPU = True              # torch/CUDA na log-posteriori se disponivel; senao numpy na CPU
PASSOS_PROGRESSO = 10        # o MH imprime ~10 vezes (a cada 10% das iteracoes)


# ---------------- Monte Carlo simples ----------------
def pontos_mc(hiper, m=M_AMOSTRAS, semente=SEMENTE):
    """M pares (k, rho) iid das Gammas ajustadas; pesos 1/M."""
    rng = np.random.default_rng(semente)
    k = rng.gamma(hiper["a_k"], 1.0 / hiper["b_k"], m)
    rho = rng.gamma(hiper["a_rho"], 1.0 / hiper["b_rho"], m)
    return {"k": k, "rho": rho, "peso": np.full(m, 1.0 / m)}


# ---------------- baseline FIXO ----------------
LIMITES_LOG_RHO_FIXO = (np.log(1e-8), np.log(1.0))


def ajustar_fixo(z, n_ep):
    """Um unico (k, rho) por maxima verossimilhanca em todos os pares (Z, N); sem priori nem hierarquia."""
    z = np.clip(np.asarray(z, dtype=float).ravel(), emissao.EPS, None)
    n_ep = np.asarray(n_ep, dtype=float).ravel()
    n, media_log_z = z.size, np.log(z).mean()

    def k_otimo(rho):
        mu = emissao.media_presente(n_ep, rho)
        s = np.mean(np.log(mu) + z / mu) - media_log_z - 1.0
        k = optimize.brentq(lambda k: np.log(k) - special.digamma(k) - s, 1e-4, 1e6)
        return k, mu, n * (k * np.log(k) - special.gammaln(k)) + (k - 1.0) * n * media_log_z - k * n * (s + media_log_z + 1.0)

    res = optimize.minimize_scalar(lambda t: -k_otimo(np.exp(t))[2], bounds=LIMITES_LOG_RHO_FIXO,
                                   method="bounded", options={"xatol": 1e-10})
    rho = float(np.exp(res.x))
    k, _, log_l = k_otimo(rho)
    na_borda = bool(min(res.x - LIMITES_LOG_RHO_FIXO[0], LIMITES_LOG_RHO_FIXO[1] - res.x) < 1e-3)
    return {"k": float(k), "rho": rho, "log_verossimilhanca": float(log_l), "n": int(n), "na_borda": na_borda}


# ---------------- log posteriori (copia da logica de ajustar_hierarquico) ----------------
class LogPosteriori:
    """log verossimilhanca marginal total na grade fixa + log hiperpriori uniforme.

    Mesma conta do `objetivo` de emissao.ajustar_hierarquico: soma sobre
    sequencias de log sum_ij pk_i pr_j exp(matriz_sij), com pesos de no
    proporcionais a densidade * k * rho (grade log-espacada). `rel` fica em
    float32 para reduzir o trafego de memoria (erro relativo ~1e-7 no log).
    Avalia varios theta de uma vez: theta de forma (C, 4) (um matvec por theta).
    """

    def __init__(self, matriz, grade_k, grade_rho, limites_log=LIMITES_LOG, usar_gpu=None):
        self.grade_k, self.grade_rho = grade_k, grade_rho
        self.log_k, self.log_r = np.log(grade_k), np.log(grade_rho)
        maximo = matriz.max(axis=(1, 2), keepdims=True)
        rel = np.exp(matriz - maximo).astype(np.float32)
        self.soma_max = float(maximo.sum())
        self.n_seq, self.nk, self.nr = matriz.shape
        self.limites = limites_log
        self.gpu = None
        if USAR_GPU if usar_gpu is None else usar_gpu:
            # import/uso de CUDA so aqui (dentro do worker, depois do fork)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    self.torch = torch
                    self.gpu = torch.device("cuda")
                    self.rel_gpu = torch.from_numpy(rel.reshape(-1, self.nr)).to(self.gpu)
            except ImportError:
                pass
        self.rel = None if self.gpu is not None else rel

    def _pesos(self, theta):
        a, b, c, d = np.exp(theta).T  # (C,)
        pk = a[:, None] * self.log_k - b[:, None] * self.grade_k
        pr = c[:, None] * self.log_r - d[:, None] * self.grade_rho
        pk = np.exp(pk - pk.max(axis=1, keepdims=True))
        pr = np.exp(pr - pr.max(axis=1, keepdims=True))
        return pk / pk.sum(axis=1, keepdims=True), pr / pr.sum(axis=1, keepdims=True)

    def log_verossimilhanca(self, theta):
        theta = np.atleast_2d(theta)
        pk, pr = self._pesos(theta)
        c = theta.shape[0]
        if self.gpu is not None:
            # uma multiplicacao matriz-matriz (S*nk, nr) @ (nr, C) para todas as cadeias; reducao em float64
            t = self.torch
            pr_g = t.from_numpy(pr.astype(np.float32)).to(self.gpu)
            por_k = (self.rel_gpu @ pr_g.T).reshape(self.n_seq, self.nk, c).double()
            pk_g = t.from_numpy(pk).to(self.gpu)
            marginal = t.einsum("skc,ck->sc", por_k, pk_g)
            return t.log(marginal).sum(dim=0).cpu().numpy() + self.soma_max
        # CPU: matriz-vetor por cadeia (mais rapido que matriz-matriz com poucas colunas; limitado por memoria)
        r = self.rel.reshape(-1, self.nr)
        pr32 = pr.astype(np.float32)
        por_k = np.stack([r @ pr32[i] for i in range(c)], axis=-1).reshape(self.n_seq, self.nk, c)
        marginal = np.einsum("skc,ck->sc", por_k.astype(np.float64), pk)
        return np.log(marginal).sum(axis=0) + self.soma_max

    def __call__(self, theta):
        theta = np.atleast_2d(theta)
        dentro = np.all((theta >= self.limites[0]) & (theta <= self.limites[1]), axis=1)
        saida = np.full(theta.shape[0], -np.inf)
        if dentro.any():
            saida[dentro] = self.log_verossimilhanca(theta[dentro])
        return saida


# ---------------- MCMC ----------------
def mcmc_hiper(logpost, inicios=INICIOS_CADEIAS, n_iter=N_ITER, burn_in=BURN_IN, semente=SEMENTE, rotulo=""):
    """Metropolis-Hastings, passeio aleatorio gaussiano, cadeias em bloco.

    Durante o burn-in, a cada JANELA_ADAPTACAO iteracoes cada cadeia atualiza (i) a
    covariancia da proposta (covariancia da janela, 2.38^2/4) e (ii) um fator de
    escala que busca aceitacao ~25%. Depois do burn-in o passo fica fixo.
    Retorna cadeias (n_cadeias, n_iter, 4) e aceitacao pos burn-in por cadeia.
    """
    rng = np.random.default_rng(semente)
    theta = np.log(np.array(inicios, dtype=float))
    c = theta.shape[0]
    lp = logpost(theta)
    cov = np.stack([np.eye(4) * 0.05 ** 2] * c)
    escala = np.ones(c)
    cadeias = np.empty((c, n_iter, 4))
    aceitou = np.zeros((c, n_iter), bool)
    passo_log = max(1, n_iter // PASSOS_PROGRESSO)
    t_ini = time.time()
    for t in range(n_iter):
        chol = np.linalg.cholesky(cov * escala[:, None, None])
        prop = theta + np.einsum("cij,cj->ci", chol, rng.standard_normal((c, 4)))
        lp_prop = logpost(prop)
        ok = np.log(rng.uniform(size=c)) < (lp_prop - lp)
        theta[ok], lp[ok] = prop[ok], lp_prop[ok]
        cadeias[:, t], aceitou[:, t] = theta, ok
        if (t + 1) % passo_log == 0:
            ini = max(0, t + 1 - passo_log)
            print(f"[MH {rotulo}] iter {t + 1}/{n_iter} aceitacao (ultimo bloco) "
                  f"{aceitou[:, ini:t + 1].mean():.2f} tempo {time.time() - t_ini:.0f} s", flush=True)
        if t < burn_in and (t + 1) % JANELA_ADAPTACAO == 0:
            for i in range(c):
                taxa = aceitou[i, t + 1 - JANELA_ADAPTACAO:t + 1].mean()
                escala[i] *= np.exp(2.0 * (taxa - ACEITACAO_ALVO))
                janela = cadeias[i, t + 1 - JANELA_ADAPTACAO:t + 1]
                if taxa > 0.02 and np.all(janela.std(axis=0) > 1e-8):
                    cov[i] = 2.38 ** 2 / 4 * np.cov(janela.T) + 1e-8 * np.eye(4)
                    escala[i] = 1.0
    return {"cadeias": cadeias, "aceitacao": aceitou[:, burn_in:].mean(axis=1),
            "aceitacao_burn_in": aceitou[:, :burn_in].mean(axis=1)}


# ---------------- diagnosticos ----------------
def split_rhat(x):
    """R-chapeu de Gelman-Rubin dividido ao meio. x: (cadeias, n) para um parametro."""
    n = x.shape[1] // 2
    metades = np.concatenate([x[:, :n], x[:, n:2 * n]], axis=0)
    w = metades.var(axis=1, ddof=1).mean()
    b = n * metades.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def autocorrelacao(x, max_lag=200):
    """Autocorrelacao media entre cadeias. x: (cadeias, n)."""
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    f = np.fft.rfft(x, 2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :max_lag] / n
    return (ac / ac[:, :1]).mean(axis=0)


def ess(x):
    """ESS multicadeia (soma das autocorrelacoes positivas iniciais, pares de lags)."""
    m, n = x.shape
    rho = autocorrelacao(x, max_lag=min(n - 1, 1000))
    soma = 0.0
    for t in range(1, len(rho) - 1, 2):
        par = rho[t] + rho[t + 1]
        if par < 0:
            break
        soma += par
    return float(m * n / (1.0 + 2.0 * soma))


def diagnosticos(res, burn_in=BURN_IN, thin=THIN):
    pos = res["cadeias"][:, burn_in:]
    fina = pos[:, ::thin]
    d = {"aceitacao": res["aceitacao"].tolist(), "aceitacao_burn_in": res["aceitacao_burn_in"].tolist(),
         "rhat": {}, "ess_thin": {}, "ess_sem_thin": {}, "media_posteriori": {}, "desvio_posteriori": {}}
    for j, nome in enumerate(NOMES_PARAM):
        v = np.exp(fina[:, :, j])
        d["rhat"][nome] = split_rhat(fina[:, :, j])
        d["ess_thin"][nome] = ess(fina[:, :, j])
        d["ess_sem_thin"][nome] = ess(pos[:, :, j])
        d["media_posteriori"][nome] = float(v.mean())
        d["desvio_posteriori"][nome] = float(v.std())
    d["rhat_max"] = max(d["rhat"].values())
    d["ess_min_thin"] = min(d["ess_thin"].values())
    d["n_amostras_agrupadas"] = int(fina.shape[0] * fina.shape[1])
    return d


# ---------------- emissao MCMC ----------------
def pontos_mcmc(res, m=M_AMOSTRAS, semente=SEMENTE, burn_in=BURN_IN, thin=THIN):
    """Sorteia m hiperparametros da posteriori agrupada (com reposicao) e (k, rho) de cada um."""
    rng = np.random.default_rng(semente)
    amostras = np.exp(res["cadeias"][:, burn_in::thin].reshape(-1, 4))
    h = amostras[rng.integers(0, len(amostras), m)]
    k = rng.gamma(h[:, 0], 1.0 / h[:, 1])
    rho = rng.gamma(h[:, 2], 1.0 / h[:, 3])
    return {"k": k, "rho": rho, "peso": np.full(m, 1.0 / m)}


def ajustar_mcmc(emissoes, hiper_mlii=None, semente=SEMENTE, n_iter=N_ITER, burn_in=BURN_IN, rotulo=""):
    """Roda o MCMC sobre a matriz de verossimilhanca das sequencias de calibracao."""
    matriz = emissao.matriz_verossimilhanca_sequencias(
        emissoes["z_sintetico"], emissoes["n_sintetico"], emissoes["id_sintetico"],
        ex.GRADE_HIER_K, ex.GRADE_HIER_RHO)
    lp = LogPosteriori(matriz, ex.GRADE_HIER_K, ex.GRADE_HIER_RHO)
    del matriz
    t0 = time.time()
    res = mcmc_hiper(lp, n_iter=n_iter, burn_in=burn_in, semente=semente, rotulo=rotulo)
    res["segundos"] = time.time() - t0
    res["logpost"] = lp
    return res


# ---------------- figuras ----------------
def figuras_mcmc(res, hiper_mlii, pasta, prefixo="mcmc", burn_in=BURN_IN, thin=THIN):
    """Trace plots (burn-in marcado), histogramas com ML-II e autocorrelacao."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(pasta, exist_ok=True)
    cad = np.exp(res["cadeias"])
    n_cad, n_iter, _ = cad.shape
    fig, eixos = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for j, (ax, nome) in enumerate(zip(eixos, NOMES_PARAM)):
        for i in range(n_cad):
            ax.plot(cad[i, :, j], lw=0.5, label=f"cadeia {i + 1}")
        ax.axvline(burn_in, color="k", ls="--", lw=1)
        ax.axhline(hiper_mlii[nome], color="r", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(nome)
    eixos[0].legend(ncol=4, fontsize=7)
    eixos[0].set_title("Trace (tracejado preto: fim do burn-in; pontilhado vermelho: ML-II)")
    eixos[-1].set_xlabel("iteracao")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{prefixo}_trace.png"), dpi=110)
    plt.close(fig)

    pos = cad[:, burn_in::thin]
    fig, eixos = plt.subplots(1, 4, figsize=(14, 3.2))
    for j, (ax, nome) in enumerate(zip(eixos, NOMES_PARAM)):
        v = pos[:, :, j].ravel()
        ax.hist(np.log(v), bins=50, density=True, color="C0", alpha=0.7)
        ax.axvline(np.log(hiper_mlii[nome]), color="r", label="ML-II")
        ax.axvline(np.log(v.mean()), color="k", ls="--", label="media post.")
        ax.set_xlabel(f"log {nome}")
    eixos[0].legend(fontsize=7)
    fig.suptitle("Posteriori dos hiperparametros (escala log)")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{prefixo}_posteriori.png"), dpi=110)
    plt.close(fig)

    fig, eixos = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    bruto = res["cadeias"][:, burn_in:]
    for j, (ax, nome) in enumerate(zip(eixos, NOMES_PARAM)):
        ac = autocorrelacao(bruto[:, :, j], max_lag=200)
        ax.plot(ac)
        ax.axvline(thin, color="r", ls="--", label=f"thin = {thin}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lag")
        ax.set_title(nome)
    eixos[0].set_ylabel("autocorrelacao")
    eixos[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{prefixo}_autocorrelacao.png"), dpi=110)
    plt.close(fig)
