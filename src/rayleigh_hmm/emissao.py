"""Emissoes continuas Gamma para a estatistica de Rayleigh acumulada Z.

Parametrizacao: Gamma(forma k, media mu), ou seja escala = mu / k.

- Ausente: MLE em Z do ESP. Sob H0, Z ~ Exp(1) (k = 1, mu = 1) em qualquer
  numero de epocas, entao a emissao Ausente nao depende do passo.
- Presente: a media cresce com o numero de epocas N acumuladas,
  mu(N) = 1 + (N - 1) * rho, em que rho e o PLV^2 verdadeiro da resposta.
  Priori em (k, rho), verossimilhanca nos Z sinteticos (ESP + senoide),
  posteriori numa grade log-espacada e emissao pela preditiva posterior
  p(z | N, dados) = soma_j w_j Gamma(z | k_j, mu_j(N)).
- Presente marginal: pdf fixa p(z | N) = integral dupla de
  Gamma(z | k, mu(N, rho)) Gamma(k | a_k, b_k) Gamma(rho | a_rho, b_rho).
  Hiperparametros (forma, taxa) por maxima verossimilhanca com um par (k, rho)
  por sequencia sintetica (grade fixa). A integral da emissao usa nos de pesos
  iguais nos quantis medios dessas Gammas; nada e aprendido durante o exame.
"""

import numpy as np
from scipy import optimize, special, stats

EPS = 1e-12


def ajustar_mle(z):
    """MLE de (k, mu) para Gamma. mu_MLE = media(z); k resolve uma equacao 1D."""
    z = np.clip(np.asarray(z, dtype=float).ravel(), EPS, None)
    mu = z.mean()
    s = np.log(mu) - np.log(z).mean()  # >= 0 pela desigualdade de Jensen
    k = optimize.brentq(lambda k: np.log(k) - special.digamma(k) - s, 1e-4, 1e4)
    return {"k": float(k), "mu": float(mu), "n": int(z.size)}


def media_presente(n_epocas, rho):
    return 1.0 + (np.asarray(n_epocas, dtype=float) - 1.0) * rho


def log_priori(k, rho, priori):
    """Prioris independentes: k ~ Gamma(forma, taxa), rho ~ Gamma(forma, taxa)."""
    return stats.gamma.logpdf(
        k, priori["k_forma"], scale=1.0 / priori["k_taxa"]
    ) + stats.gamma.logpdf(rho, priori["rho_forma"], scale=1.0 / priori["rho_taxa"])


def log_verossimilhanca_grade(z, n_epocas, grade_k, grade_rho):
    """Log-verossimilhanca Gamma(k, mu_i(rho)) em toda a grade (k, rho).

    log L = n (k log k - lgamma k) + (k - 1) soma log z - k S(rho),
    com S(rho) = soma_i [log mu_i + z_i / mu_i]; S so depende de rho.
    """
    z = np.clip(np.asarray(z, dtype=float).ravel(), EPS, None)
    mu = media_presente(np.asarray(n_epocas).ravel()[None, :], grade_rho[:, None])
    s_rho = np.sum(np.log(mu) + z[None, :] / mu, axis=1)
    k = grade_k[:, None]
    return z.size * (k * np.log(k) - special.gammaln(k)) + (k - 1.0) * np.log(z).sum() - k * s_rho[None, :]


def posteriori_em_grade(z, n_epocas, priori, grade_k, grade_rho, massa_minima=1e-10):
    """Posteriori de (k, rho) numa grade log-espacada.

    Numa grade log-espacada a celula (i, j) tem area proporcional a k_i * rho_j,
    entao a massa da celula e densidade * k * rho. Pontos com massa menor que
    `massa_minima` sao descartados da preditiva (custo), e a massa e
    renormalizada.
    """
    k, rho = np.meshgrid(grade_k, grade_rho, indexing="ij")
    log_area = np.log(k) + np.log(rho)
    log_pri = log_priori(k, rho, priori)
    log_post = log_verossimilhanca_grade(z, n_epocas, grade_k, grade_rho) + log_pri

    def normalizar(log_massa):
        massa = np.exp(log_massa - log_massa.max())
        return massa / massa.sum()

    massa_post = normalizar(log_post + log_area)
    i_map = np.unravel_index(np.argmax(log_post), log_post.shape)
    manter = massa_post > massa_minima
    return {
        "grade_k": grade_k,
        "grade_rho": grade_rho,
        "massa_posteriori": massa_post,
        "massa_priori": normalizar(log_pri + log_area),
        "map": {"k": float(k[i_map]), "rho": float(rho[i_map])},
        "media": {"k": float(np.sum(massa_post * k)), "rho": float(np.sum(massa_post * rho))},
        "na_borda": bool(
            i_map[0] in (0, len(grade_k) - 1) or i_map[1] in (0, len(grade_rho) - 1)
        ),
        "n": int(np.size(z)),
        "pontos_preditiva": {
            "k": k[manter],
            "rho": rho[manter],
            "peso": massa_post[manter] / massa_post[manter].sum(),
        },
    }


def log_pdf_gamma(z, k, mu):
    z = np.clip(np.asarray(z, dtype=float), EPS, None)
    return stats.gamma.logpdf(z, k, scale=mu / k)


def log_pdf_preditiva(z, n_epocas, pontos):
    """log p(z | N, dados) = log soma_j w_j Gamma(z | k_j, mu_j(N))."""
    z = np.clip(np.asarray(z, dtype=float), EPS, None)
    forma = z.shape
    n_epocas = np.broadcast_to(n_epocas, forma).ravel()
    mu = media_presente(n_epocas[:, None], pontos["rho"][None, :])
    termos = stats.gamma.logpdf(z.ravel()[:, None], pontos["k"][None, :], scale=mu / pontos["k"][None, :])
    return special.logsumexp(termos + np.log(pontos["peso"])[None, :], axis=1).reshape(forma)


def matriz_verossimilhanca_sequencias(z, n_epocas, ids, grade_k, grade_rho):
    """log prod_t Gamma(z_st | k, mu(N_st, rho)) em (sequencias, len(k), len(rho))."""
    z = np.clip(np.asarray(z, dtype=float).ravel(), EPS, None)
    n_epocas = np.asarray(n_epocas, dtype=float).ravel()
    _, ids = np.unique(np.asarray(ids).ravel(), return_inverse=True)
    ordem = np.argsort(ids, kind="stable")
    z, n_epocas, ids = z[ordem], n_epocas[ordem], ids[ordem]
    inicios = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    t_s = np.diff(np.r_[inicios, z.size])
    log_z_s = np.add.reduceat(np.log(z), inicios)
    mu = media_presente(n_epocas[None, :], grade_rho[:, None])
    s_rho = np.add.reduceat(np.log(mu) + z[None, :] / mu, inicios, axis=1).T  # (seq, rho)
    k = grade_k[None, :, None]
    return (t_s[:, None, None] * (k * np.log(k) - special.gammaln(k))
            + (k - 1.0) * log_z_s[:, None, None] - k * s_rho[:, None, :])


def ajustar_hierarquico(matriz, grade_k, grade_rho, inicios, limites_log=(-5.0, 11.0), limite_massa_fora=1e-3):
    """Maxima verossimilhanca marginal de (a_k, b_k, a_rho, b_rho) na grade fixa.

    Com grade log-espacada, o peso do no e prop. a densidade * k * rho. So os
    pesos mudam na otimizacao; gradiente analitico de familia exponencial,
    L-BFGS-B em log dos hiperparametros.
    """
    log_k, log_r = np.log(grade_k), np.log(grade_rho)
    maximo = matriz.max(axis=(1, 2), keepdims=True)
    rel = np.exp(matriz - maximo)  # celulas desprezadas so se < e-700 do pico da sequencia
    soma_max = float(maximo.sum())
    n_seq = matriz.shape[0]

    def objetivo(theta):
        a, b, c, d = np.exp(theta)
        pk = a * log_k - b * grade_k
        pr = c * log_r - d * grade_rho
        pk = np.exp(pk - pk.max())
        pr = np.exp(pr - pr.max())
        norm_k, norm_r = pk.sum(), pr.sum()
        pk, pr = pk / norm_k, pr / norm_r
        por_k = rel @ pr  # (seq, k)
        marginal = por_k @ pk
        resp_k = (por_k / marginal[:, None]).mean(axis=0) * pk
        resp_r = ((pk[None, :] @ rel).squeeze(1) / marginal[:, None]).mean(axis=0) * pr
        # d log verossimilhanca / d theta = (E_post - E_priori)[estatistica] * hiperparametro
        grad = np.array([(resp_k - pk) @ log_k, -(resp_k - pk) @ grade_k,
                         (resp_r - pr) @ log_r, -(resp_r - pr) @ grade_rho]) * np.exp(theta)
        return -np.log(marginal).mean(), -grad

    corridas = []
    for inicio in inicios:
        r = optimize.minimize(objetivo, np.log(inicio), jac=True, method="L-BFGS-B",
                              bounds=[limites_log] * 4, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6})
        corridas.append({"inicio": list(map(float, inicio)), "hiper": np.exp(r.x).tolist(),
                         "log_verossimilhanca": float(-r.fun * n_seq + soma_max),
                         "iteracoes": int(r.nit), "sucesso": bool(r.success)})
    melhor = max(corridas, key=lambda c: c["log_verossimilhanca"])
    a, b, c, d = melhor["hiper"]
    hiper = np.array(melhor["hiper"])
    fora_k = float(stats.gamma.cdf(grade_k[0], a, scale=1 / b) + stats.gamma.sf(grade_k[-1], a, scale=1 / b))
    fora_rho = float(stats.gamma.cdf(grade_rho[0], c, scale=1 / d) + stats.gamma.sf(grade_rho[-1], c, scale=1 / d))
    no_limite = bool(np.any(np.log(hiper) <= limites_log[0] + 1e-6) or np.any(np.log(hiper) >= limites_log[1] - 1e-6))
    dispersao = max(np.max(np.abs(np.array(x["hiper"]) / hiper - 1.0)) for x in corridas)
    return {
        "a_k": a, "b_k": b, "a_rho": c, "b_rho": d,
        "k_media": a / b, "k_desvio": a ** 0.5 / b,
        "rho_media": c / d, "rho_desvio": c ** 0.5 / d,
        "log_verossimilhanca": melhor["log_verossimilhanca"],
        "massa_fora_grade": {"k": fora_k, "rho": fora_rho},
        "hiper_no_limite": no_limite,
        "na_borda": bool(max(fora_k, fora_rho) > limite_massa_fora or no_limite),
        "dispersao_relativa_inicios": float(dispersao),
        "corridas": corridas,
    }


def nos_quadratura(hiper, j_k, j_rho):
    """Nos (k, rho) de pesos iguais nos quantis medios das Gammas ajustadas."""
    q_k = stats.gamma.ppf((np.arange(j_k) + 0.5) / j_k, hiper["a_k"], scale=1.0 / hiper["b_k"])
    q_r = stats.gamma.ppf((np.arange(j_rho) + 0.5) / j_rho, hiper["a_rho"], scale=1.0 / hiper["b_rho"])
    k, rho = np.meshgrid(q_k, q_r, indexing="ij")
    return k.ravel(), rho.ravel()
