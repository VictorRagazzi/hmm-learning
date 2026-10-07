"""B contínua global: Beta MLE em ESP; Beta MLE e MAP em estímulos >=50 dB."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta, gamma

import config
from dados import carregar_dados, preparar_protocolos, salvar_json


def estabilizar(valores):
    p = np.asarray(valores, dtype=float)
    if (p.ndim != 1 or len(p) < 2 or not np.all(np.isfinite(p))
            or np.any(p < 0) or np.any(p > 1)):
        raise ValueError("O ajuste requer pelo menos dois p-valores finitos em [0,1]")
    return np.clip(p, config.P_EPS, 1 - config.P_EPS)


def ajustar_mle(valores):
    """Máxima verossimilhança: maximiza produto das densidades dos dados."""
    p = estabilizar(valores)
    a, b, _, _ = beta.fit(p, floc=0, fscale=1)
    return {"a": float(a), "b": float(b), "mu": float(a / (a + b)),
            "kappa": float(a + b), "n": len(p),
            "n_clipped": int(np.count_nonzero(p != valores))}


def ajustar_map(valores, priori):
    """MAP = máxima verossimilhança + informação da priori em mu,kappa.

    mu=a/(a+b) e kappa=a+b. Coordenadas logit/log apenas mantêm parâmetros
    válidos na busca; não se adiciona Jacobiano ao objetivo MAP.
    """
    p = estabilizar(valores)
    u, v, shape, rate = priori
    if not np.all(np.isfinite(priori)) or min(priori) <= 0:
        raise ValueError("Parâmetros da priori devem ser positivos e finitos")
    mle = ajustar_mle(valores)

    def objetivo(coordenadas):
        mu, kappa = expit(coordenadas[0]), np.exp(coordenadas[1])
        log_likelihood = beta.logpdf(p, mu * kappa, (1 - mu) * kappa).sum()
        log_prior = beta.logpdf(mu, u, v) + gamma.logpdf(kappa, shape, scale=1 / rate)
        return -(log_likelihood + log_prior)

    candidatos = []
    for mu, kappa in ((mle["mu"], mle["kappa"]), (u / (u + v), shape / rate), (.5, 2)):
        ajuste = minimize(objetivo, [logit(mu), np.log(kappa)], method="L-BFGS-B",
                          bounds=[(-20, 20), (-20, 20)],
                          options={"ftol": 1e-13, "maxiter": 2000})
        if ajuste.success and np.isfinite(ajuste.fun):
            candidatos.append(ajuste)
    if not candidatos:
        raise RuntimeError("O ajuste MAP não convergiu")
    melhor = min(candidatos, key=lambda x: x.fun)
    if np.any(abs(melhor.x) > 19.9):
        raise ValueError("MAP na borda numérica; reveja a priori")
    mu, kappa = float(expit(melhor.x[0])), float(np.exp(melhor.x[1]))
    return {"a": mu * kappa, "b": (1 - mu) * kappa, "mu": mu, "kappa": kappa,
            "n": len(p), "n_clipped": int(np.count_nonzero(p != valores))}


def construir_matrizes_observacao(protocolos, min_db, priori):
    """Agrupa todos os pacientes/frequências; laterais nunca calibram B."""
    if len({p["detector"] for p in protocolos}) != 1:
        raise ValueError("Calibre B com um único detector por execução")
    registros = {s: [] for s in config.ESTADOS}
    for protocolo in protocolos:
        if protocolo["grupo"] != "alvo":
            continue
        for obs in protocolo["observacoes"]:
            if obs["nivel"] == "ESP":
                estado = "Ausente"
            elif obs["nivel"] >= min_db:
                estado = "Presente"
            else:
                continue
            registros[estado].append({**obs, "paciente": protocolo["paciente"],
                "frequencia": protocolo["frequencia"], "controle_hz": protocolo["controle_hz"]})
    valores = {s: [r["p_value"] for r in registros[s]] for s in config.ESTADOS}
    mle = {s: ajustar_mle(valores[s]) for s in config.ESTADOS}
    map_model = {"Ausente": mle["Ausente"], "Presente": ajustar_map(valores["Presente"], priori)}
    # Tabela didática de massas: integra a PDF em cada intervalo, não soma alturas.
    bordas = [0, .01, .05, .10, .50, 1]
    tabelas = {}
    for metodo, modelos in (("mle", mle), ("map", map_model)):
        tabelas[metodo] = [np.diff(beta.cdf(bordas, modelos[s]["a"], modelos[s]["b"])).tolist()
                          for s in config.ESTADOS]
    esp = registros["Ausente"]
    fp_freq = {}
    for frequencia in sorted({r["frequencia"] for r in esp}):
        p = [r["p_value"] for r in esp if r["frequencia"] == frequencia]
        fp_freq[str(frequencia)] = {"janelas": len(p), "taxa": float(np.mean(np.array(p) <= .05))}
    return {"detector": protocolos[0]["detector"],
            "mle": mle, "map": map_model, "priori_presente": list(priori),
            "presente_min_db": min_db, "estados": config.ESTADOS,
            "bordas_p": bordas, "matriz": tabelas, "calibracao": registros,
            "fp_bruto_esp": {"alpha": .05, "global": float(np.mean(np.array(valores["Ausente"]) <= .05)),
                             "por_frequencia": fp_freq},
            "descricao": "B_s(p)=BetaPDF(p;a_s,b_s). Matriz por intervalos é somente ilustrativa."}


def log_emissoes(p_values, modelos):
    p = np.clip(np.asarray(p_values), config.P_EPS, 1 - config.P_EPS)
    return np.stack([beta.logpdf(p, modelos[s]["a"], modelos[s]["b"])
                     for s in config.ESTADOS], axis=-1)


def salvar_distribuicoes(resultado):
    """Curvas para a apresentação: PDF/CDF de B e prioris dos parâmetros."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(config.OUTPUTS_DIR / ".matplotlib"))
    # O backend Agg grava figuras sem abrir uma interface gráfica.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    p = np.linspace(.0001, .9999, 1000)
    fig, eixos = plt.subplots(2, 2, figsize=(11, 7))
    for coluna, estado in enumerate(config.ESTADOS):
        for metodo in ("mle", "map"):
            modelo = resultado[metodo][estado]
            estilo = "--" if metodo == "mle" else "-"
            largura = 2.5 if metodo == "mle" else 1.2
            eixos[0, coluna].plot(p, beta.pdf(p, modelo["a"], modelo["b"]), estilo,
                                  linewidth=largura, label=metodo.upper())
            eixos[1, coluna].plot(p, beta.cdf(p, modelo["a"], modelo["b"]), estilo,
                                  linewidth=largura, label=metodo.upper())
        for linha, funcao in enumerate(("Densidade (PDF)", "Probabilidade acumulada (CDF)")):
            eixo = eixos[linha, coluna]
            eixo.set(xlabel=f"p-valor ({resultado['detector']})", ylabel=funcao, title=estado)
            eixo.legend()
    # A PDF Presente cresce perto de zero; escala log permite ver também a cauda.
    eixos[0, 1].set_yscale("log")
    eixos[0, 1].set_ylabel("Densidade (PDF), escala log")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "distribuicoes_B.png", dpi=160)
    fig.savefig(config.OUTPUTS_DIR / "distribuicoes_B.pdf")
    plt.close(fig)
    u, v, shape, rate = resultado["priori_presente"]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 3))
    eixos[0].plot(p, beta.pdf(p, u, v))
    eixos[0].set(xlabel="μ", ylabel="Densidade", title=f"Priori μ ~ Beta({u:g},{v:g})")
    k = np.linspace(.001, gamma.ppf(.999, shape, scale=1 / rate), 1000)
    eixos[1].plot(k, gamma.pdf(k, shape, scale=1 / rate))
    eixos[1].set(xlabel="κ", ylabel="Densidade", title=f"Priori κ ~ Gamma({shape:g},{rate:g}), shape/rate")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "prioris.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    dados = carregar_dados()
    cfg = config.configuracao()
    protocolos = preparar_protocolos(dados, cfg["janela"], cfg["passo"])
    resultado = construir_matrizes_observacao(protocolos, cfg["presente_min_db"], cfg["priori"])
    resultado["configuracao"] = cfg
    salvar_json(resultado, config.RESULTS_DIR / "observation_matrix.json")
    salvar_distribuicoes(resultado)
    for metodo in ("mle", "map"):
        print(f"B {metodo.upper()} — parâmetros das densidades:", resultado[metodo])
        print("Massas por intervalo de p:", np.asarray(resultado["matriz"][metodo]))
    print("FP bruto ESP:", resultado["fp_bruto_esp"])
