"""Comparação LOSO: Rayleigh bruto, HMM Beta MLE e HMM Beta MAP.

Execute da raiz: .venv/bin/python src/map_beta_hmm_loso.py
A priori só modifica Presente. Cada emissão é uma única Beta por estado.
"""

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit, digamma
from scipy.stats import beta, gamma

import continuous_rayleigh_hmm as crh
import phase_transition_hmm as phase
import compare_histogram_phase_rayleigh as protocol_tools

# CONFIGURAÇÕES: Gamma usa shape/rate (scale = 1/rate).
MU_PRIOR_ALPHA = 2.0
MU_PRIOR_BETA = 4.0
KAPPA_PRIOR_SHAPE = 2.0
KAPPA_PRIOR_RATE = 1.0
PRESENT_MIN_DB = 50
WINDOW_SIZE_EPOCHS = 120
WINDOW_STEP_EPOCHS = 60
MATRIX_A = [[0.99, 0.01], [0.15, 0.85]]
PI = [0.95, 0.05]
MIN_CONSECUTIVE = 4
MIN_PERCENT = 0.1  # Fração entre 0 e 1; None desativa (0.10 = 10%).
MODO_REGRA_DECISAO = "OR"  # "OR": basta um critério; "AND": exige ambos.
RAW_ALPHA = 0.005
DATA_DIR = "data"
OUTPUT_FILE = Path("results/map_beta_hmm_loso.json")
OUTPUT_DIR = Path("outputs/map_beta_hmm_loso")
P_EPS = 1e-9
GRID_SIZE = 401
METHODS = {"bruto": "Detector bruto", "mle": "HMM sem Bayes (MLE)",
           "map": "HMM com Bayes (MAP)"}


def stabilize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Calibração requer pelo menos dois p-valores finitos")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-valores devem pertencer a [0,1]")
    return np.clip(values, P_EPS, 1 - P_EPS)


def log_likelihood(mu, kappa, p):
    a, b = mu * kappa, (1 - mu) * kappa
    return ((a - 1) * np.log(p).sum() + (b - 1) * np.log1p(-p).sum()
            - len(p) * betaln(a, b))


def log_posterior(mu, kappa, p, prior):
    u, v, shape, rate = prior
    return (log_likelihood(mu, kappa, p) + beta.logpdf(mu, u, v)
            + gamma.logpdf(kappa, shape, scale=1 / rate))


def fit_map(values, prior):
    """Maximiza densidade em (mu,kappa), sem Jacobiano das coordenadas de busca.

    O Jacobiano só entra na integração da posterior para os gráficos.
    """
    p = stabilize(values)
    u, v, shape, rate = prior
    if min(prior) <= 0 or not np.all(np.isfinite(prior)):
        raise ValueError("Todos os hiperparâmetros devem ser positivos e finitos")
    # Prioris singulares de mu podem gerar MAP na borda dependendo dos dados.
    # Não retornar silenciosamente uma solução limitada pela caixa numérica.
    mle = crh._fit_beta(p)
    mu_mle, k_mle = mle["a"] / (mle["a"] + mle["b"]), mle["a"] + mle["b"]
    s1, s2, n = np.log(p).sum(), np.log1p(-p).sum(), len(p)

    def objective(z):
        mu, kappa = expit(z[0]), np.exp(z[1])
        a, b = mu * kappa, (1 - mu) * kappa
        da = s1 - n * (digamma(a) - digamma(kappa))
        db = s2 - n * (digamma(b) - digamma(kappa))
        dmu = kappa * (da - db) + (u - 1) / mu - (v - 1) / (1 - mu)
        dk = mu * da + (1 - mu) * db + (shape - 1) / kappa - rate
        value = -log_posterior(mu, kappa, p, prior)
        grad = -np.array([dmu * mu * (1 - mu), dk * kappa])
        return value, grad

    starts = [(mu_mle, k_mle), (u / (u + v), shape / rate), (.5, 2.)]
    candidates = [minimize(objective, [logit(m), np.log(k)], jac=True,
                           method="L-BFGS-B", bounds=[(-20, 20), (-20, 20)],
                           options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 2000})
                  for m, k in starts]
    good = [c for c in candidates if c.success and np.isfinite(c.fun)]
    if not good:
        raise RuntimeError(f"MAP não convergiu: {[c.message for c in candidates]}")
    best = min(good, key=lambda c: c.fun)
    if np.any(np.abs(best.x) > 19.9):
        raise ValueError("MAP atingiu borda numérica; reveja priori/modelo")
    mu, kappa = float(expit(best.x[0])), float(np.exp(best.x[1]))
    return {"a": mu * kappa, "b": (1 - mu) * kappa, "mu": mu, "kappa": kappa,
            "n": len(p), "n_clipped": int(np.sum(p != values)),
            "log_posterior": float(-best.fun), "gradient_norm": float(np.linalg.norm(best.jac)),
            "successful_starts": len(good), "map_coordinates": "mu,kappa"}


def calibration(protocols, excluded, min_db):
    """Retém uma pessoa inteira; laterais nunca entram no ajuste."""
    records = {"Ausente": [], "Presente": []}
    for protocol in protocols:
        if protocol["participante"] == excluded or protocol["grupo"] != "estimulo":
            continue
        for f in protocol["fases"]:
            if f["nivel"] == "ESP":
                state = "Ausente"
            elif f["nivel"] >= min_db:
                state = "Presente"
            else:
                continue
            for i, value in enumerate(f["p_values"]):
                records[state].append({"participante": protocol["participante"],
                    "arquivo": f["arquivo"], "nivel_db": f["nivel"], "canal": f["canal"],
                    "fs": f["fs"], "frequencia": f["frequencia"],
                    "controle_hz": protocol["controle_hz"],
                    "intervalo_epocas": f["intervalos_locais"][i],
                    "valor_rayleigh": f["valores_ord"][i], "p_value": value})
    return {s: np.asarray([r["p_value"] for r in rows]) for s, rows in records.items()}, records


def posterior_grid(values, prior, fitted, size=GRID_SIZE):
    """Quadratura local adaptativa da posterior conjunta para visualização.

    Integra em mu e log(kappa), incluindo o Jacobiano kappa. Não muda B.
    """
    p = stabilize(values)
    center = np.array([fitted["mu"], np.log(fitted["kappa"])])
    def objective(x):
        return -log_posterior(x[0], np.exp(x[1]), p, prior)
    h = np.array([1e-5, 1e-4])
    hessian = np.zeros((2, 2))
    for i in range(2):
        ei = np.eye(2)[i] * h[i]
        hessian[i, i] = (objective(center + ei) - 2*objective(center)
                         + objective(center - ei)) / h[i]**2
    e0, e1 = np.array([h[0], 0]), np.array([0, h[1]])
    hessian[0, 1] = hessian[1, 0] = (
        objective(center+e0+e1)-objective(center+e0-e1)
        -objective(center-e0+e1)+objective(center-e0-e1))/(4*h[0]*h[1])
    if np.min(np.linalg.eigvalsh(hessian)) <= 0:
        raise RuntimeError("Posterior sem curvatura positiva; quadratura local inadequada")
    sd = np.sqrt(np.diag(np.linalg.inv(hessian)))
    for width in (8, 12, 20):
        mu = np.linspace(max(1e-8, center[0]-width*sd[0]),
                         min(1-1e-8, center[0]+width*sd[0]), size)
        log_k = np.linspace(center[1]-width*sd[1], center[1]+width*sd[1], size)
        k = np.exp(log_k)
        log_density = log_posterior(mu[:, None], k[None, :], p, prior) + log_k[None, :]
        joint = np.exp(log_density - np.max(log_density))
        norm = np.trapezoid(np.trapezoid(joint, log_k, axis=1), mu)
        joint /= norm
        pmu = np.trapezoid(joint, log_k, axis=1)
        plogk = np.trapezoid(joint, mu, axis=0)
        edge_mass = (np.trapezoid(pmu[:5], mu[:5]) + np.trapezoid(pmu[-5:], mu[-5:])
                     + np.trapezoid(plogk[:5], log_k[:5])
                     + np.trapezoid(plogk[-5:], log_k[-5:]))
        if edge_mass < 1e-6:
            return {"mu": mu, "mu_pdf": pmu, "kappa": k, "kappa_pdf": plogk/k,
                    "joint_logk": joint, "edge_mass": float(edge_mass), "width_sd": width}
    raise RuntimeError("Posterior não coberta pela grade; ampliar quadratura")


def detection_rule(consecutive=None, percent=None, mode=None):
    """Valida critérios explícitos; None desativa cada limiar."""
    if consecutive is None and percent is None:
        raise ValueError("Ative ao menos um critério de detecção")
    if consecutive is not None and (isinstance(consecutive, bool)
            or not isinstance(consecutive, (int, np.integer)) or consecutive < 1):
        raise ValueError("Consecutivas deve ser inteiro >=1 ou None")
    if percent is not None and (not np.isfinite(percent) or not 0 < percent <= 1):
        raise ValueError("Porcentagem deve ser fração em (0,1] ou None")
    if mode not in ("OR", "AND"):
        raise ValueError("Modo da regra deve ser OR ou AND")
    return {"consecutivas": consecutive, "min_percent": percent, "modo_regra_decisao": mode}


def first_detection(binary, ends, phases, rule, phase_min=0):
    """Primeiro disparo causal: fração de positivos no prefixo desde ESP.

    Denominador inclui somente janelas já observadas; nunca usa o futuro.
    Contadores continuam entre intensidades, como o delta do HMM.
    """
    consecutive, positives = 0, 0
    for index, (positive, end, phase_index) in enumerate(zip(binary, ends, phases, strict=True), 1):
        consecutive = consecutive + 1 if positive else 0
        positives += int(positive)
        criteria = []
        if rule["consecutivas"] is not None:
            criteria.append(consecutive >= rule["consecutivas"])
        if rule["min_percent"] is not None:
            criteria.append(positives / index >= rule["min_percent"])
        passed = all(criteria) if rule["modo_regra_decisao"] == "AND" else any(criteria)
        if phase_index >= phase_min and passed:
            return int(end), int(phase_index)
    return None


def evaluate(protocols, models=None, rule=None, matrix_a=None, pi=None):
    matrix_a = MATRIX_A if matrix_a is None else matrix_a
    pi = PI if pi is None else pi
    if rule is None:
        rule = detection_rule(MIN_CONSECUTIVE, MIN_PERCENT, MODO_REGRA_DECISAO)
    else:
        rule = detection_rule(rule["consecutivas"], rule["min_percent"], rule["modo_regra_decisao"])
    details = []
    for protocol in protocols:
        p = np.concatenate([f["p_values"] for f in protocol["fases"]])
        ends = np.concatenate([f["inicio_global"] + np.asarray(f["finais_locais"], dtype=int)
                               for f in protocol["fases"]])
        phases = np.concatenate([np.full(len(f["p_values"]), i, dtype=int)
                                 for i, f in enumerate(protocol["fases"])])
        binary = p <= RAW_ALPHA if models is None else phase._estados_hmm(p, matrix_a, pi, models)
        first = first_detection(binary, ends, phases, rule)
        detection = first_detection(binary, ends, phases, rule, phase_min=1)
        esp = phases == 0
        fp_esp = first_detection(binary[esp], ends[esp], phases[esp], rule) is not None
        total = detection[0] if detection else protocol["n_epocas_total"]
        since_stimulus = total - protocol["fases"][1]["inicio_global"]
        level_index = detection[1] if detection else None
        details.append({"participante": protocol["participante"], "grupo": protocol["grupo"],
            "frequencia_indice": protocol["frequencia_indice"],
            "frequencia": protocol["fases"][0]["frequencia"], "fp_esp": fp_esp,
            "detectou": detection is not None,
            "primeiro_nivel_db": phase.LEVELS[level_index] if detection else None,
            "tempo_total_epocas": total, "tempo_desde_inicio_estimulo_epocas": since_stimulus,
            "disparo_em_qualquer_fase": first is not None,
            "primeiro_disparo_total_s": first[0] if first else None,
            "tempo_deteccao_desde_30_s": since_stimulus if detection else None,
            "tempo_deteccao_total_s": total if detection else None,
            "tempo_na_intensidade_s": total-protocol["fases"][level_index]["inicio_global"] if detection else None})
    return details


def load_protocols(return_coefficients=False):
    esp, stimulated = crh._carregar_coeficientes(DATA_DIR, "rayleigh")
    phase._adicionar_laterais_esp(esp, "rayleigh")
    manifest = []
    for item in esp + stimulated:
        path = Path(item["arquivo"])
        with h5py.File(path, "r") as file:
            fs = float(np.asarray(file["Fs"]).ravel()[0])
            if not np.isclose(file["x"].shape[2] / fs, 1):
                raise ValueError("Relógio requer épocas de um segundo")
            item["fs"] = fs
            metadata = {"arquivo": str(path), "fs": fs, "shape": list(file["x"].shape),
                        "freqEstim": np.asarray(file["freqEstim"]).ravel().tolist(),
                        "binsM": np.asarray(file["binsM"]).ravel().tolist()}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        manifest.append({**metadata, "sha256": digest.hexdigest()})
    protocols = protocol_tools.preparar_protocolos(
        esp, stimulated, WINDOW_SIZE_EPOCHS, WINDOW_STEP_EPOCHS)
    controls = {(p["participante"], p["frequencia_indice"]): p["fases"][0]["frequencia"]
                for p in protocols if p["grupo"] == "lateral"}
    for p in protocols:
        p["controle_hz"] = controls[p["participante"], p["frequencia_indice"]]
    if return_coefficients:
        return protocols, manifest, esp, stimulated
    return protocols, manifest


def run(prior, min_db, rule=None):
    rule = rule or detection_rule(MIN_CONSECUTIVE, MIN_PERCENT, MODO_REGRA_DECISAO)
    matrix, pi = np.asarray(MATRIX_A), np.asarray(PI)
    if matrix.shape != (2, 2) or np.any(matrix <= 0) or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("A precisa ser 2×2, positiva e ter linhas somando 1")
    if pi.shape != (2,) or np.any(pi <= 0) or not np.isclose(pi.sum(), 1):
        raise ValueError("pi precisa ser positiva e somar 1")
    protocols, manifest = load_protocols()
    subjects = sorted({p["participante"] for p in protocols})
    folds, details = {}, {m: [] for m in METHODS}
    for subject in subjects:
        values, records = calibration(protocols, subject, min_db)
        mle = {s: crh._fit_beta(stabilize(p)) for s, p in values.items()}
        fitted = fit_map(values["Presente"], prior)
        map_models = {"Ausente": mle["Ausente"], "Presente": fitted}
        test = [p for p in protocols if p["participante"] == subject]
        for method, models in (("bruto", None), ("mle", mle), ("map", map_models)):
            details[method].extend(evaluate(test, models, rule))
        holdout = np.concatenate([f["p_values"] for p in test if p["grupo"] == "estimulo"
                                  for f in p["fases"] if f["nivel"] != "ESP" and f["nivel"] >= min_db])
        holdout = stabilize(holdout)
        folds[subject] = {"treino": [s for s in subjects if s != subject], "teste": subject,
            "calibracao": records, "emissoes_mle": mle, "emissoes_map": map_models,
            "logpdf_media_presente_teste": {m: float(np.mean(beta.logpdf(
                holdout, models["Presente"]["a"], models["Presente"]["b"])))
                for m, models in (("mle", mle), ("map", map_models))}}
        print(f"LOSO {subject}: ESP={len(values['Ausente'])}, >= {min_db} dB={len(values['Presente'])}; "
              f"MAP mu={fitted['mu']:.6f}, kappa={fitted['kappa']:.6f}", flush=True)
    return {"metodo": "Beta única MAP Presente, Beta MLE Ausente; LOSO das emissões",
        "configuracao": {"prior_presente": dict(zip(("mu_alpha", "mu_beta", "kappa_shape", "kappa_rate"), prior)),
            "present_min_db": min_db, "detector": "rayleigh", "canal": crh.gom.CHANNEL_INDEX,
            "window": WINDOW_SIZE_EPOCHS, "step": WINDOW_STEP_EPOCHS,
            "A": MATRIX_A, "pi": PI, **rule, "raw_alpha": RAW_ALPHA,
            "denominador_percentual": "janelas observadas no prefixo desde ESP; continua entre fases",
            "avaliacao_db": list(phase.LEVELS), "p_clip": P_EPS,
            "origem_parametros_temporais": "configuração histórica fixada; sem busca nova"},
        "manifesto_dados": manifest, "protocolos": protocols, "dobras": folds,
        "comparacao": {m: {"nome": METHODS[m], "metricas": protocol_tools.resumir(d), "detalhes": d}
                       for m, d in details.items()},
        "pareamento_map_mle": {"metodo_1": "map", "metodo_2": "mle",
            "campos_herdados": "hmm=MAP; rayleigh=MLE (nomes do utilitário histórico)",
            **protocol_tools.comparar_pareado(details["map"], details["mle"])},
        "limitacoes": ["LOSO separa emissão; A/pi/janela/alpha históricos foram selecionados nestes participantes.",
            "Estímulo >= corte é proxy de resposta, não confirmação fisiológica.",
            "Sobreposição e dependência por pessoa: likelihood iid é aproximada.",
            "MAP depende da parametrização: densidade maximizada em mu,kappa.",
            "Exploração de prioris após ver métricas externas não é validação independente."]}


def plot_fold(subject, fold, prior, directory):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/map-beta-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.asarray([r["p_value"] for r in fold["calibracao"]["Presente"]])
    mle, fitted = fold["emissoes_mle"]["Presente"], fold["emissoes_map"]["Presente"]
    grid = posterior_grid(p, prior, fitted)
    # Grade refinada verifica estabilidade das marginais, além da massa nas bordas.
    refined = posterior_grid(p, prior, fitted, size=2*GRID_SIZE-1)
    errors = [float(np.max(np.abs(grid[key]-refined[key][::2])) / np.max(refined[key]))
              for key in ("mu_pdf", "kappa_pdf")]
    if max(errors) > 0.01:
        raise RuntimeError("Quadratura posterior não convergiu com refinamento")
    u, v, shape, rate = prior
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.linspace(1e-4, 1-1e-4, 1200)
    ax = axes[0, 0]
    ax.plot(x, beta.pdf(x, u, v), label=f"Priori Beta({u:g},{v:g})")
    ax.plot(grid["mu"], grid["mu_pdf"], label="Posterior marginal μ | dados")
    ax.axvline(fitted["mu"], color="black", linestyle="--", label="μ MAP conjunto")
    ax.axvline(mle["a"]/(mle["a"]+mle["b"]), color="tab:orange", linestyle=":", label="μ MLE")
    ax.set(xlabel="μ — média dos p-valores", ylabel="Densidade", title="Influência dos dados sobre μ")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    k = np.linspace(1e-4, max(gamma.ppf(.995, shape, scale=1/rate), grid["kappa"][-1]), 1200)
    ax.plot(k, gamma.pdf(k, shape, scale=1/rate), label=f"Priori Gamma({shape:g},{rate:g}) shape/rate")
    ax.plot(grid["kappa"], grid["kappa_pdf"], label="Posterior marginal κ | dados")
    ax.axvline(fitted["kappa"], color="black", linestyle="--", label="κ MAP conjunto")
    ax.axvline(mle["a"]+mle["b"], color="tab:orange", linestyle=":", label="κ MLE")
    ax.set(xlabel="κ — concentração", ylabel="Densidade", title="Influência dos dados sobre κ")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.hist(p, bins=np.linspace(0, 1, 31), density=True, alpha=.35, label=f"Dados de treino (n={len(p)})")
    ax.plot(x, beta.pdf(x, mle["a"], mle["b"]), label="Beta ajustada só aos dados (MLE)")
    ax.plot(x, beta.pdf(x, fitted["a"], fitted["b"]), linestyle="--", label="Beta única usada no HMM (MAP)")
    ax.set(xlabel="p-valor Rayleigh", ylabel="Densidade", title="Dados e emissão Presente (escala log em y)", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.contourf(grid["mu"], grid["kappa"],
                (grid["joint_logk"]/grid["kappa"][None, :]).T, levels=20, cmap="Blues")
    ax.scatter([fitted["mu"]], [fitted["kappa"]], color="black", marker="x", label="MAP conjunto")
    ax.set(xlabel="μ", ylabel="κ", title="Posterior conjunta em μ,κ (zoom)")
    ax.legend(fontsize=8)
    fig.suptitle(f"LOSO — participante retido: {subject}\n"
                 "Posterior dos parâmetros para visualizar; emissão = uma única Beta no MAP", fontsize=13)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(directory / f"prior_dados_posterior_{subject}.{ext}", dpi=160)
    plt.close(fig)
    with (directory / f"posterior_parametros_{subject}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mu", "prior_mu_pdf", "posterior_mu_pdf", "kappa", "prior_kappa_pdf", "posterior_kappa_pdf"])
        for m, fm, k, fk in zip(grid["mu"], grid["mu_pdf"], grid["kappa"], grid["kappa_pdf"], strict=True):
            writer.writerow([m, beta.pdf(m, u, v), fm, k, gamma.pdf(k, shape, scale=1/rate), fk])
    return {"edge_mass": grid["edge_mass"], "refinement_relative_error": errors}


def write_report(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    prior = tuple(result["configuracao"]["prior_presente"].values())
    rows = []
    for method, item in result["comparacao"].items():
        g = item["metricas"]["global"]
        rows.append({"Método": METHODS[method], "Detecção": f"{g['detectados']}/{g['n_alvos']} ({g['taxa_deteccao']:.2%})",
            "FP lateral": f"{g['fp_lateral']}/{g['n_laterais']} ({g['taxa_fp_lateral']:.2%})",
            "FP alvo ESP": str(g["fp_alvo_esp"]), "Acurácia balanceada": f"{g['acuracia_balanceada_fp_lateral']:.2%}",
            "Tempo médio/mediano (s)": f"{g['tempo_desde_30_medio_detectados_s']:.1f} / {g['tempo_desde_30_mediano_detectados_s']:.1f}"
            if g["detectados"] else "—"})
    with (directory / "comparacao.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = list(rows[0])
    table = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n"
    table += "\n".join("| " + " | ".join(row.values()) + " |" for row in rows)
    with (directory / "parametros_por_dobra.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["teste", "estado", "metodo", "n", "a", "b", "mu", "kappa"])
        for subject, fold in result["dobras"].items():
            for method in ("mle", "map"):
                for state, fitted in fold[f"emissoes_{method}"].items():
                    a, b = fitted["a"], fitted["b"]
                    writer.writerow([subject, state, method, fitted["n"], a, b, a/(a+b), a+b])
    result["quadratura_posterior"] = {s: plot_fold(s, f, prior, directory) for s, f in result["dobras"].items()}
    notes = ("Calibração Presente ≥ " + str(result["configuracao"]["present_min_db"]) + " dB em treino; avaliação ESP e 30–70 dB. "
        f"Ausente MLE compartilhado. Mesma janela/passo {result['configuracao']['window']}/{result['configuracao']['step']}, "
        "A, pi e regra temporal compartilhados; alpha aplicado apenas ao bruto. "
        f"Regra: consecutivas={result['configuracao']['consecutivas']}, "
        f"fração mínima={result['configuracao'].get('min_percent')}, "
        f"modo={result['configuracao'].get('modo_regra_decisao', 'OR')}. "
        "Porcentagem usa somente o prefixo observado desde ESP, sem reiniciar entre intensidades. "
        "Tempo desde 30 dB somente entre detectados; não detectados têm tempo null. "
        "LOSO das emissões; parâmetros temporais históricos não foram escolhidos em dobras independentes. "
        "Dados estimulados são proxy de resposta. Não houve busca nem seleção automática de priori.")
    readme = "# Comparação Beta MAP / MLE em LOSO\n\n" + table + "\n\n" + notes
    readme += "\n\nEdite as quatro constantes MU_PRIOR_ALPHA/BETA e KAPPA_PRIOR_SHAPE/RATE no topo de src/map_beta_hmm_loso.py. "
    readme += "Ou use --mu-prior 2 4 --kappa-prior 2 1 --present-min-db 50 --output-dir outputs/map_beta_hmm_loso. "
    readme += "Critérios: --min-consecutive 4 --min-percent 0.10 --decision-mode AND. Use none para desativar um limiar. "
    readme += "Execute novamente para refazer o ajuste. --plots-only recria as figuras a partir do JSON salvo, sem usar novas prioris. "
    readme += "Não misture posterior dos parâmetros (μ,κ) e densidade dos p-valores. A emissão MAP integra 1 e pode exceder 1.\n"
    (directory / "README.md").write_text(readme, encoding="utf-8")
    html_table = "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"
    html_table += "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in r.values()) + "</tr>" for r in rows) + "</table>"
    figures = "".join(f'<h2>Participante retido: {html.escape(s)}</h2><a href="prior_dados_posterior_{s}.pdf">PDF</a> · '
                      f'<a href="prior_dados_posterior_{s}.svg">SVG</a><img src="prior_dados_posterior_{s}.png">' for s in result["dobras"])
    (directory / "index.html").write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Beta MAP — LOSO</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;color:#172635}table{border-collapse:collapse;width:100%}'
        'td,th{padding:12px;border:1px solid #ccd5df}img{width:100%;margin:20px 0}p{line-height:1.6}</style>'
        '<h1>Detector bruto × HMM MLE × HMM MAP</h1>' + html_table + '<p>'+html.escape(notes)+'</p>'
        '<p>Priori Presente: μ ~ Beta('+str(prior[0])+','+str(prior[1])+'); κ ~ Gamma('+str(prior[2])+','+str(prior[3])+') shape/rate. '
        'Cada figura usa somente dados dos dez participantes de treino. O MAP conjunto pode diferir do modo de cada marginal.</p>'
        + figures + '</html>', encoding="utf-8")
    print(table, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu-prior", type=float, nargs=2, default=[MU_PRIOR_ALPHA, MU_PRIOR_BETA], metavar=("ALPHA", "BETA"))
    parser.add_argument("--kappa-prior", type=float, nargs=2, default=[KAPPA_PRIOR_SHAPE, KAPPA_PRIOR_RATE], metavar=("SHAPE", "RATE"))
    parser.add_argument("--present-min-db", type=int, default=PRESENT_MIN_DB, choices=[30, 40, 50, 60, 70])
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plots-only", action="store_true")
    def optional_int(value):
        return None if value.lower() == "none" else int(value)
    def optional_float(value):
        return None if value.lower() == "none" else float(value)
    parser.add_argument("--min-consecutive", type=optional_int, default=MIN_CONSECUTIVE,
                        help="Mínimo consecutivo; 'none' desativa")
    parser.add_argument("--min-percent", type=optional_float, default=MIN_PERCENT,
                        help="Fração mínima no prefixo causal, ex. 0.10; 'none' desativa")
    parser.add_argument("--decision-mode", choices=["OR", "AND"], default=MODO_REGRA_DECISAO)
    args = parser.parse_args()
    prior = tuple(args.mu_prior + args.kappa_prior)
    if min(prior) <= 0 or not np.all(np.isfinite(prior)):
        parser.error("Hiperparâmetros precisam ser positivos e finitos")
    if args.plots_only:
        result = json.loads(args.output.read_text())
    else:
        try:
            rule = detection_rule(args.min_consecutive, args.min_percent, args.decision_mode)
        except ValueError as error:
            parser.error(str(error))
        result = run(prior, args.present_min_db, rule)
    if args.plots_only:
        print("Recriando figuras com priori/corte do JSON salvo; para novos valores execute sem --plots-only.")
    write_report(result, args.output_dir)
    if not args.plots_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Relatório: {args.output_dir / 'index.html'}; resultados: {args.output}")


if __name__ == "__main__":
    main()
