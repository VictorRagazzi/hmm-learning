"""Emissão preditiva Beta experimental; baseline MLE e protocolo fixos.

Execute da raiz. Não executa search, não altera artefatos históricos.
Veja docs/agents/bayesian_beta_hmm.md para hipóteses e reprodução.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assr-matplotlib")
os.environ.setdefault("PYTENSOR_FLAGS", "compiledir=/tmp/assr-pytensor")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/assr-cache")
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.special import logsumexp, betaln
from scipy.integrate import quad
from scipy.stats import beta, gamma
import continuous_rayleigh_hmm as crh
import phase_transition_hmm as phase
from compare_histogram_phase_rayleigh import avaliar, resumir

STATES = ("Ausente", "Presente")
SOURCE = Path("results/phase_transition_hmm.json")
OBS_SOURCE = Path("results/compare_histogram_phase_rayleigh.json")
OUTPUT = Path("results/bayesian_beta_hmm.json")
ASSETS = Path("results/bayesian_beta_samples")
FIGURES = Path("outputs/bayesian_beta_hmm")
SEED = 20260930
CHAINS, DRAWS, TUNE = 4, 2000, 2000
TARGET_ACCEPT = 0.95
P_EPS = 1e-9
PREDICTIVE_DRAWS = 2000
PRIOR_DRAWS = 10000
# (u, v, shape, rate). Predefinidas por argumentos preditivos, não por FP.
PRIORS = {
    "principal": {"Ausente": (8, 8, 2, 1), "Presente": (2, 4, 2, 1)},
    "ampla": {"Ausente": (2, 2, 2, 0.5), "Presente": (1, 2, 2, 0.5)},
    "simetrica": {"Ausente": (2, 2, 2, 1), "Presente": (2, 2, 2, 1)},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stabilize(p):
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p deve ser finito e pertencer a [0,1]")
    return np.clip(p, P_EPS, 1 - P_EPS)


def predictive_logpdf(p, samples, clip=True):
    """Média de densidades, nunca média de logs ou plug-in nas médias."""
    p = stabilize(p) if clip else np.asarray(p, dtype=float)
    a, b = np.asarray(samples["a"]), np.asarray(samples["b"])
    if a.ndim != 1 or a.shape != b.shape or not len(a) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Amostras Beta inválidas")
    result = []
    for chunk in np.array_split(p, max(1, int(np.ceil(len(p) / 128)))):
        result.append(logsumexp(beta.logpdf(chunk[:, None], a, b), axis=1) - np.log(len(a)))
    return np.concatenate(result)


def causal_states(log_emissions, matrix, pi):
    if not len(log_emissions):
        return np.zeros(0, dtype=bool), np.empty((0, 2))
    delta = np.log(pi) + log_emissions[0]
    scores = [delta.copy()]
    states = [int(np.argmax(delta))]
    for emission in log_emissions[1:]:
        delta = np.max(delta[:, None] + np.log(matrix), axis=0) + emission
        scores.append(delta.copy())
        states.append(int(np.argmax(delta)))
    return np.asarray(states, dtype=bool), np.asarray(scores)


def load_fixed():
    config = json.loads(SOURCE.read_text())["por_detector"]["rayleigh"]["por_configuracao"]["window_120_step_60"]
    saved = json.loads(OBS_SOURCE.read_text())["configuracoes"]["continuo_window_120_step_60"]
    h = config["hmm"]
    assert (config["window_size_epochs"], config["window_step_epochs"]) == (120, 60)
    np.testing.assert_allclose(h["matriz_a"], [[.99, .01], [.15, .85]])
    np.testing.assert_allclose(h["pi_inicial"], [.95, .05])
    assert h["min_consecutive"] == 1
    protocols = saved["protocolos"]
    assert len({p["participante"] for p in protocols}) == 11
    assert all(sum(p["grupo"] == group for p in protocols) == 88 for group in ("estimulo", "lateral"))
    data = {s: [] for s in STATES}
    for p in protocols:
        assert [f["nivel"] for f in p["fases"]] == list(phase.LEVELS)
        for f in p["fases"]:
            np.testing.assert_allclose(f["p_values"], np.exp(-120 * np.asarray(f["valores_ord"])), rtol=1e-12)
            assert all(end-start == 120 and end <= f["n_epocas"] for start, end in f["intervalos_locais"])
            if p["grupo"] == "estimulo":
                data["Ausente" if f["nivel"] == "ESP" else "Presente"].extend(f["p_values"])
    assert [len(data[s]) for s in STATES] == [104, 1744]
    return config, saved, protocols, {s: np.asarray(v) for s, v in data.items()}


def audit_data(protocols):
    """Verifica observações salvas contra EEG autêntico sem recalibrar detector."""
    import detectors
    import get_obs_matrix as gom
    records = {}
    for p in protocols:
        for f in p["fases"]:
            records.setdefault(f["arquivo"], []).append((p, f))
    manifest = []
    for path, entries in sorted(records.items()):
        with h5py.File(path, "r") as mat:
            fs = float(np.asarray(mat["Fs"]).ravel()[0])
            targets = np.asarray(mat["freqEstim"]).ravel()
            controls = np.asarray(mat["binsM"]).ravel()
            shape = mat["x"].shape
            channel = entries[0][1]["canal"]
            x = np.asarray(mat["x"][channel:channel+1])
        assert np.isclose(shape[2] / fs, 1)
        for p, f in entries:
            assert f["canal"] == channel and f["fs"] == fs and f["n_epocas"] == shape[1]
            freq = (targets if p["grupo"] == "estimulo" else controls)[p["frequencia_indice"]]
            assert f["frequencia"] == freq
            coeff = gom.extrair_coeficientes_epocas(x, 0, fs, freq)[0]
            values, intervals = detectors.calcular_estatistica_janelas(coeff, detectors.RAYLEIGH, 120, 60)
            np.testing.assert_allclose(values, f["valores_ord"], rtol=1e-11, atol=1e-14)
            assert [list(i) for i in intervals] == f["intervalos_locais"]
        manifest.append({"arquivo": path, "sha256": digest(path), "fs": fs,
                         "shape": list(shape), "canal": channel,
                         "freqEstim": targets.tolist(), "binsM": controls.tolist(),
                         "n_janelas_por_frequencia": len(entries[0][1]["p_values"])})
    return manifest


def build_model(p, prior):
    """Likelihood Beta exata em estatísticas suficientes e prioris explícitas."""
    import pymc as pm
    import pytensor.tensor as pt
    u, v, shape, rate = prior
    with pm.Model() as model:
        mu = pm.Beta("mu", alpha=u, beta=v)
        kappa = pm.Gamma("kappa", alpha=shape, beta=rate)
        a = pm.Deterministic("a", mu * kappa)
        b = pm.Deterministic("b", (1 - mu) * kappa)
        observed = stabilize(p)
        pm.Potential("beta_loglik", (a-1)*float(np.log(observed).sum())
                     + (b-1)*float(np.log1p(-observed).sum())
                     - len(observed)*(pt.gammaln(a)+pt.gammaln(b)-pt.gammaln(a+b)))
    return model


def sample_posterior(p, prior, seed, path, reuse=False):
    import pymc as pm
    import arviz as az
    specification = json.dumps({"prior": list(prior), "seed": seed, "chains": CHAINS,
                                "draws": DRAWS, "tune": TUNE, "target_accept": TARGET_ACCEPT,
                                "data_sha256": hashlib.sha256(stabilize(p).tobytes()).hexdigest()}, sort_keys=True)
    if reuse and path.exists():
        idata = az.from_netcdf(path)
        if idata.attrs.get("specification") != specification:
            raise ValueError(f"Amostras incompatíveis com dados/configuração: {path}")
        return idata
    with build_model(p, prior):
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                          random_seed=seed, target_accept=TARGET_ACCEPT,
                          progressbar=False, compute_convergence_checks=True)
    idata.attrs["specification"] = specification
    idata.to_netcdf(path)
    return idata


def posterior_summary(idata):
    import arviz as az
    table = az.summary(idata, var_names=["mu", "kappa", "a", "b"], hdi_prob=.95, round_to=8)
    summary = table.to_dict(orient="index")
    for name in summary:
        values = idata.posterior[name].values.ravel()
        summary[name]["equal_tail_95"] = np.quantile(values, [.025, .975]).tolist()
    diagnostics = {"divergences": int(idata.sample_stats.diverging.sum()),
                   "max_rhat": float(table.r_hat.max()),
                   "min_ess_bulk": float(table.ess_bulk.min()),
                   "min_ess_tail": float(table.ess_tail.min()),
                   "bfmi_per_chain": az.bfmi(idata).tolist()}
    diagnostics["passed"] = (diagnostics["divergences"] == 0 and diagnostics["max_rhat"] < 1.01
                              and min(diagnostics["min_ess_bulk"], diagnostics["min_ess_tail"]) > 400
                              and min(diagnostics["bfmi_per_chain"]) > .3)
    return summary, diagnostics


def fixed_samples(idata, seed):
    rng = np.random.default_rng(seed)
    count = idata.posterior.mu.size
    idx = np.sort(rng.choice(count, min(PREDICTIVE_DRAWS, count), replace=False))
    return {**{s: idata.posterior[s].values.ravel()[idx] for s in ("mu", "kappa", "a", "b")},
            "flat_indices": idx}


def evaluate(protocols, config, samples=None):
    binaries, traces = [], []
    h = config["hmm"]
    for protocol in protocols:
        p = np.concatenate([f["p_values"] for f in protocol["fases"]])
        ends = np.concatenate([f["inicio_global"] + np.asarray(f["finais_locais"], dtype=int) for f in protocol["fases"]])
        phases = np.concatenate([np.full(len(f["p_values"]), i, dtype=int) for i, f in enumerate(protocol["fases"])])
        if samples is None:
            logs = np.column_stack([beta.logpdf(stabilize(p), config["modelos_emissao_beta"][s]["a"],
                                               config["modelos_emissao_beta"][s]["b"]) for s in STATES])
        else:
            logs = np.column_stack([predictive_logpdf(p, samples[s]) for s in STATES])
        states, delta = causal_states(logs, h["matriz_a"], h["pi_inicial"])
        for end in range(1, len(p) + 1):
            np.testing.assert_array_equal(causal_states(logs[:end], h["matriz_a"], h["pi_inicial"])[0], states[:end])
        if samples is None:
            np.testing.assert_array_equal(states, phase._estados_hmm(p, h["matriz_a"], h["pi_inicial"], config["modelos_emissao_beta"]))
        binaries.append(states)
        traces.append({"participante": protocol["participante"], "grupo": protocol["grupo"],
                       "frequencia_indice": protocol["frequencia_indice"], "finais": ends.tolist(),
                       "fase_indices": phases.tolist(), "p_values": p.tolist(),
                       "log_emissoes": logs.tolist(), "delta": delta.tolist(), "estados": states.tolist()})
    details = avaliar(protocols, binaries, h["min_consecutive"])
    summary = resumir(details)
    summary["global"]["taxa_fp_alvo_esp"] = summary["global"]["fp_alvo_esp"] / 88
    for key in ("por_paciente", "por_frequencia"):
        for item in summary[key].values():
            item["taxa_fp_alvo_esp"] = item["fp_alvo_esp"] / item["n_alvos"]
    return {"metricas": summary, "detalhes": details, "trajetorias": traces}


def paired(baseline, alternative):
    pairs = []
    for a, b in zip(baseline["detalhes"], alternative["detalhes"], strict=True):
        if a["grupo"] != "estimulo":
            continue
        pairs.append({"participante": a["participante"], "frequencia": a["frequencia"],
                      "mle_detectou": a["detectou"], "bayes_detectou": b["detectou"],
                      "mle_tempo_s": a["tempo_deteccao_desde_30_s"],
                      "bayes_tempo_s": b["tempo_deteccao_desde_30_s"],
                      "delta_tempo_s": b["tempo_deteccao_desde_30_s"] - a["tempo_deteccao_desde_30_s"]
                      if a["detectou"] and b["detectou"] else None})
    changes = [p["delta_tempo_s"] for p in pairs if p["delta_tempo_s"] is not None]
    return {"ambos": len(changes), "bayes_antes": sum(t < 0 for t in changes),
            "mle_antes": sum(t > 0 for t in changes), "empates": sum(t == 0 for t in changes),
            "somente_mle": sum(p["mle_detectou"] and not p["bayes_detectou"] for p in pairs),
            "somente_bayes": sum(p["bayes_detectou"] and not p["mle_detectou"] for p in pairs),
            "diferenca_media_s_pareada": float(np.mean(changes)) if changes else None, "pares": pairs}


def compare_emissions(baseline, alternative, models, samples):
    differences = []
    changed = {g: {"janelas": 0, "trajetorias": 0, "total_janelas": 0} for g in ("estimulo", "lateral")}
    for a, b in zip(baseline["trajetorias"], alternative["trajetorias"], strict=True):
        diff = np.asarray(b["log_emissoes"]) - np.asarray(a["log_emissoes"])
        differences.extend(diff.tolist())
        states = np.asarray(a["estados"]) != np.asarray(b["estados"])
        changed[a["grupo"]]["janelas"] += int(states.sum())
        changed[a["grupo"]]["trajetorias"] += int(states.any())
        changed[a["grupo"]]["total_janelas"] += len(states)
    differences = np.asarray(differences)
    grid = np.array([1e-9, 1e-5, .001, .01, .05, .5, .95, 1-1e-9])
    return {"mudancas_estado_causal": changed,
            "emissoes_por_estado": {s: {
                "delta_logpdf_observado_quantiles": np.quantile(differences[:, i], [.025, .5, .975]).tolist(),
                "p_referencia": grid.tolist(), "logpdf_mle": beta.logpdf(grid, models[s]["a"], models[s]["b"]).tolist(),
                "logpdf_preditiva": predictive_logpdf(grid, samples[s]).tolist()} for i, s in enumerate(STATES)}}


def prior_predictive(prior, seed):
    rng = np.random.default_rng(seed)
    u, v, shape, rate = prior
    mu = rng.beta(u, v, PRIOR_DRAWS)
    kappa = rng.gamma(shape, scale=1 / rate, size=PRIOR_DRAWS)
    a, b = mu*kappa, (1-mu)*kappa
    p = rng.beta(a, b)
    return {"mu": mu, "kappa": kappa, "a": a, "b": b, "p": p}


def numerical_checks(samples):
    # Duas metades com x=exp(-t), jacobiano exp(-t): preserva caudas
    # singulares sem formar x numericamente igual a 0 ou 1.
    a, b = samples["a"], samples["b"]
    def half(a, b):
        def integrand(t):
            logs = -a*t + (b-1)*np.log1p(-np.exp(-t)) - betaln(a, b)
            return float(np.exp(logsumexp(logs)-np.log(len(a))))
        return quad(integrand, np.log(2), np.inf, epsabs=1e-9)[0]
    mass = half(a, b) + half(b, a)
    # CDF da mistura é uma verificação exata de massa e mede clipping.
    mass_cdf = float(np.mean(beta.cdf(1, samples["a"], samples["b"]) - beta.cdf(0, samples["a"], samples["b"])))
    tail = float(np.mean(beta.cdf(P_EPS, samples["a"], samples["b"]) + beta.sf(1-P_EPS, samples["a"], samples["b"])))
    p = np.array([0, 1e-300, P_EPS, .01, .5, 1-P_EPS, 1])
    logs = predictive_logpdf(p, samples)
    direct = np.log(np.mean(beta.pdf(stabilize(p)[:, None], samples["a"], samples["b"]), axis=1))
    np.testing.assert_allclose(logs, direct, atol=1e-11)
    assert np.all(np.isfinite(logs)) and np.isclose(mass_cdf, 1) and np.isclose(mass, 1)
    return {"mass_cdf": mass_cdf, "mass_quadrature": mass,
            "mass_outside_clip_interval": tail, "extreme_logpdf": logs.tolist(),
            "logsumexp_matches_direct": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        from plot_bayesian_beta_hmm import generate
        generate(json.loads(OUTPUT.read_text()))
        return
    config, saved, protocols, data = load_fixed()
    baseline = evaluate(protocols, config)
    g = baseline["metricas"]["global"]
    assert (g["detectados"], g["fp_lateral"]) == (70, 8)
    assert np.isclose(g["tempo_desde_30_medio_detectados_s"], 734, atol=1)
    assert g["tempo_desde_30_mediano_detectados_s"] == 730
    for actual, reference in zip(baseline["detalhes"], saved["detalhes_hmm_antigo"], strict=True):
        assert actual == reference
    for s in STATES:
        fit = crh._fit_beta(data[s])
        np.testing.assert_allclose([fit["a"], fit["b"]], [config["modelos_emissao_beta"][s][k] for k in ("a", "b")], rtol=1e-10)
    print("Baseline reproduzida:", g, flush=True)
    manifest = audit_data(protocols)
    result = {"status": "experimental_in_sample", "busca_executada": False,
              "seed": SEED, "configuracao_historica": config,
              "mcmc": {"chains": CHAINS, "draws": DRAWS, "tune": TUNE, "target_accept": TARGET_ACCEPT,
                       "sampler": "PyMC NUTS", "predictive_draws": PREDICTIVE_DRAWS},
              "versions": {name: importlib.metadata.version(name) for name in ("pymc", "arviz", "numpy", "scipy", "matplotlib", "h5py", "pytensor")},
              "python_version": sys.version,
              "sources": {str(p): digest(p) for p in (SOURCE, OBS_SOURCE,
                          Path("src/continuous_rayleigh_hmm.py"), Path("src/phase_transition_hmm.py"),
                          Path("src/get_obs_matrix.py"), Path("src/detectors.py"),
                          Path("src/bayesian_beta_hmm.py"), Path("src/plot_bayesian_beta_hmm.py"))},
              "manifest_dados": manifest, "protocolos": protocols,
              "calibracao": {s: {"n_janelas": len(data[s]), "n_clipped": int(np.sum(data[s] != stabilize(data[s]))),
                                   "n_arquivos_contribuintes": sum(m["n_janelas_por_frequencia"] > 0 and
                                    (("ESP.mat" in m["arquivo"]) == (s == "Ausente")) for m in manifest)} for s in STATES},
              "baseline_mle": baseline, "prioris": PRIORS, "alternativas": {},
              "limitacoes": ["Exploratório in-sample, Presente real é proxy; não validação clínica.",
                             "Janelas sobrepostas e mesmo participante: likelihood fatorizada pode subestimar incerteza.",
                             "Marginais preditivas por janela não integram parâmetros compartilhados conjuntamente na trajetória; A/pi fixas."]}
    for i, (name, priors) in enumerate(PRIORS.items()):
        samples, posterior, diagnostics, checks, prior_stats = {}, {}, {}, {}, {}
        for j, s in enumerate(STATES):
            seed = SEED + i * 100 + j
            prior = prior_predictive(priors[s], seed)
            np.savez_compressed(ASSETS / f"prior_{name}_{s}.npz", **prior)
            prior_stats[s] = {"p_quantiles": np.quantile(prior["p"], [.025, .5, .975]).tolist(),
                              "Pr_p_below_005": float(np.mean(beta.cdf(.05, prior["a"], prior["b"]))),
                              "Pr_a_b_below_1": float(np.mean((prior["a"] < 1) & (prior["b"] < 1)))}
            idata = sample_posterior(data[s], priors[s], seed, ASSETS / f"posterior_{name}_{s}.nc", args.reuse_samples)
            posterior[s], diagnostics[s] = posterior_summary(idata)
            if not diagnostics[s]["passed"]:
                raise RuntimeError(f"Diagnóstico reprovado {name}/{s}: {diagnostics[s]}")
            samples[s] = fixed_samples(idata, seed+1000)
            np.savez_compressed(ASSETS / f"predictive_{name}_{s}.npz", **samples[s])
            checks[s] = numerical_checks(samples[s])
        comparison = evaluate(protocols, config, samples)
        result["alternativas"][name] = {"posterior": posterior, "diagnostics": diagnostics,
                                       "numerical_checks": checks, "prior_predictive": prior_stats,
                                       **comparison, "pareamento_com_mle": paired(baseline, comparison),
                                       "comparacao_emissoes_estados": compare_emissions(baseline, comparison, config["modelos_emissao_beta"], samples)}
        print(name, comparison["metricas"]["global"], diagnostics, flush=True)
    result["recuperacao_simulada"] = {}
    for i, (mu, kappa) in enumerate(((.5, 2.), (.25, 4.))):
        seed = SEED + 10000 + i
        p = np.random.default_rng(seed).beta(mu*kappa, (1-mu)*kappa, 1500)
        np.savez_compressed(ASSETS / f"simulated_{i}.npz", p=p, mu=mu, kappa=kappa)
        idata = sample_posterior(p, PRIORS["principal"][STATES[i]], seed, ASSETS / f"recovery_{i}.nc", args.reuse_samples)
        summary, diagnostics = posterior_summary(idata)
        truth = {"mu": mu, "kappa": kappa, "a": mu*kappa, "b": (1-mu)*kappa}
        covered = {k: summary[k]["equal_tail_95"][0] < v < summary[k]["equal_tail_95"][1] for k, v in truth.items()}
        z = {k: (summary[k]["mean"]-v)/summary[k]["sd"] for k,v in truth.items()}
        assert diagnostics["passed"] and all(abs(v) < 3 for v in z.values())
        result["recuperacao_simulada"][str(i)] = {"truth": truth, "summary": summary, "diagnostics": diagnostics,
                                                "covered_95": covered, "recovery_standardized_error": z}
    result["sample_files"] = {str(p): digest(p) for p in sorted(ASSETS.glob("*"))}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    from plot_bayesian_beta_hmm import generate
    generate(result)


if __name__ == "__main__":
    main()
