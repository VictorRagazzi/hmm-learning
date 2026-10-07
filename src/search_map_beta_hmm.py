"""Busca em grade de HMM Beta MLE/MAP: in-sample ou treinos LOSO.

Entrada compatível: search_hmm_parameters.py --continuous-map [opções].
Prioris/corte fixos por execução; não usa senoide nem discretização.
"""

import argparse
import csv
import hashlib
import html
import json
import os
import tempfile
from itertools import product
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta

import map_beta_hmm_loso as model

# Grades equivalentes ao search discreto para A, pi e regras, mais janelamento.
WINDOW_SIZE_VALUES = (10, 20, 30, 60, 90, 120, 180)
STEP_MODES = ("half_overlap", "no_overlap")
MIN_CONSECUTIVE_VALUES = tuple(range(1, 13))
MIN_PERCENT_VALUES = (.01, .025, .05, .075, .09, .10, .20, .30, .40, .50,
                      .60, .70, .80, .90, 1.)
DECISION_MODES = ("OR", "AND")
PERSISTENCE_VALUES = (.99, .95, .90, .85, .80, .75, .70, .65, .60, .55, .50)
PI_ABSENT_VALUES = PERSISTENCE_VALUES
MAX_TRAIN_LATERAL_FP = .05
CANDIDATE_BATCH_SIZE = 64
OUTPUT = Path("results/search_map_beta_hmm.json")
OUTPUT_DIR = Path("outputs/search_map_beta_hmm")
IN_SAMPLE_OUTPUT = Path("results/search_map_beta_hmm_in_sample.json")
IN_SAMPLE_OUTPUT_DIR = Path("outputs/search_map_beta_hmm_in_sample")


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        name = stream.name
    os.replace(name, path)


def build_grid(quick=False, windows=None, steps=None):
    sizes = tuple(windows or ((60, 120) if quick else WINDOW_SIZE_VALUES))
    modes = tuple(steps or STEP_MODES)
    persist = (.85, .99) if quick else PERSISTENCE_VALUES
    pi_values = (.5, .95) if quick else PI_ABSENT_VALUES
    matrices = [[[a, 1-a], [1-b, b]] for a, b in product(persist, persist)]
    heuristic_path = Path("results/transition_matrix.json")
    heuristic = json.loads(heuristic_path.read_text())["matriz"]
    # Somente candidatas numéricas; não interpretar como fisiologia estimada.
    for matrix in (model.MATRIX_A, heuristic):
        if not any(np.allclose(matrix, other) for other in matrices):
            matrices.append(matrix)
    candidates = [{"A": a, "pi": [p, 1-p]} for a, p in product(matrices, pi_values)]
    consecutive = (1, 4) if quick else MIN_CONSECUTIVE_VALUES
    percentages = (.1, .5) if quick else MIN_PERCENT_VALUES
    rules = [model.detection_rule(c, p, mode)
             for c, p, mode in product(consecutive, percentages, DECISION_MODES)]
    rules += [model.detection_rule(c, None, "OR") for c in consecutive]
    rules += [model.detection_rule(None, p, "OR") for p in percentages]
    configurations = [(size, max(1, size//2) if mode == "half_overlap" else size)
                      for size, mode in product(sizes, modes)]
    return {"windows_steps": [list(c) for c in configurations], "candidates": candidates,
            "rules": rules, "max_train_lateral_fp": MAX_TRAIN_LATERAL_FP,
            "methods": ["bruto", "mle", "map"], "quick": quick}


def prepare_batch(protocols):
    arrays = [np.concatenate([f["p_values"] for f in p["fases"]]) for p in protocols]
    phases = [np.concatenate([np.full(len(f["p_values"]), i) for i, f in enumerate(p["fases"])])
              for p in protocols]
    length = max(map(len, arrays), default=0)
    p = np.full((len(protocols), length), .5)
    valid, scope = np.zeros_like(p, dtype=bool), np.zeros_like(p, dtype=bool)
    target = np.array([p["grupo"] == "estimulo" for p in protocols])
    for i, (a, f) in enumerate(zip(arrays, phases, strict=True)):
        p[i, :len(a)] = a
        valid[i, :len(a)] = True
        scope[i, :len(a)] = f >= 1 if target[i] else True
    return p, valid, scope, target


def batch_states(emissions, candidates):
    """Todos os estados terminais causais, sem backtracking nem mutação global."""
    count, subjects, length = len(candidates), emissions.shape[0], emissions.shape[1]
    states = np.zeros((count, subjects, length), dtype=bool)
    if length == 0:
        return states
    log_a = np.log(np.asarray([c["A"] for c in candidates]) + 1e-300)
    delta = np.log(np.asarray([c["pi"] for c in candidates])[:, None, :] + 1e-300) + emissions[None, :, 0]
    states[:, :, 0] = delta[:, :, 1] > delta[:, :, 0]
    for t in range(1, length):
        d0 = np.maximum(delta[:, :, 0]+log_a[:, None, 0, 0], delta[:, :, 1]+log_a[:, None, 1, 0])
        d1 = np.maximum(delta[:, :, 0]+log_a[:, None, 0, 1], delta[:, :, 1]+log_a[:, None, 1, 1])
        delta[:, :, 0] = d0 + emissions[None, :, t, 0]
        delta[:, :, 1] = d1 + emissions[None, :, t, 1]
        states[:, :, t] = delta[:, :, 1] > delta[:, :, 0]
    return states


def rule_scores(states, valid, scope, target, rules):
    """Avalia regras sem recalcular Viterbi. AND exige coincidência no prefixo."""
    states = states & valid[None]
    count, sequences, length = states.shape
    runs = np.zeros(states.shape, dtype=np.int16)
    if length:
        runs[:, :, 0] = states[:, :, 0]
        for t in range(1, length):
            runs[:, :, t] = (runs[:, :, t-1]+1)*states[:, :, t]
        fractions = states.cumsum(axis=2) / np.arange(1, length+1)
        max_run = np.max(np.where(scope[None], runs, 0), axis=2)
        max_fraction = np.max(np.where(scope[None], fractions, -1), axis=2)
    else:
        fractions = np.empty(states.shape)
        max_run = np.zeros((count, sequences))
        max_fraction = np.full((count, sequences), -1.)
    conditional_fraction = {}
    scores = np.empty((count, len(rules), 2), dtype=int)
    for j, rule in enumerate(rules):
        c, p, mode = rule["consecutivas"], rule["min_percent"], rule["modo_regra_decisao"]
        if c is None:
            detected = max_fraction >= p
        elif p is None:
            detected = max_run >= c
        elif mode == "OR":
            detected = (max_run >= c) | (max_fraction >= p)
        else:
            if c not in conditional_fraction:
                conditional_fraction[c] = np.max(np.where(scope[None] & (runs >= c), fractions, -1), axis=2) if length else max_fraction
            detected = conditional_fraction[c] >= p
        scores[:, j, 0] = detected[:, target].sum(axis=1)
        scores[:, j, 1] = detected[:, ~target].sum(axis=1)
    return scores


def select_best(protocols, fitted, method, grid):
    p, valid, scope, target = prepare_batch(protocols)
    n_targets, n_controls = int(target.sum()), int((~target).sum())
    if method == "bruto":
        emissions = None
        candidates = [{"A": None, "pi": None}]
    else:
        clipped = np.clip(p, model.P_EPS, 1-model.P_EPS)
        emissions = np.stack([beta.logpdf(clipped, fitted[s]["a"], fitted[s]["b"])
                              for s in ("Ausente", "Presente")], axis=2)
        candidates = grid["candidates"]
    best, excluded = None, 0
    for start in range(0, len(candidates), CANDIDATE_BATCH_SIZE):
        batch = candidates[start:start+CANDIDATE_BATCH_SIZE]
        states = (p[None] <= model.RAW_ALPHA) if method == "bruto" else batch_states(emissions, batch)
        scores = rule_scores(states, valid, scope, target, grid["rules"])
        det, fp = scores[:, :, 0], scores[:, :, 1]
        eligible = fp/n_controls <= grid["max_train_lateral_fp"]
        excluded += int((~eligible).sum())
        # Mesmo objetivo do search antigo. Desempata por detecção, FP, ordem da grade.
        balance = (det/n_targets + 1-fp/n_controls)/2
        ordering = np.lexsort((-fp.ravel(), det.ravel(), np.where(eligible, balance, -np.inf).ravel()))
        index = int(ordering[-1])
        i, j = np.unravel_index(index, det.shape)
        if not eligible[i, j]:
            continue
        key = (float(balance[i, j]), int(det[i, j]), -int(fp[i, j]))
        if best is None or key > tuple(best["score"]):
            best = {"score": list(key), **batch[i], "rule": grid["rules"][j],
                    "train_metrics": {"deteccao": int(det[i, j]), "n_alvos": n_targets,
                        "fp_lateral": int(fp[i, j]), "n_laterais": n_controls,
                        "taxa_deteccao": float(det[i, j]/n_targets),
                        "taxa_fp_lateral": float(fp[i, j]/n_controls),
                        "acuracia_balanceada": float(balance[i, j])}}
    return best, {"evaluated": len(candidates)*len(grid["rules"]), "excluded_fp": excluded}


def evaluate_selected(protocols, selected, models):
    return model.evaluate(protocols, models, selected["rule"], selected["A"], selected["pi"])


def load_windows(grid):
    _, manifest, esp, stimulated = model.load_protocols(return_coefficients=True)
    windows = {}
    for size, step in grid["windows_steps"]:
        protocols = model.protocol_tools.preparar_protocolos(esp, stimulated, size, step)
        controls = {(p["participante"], p["frequencia_indice"]): p["fases"][0]["frequencia"]
                    for p in protocols if p["grupo"] == "lateral"}
        for protocol in protocols:
            protocol["controle_hz"] = controls[protocol["participante"], protocol["frequencia_indice"]]
        windows[f"{size}/{step}"] = protocols
    return windows, manifest


def run(grid, prior, min_db, path, resume=False, validation="in-sample"):
    if validation not in ("in-sample", "loso"):
        raise ValueError("Validação deve ser in-sample ou loso")
    windows, manifest = load_windows(grid)
    specification = {"grid": grid, "prior": list(prior), "present_min_db": min_db,
                     "validation": validation,
                     "raw_alpha": model.RAW_ALPHA, "manifest": manifest,
                     "source_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                         for f in ("src/search_map_beta_hmm.py", "src/map_beta_hmm_loso.py",
                                   "src/detectors.py", "src/continuous_rayleigh_hmm.py",
                                   "src/compare_histogram_phase_rayleigh.py", "src/phase_transition_hmm.py")}}
    result = {"status": "running", "specification": specification, "folds": {}, "deployment_fit": None}
    if resume and path.exists():
        result = json.loads(path.read_text())
        if result["specification"] != specification:
            raise ValueError("Checkpoint difere em grade, dados, prioris ou código; use outra saída")
    subjects = sorted({p["participante"] for p in next(iter(windows.values()))})
    for heldout in (subjects + [None] if validation == "loso" else [None]):
        name = heldout or ("todos_in_sample" if validation == "in-sample" else "todos_deployment")
        if (heldout and heldout in result["folds"]) or (heldout is None and result["deployment_fit"] is not None):
            print(f"Checkpoint: {name} já concluído", flush=True)
            continue
        best, totals = {m: None for m in grid["methods"]}, {m: {"evaluated": 0, "excluded_fp": 0} for m in grid["methods"]}
        started = time.monotonic()
        for label, protocols in windows.items():
            train = [p for p in protocols if p["participante"] != heldout]
            values, _ = model.calibration(protocols, heldout, min_db)
            mle = {s: model.crh._fit_beta(model.stabilize(p)) for s, p in values.items()}
            map_fit = {"Ausente": mle["Ausente"], "Presente": model.fit_map(values["Presente"], prior)}
            for method in grid["methods"]:
                fitted = None if method == "bruto" else mle if method == "mle" else map_fit
                selected, counts = select_best(train, fitted, method, grid)
                for k, v in counts.items():
                    totals[method][k] += v
                if selected is not None and (best[method] is None or tuple(selected["score"]) > tuple(best[method]["score"])):
                    best[method] = {**selected, "window": int(label.split('/')[0]), "step": int(label.split('/')[1]),
                                    "models": fitted, "window_label": label}
            print(f"{name}: janela/passo {label}; {time.monotonic()-started:.1f}s", flush=True)
        fold = {"train_subjects": [s for s in subjects if s != heldout], "test_subject": heldout,
                "selected": best, "counts": totals, "test": {}, "in_sample_evaluation": {}}
        for method, selected in best.items():
            if selected is None:
                # Dobras sem candidato não são descartadas nem ajustadas pelo teste.
                raise RuntimeError(f"{name}/{method}: nenhum candidato sob teto FP; ampliar grade explicitamente")
            protocols = windows[selected["window_label"]]
            train = [p for p in protocols if p["participante"] != heldout]
            training_events = evaluate_selected(train, selected, selected["models"])
            summary = model.protocol_tools.resumir(training_events)["global"]
            assert summary["detectados"] == selected["train_metrics"]["deteccao"]
            assert summary["fp_lateral"] == selected["train_metrics"]["fp_lateral"]
            selected["training_full_metrics"] = summary
            if heldout is None:
                fold["in_sample_evaluation"][method] = {
                    "events": training_events, "metrics": model.protocol_tools.resumir(training_events)}
            if heldout is not None:
                test = [p for p in protocols if p["participante"] == heldout]
                events = evaluate_selected(test, selected, selected["models"])
                fold["test"][method] = {"events": events, "metrics": model.protocol_tools.resumir(events)}
        if heldout is None:
            result["deployment_fit"] = fold
        else:
            result["folds"][heldout] = fold
        atomic_save(result, path)
    result["comparison"] = {}
    for method in grid["methods"]:
        events = (result["deployment_fit"]["in_sample_evaluation"][method]["events"]
                  if validation == "in-sample" else
                  [d for fold in result["folds"].values() for d in fold["test"][method]["events"]])
        result["comparison"][method] = {"metrics": model.protocol_tools.resumir(events), "events": events}
    result["status"] = "complete"
    result["limitations"] = ["Busca exaustiva na grade definida, não ótimo contínuo global nem Baum–Welch.",
        ("Emissões, hiperparâmetros e avaliação nos mesmos onze sujeitos: vazamento deliberado in-sample."
         if validation == "in-sample" else
         "Emissões e hiperparâmetros selecionados nos dez sujeitos de treino; teste externo intocado."),
        "Teto FP aplicado à amostra de ajuste, não garante 5% em novos pacientes.",
        "Prioris/corte e desenho da grade já foram discutidos com dados destes pacientes; não validação clínica.",
        "Deployment é reajuste em todos os sujeitos; suas métricas de treino não são externas.",
        "Sobreposição/dependência de janelas e proxy Presente real preservados."]
    atomic_save(result, path)
    return result


def report(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    in_sample = result["specification"].get("validation", "loso") == "in-sample"
    description = ("Ajuste de B, seleção de parâmetros e avaliação nos mesmos onze pacientes. Resultado exploratório in-sample."
                   if in_sample else
                   "A tabela usa somente predições externas. Cada método seleciona sua configuração no treino de cada dobra.")
    title = "Search Beta contínua MAP/MLE — " + ("in-sample" if in_sample else "LOSO")
    rows = []
    for method, item in result["comparison"].items():
        g = item["metrics"]["global"]
        rows.append({"Método": model.METHODS[method], "Detecção": f"{g['detectados']}/{g['n_alvos']} ({g['taxa_deteccao']:.2%})",
                     "FP lateral": f"{g['fp_lateral']}/{g['n_laterais']} ({g['taxa_fp_lateral']:.2%})",
                     "FP alvo ESP": str(g['fp_alvo_esp']), "BA lateral": f"{g['acuracia_balanceada_fp_lateral']:.2%}",
                     "Tempo médio (s)": f"{g['tempo_desde_30_medio_detectados_s']:.1f}" if g['detectados'] else "—"})
    with (directory/'comparacao.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = '| '+' | '.join(rows[0])+' |\n|'+'|'.join(['---']*len(rows[0]))+'|\n'
    table += '\n'.join('| '+' | '.join(r.values())+' |' for r in rows)
    lines = ['# '+title, '', table, '', description,
        'Critérios percentuais são causais desde ESP. Teto lateral na amostra de ajuste: 5%.',
        '', '## Configuração global em todos os pacientes (métricas de ajuste)', '']
    for method, selected in result['deployment_fit']['selected'].items():
        lines += [f"### {model.METHODS[method]}", '', '```json', json.dumps(selected, indent=2, ensure_ascii=False), '```', '']
    lines += ['## Limites', ''] + result['limitations']
    (directory/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    body = '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in rows[0])+'</tr>'
    body += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r.values())+'</tr>' for r in rows)+'</table>'
    body += '<p>'+html.escape(description)+'</p>'
    body += '<h2>Configurações finais para uso (ajuste em todos; métricas de treino)</h2><pre>'+html.escape(json.dumps(result['deployment_fit']['selected'], indent=2, ensure_ascii=False))+'</pre>'
    if not in_sample:
        body += '<h2>Configurações por dobra</h2><pre>'+html.escape(json.dumps({s: f['selected'] for s,f in result['folds'].items()}, indent=2, ensure_ascii=False))+'</pre>'
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
        '<style>body{font:16px system-ui;margin:40px auto;max-width:1200px}td,th{border:1px solid #aaa;padding:12px}'
        'table{border-collapse:collapse}pre{background:#eff3f8;padding:20px;overflow:auto}</style><h1>'+html.escape(title)+'</h1>'+body, encoding='utf-8')
    print(table, flush=True)
    for m, selected in result['deployment_fit']['selected'].items():
        print(m, 'deployment:', selected['window_label'], selected['rule'], 'A=', selected['A'], 'pi=', selected['pi'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', choices=['in-sample', 'loso'], default='in-sample',
                        help='Padrão: todos os pacientes para ajuste e avaliação; loso mantém teste por sujeito')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--quick', action='store_true', help='Grade reduzida apenas para diagnóstico; não busca completa')
    parser.add_argument('--windows', type=int, nargs='+', help='Subconjunto de tamanhos de janela')
    parser.add_argument('--step-modes', choices=STEP_MODES, nargs='+')
    parser.add_argument('--mu-prior', type=float, nargs=2, default=[model.MU_PRIOR_ALPHA, model.MU_PRIOR_BETA])
    parser.add_argument('--kappa-prior', type=float, nargs=2, default=[model.KAPPA_PRIOR_SHAPE, model.KAPPA_PRIOR_RATE])
    parser.add_argument('--present-min-db', type=int, choices=[30,40,50,60,70], default=model.PRESENT_MIN_DB)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.windows and (len(set(args.windows)) != len(args.windows) or any(w < 3 for w in args.windows)):
        parser.error('Janelas únicas, cada uma >=3 épocas')
    prior = tuple(args.mu_prior+args.kappa_prior)
    if not np.all(np.isfinite(prior)) or min(prior) <= 0:
        parser.error('Prioris precisam de hiperparâmetros positivos/finito')
    grid = build_grid(args.quick, args.windows, args.step_modes)
    path = args.output or (IN_SAMPLE_OUTPUT if args.validation == 'in-sample' else OUTPUT)
    directory = args.output_dir or (IN_SAMPLE_OUTPUT_DIR if args.validation == 'in-sample' else OUTPUT_DIR)
    result = run(grid, prior, args.present_min_db, path, args.resume, args.validation)
    report(result, directory)


if __name__ == '__main__':
    main()
