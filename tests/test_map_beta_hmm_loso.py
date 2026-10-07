"""MAP em coordenadas físicas, exclusão por sujeito e quadratura posterior."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, gamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import map_beta_hmm_loso as model


class MapBetaTests(unittest.TestCase):
    def test_map_matches_direct_density_optimization(self):
        p = np.random.default_rng(4).beta(.4, 1.2, 100)
        prior = (2., 4., 2., 1.)
        fitted = model.fit_map(p, prior)
        def independent(x):
            mu, k = x
            return -(beta.logpdf(p, mu*k, (1-mu)*k).sum()
                     + beta.logpdf(mu, 2, 4) + gamma.logpdf(k, 2, scale=1))
        direct = minimize(independent, [.3, 2], method="Nelder-Mead",
                          bounds=[(.0001, .9999), (.0001, 100)],
                          options={"xatol": 1e-10, "fatol": 1e-10})
        self.assertTrue(direct.success)
        np.testing.assert_allclose([fitted["mu"], fitted["kappa"]], direct.x, rtol=1e-6)
        self.assertAlmostEqual(-fitted["log_posterior"], direct.fun, places=8)
        self.assertAlmostEqual(beta.cdf(1, fitted["a"], fitted["b"]), 1.)

    def test_prior_changes_fit_and_extremes_are_finite(self):
        values = np.array([0., 1., .01, .05, .2, .5])
        weak = model.fit_map(values, (2, 4, 2, 1))
        strong = model.fit_map(values, (80, 20, 2, 1))
        self.assertGreater(strong["mu"], weak["mu"])
        self.assertEqual(weak["n_clipped"], 2)
        for invalid in ([np.nan, .5], [-.1, .5], [.5, 1.1]):
            with self.assertRaises(ValueError): model.fit_map(np.array(invalid), (2, 4, 2, 1))

    def test_loso_excludes_every_phase_and_laterals(self):
        protocols = []
        for subject in ("A", "B"):
            for group in ("estimulo", "lateral"):
                phases = [{"nivel": level, "p_values": [.1, .2], "arquivo": "sample.mat",
                    "canal": 0, "fs": 1000, "frequencia": 81, "valores_ord": [.01, .02],
                    "intervalos_locais": [[0, 10], [5, 15]]} for level in ("ESP", 30, 40, 50, 60, 70)]
                protocols.append({"participante": subject, "grupo": group,
                                  "controle_hz": 82, "fases": phases})
        values, records = model.calibration(protocols, "A", 50)
        self.assertEqual(len(values["Ausente"]), 2)
        self.assertEqual(len(values["Presente"]), 6)
        self.assertEqual({r["participante"] for rows in records.values() for r in rows}, {"B"})
        self.assertEqual({r["nivel_db"] for r in records["Presente"]}, {50, 60, 70})

    def test_posterior_quadrature_integrates_one(self):
        p = np.random.default_rng(5).beta(.4, 1.2, 100)
        prior = (2, 4, 2, 1)
        grid = model.posterior_grid(p, prior, model.fit_map(p, prior))
        self.assertAlmostEqual(np.trapezoid(grid["mu_pdf"], grid["mu"]), 1., places=10)
        self.assertAlmostEqual(np.trapezoid(grid["kappa_pdf"], grid["kappa"]), 1., places=4)
        self.assertLess(grid["edge_mass"], 1e-6)

    def test_detection_and_or_disabled_and_empty(self):
        binary = [False, True, False, True, True]
        ends, phases = [10, 20, 30, 40, 50], [0]*5
        self.assertEqual(model.first_detection(binary, ends, phases,
            model.detection_rule(2, .5, "OR")), (20, 0))
        self.assertEqual(model.first_detection(binary, ends, phases,
            model.detection_rule(2, .5, "AND")), (50, 0))
        self.assertEqual(model.first_detection(binary, ends, phases,
            model.detection_rule(None, .5, "AND")), (20, 0))
        self.assertEqual(model.first_detection(binary, ends, phases,
            model.detection_rule(2, None, "OR")), (50, 0))
        self.assertIsNone(model.first_detection([], [], [], model.detection_rule(1, None, "OR")))
        for args in ((None, None, "OR"), (0, .5, "OR"), (1, 0, "OR"),
                     (1, 1.1, "AND"), (1, float('nan'), "OR"), (1, .1, "bad")):
            with self.assertRaises(ValueError): model.detection_rule(*args)

    def test_percentage_is_prefix_causal_and_continues_between_phases(self):
        rule = model.detection_rule(None, .6, "OR")
        binary, ends, phases = [True, False, True, False], [10, 20, 30, 40], [0, 0, 1, 1]
        self.assertEqual(model.first_detection(binary, ends, phases, rule, phase_min=1), (30, 1))
        # No reset at 30 dB: one positive after four negative ESP windows is 1/5.
        self.assertIsNone(model.first_detection([False]*4+[True], list(range(5)), [0]*4+[1], rule, phase_min=1))
        # Future negatives cannot postpone a decision already available.
        self.assertEqual(model.first_detection(binary[:3], ends[:3], phases[:3], rule, phase_min=1), (30, 1))

    def test_default_evaluator_reproduces_saved_events(self):
        import json
        path = Path(__file__).resolve().parents[1] / "results/map_beta_hmm_loso.json"
        saved = json.loads(path.read_text())
        old_rule = model.detection_rule(1, None, "OR")
        for subject, fold in saved["dobras"].items():
            protocols = [p for p in saved["protocolos"] if p["participante"] == subject]
            for method, models in (("bruto", None), ("mle", fold["emissoes_mle"]), ("map", fold["emissoes_map"])):
                expected = model.protocol_tools.avaliar(protocols, [
                    np.concatenate([f["p_values"] for f in p["fases"]]) <= model.RAW_ALPHA
                    if models is None else model.phase._estados_hmm(
                        np.concatenate([f["p_values"] for f in p["fases"]]), model.MATRIX_A, model.PI, models)
                    for p in protocols], 1)
                self.assertEqual(model.evaluate(protocols, models, old_rule), expected)


if __name__ == "__main__":
    unittest.main()
