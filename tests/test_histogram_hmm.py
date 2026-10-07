"""Verificações numéricas da emissão e do caminho de máxima verossimilhança."""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import histogram_hmm as hh


class HistogramTests(unittest.TestCase):
    def setUp(self):
        records = [
            {"estado_calibracao": state, "valor_msc": value}
            for state, values in (("Ausente", [0, .1, .2]),
                                  ("Presente", [.7, .9, 1]))
            for value in values
        ]
        self.model = hh.ajustar_histogramas(records, 4)

    def test_mass_and_density_and_unseen_values(self):
        widths = np.diff(self.model["edges"])
        for state in hh.ESTADOS:
            row = self.model["estados"][state]
            self.assertEqual(sum(row["contagens"]), 3)
            self.assertAlmostEqual(sum(row["probabilidades"]), 1)
            self.assertAlmostEqual(np.dot(row["densidades"], widths), 1)
        result = hh.log_emissoes([0, .031, .25, 1], self.model)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_equal(result[0], result[1])
        with self.assertRaises(ValueError):
            hh.log_emissoes([1.1], self.model)

    def test_viterbi_matches_exhaustive_search_and_bin_masses(self):
        observations = [.05, .8, .95, .2]
        a = np.array([[.8, .2], [.3, .7]])
        pi = np.array([.7, .3])
        emissions = hh.log_emissoes(observations, self.model)
        def score(path, logs):
            return (np.log(pi[path[0]]) + logs[0, path[0]]
                    + sum(np.log(a[path[t-1], path[t]]) + logs[t, path[t]]
                          for t in range(1, len(path))))
        paths = list(itertools.product(range(2), repeat=4))
        expected = max(paths, key=lambda p: score(p, emissions))
        result = hh.viterbi(observations, a, pi, self.model)
        self.assertEqual(result, [hh.ESTADOS[i] for i in expected])
        masses = emissions + np.log(.25)
        self.assertEqual(expected, max(paths, key=lambda p: score(p, masses)))
        self.assertEqual(hh.viterbi([], a, pi, self.model), [])


if __name__ == "__main__":
    unittest.main()
