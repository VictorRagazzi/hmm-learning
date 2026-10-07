"""Causalidade, relógio de fases e denominador de FP lateral."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare_histogram_phase_rayleigh as cp
import histogram_hmm as hh


class PhaseTests(unittest.TestCase):
    def test_causal_terminal_matches_each_prefix_without_future(self):
        records = [{"estado_calibracao": s, "valor_msc": x}
                   for s, xs in (("Ausente", [.01, .02]), ("Presente", [.8, .9]))
                   for x in xs]
        model = hh.ajustar_histogramas(records, 4)
        a, pi = [[.95, .05], [.15, .85]], [.8, .2]
        observations = [.02, .1, .8, .9, .05]
        online = cp.estados_causais(observations, a, pi, model)
        for end in range(1, len(observations) + 1):
            self.assertEqual(online[end-1],
                hh.viterbi(observations[:end], a, pi, model)[-1] == "Presente")
        np.testing.assert_equal(online[:3], cp.estados_causais(observations[:3], a, pi, model))

    def test_fp_esp_lateral_and_censored_time(self):
        def protocol(group):
            return {"participante": "Teste", "grupo": group,
                "frequencia_indice": 0, "n_epocas_total": 12,
                "fases": [{"nivel": "ESP", "frequencia": 81 if group == "estimulo" else 82,
                           "inicio_global": 0, "n_epocas": 2, "p_values": [.5, .5],
                           "finais_locais": [1, 2]},
                          {"nivel": 30, "frequencia": 81 if group == "estimulo" else 82,
                           "inicio_global": 2, "n_epocas": 10, "p_values": [.5, .5],
                           "finais_locais": [5, 10]}]}
        protocols = [protocol("estimulo"), protocol("lateral")]
        # Alvo não detectado; lateral dispara somente em ESP.
        details = cp.avaliar(protocols, [np.zeros(4, bool), np.array([1, 1, 0, 0], bool)], 2)
        result = cp.resumir(details)["global"]
        self.assertEqual(result["taxa_fp_lateral"], 1)
        self.assertEqual(result["fp_lateral_apos_30"], 0)
        self.assertIsNone(result["tempo_desde_30_medio_detectados_s"])
        self.assertEqual(result["duracao_media_desde_30_inclui_nao_detectados_s"], 10)
        # Consecutividade atravessa a fronteira; o relógio soma duração ESP.
        details = cp.avaliar(protocols, [np.array([0, 1, 1, 0], bool), np.zeros(4, bool)], 2)
        self.assertEqual(details[0]["tempo_deteccao_total_s"], 7)
        self.assertEqual(details[0]["tempo_deteccao_desde_30_s"], 5)
        self.assertFalse(details[0]["fp_esp"])


if __name__ == "__main__":
    unittest.main()
