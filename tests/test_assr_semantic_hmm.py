import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import assr_semantic_hmm as assr


class TestAssrSemanticHmm(unittest.TestCase):
    def test_discretizacao_adapta_quantidade_de_niveis(self):
        for quantidade in (1, 3, 4, 6):
            niveis = [f"n{i}" for i in range(quantidade)]
            self.assertEqual(len(assr.limiares_discretizacao(quantidade)), quantidade - 1)
            self.assertEqual(assr.discretizar(0.0, niveis), niveis[0])
            self.assertEqual(assr.discretizar(300.0, niveis), niveis[-1])

    def test_csm_distingue_fase_estavel_de_aleatoria(self):
        fs, epocas = 1000, 20
        tempo = np.arange(fs) / fs
        frequencias = np.asarray(assr.FREQUENCIAS_ESPERADAS_HZ, dtype=float)
        rng = np.random.default_rng(7)
        coerente = np.zeros((fs, epocas, 2))
        aleatorio = np.zeros_like(coerente)
        for epoca in range(epocas):
            for frequencia in frequencias:
                coerente[:, epoca, :] += np.sin(2 * np.pi * frequencia * tempo)[:, None]
                fases = rng.uniform(0, 2 * np.pi, size=2)
                aleatorio[:, epoca, :] += np.sin(2 * np.pi * frequencia * tempo[:, None] + fases)
        forte = assr.calcular_estatistica_banda(coerente, fs, frequencias)
        fraca = assr.calcular_estatistica_banda(aleatorio, fs, frequencias)
        self.assertGreater(forte, fraca)

    def test_inferencia_por_lookup(self):
        tabela = {
            "repouso": dict(ausente="repouso", fraco="repouso", forte="resposta", muito_forte="resposta"),
            "resposta": dict(ausente="repouso", fraco="resposta", forte="resposta", muito_forte="resposta"),
        }
        with patch.object(assr, "_chamar_ollama", side_effect=AssertionError("LLM online")):
            resultado = assr.inferir_sequencia([0.0, 2.5, 3.1], tabela)
        self.assertEqual(resultado["estados"], ["repouso", "resposta", "resposta"])

    def test_compilacao_faz_uma_chamada_de_notas_e_uma_por_estado(self):
        respostas = [
            {"notas": {"repouso": "nota r", "resposta": "nota a"}},
            {"transicoes": dict(ausente="repouso", fraco="repouso", forte="resposta", muito_forte="resposta")},
            {"transicoes": dict(ausente="repouso", fraco="resposta", forte="resposta", muito_forte="resposta")},
        ]
        with patch.object(assr, "_chamar_ollama", side_effect=respostas) as chamada:
            compilacao = assr.compilar_tabela_semantica("modelo-teste")
        self.assertEqual(chamada.call_count, 1 + len(assr.ESTADOS))
        self.assertEqual(compilacao["tabela"]["resposta"]["fraco"], "resposta")


if __name__ == "__main__":
    unittest.main()
