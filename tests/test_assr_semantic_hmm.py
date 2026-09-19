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
        tabela = {}
        for historico in assr.gerar_historicos():
            tabela[assr.chave_historico(historico)] = {
                assr.NIVEIS_OBSERVACAO[0]: assr.ESTADO_INICIAL,
                assr.NIVEIS_OBSERVACAO[1]: historico[-1],
                assr.NIVEIS_OBSERVACAO[2]: assr.ESTADO_RESPOSTA,
                assr.NIVEIS_OBSERVACAO[3]: assr.ESTADO_RESPOSTA,
            }
        with patch.object(assr, "_chamar_ollama", side_effect=AssertionError("LLM online")):
            resultado = assr.inferir_sequencia([0.0, 2.5, 3.1], tabela)
        self.assertEqual(
            resultado["estados"],
            [assr.ESTADO_INICIAL, assr.ESTADO_RESPOSTA, assr.ESTADO_RESPOSTA],
        )
        self.assertAlmostEqual(resultado["fracao_resposta"], 2 / 3)
        self.assertEqual(resultado["maior_sequencia_resposta"], 2)
        self.assertTrue(resultado["detectou"])
        historico_inicial = [assr.ESTADO_INICIAL] * assr.ORDEM_CONTEXTO
        historico_com_resposta = historico_inicial[1:] + [assr.ESTADO_RESPOSTA]
        self.assertEqual(
            resultado["historicos_consultados"],
            [
                assr.chave_historico(historico_inicial),
                assr.chave_historico(historico_inicial),
                assr.chave_historico(historico_com_resposta),
            ],
        )

    def test_compilacao_faz_duas_chamadas_por_historico(self):
        historicos = assr.gerar_historicos()
        respostas = [{"expert_note": f"nota {i}"} for i in range(len(historicos))]
        respostas += [
            {
                "transicoes": {
                    assr.NIVEIS_OBSERVACAO[0]: assr.ESTADO_INICIAL,
                    assr.NIVEIS_OBSERVACAO[1]: historico[-1],
                    assr.NIVEIS_OBSERVACAO[2]: assr.ESTADO_RESPOSTA,
                    assr.NIVEIS_OBSERVACAO[3]: assr.ESTADO_RESPOSTA,
                }
            }
            for historico in historicos
        ]
        with patch.object(assr, "_chamar_ollama", side_effect=respostas) as chamada:
            compilacao = assr.compilar_tabela_semantica("modelo-teste")
        self.assertEqual(chamada.call_count, 2 * len(assr.ESTADOS) ** assr.ORDEM_CONTEXTO)
        self.assertEqual(len(compilacao["notas_especialista"]), len(historicos))
        self.assertEqual(
            compilacao["tabela"][assr.chave_historico(historicos[0])][
                assr.NIVEIS_OBSERVACAO[2]
            ],
            assr.ESTADO_RESPOSTA,
        )


if __name__ == "__main__":
    unittest.main()
