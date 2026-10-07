"""Verificações matemáticas e reprodução do MVP com os EEG reais."""

import sys
from itertools import product
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, gamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from dados import carregar_dados, preparar_protocolos
from detectors import DETECTORES, calcular_janelas
from get_obs_matrix import ajustar_map, construir_matrizes_observacao, log_emissoes
from get_tran_matrix import construir_matriz_transicao
from hmm_inference import avaliar, primeiro_disparo, resumir
from search_hmm_parameters import preparar_lote, contar_regras, construir_grade, escolher_parametros
from viterbi import viterbi, viterbi_lote


class MatematicaTests(unittest.TestCase):
    def test_viterbi_contra_todos_os_caminhos_e_prefixos(self):
        a, pi = [[.8, .2], [.4, .6]], [.9, .1]
        b = np.log([[.6, .1], [.4, .7], [.3, .9], [.8, .1]])
        caminho, causais = viterbi(b, a, pi)
        def pontuacao(c):
            return np.log(pi[c[0]]) + b[0, c[0]] + sum(
                np.log(a[c[t-1]][c[t]]) + b[t, c[t]] for t in range(1, len(c)))
        melhor = max(product(range(2), repeat=len(b)), key=pontuacao)
        self.assertEqual(caminho.tolist(), list(melhor))
        for n in range(1, len(b) + 1):
            prefixo, _ = viterbi(b[:n], a, pi)
            self.assertEqual(causais[n-1], prefixo[-1])
        self.assertEqual(len(viterbi(np.empty((0, 2)), a, pi)[0]), 0)
        # Probabilidade zero precisa proibir a transição, sem substituir por epsilon.
        caminho, _ = viterbi(b, [[1, 0], [0, 1]], [1, 0])
        self.assertEqual(caminho.tolist(), [0]*len(b))

    def test_map_contra_otimizacao_independente_e_priori(self):
        p = np.random.default_rng(4).beta(.4, 1.2, 100)
        ajuste = ajustar_map(p, (2, 4, 2, 1))
        def objetivo(x):
            mu, k = x
            return -(beta.logpdf(p, mu*k, (1-mu)*k).sum()
                     + beta.logpdf(mu, 2, 4) + gamma.logpdf(k, 2, scale=1))
        direto = minimize(objetivo, [.3, 2], method="Nelder-Mead",
                          bounds=[(.0001, .9999), (.0001, 100)],
                          options={"xatol": 1e-10, "fatol": 1e-10})
        self.assertTrue(direto.success)
        np.testing.assert_allclose([ajuste['mu'], ajuste['kappa']], direto.x, rtol=1e-6)
        forte = ajustar_map(p, (80, 20, 2, 1))
        self.assertGreater(forte['mu'], ajuste['mu'])
        extremo = ajustar_map([0, 1, .05, .2, .4], (2, 4, 2, 1))
        self.assertEqual(extremo['n_clipped'], 2)
        with self.assertRaises(ValueError):
            ajustar_map([float('nan'), .5], (2, 4, 2, 1))

    def test_decisao_causal_continua_e_and_no_mesmo_instante(self):
        obs = [{'nivel': nivel, 'fim_s': t} for t, nivel in enumerate(['ESP']*4+[30]*5)]
        regra = {'consecutivas': None, 'percentual': .6, 'modo': 'OR'}
        self.assertIsNone(primeiro_disparo([0]*4+[1], obs[:5], regra, True))
        self.assertEqual(primeiro_disparo([1, 1, 1, 0, 1], obs[:5], regra, True)['fim_s'], 4)
        # A corrida 2 ocorreu antes do percentual .7: AND nunca passa.
        regra = {'consecutivas': 2, 'percentual': .7, 'modo': 'AND'}
        self.assertIsNone(primeiro_disparo([1, 0, 1, 0, 0, 1, 1], obs[:7], regra))
        regra['modo'] = 'OR'
        self.assertEqual(primeiro_disparo([1, 0, 1, 0, 0, 1, 1], obs[:7], regra)['fim_s'], 0)
        regra = {'consecutivas': 2, 'percentual': None, 'modo': 'OR'}
        self.assertEqual(primeiro_disparo([0, 0, 0, 1, 1], obs[:5], regra, True)['fim_s'], 4)

    def test_busca_lote_e_regras_contra_decisao_individual(self):
        rng = np.random.default_rng(7)
        protocolos = []
        for i, n in enumerate((0, 7, 3, 8)):
            protocolos.append({'grupo': 'alvo' if i < 2 else 'lateral',
                               'observacoes': [{'p_value': float(p), 'nivel': 'ESP' if t < 2 else 30,
                                                'fim_s': t} for t, p in enumerate(rng.uniform(.01, .99, n))]})
        modelos = {'Ausente': {'a': 1, 'b': 1}, 'Presente': {'a': .3, 'b': 1}}
        candidatos = [{'A': [[.8, .2], [.45, .55]], 'pi': [.9, .1]},
                      {'A': [[.5, .5], [.1, .9]], 'pi': [.5, .5]}]
        grade = construir_grade(True)
        p, validos, escopo, alvos = preparar_lote(protocolos)
        estados = viterbi_lote(log_emissoes(p, modelos), candidatos)
        contagens = contar_regras(estados, validos, escopo, alvos, grade['regras'])
        for i, candidato in enumerate(candidatos):
            for j, regra in enumerate(grade['regras']):
                det, fp = 0, 0
                for k, protocolo in enumerate(protocolos):
                    obs = protocolo['observacoes']
                    _, scalar = viterbi(log_emissoes([o['p_value'] for o in obs], modelos),
                                        candidato['A'], candidato['pi'])
                    np.testing.assert_array_equal(estados[i, k, :len(obs)], scalar)
                    disparou = primeiro_disparo(scalar, obs, regra, bool(alvos[k])) is not None
                    if alvos[k]: det += disparou
                    else: fp += disparou
                self.assertEqual(contagens[i, j].tolist(), [det, fp])
        vazios = [{**p, 'observacoes': []} for p in protocolos]
        p, validos, escopo, alvos = preparar_lote(vazios)
        estados = viterbi_lote(log_emissoes(p, modelos), candidatos)
        self.assertFalse(contar_regras(estados, validos, escopo, alvos, grade['regras']).any())


class EEGRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dados = carregar_dados()
        cls.protocolos = preparar_protocolos(cls.dados, 90, 45, "rayleigh")
        cls.b = construir_matrizes_observacao(cls.protocolos, 50, [2, 4, 2, 1])

    def test_metadados_concatenacao_e_calibracao(self):
        self.assertEqual(len(self.dados), 11)
        self.assertEqual(sum(len(f) for f in self.dados.values()), 66)
        self.assertEqual(len(self.protocolos), 176)
        for protocolo in self.protocolos:
            self.assertEqual([f['nivel'] for f in protocolo['fases']], list(config.NIVEIS))
            for o in protocolo['observacoes']:
                fase = next(f for f in protocolo['fases'] if f['nivel'] == o['nivel'])
                inicio, fim = o['intervalo_epocas']
                self.assertEqual(fim-inicio, 90)
                self.assertLessEqual(fim, fase['n_epocas'])
                self.assertEqual(o['fim_s'], fase['inicio_s']+fim)
        self.assertEqual(len(self.b['calibracao']['Ausente']), 128)
        self.assertEqual(len(self.b['calibracao']['Presente']), 672)
        # 70 dB tem gravações curtas demais para M=90; mantém-se a fase na avaliação.
        self.assertEqual({r['nivel'] for r in self.b['calibracao']['Presente']}, {50, 60})
        self.assertEqual({r['paciente'] for r in self.b['calibracao']['Presente']}, set(self.dados))
        for metodo in ('mle', 'map'):
            np.testing.assert_allclose(np.sum(self.b['matriz'][metodo], axis=1), 1)
            for estado in config.ESTADOS:
                m = self.b[metodo][estado]
                self.assertEqual(beta.cdf(1, m['a'], m['b']) - beta.cdf(0, m['a'], m['b']), 1)
        np.testing.assert_allclose([self.b['map']['Presente']['a'], self.b['map']['Presente']['b']],
                                   [.3035875493544358, .9561090720331704], rtol=1e-6)

    def test_a_calculada_com_o_mesmo_janelamento(self):
        a = construir_matriz_transicao(self.dados, 90, 45)
        np.testing.assert_allclose(a['matriz'], [[.5, .5], [44/283, 239/283]])
        np.testing.assert_allclose(np.sum(a['matriz'], axis=1), 1)
        self.assertEqual(a['resumo']['Ausente']['janelas'], 16)
        self.assertEqual(a['resumo']['Presente']['janelas'], 283)

    def test_reproducao_da_versao_escolhida(self):
        cfg = config.configuracao()
        for metodo in ('mle', 'map'):
            resultado = avaliar(self.protocolos, self.b[metodo], cfg['A'], cfg['pi'], cfg['regra'])
            metricas = resumir(resultado)
            self.assertEqual(metricas['global']['detectados'], 66)
            self.assertEqual(metricas['global']['fp_lateral'], 4)
            self.assertEqual(metricas['global']['fp_alvo_esp'], 0)
            self.assertAlmostEqual(metricas['global']['tempo_medio_detectados_s'], 784.6212121212121)
            self.assertEqual(metricas['global']['tempo_mediano_detectados_s'], 737.5)
            self.assertAlmostEqual(metricas['global']['acuracia_balanceada'], .8522727272727273)
            self.assertEqual(len(metricas['por_frequencia']), 8)
            self.assertEqual(len(metricas['por_paciente']), 11)

    def test_referencia_detector_bruto(self):
        # Baseline histórica: Rayleigh p<=.005, janela/passo 120/60,
        # fração de positivos .025. Serve só para testar a comparação anterior.
        protocolos = preparar_protocolos(self.dados, 120, 60, 'rayleigh')
        regra = {'consecutivas': None, 'percentual': .025, 'modo': 'OR'}
        tempos, fp = [], 0
        for p in protocolos:
            obs = p['observacoes']
            estados = [o['p_value'] <= .005 for o in obs]
            if p['grupo'] == 'alvo':
                disparo = primeiro_disparo(estados, obs, regra, True)
                if disparo is not None:
                    tempos.append(disparo['fim_s'] - p['fases'][1]['inicio_s'])
            else:
                fp += primeiro_disparo(estados, obs, regra) is not None
        self.assertEqual(len(tempos), 58)
        self.assertEqual(fp, 3)
        self.assertAlmostEqual((len(tempos)/88 + 1-fp/88)/2, .8125)
        self.assertAlmostEqual(np.mean(tempos), 824.5172413793103)
        self.assertEqual(np.median(tempos), 810.)

    def test_todos_detectores_b_viterbi_e_busca(self):
        cfg = config.configuracao()
        candidatos = [{'A': cfg['A'], 'pi': cfg['pi']}]
        for detector in DETECTORES:
            with self.subTest(detector=detector):
                protocolos = preparar_protocolos(self.dados, 90, 45, detector)
                b = construir_matrizes_observacao(protocolos, 50, [2, 4, 2, 1])
                self.assertEqual(b['detector'], detector)
                self.assertEqual(len(b['calibracao']['Ausente']), 128)
                for metodo in ('mle', 'map'):
                    np.testing.assert_allclose(np.sum(b['matriz'][metodo], axis=1), 1)
                    resultado = avaliar(protocolos, b[metodo], cfg['A'], cfg['pi'], cfg['regra'])
                    self.assertEqual(len(resultado), 176)
                    escolhido = escolher_parametros(protocolos, b[metodo], candidatos,
                                                     construir_grade(True)['regras'])
                    if escolhido is not None:
                        eventos = avaliar(protocolos, b[metodo], cfg['A'], cfg['pi'], escolhido['regra'])
                        resumo = resumir(eventos)['global']
                        self.assertEqual(resumo['detectados'], escolhido['score'][1])
                        self.assertEqual(resumo['fp_lateral'], -escolhido['score'][2])
                if detector == 'spectral_f':
                    for protocolo in protocolos:
                        for obs in protocolo['observacoes']:
                            self.assertEqual(len(obs['ruido_hz']), 2)
                            self.assertNotIn(protocolo['frequencia'], obs['ruido_hz'])
                            self.assertFalse(set(obs['ruido_hz']) & set(self.dados[protocolo['paciente']]['ESP']['alvos']))

    def test_distribuicoes_nulas_e_equivalencia_msc_csm(self):
        rng = np.random.default_rng(14)
        ruido = rng.normal(size=(12000, 3)) + 1j*rng.normal(size=(12000, 3))
        resultados = {}
        for detector in DETECTORES:
            entrada = ruido if detector == 'spectral_f' else ruido[:, 0]
            _, p, _ = calcular_janelas(entrada, 20, 20, detector)
            resultados[detector] = np.asarray(p)
            self.assertLess(abs(np.mean(resultados[detector] <= .05) - .05), .03, detector)
        np.testing.assert_allclose(resultados['msc'], resultados['csm'], atol=1e-11)


if __name__ == '__main__':
    unittest.main()
