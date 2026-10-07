"""Busca vetorizada versus inferência e regras causais escalares."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import contextlib
import io

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import search_map_beta_hmm as search
import map_beta_hmm_loso as model


class SearchTests(unittest.TestCase):
    def test_batched_viterbi_and_every_prefix(self):
        from scipy.stats import beta
        p = np.array([[.6, .3, .01, .8], [.4, .001, .9, .2]])
        models = {'Ausente': {'a': 1.2, 'b': .8}, 'Presente': {'a': .3, 'b': 1.4}}
        emissions = np.stack([beta.logpdf(p, models[s]['a'], models[s]['b']) for s in models], axis=2)
        candidates = [{'A': [[.9,.1],[.2,.8]], 'pi': [.95,.05]},
                      {'A': [[.5,.5],[.5,.5]], 'pi': [.5,.5]}]
        states = search.batch_states(emissions, candidates)
        for i,c in enumerate(candidates):
            for j in range(2):
                for end in range(1,5):
                    expected = model.phase._estados_hmm(p[j,:end], c['A'], c['pi'], models)
                    np.testing.assert_array_equal(states[i,j,:end], expected)

    def test_vector_rules_equal_scalar_including_and_coincidence(self):
        rng = np.random.default_rng(12)
        states = rng.integers(0, 2, (3, 6, 12)).astype(bool)
        target = np.array([True]*3 + [False]*3)
        valid = np.ones((6,12), dtype=bool)
        valid[0,8:] = False
        valid[1] = False
        scope = valid.copy()
        scope[:3,:3] = False
        rules = [model.detection_rule(c,p,mode) for c in (1,2,4,None)
                 for p in (.1,.5,.9,None) for mode in ('OR','AND') if c is not None or p is not None]
        scores = search.rule_scores(states, valid, scope, target, rules)
        for candidate in range(3):
            for j,rule in enumerate(rules):
                flags = []
                for sequence in range(6):
                    length = int(valid[sequence].sum())
                    phases = [0 if t<3 else 1 for t in range(length)]
                    flags.append(model.first_detection(states[candidate,sequence,:length], range(length),
                        phases, rule, phase_min=1 if target[sequence] else 0) is not None)
                self.assertEqual(scores[candidate,j].tolist(), [sum(flags[:3]),sum(flags[3:])])

    def test_empty_and_padding_never_trigger(self):
        rules = [model.detection_rule(1,.1,'OR'), model.detection_rule(None,.1,'AND')]
        for length in (0,5):
            scores = search.rule_scores(np.ones((2,2,length),dtype=bool),
                np.zeros((2,length),dtype=bool), np.zeros((2,length),dtype=bool), np.array([True,False]), rules)
            self.assertTrue(np.all(scores == 0))

    def test_grid_contains_requested_dimensions_and_valid_probabilities(self):
        grid = search.build_grid()
        self.assertEqual(len(grid['windows_steps']),14)
        self.assertEqual(len(grid['rules']),387)
        for c in grid['candidates']:
            np.testing.assert_allclose(np.array(c['A']).sum(axis=1),1)
            self.assertAlmostEqual(sum(c['pi']),1)
        self.assertTrue(any(np.allclose(c['A'],model.MATRIX_A) for c in grid['candidates']))

    def test_in_sample_uses_every_subject_and_separate_checkpoint_mode(self):
        protocols = []
        for subject in ('A','B'):
            for group in ('estimulo','lateral'):
                phases = []
                for i,level in enumerate(model.phase.LEVELS):
                    phases.append({'nivel':level, 'p_values':[.02,.2,.8],
                        'arquivo':f'{subject}{level}.mat', 'canal':0, 'fs':1000,
                        'frequencia':81 if group=='estimulo' else 82,
                        'inicio_global':i*30, 'n_epocas':30, 'finais_locais':[10,20,30],
                        'intervalos_locais':[[0,10],[10,20],[20,30]], 'valores_ord':[.01,.02,.03]})
                protocols.append({'participante':subject,'grupo':group,'frequencia_indice':0,
                    'controle_hz':82,'fases':phases,'n_epocas_total':180})
        grid = {'windows_steps':[[10,10]], 'candidates':[], 'rules':[],
                'methods':['bruto','mle','map'], 'max_train_lateral_fp':.05}
        def select(train,fit,method,grid):
            targets=sum(p['grupo']=='estimulo' for p in train)
            return ({'score':[.5,0,0],'A':model.MATRIX_A,'pi':model.PI,
                'rule':model.detection_rule(99,None,'OR'),
                'train_metrics':{'deteccao':0,'fp_lateral':0,'taxa_fp_lateral':0,
                                 'n_alvos':targets,'n_laterais':targets}},
                    {'evaluated':1,'excluded_fp':0})
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(search,'load_windows',return_value=({'10/10':protocols},[])), \
             patch.object(search,'select_best',side_effect=select):
            path=Path(tmp)/'sample.json'
            result=search.run(grid,(2,4,2,1),50,path,validation='in-sample')
            self.assertEqual(result['folds'],{})
            self.assertEqual(result['deployment_fit']['train_subjects'],['A','B'])
            self.assertIsNone(result['deployment_fit']['test_subject'])
            fitted=result['deployment_fit']['selected']['map']['models']
            self.assertEqual(fitted['Ausente']['n'],6)
            self.assertEqual(fitted['Presente']['n'],18)
            self.assertEqual(len(result['comparison']['map']['events']),4)
            self.assertEqual(result['comparison']['map']['metrics']['global']['n_alvos'],2)
            self.assertIn('vazamento deliberado',result['limitations'][1])
            resumed=search.run(grid,(2,4,2,1),50,path,resume=True,validation='in-sample')
            self.assertEqual(result,resumed)
            with self.assertRaises(ValueError):
                search.run(grid,(2,4,2,1),50,path,resume=True,validation='loso')


if __name__ == '__main__':
    unittest.main()
