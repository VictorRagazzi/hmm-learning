"""Contrato numérico da emissão preditiva e da inferência causal."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.stats import beta
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bayesian_beta_hmm as model


class BayesianEmissionTests(unittest.TestCase):
    def test_compressed_likelihood_equals_beta_likelihood(self):
        p=np.array([0,.001,.2,.7,1.])
        m=model.build_model(p,model.PRIORS["principal"]["Ausente"])
        likelihood=m.compile_logp(vars=[m["beta_loglik"]])
        for mu,kappa in ((.3,1.5),(.8,4.),(.5,2.)):
            point={"mu_logodds__":np.log(mu/(1-mu)),"kappa_log__":np.log(kappa)}
            expected=beta.logpdf(model.stabilize(p),mu*kappa,(1-mu)*kappa).sum()
            self.assertAlmostEqual(likelihood(point),expected,places=8)

    def test_mixture_is_mean_of_densities(self):
        samples = {"a": np.array([.3, 4]), "b": np.array([2., .8])}
        p = np.array([.01, .2, .8])
        actual = model.predictive_logpdf(p, samples)
        expected = np.log((beta.pdf(p, .3, 2) + beta.pdf(p, 4, .8))/2)
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.allclose(actual, (beta.logpdf(p,.3,2)+beta.logpdf(p,4,.8))/2))
        self.assertFalse(np.allclose(actual, beta.logpdf(p,2.15,1.4)))

    def test_extremes_and_underflow(self):
        samples = {"a": np.array([1000.,1100.]), "b": np.array([1200.,1300.])}
        logs = model.predictive_logpdf(np.array([0,1e-200,.5,1]),samples)
        self.assertTrue(np.all(np.isfinite(logs)))
        self.assertLess(logs[0],-10000)
        self.assertEqual(logs[0],logs[1])
        for invalid in ([np.nan],[-.1],[1.1]):
            with self.assertRaises(ValueError): model.predictive_logpdf(invalid,samples)

    def test_integral_with_singular_edges(self):
        samples={"a":np.array([.2,2.,1.]),"b":np.array([.3,1.,4.])}
        checks=model.numerical_checks(samples)
        self.assertAlmostEqual(checks["mass_quadrature"],1.,places=7)
        self.assertGreater(checks["mass_outside_clip_interval"],0)

    def test_viterbi_terminal_matches_enumeration_and_prefixes(self):
        from itertools import product
        matrix=np.array([[.99,.01],[.15,.85]])
        pi=np.array([.95,.05])
        logs=np.array([[0,1],[0,6],[1,0],[0,8]])
        states,_=model.causal_states(logs,matrix,pi)
        for n in range(1,5):
            candidates=[]
            for path in product((0,1),repeat=n):
                score=np.log(pi[path[0]])+logs[0,path[0]]
                for i in range(1,n): score+=np.log(matrix[path[i-1],path[i]])+logs[i,path[i]]
                candidates.append((score,path))
            best=max(candidates,key=lambda c:c[0])[1]
            self.assertEqual(states[n-1],bool(best[-1]))
            np.testing.assert_array_equal(model.causal_states(logs[:n],matrix,pi)[0],states[:n])
        self.assertEqual(len(model.causal_states(np.empty((0,2)),matrix,pi)[0]),0)

    def test_saved_protocol_baseline(self):
        config,saved,protocols,data=model.load_fixed()
        baseline=model.evaluate(protocols,config)
        self.assertEqual(baseline["detalhes"],saved["detalhes_hmm_antigo"])
        self.assertEqual([len(data[s]) for s in model.STATES],[104,1744])
        g=baseline["metricas"]["global"]
        self.assertEqual((g["detectados"],g["fp_alvo_esp"],g["fp_lateral"]),(70,0,8))
        self.assertEqual(g["tempo_desde_30_medio_detectados_s"],734)
        self.assertEqual(g["tempo_desde_30_mediano_detectados_s"],730)
        self.assertTrue(all(d["tempo_deteccao_desde_30_s"] is None for d in baseline["detalhes"] if not d["detectou"]))


if __name__ == "__main__":
    unittest.main()
