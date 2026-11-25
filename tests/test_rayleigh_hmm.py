"""Forward-backward, MLE Gamma e posteriori em grade do experimento rayleigh_hmm."""
import itertools
import os
import sys
import unittest

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rayleigh_hmm"))

import dados  # noqa: E402
import emissao  # noqa: E402
import avaliacao  # noqa: E402
import experimento  # noqa: E402
import hmm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rayleigh_hmm", "monte_carlo"))
import mcmc_completo  # noqa: E402
import mc_comum  # noqa: E402


class TesteRayleighHMM(unittest.TestCase):
    def test_forward_backward_igual_a_enumeracao(self):
        rng = np.random.default_rng(1)
        T = 5
        log_em = rng.normal(size=(3, T, 2))
        pi = np.array([0.7, 0.3])
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        gama = hmm.posteriori_estados(log_em, pi, A)
        for n in range(3):
            conjunta = {}
            for caminho in itertools.product([0, 1], repeat=T):
                p = pi[caminho[0]] * np.exp(log_em[n, 0, caminho[0]])
                for t in range(1, T):
                    p *= A[caminho[t - 1], caminho[t]] * np.exp(log_em[n, t, caminho[t]])
                conjunta[caminho] = p
            total = sum(conjunta.values())
            for t in range(T):
                esperado = sum(p for c, p in conjunta.items() if c[t] == 1) / total
                assert np.isclose(gama[n, t], esperado)


    def test_viterbi_igual_a_forca_bruta_em_cada_prefixo(self):
        rng = np.random.default_rng(7)
        T = 7
        for _ in range(20):
            log_em = rng.normal(scale=2.0, size=(3, T, 2))
            p1, a01, a10 = rng.uniform(0.02, 0.98, size=3)
            pi = np.array([1 - p1, p1])
            A = np.array([[1 - a01, a01], [a10, 1 - a10]])
            det = hmm.viterbi_alguma_presente(log_em, pi, A)
            for n in range(3):
                for t in range(1, T + 1):
                    def log_p(c):
                        v = np.log(pi[c[0]]) + log_em[n, 0, c[0]]
                        for j in range(1, t):
                            v += np.log(A[c[j - 1], c[j]]) + log_em[n, j, c[j]]
                        return v
                    melhor = max(itertools.product([0, 1], repeat=t), key=log_p)
                    self.assertEqual(det[n, t - 1], any(melhor))

    def test_viterbi_causal(self):
        rng = np.random.default_rng(8)
        log_em = rng.normal(scale=2.0, size=(50, 8, 2))
        pi = np.array([0.8, 0.2])
        A = np.array([[0.9, 0.1], [0.3, 0.7]])
        det = hmm.viterbi_alguma_presente(log_em, pi, A)
        for t in range(7):
            alterado = log_em.copy()
            alterado[:, t + 1:] = rng.normal(scale=2.0, size=alterado[:, t + 1:].shape)
            novo = hmm.viterbi_alguma_presente(alterado, pi, A)
            np.testing.assert_array_equal(novo[:, : t + 1], det[:, : t + 1])

    def test_viterbi_empate_fica_com_ausente(self):
        det = hmm.viterbi_alguma_presente(np.zeros((1, 4, 2)), np.array([0.5, 0.5]), np.full((2, 2), 0.5))
        self.assertFalse(det.any())

    def test_filtrada_igual_a_suavizada_no_ultimo_passo_e_causal(self):
        rng = np.random.default_rng(4)
        log_em = rng.normal(size=(4, 6, 2))
        pi = np.array([0.6, 0.4])
        A = np.array([[0.8, 0.2], [0.1, 0.9]])
        filtrada = hmm.posteriori_filtrada(log_em, pi, A)
        np.testing.assert_allclose(filtrada[:, -1], hmm.posteriori_estados(log_em, pi, A)[:, -1])
        alterado = log_em.copy()
        alterado[:, 4:] = rng.normal(size=(4, 2, 2))
        np.testing.assert_allclose(hmm.posteriori_filtrada(alterado, pi, A)[:, :4], filtrada[:, :4])

    def test_mle_gamma_recupera_parametros(self):
        z = np.random.default_rng(2).gamma(2.5, 3.0 / 2.5, size=20000)
        mle = emissao.ajustar_mle(z)
        assert abs(mle["k"] - 2.5) < 0.1 and abs(mle["mu"] - 3.0) < 0.05


    def test_posteriori_recupera_rho_e_preditiva_integra_1(self):
        rng = np.random.default_rng(3)
        n_ep = rng.choice([10, 50, 200, 480], size=4000)
        k_real, rho_real = 1.5, 0.02
        mu = emissao.media_presente(n_ep, rho_real)
        z = rng.gamma(k_real, mu / k_real)
        priori = {"k_forma": 2.0, "k_taxa": 1.0, "rho_forma": 2.0, "rho_taxa": 200.0}
        post = emissao.posteriori_em_grade(z, n_ep, priori, np.logspace(-1, 1.5, 200), np.logspace(-5, -0.5, 200))
        self.assertAlmostEqual(post["map"]["k"], k_real, delta=0.1)
        self.assertAlmostEqual(post["map"]["rho"], rho_real, delta=0.002)
        grade = np.linspace(1e-6, 200, 400001)
        dens = np.exp(emissao.log_pdf_preditiva(grade, 100, post["pontos_preditiva"]))
        self.assertAlmostEqual(np.trapezoid(dens, grade), 1.0, delta=0.01)


    def test_rayleigh_acumulado_mascara_toda_true_igual_ao_sem_mascara(self):
        rng = np.random.default_rng(5)
        janelas = rng.normal(size=(6, 10, 3)) + 1j * rng.normal(size=(6, 10, 3))
        z0, n0 = dados.rayleigh_z_acumulado(janelas)
        z1, n1 = dados.rayleigh_z_acumulado(janelas, np.ones((6, 10), dtype=bool))
        np.testing.assert_allclose(z1, z0)
        np.testing.assert_array_equal(n1, n0)
        np.testing.assert_array_equal(n0, 10 * np.arange(1, 7))

    def test_rayleigh_acumulado_conta_so_validas_e_ignora_invalidas(self):
        rng = np.random.default_rng(6)
        janelas = rng.normal(size=(4, 10, 2)) + 1j * rng.normal(size=(4, 10, 2))
        valida = np.ones((4, 10), dtype=bool)
        valida[0, :3] = False
        valida[2, 5] = False
        z0, n = dados.rayleigh_z_acumulado(janelas, valida)
        np.testing.assert_array_equal(n, [7, 17, 26, 36])
        alterado = janelas.copy()
        alterado[0, :3] = np.exp(1j * rng.uniform(0, 6, size=(3, 2))) * 100
        alterado[2, 5] = 50j
        z1, _ = dados.rayleigh_z_acumulado(alterado, valida)
        np.testing.assert_allclose(z1, z0)
        z_vazio, n_vazio = dados.rayleigh_z_acumulado(janelas[:1], np.zeros((1, 10), dtype=bool))
        assert n_vazio[0] == 0 and np.all(z_vazio == 0)

    def test_modos_de_sinal_tem_forma_certa(self):
        x = np.random.default_rng(7).normal(size=(5, 8, 40))
        canal = dados.combinar_canais(x, "canal", [2])
        np.testing.assert_array_equal(canal, x[2])
        media = dados.combinar_canais(x, "media", [0, 1, 3])
        np.testing.assert_allclose(media, x[[0, 1, 3]].mean(axis=0))
        pca = dados.combinar_canais(x, "pca", [0, 1, 3])
        assert canal.shape == media.shape == pca.shape == (8, 40)
        assert pca.std() > 0

    def test_pca_recupera_fonte_comum(self):
        rng = np.random.default_rng(8)
        fonte = rng.normal(size=(6, 50))
        x = np.stack([g * fonte + 0.05 * rng.normal(size=fonte.shape) for g in (1.0, -2.0, 0.5)])
        pca = dados.combinar_canais(x, "pca", [0, 1, 2])
        assert abs(np.corrcoef(pca.ravel(), fonte.ravel())[0, 1]) > 0.99

    def test_rejeicao_marca_epoca_de_variancia_alta(self):
        sinal = np.random.default_rng(9).normal(size=(20, 100))
        sinal[7] *= 10.0
        valida = dados.epocas_validas(sinal, 3.0)
        assert not valida[7] and valida.sum() == 19
        assert dados.epocas_validas(sinal, None).all()

    def test_truncar_sequencia_acumulada_igual_a_regra_no_prefixo(self):
        rng = np.random.default_rng(10)
        comprimentos = [8, 12, 12, 30, 30, 5]
        scores = [rng.uniform(size=n) for n in comprimentos]
        duracoes = [1.0] * len(scores)
        passos = np.array([6, 9, 12, 20, 7, 4])
        regra = {"familia": "fracao_ou_consecutivas", "fracao": 0.3, "consecutivas": 2}
        grupos = avaliacao.preparar(scores, duracoes, passos)
        det, tempo = avaliacao.exame(grupos, len(scores), avaliacao.bruto_detecta, 0.4, regra)
        prefixos = [s[: min(len(s), k)] for s, k in zip(scores, passos)]
        det2, tempo2 = avaliacao.exame(avaliacao.preparar(prefixos, duracoes), len(scores),
                                       avaliacao.bruto_detecta, 0.4, regra)
        np.testing.assert_array_equal(det, det2)
        np.testing.assert_array_equal(tempo, tempo2)
        assert det.any() and (~det).any()
        # o tempo nunca passa do trecho mantido
        assert np.all(tempo[det] <= passos[det] * experimento.EPOCAS_POR_JANELA)

    def test_deteccao_acumulada_monotonica_e_termina_na_taxa_final(self):
        rng = np.random.default_rng(11)
        tempo = np.where(rng.uniform(size=200) < 0.6, rng.integers(30, 480, size=200) // 10 * 10.0, np.nan)
        t = np.arange(0, 501)
        curva = avaliacao.curva_acumulada(tempo, t)
        assert np.all(np.diff(curva) >= 0)
        assert curva[0] == 0.0
        self.assertAlmostEqual(curva[-1], np.mean(~np.isnan(tempo)))

    def test_injetar_senoide_deterministica_e_snr_pedida(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(200000, 2)) + 1j * rng.normal(size=(200000, 2))
        y = dados.injetar_senoide(x, -15.0)
        np.testing.assert_array_equal(y, dados.injetar_senoide(x, -15.0))
        potencia = np.mean(np.abs(y) ** 2, axis=0)
        s_sinal = np.abs(y.mean(axis=0)) ** 2 - potencia / len(y)  # estimador sem vies
        snr_db = 10 * np.log10(s_sinal / (potencia - s_sinal))
        np.testing.assert_allclose(snr_db, -15.0, atol=0.3)

    def test_hierarquico_recupera_hiperparametros(self):
        rng = np.random.default_rng(12)
        a_k, b_k, a_r, b_r = 6.0, 2.0, 3.0, 100.0
        n_seq, T = 1500, 12
        k = rng.gamma(a_k, 1 / b_k, size=(n_seq, 1))
        rho = rng.gamma(a_r, 1 / b_r, size=(n_seq, 1))
        n_ep = np.tile(10 * np.arange(1, T + 1), (n_seq, 1))
        z = rng.gamma(k, emissao.media_presente(n_ep, rho) / k)
        ids = np.repeat(np.arange(n_seq), T)
        gk, gr = np.geomspace(0.05, 100, 70), np.geomspace(1e-4, 1, 180)
        matriz = emissao.matriz_verossimilhanca_sequencias(z.ravel(), n_ep.ravel(), ids, gk, gr)
        h = emissao.ajustar_hierarquico(matriz, gk, gr, [(2.0, 1.0, 1.0, 20.0), (10.0, 2.0, 5.0, 200.0)])
        assert not h["na_borda"] and h["dispersao_relativa_inicios"] < 1e-3
        self.assertAlmostEqual(h["k_media"], a_k / b_k, delta=0.3)
        self.assertAlmostEqual(h["rho_media"], a_r / b_r, delta=0.003)
        self.assertAlmostEqual(h["k_desvio"], a_k ** 0.5 / b_k, delta=0.35)
        self.assertAlmostEqual(h["rho_desvio"], a_r ** 0.5 / b_r, delta=0.003)

    def test_ids_nao_contiguos_igual_a_soma_por_sequencia(self):
        z = np.array([1.0, 2.0, 3.0, 4.0])
        n_ep = np.array([10.0, 10.0, 20.0, 20.0])
        gk, gr = np.array([1.5]), np.array([0.02])
        m = emissao.matriz_verossimilhanca_sequencias(z, n_ep, np.array([0, 1, 0, 1]), gk, gr)
        esperado = emissao.log_pdf_gamma(np.array([1.0, 3.0]), 1.5, emissao.media_presente(np.array([10.0, 20.0]), 0.02)).sum()
        self.assertAlmostEqual(m[0, 0, 0], esperado)

    def test_emissao_marginal_coincide_com_integral_direta(self):
        hiper = {"a_k": 3.0, "b_k": 1.5, "a_rho": 2.0, "b_rho": 100.0}
        z, n_ep = 4.0, 100
        emissoes = {"ausente": {"k": 1.0, "mu": 1.0},
                    "marginal": {"pontos": experimento.pontos_marginal(hiper, 100, 200)}}
        log_em = experimento.log_emissoes({"z": np.array([z]), "n_ep": np.array([n_ep])}, emissoes, "marginal")

        def integrando(rho, k):
            return (stats.gamma.pdf(z, k, scale=emissao.media_presente(n_ep, rho) / k)
                    * stats.gamma.pdf(k, 3.0, scale=1 / 1.5) * stats.gamma.pdf(rho, 2.0, scale=1 / 100.0))

        direta, _ = integrate.dblquad(integrando, 0.0, 25.0, 0.0, 0.5, epsabs=1e-7, epsrel=1e-6)
        self.assertAlmostEqual(np.exp(log_em[0, 1]) / direta, 1.0, delta=2e-3)

    def test_calibracao_real_usa_todos_os_alvos_com_estimulo(self):
        def seq(tipo, condicao, T):
            return {"tipo": tipo, "condicao": condicao, "z": np.arange(1.0, T + 1), "n_ep": 10.0 * np.arange(1, T + 1)}

        sequencias = [seq("alvo", "ESP", 3), seq("lateral", "50dB", 3), seq("alvo", "30dB", 4), seq("alvo", "70dB", 2)]
        z, n, ids = experimento.gerar_presente_real(sequencias)
        self.assertEqual(z.size, 6)
        self.assertEqual(n.size, 6)
        self.assertEqual(sorted(set(ids.tolist())), [0, 1])
        self.assertEqual(np.bincount(ids).tolist(), [4, 2])

    def test_mcmc_completo_dados_simulados(self):
        rng = np.random.default_rng(7)
        a_k, b_k, a_r, b_r = 4.0, 2.0, 3.0, 150.0
        n_seq, T = 40, 15
        k, rho = rng.gamma(a_k, 1 / b_k, n_seq), rng.gamma(a_r, 1 / b_r, n_seq)
        n_ep = np.tile(10.0 * np.arange(1, T + 1), n_seq)
        ids = np.repeat(np.arange(n_seq), T)
        mu = emissao.media_presente(n_ep, rho[ids])
        z = rng.gamma(k[ids], mu / k[ids])
        res = mcmc_completo.mcmc_completo(z, n_ep, ids, n_iter=600, burn_in=300, semente=3,
                                          idx_exemplos=[0, 5], thin=5)
        self.assertEqual(res["cadeias"].shape, (4, 600, 4))
        self.assertEqual(res["latentes_exemplos"].shape, (4, 600, 2, 2))
        self.assertEqual(res["latentes_thin"].shape, (4, 60, n_seq, 2))
        self.assertEqual(res["aceitacao"].shape, (4,))
        for v in (res["cadeias"], res["latentes_exemplos"], res["latentes_thin"]):
            self.assertTrue(np.all(np.isfinite(v)))
        medias = np.exp(res["cadeias"][:, 300:]).mean(axis=(0, 1))
        for obtido, verdadeiro in zip(medias, (a_k, b_k, a_r, b_r)):
            self.assertLess(abs(np.log(obtido / verdadeiro)), 1.2)

    def test_ajustar_fixo_dados_simulados(self):
        rng = np.random.default_rng(11)
        k, rho, n = 2.0, 0.02, 3000
        n_ep = rng.integers(10, 481, n)
        mu = emissao.media_presente(n_ep, rho)
        z = rng.gamma(k, mu / k)
        r = mc_comum.ajustar_fixo(z, n_ep)
        self.assertLess(abs(r["k"] / k - 1), 0.10)
        self.assertLess(abs(r["rho"] / rho - 1), 0.10)
        self.assertFalse(r["na_borda"])


def _somas_simuladas(x):
    """Somas acumuladas (como detectores.carregar_somas) de x complexo (canais, epocas, freqs); canal 0 = FC."""
    import detectores
    espectro = x  # ja no dominio da frequencia; os bins sao os indices 0..F-1
    bins_viz = np.arange(espectro.shape[2] - 20, espectro.shape[2])
    bins = np.arange(8)
    somas = detectores._somas_do_tipo(espectro, bins, bins_viz)
    return {"alvo": somas, "lateral": somas, "n_viz": bins_viz.size}


def _ruido_complexo(rng, canais, epocas, freqs):
    return (rng.normal(size=(canais, epocas, freqs)) + 1j * rng.normal(size=(canais, epocas, freqs))) / np.sqrt(2)


class TesteDetectores(unittest.TestCase):
    def test_rayleigh_por_somas_igual_ao_acumulado(self):
        import detectores
        rng = np.random.default_rng(3)
        x = _ruido_complexo(rng, 16, 120, 40)
        somas = _somas_simuladas(x)
        fim = (np.arange(12) + 1) * 10
        z, n = detectores.z_intervalo(somas, "rayleigh", "alvo", np.zeros_like(fim), fim)
        esperado, n_esp = dados.rayleigh_z_acumulado(dados.janelar(x[0][:, :8], 10))
        np.testing.assert_allclose(z, esperado, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(n, n_esp)

    def test_intervalo_com_cap_igual_ao_calculo_direto(self):
        import detectores
        rng = np.random.default_rng(4)
        x = _ruido_complexo(rng, 16, 100, 40)
        somas = _somas_simuladas(x)
        z, _ = detectores.z_intervalo(somas, "rayleigh", "alvo", np.array([40]), np.array([70]))
        fases = x[0][40:70, :8] / np.abs(x[0][40:70, :8])
        np.testing.assert_allclose(z[0], np.abs(fases.sum(axis=0)) ** 2 / 30, rtol=1e-9)

    def test_calibracao_h0_dos_detectores(self):
        import detectores
        rng = np.random.default_rng(5)
        n_rep, n = 4000, 20
        x = _ruido_complexo(rng, 16, n, n_rep + 20)  # n_rep repeticoes independentes nos bins 0..n_rep-1 e vizinhos no fim
        espectro = x
        bins_viz = np.arange(n_rep, n_rep + 20)
        somas = {"alvo": detectores._somas_do_tipo(espectro, np.arange(n_rep), bins_viz), "n_viz": 20}
        for det in ("msc", "hotelling", "mmsc", "f_espectral"):
            z, _ = detectores.z_intervalo(somas, det, "alvo", np.array([0]), np.array([n]))
            self.assertTrue(np.all(np.isfinite(z)), det)
            self.assertTrue(0.035 <= np.mean(np.exp(-z) <= 0.05) <= 0.065, det)
            self.assertLess(abs(z.mean() - 1.0), 0.1, det)

    def test_logsf_fechado_igual_ao_scipy(self):
        import detectores
        rng = np.random.default_rng(6)
        x = _ruido_complexo(rng, 16, 30, 28)
        somas = _somas_simuladas(x)
        z, _ = detectores.z_intervalo(somas, "hotelling", "alvo", np.array([0]), np.array([30]))
        for i in range(8):
            pts = np.stack([x[0][:, i].real, x[0][:, i].imag], axis=1)
            m, c = pts.mean(axis=0), np.cov(pts.T, ddof=1)
            f = 28 / (2 * 29) * 30 * m @ np.linalg.solve(c, m)
            self.assertAlmostEqual(z[0, i], -stats.f.logsf(f, 2, 28), places=8)
        z, _ = detectores.z_intervalo(somas, "mmsc", "alvo", np.array([0]), np.array([30]))
        for i in range(8):
            y = x[[0, 13, 15, 1], :, i]
            v, s = y.sum(axis=1), y @ y.conj().T
            k2 = (v.conj() @ np.linalg.solve(s, v)).real / 30
            self.assertAlmostEqual(z[0, i], -stats.beta.logsf(k2, 4, 26), places=8)

    def test_msc_detecta_sinal_forte(self):
        import detectores
        rng = np.random.default_rng(7)
        x = _ruido_complexo(rng, 16, 60, 40)
        x[:, :, :8] += 1.5  # resposta coerente (fase constante) nos 8 bins alvo
        somas = _somas_simuladas(x)
        for det in detectores.DETECTORES:
            z, _ = detectores.z_intervalo(somas, det, "alvo", np.array([0]), np.array([60]))
            self.assertTrue(np.all(z > 20), det)


if __name__ == "__main__":
    unittest.main()
