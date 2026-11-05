# Exploratorio: HMM observando o p-valor acumulado (in-sample, 11 participantes)

## B1 (checagem: mudanca de variaveis do modelo Gamma)

Viterbi Z vs p(B1) em 1296 combinacoes (pi, A): 0 divergencias em 37013760 celulas. No ponto P1: Z {'n_det': 252, 'n_fp_lat': 36, 'n_fp_esp': 5, 'tempo': 132.3015873015873} vs B1 {'n_det': 252, 'n_fp_lat': 36, 'n_fp_esp': 5, 'tempo': 132.3015873015873}. Identico: True.

## B2: ajustes

Ausente Beta: a0=1.1045, b0=0.9539 (n=1160). Log-verossim. em p (mesmos 1160 dados): Beta 10.33; Uniforme 0.00; Gamma(Z)+jacobiano 10.30.

Presente: a_beta=1.4582, b_beta=0.4280 (media beta 3.407, dp 2.821); a_rho=0.4875, b_rho=23.8239 (media rho 0.0205, dp 0.0293). Log-verossim. marginal 10138.0 (Gamma-Z + jacobiano: 11198.5). Massa fora da grade {'beta': 0.00027169751319996023, 'rho': 7.0565605947375e-05}; hiper no limite False; massa da posteriori nas bordas {'beta_min': 1.8244852818460984e-09, 'beta_max': 3.6766328307613762e-53, 'rho_min': 1.1297291315956243e-06, 'rho_max': 2.964843279687104e-14}; dispersao entre inicios 1.8e-06; na_borda=False.

Corridas: inicio [9.0, 6.0, 1.0, 40.0] -> ll 10137.98, it 15, ok True; inicio [4.0, 3.0, 2.0, 100.0] -> ll 10137.98, it 15, ok True; inicio [20.0, 2.0, 0.7, 30.0] -> ll 10137.98, it 15, ok True; inicio [1.26, 0.225, 0.67, 18.6] -> ll 10137.98, it 13, ok True

bruto_referencia: det 252/440, FP lat 35/528, FP ESP 4/88, tempo medio/mediano 134.7/100.0.

B2 busca: 186624 candidatos; viaveis 19785; com dif. pareada <= 0: 0; pontos de Pareto 9.

## Vencedores B2 e referencias

| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | McNemar so_bruto-so_hmm (alvos / laterais / ESP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 Gamma-Z (referencia) | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 252/440 (57.3%) | 36/528 | 5/88 | 132.3 / 85.0 | 244 | 53/115/76 | +2.7 | 2.5e-06 | 8-8 p=1 / 7-8 p=1 / 0-1 p=1 |
| a_chave_ordem | 0.005 | 0.01 | 0.5 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 253/440 (57.5%) | 36/528 | 3/88 | 135.0 / 100.0 | 247 | 49/109/89 | +4.3 | 1.2e-07 | 5-6 p=1 / 5-6 p=1 / 1-0 p=1 |
| b_vazio_menor_dif_det_ge_bruto | 0.005 | 0.01 | 0.5 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 253/440 (57.5%) | 36/528 | 3/88 | 135.0 / 100.0 | 247 | 49/109/89 | +4.3 | 1.2e-07 | 5-6 p=1 / 5-6 p=1 / 1-0 p=1 |
| B2_em_P1 | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 269/440 (61.1%) | 58/528 | 9/88 | 129.8 / 90.0 | 249 | 93/66/90 | -6.9 | 0.0078 | 3-20 p=0.00049 / 0-23 p=2.4e-07 / 0-5 p=0.062 |

## Fronteira de Pareto B2

| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | McNemar so_bruto-so_hmm (alvos / laterais / ESP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 0.005 | 0.01 | 0.5 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 253/440 (57.5%) | 36/528 | 3/88 | 135.0 / 100.0 | 247 | 49/109/89 | +4.3 | 1.2e-07 | 5-6 p=1 / 5-6 p=1 / 1-0 p=1 |
| P2 | 0.1 | 1e-05 | 0.5 | fracao=0.4 (fracao) | 181/440 (41.1%) | 33/528 | 4/88 | 131.2 / 70.0 | 177 | 35/96/46 | +41.7 | 3.5e-12 | 75-4 p=5.2e-18 / 14-12 p=0.85 / 1-1 p=1 |
| P3 | 0.2 | 0.001 | 0.5 | fracao=0.7 (fracao) | 130/440 (29.5%) | 36/528 | 3/88 | 126.5 / 70.0 | 126 | 22/64/40 | +56.0 | 1.1e-07 | 126-4 p=1.7e-32 / 19-20 p=1 / 2-1 p=1 |
| P4 | 0.2 | 0.0005 | 0.5 | fracao=0.7 (fracao) | 129/440 (29.3%) | 36/528 | 3/88 | 124.5 / 70.0 | 125 | 21/64/40 | +54.5 | 1.4e-07 | 127-4 p=8.9e-33 / 19-20 p=1 / 2-1 p=1 |
| P5 | 0.2 | 0.0001 | 0.5 | fracao=0.7 (fracao) | 125/440 (28.4%) | 36/528 | 3/88 | 113.7 / 70.0 | 121 | 21/60/40 | +45.5 | 9.8e-07 | 131-4 p=6.3e-34 / 19-20 p=1 / 2-1 p=1 |
| P6 | 0.2 | 0.01 | 0.5 | fracao=0.8 (fracao) | 107/440 (24.3%) | 32/528 | 2/88 | 113.6 / 50.0 | 102 | 19/60/23 | +50.8 | 1.6e-05 | 150-5 p=3.2e-38 / 24-21 p=0.77 / 3-1 p=0.62 |
| P7 | 0.2 | 0.005 | 0.5 | fracao=0.8 (fracao) | 104/440 (23.6%) | 32/528 | 2/88 | 109.7 / 50.0 | 100 | 19/58/23 | +46.5 | 4.2e-05 | 152-4 p=5.3e-40 / 24-21 p=0.77 / 3-1 p=0.62 |
| P8 | 0.2 | 0.001 | 0.5 | fracao=0.8 (fracao) | 101/440 (23.0%) | 31/528 | 2/88 | 102.6 / 50.0 | 97 | 19/55/23 | +39.7 | 0.00017 | 155-4 p=7.2e-41 / 24-20 p=0.65 / 3-1 p=0.62 |
| P9 | 0.2 | 1e-05 | 0.5 | fracao=0.8 (fracao) | 100/440 (22.7%) | 31/528 | 2/88 | 102.1 / 50.0 | 96 | 19/54/23 | +39.9 | 0.0002 | 156-4 p=3.7e-41 / 24-20 p=0.65 / 3-1 p=0.62 |

## Pareado B2 vs P1 Gamma-Z (n pares, dif. media s)

- a_chave_ordem: 249 pares, +1.3 s
- b_vazio_menor_dif_det_ge_bruto: 249 pares, +1.3 s
- B2_em_P1: 252 pares, -10.0 s

### Por intensidade: B2 a_chave_ordem vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 40/88 | 8/88 | 6/88 | 206.6 | 209.5 |
| 40dB | 66/88 | 65/88 | 6/88 | 6/88 | 188.9 | 194.0 |
| 50dB | 65/88 | 64/88 | 11/88 | 11/88 | 120.2 | 119.8 |
| 60dB | 49/88 | 49/88 | 5/88 | 9/88 | 79.8 | 80.6 |
| 70dB | 31/88 | 35/88 | 1/88 | 1/88 | 41.6 | 44.0 |

### Por intensidade: B2 b_vazio_menor_dif_det_ge_bruto vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 40/88 | 8/88 | 6/88 | 206.6 | 209.5 |
| 40dB | 66/88 | 65/88 | 6/88 | 6/88 | 188.9 | 194.0 |
| 50dB | 65/88 | 64/88 | 11/88 | 11/88 | 120.2 | 119.8 |
| 60dB | 49/88 | 49/88 | 5/88 | 9/88 | 79.8 | 80.6 |
| 70dB | 31/88 | 35/88 | 1/88 | 1/88 | 41.6 | 44.0 |

Tempo de execucao: 139 s.
