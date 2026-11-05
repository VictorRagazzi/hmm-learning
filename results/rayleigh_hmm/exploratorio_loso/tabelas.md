# Exploratorio LOSO (leave-one-subject-out), N = 11 participantes

Resultados FORA DA AMOSTRA por participante: tudo que e ajustado/escolhido usa so os outros 10; as decisoes sao do participante retirado, concatenadas nos 11 folds. Verdade-terreno: todos os alvos com estimulo = presente. bruto_referencia (alfa 0,05, 3 consecutivas) nao ajusta nada.

Grade HMM estendida: pi 12 x A(Aus->Pres) 12 x A(Pres->Aus) 9 x 144 regras; FP lateral de treino <= 0.07.

## Geral (pooled, fora da amostra)

| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |
|---|---|---|---|---|
| HMM-REAL | 250/440 (56.8%) | 39/528 (7.4%) | 5/88 (5.7%) | 134.3 / 90.0 |
| HMM-SINTETICO | 253/440 (57.5%) | 41/528 (7.8%) | 5/88 (5.7%) | 131.9 / 90.0 |
| bruto_otimizado_LOSO | 252/440 (57.3%) | 35/528 (6.6%) | 4/88 (4.5%) | 136.7 / 100.0 |
| bruto_referencia | 252/440 (57.3%) | 35/528 (6.6%) | 4/88 (4.5%) | 134.7 / 100.0 |
| HMM in-sample (referencia) | 252/440 (57.3%) | 36/528 | 5/88 | 132.3 / 85.0 |
| bruto_ref in-sample (referencia) | 252/440 (57.3%) | 35/528 | 4/88 | 134.7 / 100.0 |

## Testes pareados (A vs B; McNemar so_A-so_B; Wilcoxon nos alvos detectados por ambos)

| A vs B | McNemar alvos (so_A-so_B, p) | laterais | ESP | Wilcoxon tempo (n, A mais rapido/mais lento/igual, dif media A-B, p) |
|---|---|---|---|---|
| HMM-REAL vs bruto_referencia | 5-7, p=0.77 | 10-6, p=0.45 | 2-1, p=1 | n=245, 63/104/78, +3.7, p=0.0002 |
| HMM-SINTETICO vs bruto_referencia | 10-9, p=1 | 10-4, p=0.18 | 1-0, p=1 | n=243, 56/110/77, +2.6, p=0.00018 |
| bruto_otimizado_LOSO vs bruto_referencia | 0-0, p=1 | 0-0, p=1 | 0-0, p=1 | n=252, 2/18/232, +2.0, p=0.00027 |
| HMM-REAL vs HMM-SINTETICO | 3-6, p=0.51 | 4-6, p=0.75 | 1-1, p=1 | n=247, 69/55/123, +0.7, p=0.8 |

## Por intensidade (det n/88 / FP lat n/88 / tempo medio s)

| Intensidade | HMM-REAL det / FP lat / tempo | HMM-SINTETICO det / FP lat / tempo | bruto_otimizado_LOSO det / FP lat / tempo | bruto_referencia det / FP lat / tempo |
|---|---|---|---|---|
| 30dB | 41/88 / 7/88 / 211.5 | 41/88 / 8/88 / 197.3 | 41/88 / 8/88 / 211.5 | 41/88 / 8/88 / 206.6 |
| 40dB | 65/88 / 6/88 / 188.8 | 65/88 / 7/88 / 189.5 | 66/88 / 6/88 / 193.0 | 66/88 / 6/88 / 188.9 |
| 50dB | 63/88 / 10/88 / 118.4 | 63/88 / 11/88 / 117.3 | 65/88 / 11/88 / 120.8 | 65/88 / 11/88 / 120.2 |
| 60dB | 49/88 / 10/88 / 78.4 | 50/88 / 10/88 / 80.8 | 49/88 / 5/88 / 79.8 | 49/88 / 5/88 / 79.8 |
| 70dB | 32/88 / 3/88 / 41.6 | 34/88 / 1/88 / 44.7 | 31/88 / 1/88 / 41.6 | 31/88 / 1/88 / 41.6 |

## Por participante held-out (det n/40 / FP lat n/48 / FP ESP n/8)

| Participante | HMM-REAL det/FP lat/FP ESP | HMM-SINTETICO det/FP lat/FP ESP | bruto_otimizado_LOSO det/FP lat/FP ESP | bruto_referencia det/FP lat/FP ESP |
|---|---|---|---|---|
| Ab | 25/40 / 6/48 / 0/8 | 27/40 / 7/48 / 0/8 | 26/40 / 7/48 / 0/8 | 26/40 / 7/48 / 0/8 |
| An | 28/40 / 8/48 / 0/8 | 29/40 / 8/48 / 1/8 | 28/40 / 8/48 / 1/8 | 28/40 / 8/48 / 1/8 |
| Bb | 20/40 / 3/48 / 0/8 | 20/40 / 3/48 / 0/8 | 21/40 / 4/48 / 0/8 | 21/40 / 4/48 / 0/8 |
| Er | 28/40 / 5/48 / 2/8 | 27/40 / 5/48 / 2/8 | 28/40 / 2/48 / 1/8 | 28/40 / 2/48 / 1/8 |
| Lu | 18/40 / 5/48 / 1/8 | 18/40 / 3/48 / 0/8 | 19/40 / 4/48 / 0/8 | 19/40 / 4/48 / 0/8 |
| Qu | 22/40 / 3/48 / 0/8 | 23/40 / 3/48 / 0/8 | 22/40 / 2/48 / 0/8 | 22/40 / 2/48 / 0/8 |
| Sa | 34/40 / 3/48 / 0/8 | 33/40 / 2/48 / 0/8 | 33/40 / 2/48 / 0/8 | 33/40 / 2/48 / 0/8 |
| So | 24/40 / 0/48 / 0/8 | 24/40 / 3/48 / 0/8 | 23/40 / 1/48 / 0/8 | 23/40 / 1/48 / 0/8 |
| Ti | 12/40 / 3/48 / 1/8 | 12/40 / 3/48 / 1/8 | 13/40 / 3/48 / 1/8 | 13/40 / 3/48 / 1/8 |
| Vi | 15/40 / 1/48 / 1/8 | 16/40 / 1/48 / 1/8 | 15/40 / 1/48 / 1/8 | 15/40 / 1/48 / 1/8 |
| Wr | 24/40 / 2/48 / 0/8 | 24/40 / 3/48 / 0/8 | 24/40 / 1/48 / 0/8 | 24/40 / 1/48 / 0/8 |

## Por fold: HMM

| Fold (h) | Braco | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | FP lat treino | FP lat held-out | Ausente k, mu | a_k, b_k | a_rho, b_rho | k medio | rho medio | borda/limite | convergiu | SNRs usadas (dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ab | REAL | 0.01 | 0.01 | 0.2 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.060 | 0.125 | 0.954, 0.875 | 1.33, 0.258 | 0.286, 8.05 | 5.14 | 0.0355 | True/False | True | - |
| Ab | SINTETICO | 0.01 | 0.01 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.146 | 0.954, 0.875 | 1.34, 0.256 | 0.624, 17.4 | 5.22 | 0.0359 | False/False | True | -23.6, -18.9, -14.8, -13.2, -11.3 |
| An | REAL | 0.2 | 0.2 | 0.5 | fracao=0.2, consecutivas=3 (fracao_ou_consecutivas) | 0.069 | 0.167 | 1.016, 0.923 | 1.38, 0.288 | 0.292, 9.14 | 4.79 | 0.0319 | True/False | True | - |
| An | SINTETICO | 0.01 | 0.005 | 0.2 | consecutivas=1 (consecutivas) | 0.069 | 0.167 | 1.016, 0.923 | 1.32, 0.249 | 0.631, 18 | 5.32 | 0.0351 | False/False | True | -24.6, -18.9, -15.4, -13.4, -11.9 |
| Bb | REAL | 0.05 | 0.02 | 0.3 | consecutivas=1 (consecutivas) | 0.067 | 0.062 | 0.957, 0.893 | 1.4, 0.273 | 0.318, 9.43 | 5.13 | 0.0337 | True/False | True | - |
| Bb | SINTETICO | 0.01 | 0.005 | 0.3 | consecutivas=1 (consecutivas) | 0.069 | 0.062 | 0.957, 0.893 | 1.2, 0.21 | 0.669, 18.5 | 5.7 | 0.0362 | False/False | True | -23.3, -18.4, -15.3, -12.9, -11.9 |
| Er | REAL | 0.0001 | 0.02 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.104 | 0.964, 0.844 | 1.35, 0.266 | 0.287, 8.32 | 5.05 | 0.0345 | True/False | True | - |
| Er | SINTETICO | 0.001 | 0.001 | 0.2 | consecutivas=1 (consecutivas) | 0.069 | 0.104 | 0.964, 0.844 | 1.42, 0.281 | 0.634, 17.6 | 5.04 | 0.036 | False/False | True | -23.9, -18.7, -15.1, -13.2, -11.3 |
| Lu | REAL | 0.05 | 0.02 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.104 | 0.957, 0.869 | 1.35, 0.265 | 0.308, 8.55 | 5.09 | 0.036 | True/False | True | - |
| Lu | SINTETICO | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.069 | 0.062 | 0.957, 0.869 | 1.23, 0.213 | 0.678, 18.1 | 5.78 | 0.0374 | False/False | True | -23.5, -18.3, -14.7, -13.3, -11.3 |
| Qu | REAL | 0.05 | 0.02 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.062 | 0.952, 0.869 | 1.33, 0.256 | 0.299, 8.28 | 5.21 | 0.0361 | True/False | True | - |
| Qu | SINTETICO | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.069 | 0.062 | 0.952, 0.869 | 1.17, 0.193 | 0.644, 17.7 | 6.05 | 0.0364 | False/False | True | -23.7, -18.8, -14.9, -13.1, -11.3 |
| Sa | REAL | 0.2 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.067 | 0.062 | 0.939, 0.894 | 1.56, 0.355 | 0.337, 14.3 | 4.4 | 0.0235 | True/False | True | - |
| Sa | SINTETICO | 0.1 | 0.05 | 0.5 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.042 | 0.939, 0.894 | 1.33, 0.282 | 0.6, 22.6 | 4.71 | 0.0266 | False/False | True | -23.4, -18.7, -16.1, -15.6, -13.2 |
| So | REAL | 0.05 | 0.02 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.069 | 0.000 | 0.955, 0.877 | 1.36, 0.267 | 0.297, 8.37 | 5.08 | 0.0355 | True/False | True | - |
| So | SINTETICO | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.069 | 0.062 | 0.955, 0.877 | 1.32, 0.224 | 0.701, 18.3 | 5.92 | 0.0383 | False/False | True | -23.6, -18.5, -14.8, -13.1, -11.4 |
| Ti | REAL | 0.005 | 0.005 | 0.2 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.062 | 0.951, 0.860 | 1.34, 0.259 | 0.321, 8.95 | 5.18 | 0.0358 | True/False | True | - |
| Ti | SINTETICO | 0.005 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.069 | 0.062 | 0.951, 0.860 | 1.16, 0.19 | 0.681, 17.6 | 6.07 | 0.0387 | False/False | True | -23.3, -18.4, -14.7, -13.0, -11.5 |
| Vi | REAL | 0.0001 | 0.02 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.065 | 0.021 | 0.959, 0.862 | 1.3, 0.246 | 0.312, 8.54 | 5.27 | 0.0366 | True/False | True | - |
| Vi | SINTETICO | 0.005 | 0.001 | 0.3 | consecutivas=1 (consecutivas) | 0.069 | 0.021 | 0.959, 0.862 | 1.2, 0.201 | 0.697, 19 | 5.96 | 0.0366 | False/False | True | -23.4, -18.3, -14.7, -13.0, -11.2 |
| Wr | REAL | 0.05 | 0.02 | 0.3 | consecutivas=1 (consecutivas) | 0.065 | 0.042 | 0.933, 0.889 | 1.38, 0.278 | 0.294, 8.52 | 4.97 | 0.0346 | True/False | True | - |
| Wr | SINTETICO | 0.005 | 0.0005 | 0.2 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.062 | 0.933, 0.889 | 1.22, 0.211 | 0.68, 17.9 | 5.78 | 0.0379 | False/False | True | -24.0, -18.6, -15.0, -13.4, -11.3 |

## Por fold: bruto otimizado

| Fold (h) | alfa | Regra | FP lat treino | FP lat held-out |
|---|---|---|---|---|
| Ab | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.058 | 0.146 |
| An | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.056 | 0.167 |
| Bb | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.065 | 0.083 |
| Er | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.069 | 0.042 |
| Lu | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.065 | 0.083 |
| Qu | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.069 | 0.042 |
| Sa | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.069 | 0.042 |
| So | 0.05 | fracao=0.2, consecutivas=3 (fracao_e_consecutivas) | 0.067 | 0.021 |
| Ti | 0.05 | fracao=0.8, consecutivas=3 (fracao_ou_consecutivas) | 0.067 | 0.062 |
| Vi | 0.05 | fracao=0.2, consecutivas=3 (fracao_e_consecutivas) | 0.067 | 0.021 |
| Wr | 0.05 | fracao=0.2, consecutivas=3 (fracao_e_consecutivas) | 0.067 | 0.021 |

## FP lateral held-out acima de 7% (folds)

- REAL: 4 de 11 (Ab, An, Er, Lu)
- SINTETICO: 3 de 11 (Ab, An, Er)
- bruto_otimizado_LOSO: 4 de 11 (Ab, An, Bb, Lu)

SNR com os 11 participantes: -23.6, -18.6, -15.0, -13.3, -11.5 dB (documento: -23.5, -18.5, -15.0, -13.5, -11.5).

Tempo de execucao: 206 s.
