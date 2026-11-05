# Exploratorio LOSO (leave-one-subject-out), N = 11 participantes

Resultados FORA DA AMOSTRA por participante: tudo que e ajustado/escolhido usa so os outros 10; as decisoes sao do participante retirado, concatenadas nos 11 folds. Verdade-terreno: todos os alvos com estimulo = presente. bruto_referencia (alfa 0,05, 3 consecutivas) nao ajusta nada.

Grade HMM estendida: pi 12 x A(Aus->Pres) 12 x A(Pres->Aus) 9 x 144 regras; FP lateral de treino <= teto do fold (tabela abaixo).

## Geral (pooled, fora da amostra)

| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |
|---|---|---|---|---|
| HMM-REAL | 238/440 (54.1%) | 37/528 (7.0%) | 6/88 (6.8%) | 124.1 / 80.0 |
| HMM-SINTETICO | 235/440 (53.4%) | 50/528 (9.5%) | 6/88 (6.8%) | 129.6 / 80.0 |
| bruto_otimizado_LOSO | 244/440 (55.5%) | 47/528 (8.9%) | 3/88 (3.4%) | 131.6 / 90.0 |
| bruto_referencia | 246/440 (55.9%) | 45/528 (8.5%) | 3/88 (3.4%) | 132.9 / 95.0 |
| HMM in-sample (hmm_viterbi) (referencia) | 241/440 (54.8%) | 45/528 | 7/88 | 121.4 / 70.0 |
| bruto_ref in-sample (referencia) | 246/440 (55.9%) | 45/528 | 3/88 | 132.9 / 95.0 |

## Testes pareados (A vs B; McNemar so_A-so_B; Wilcoxon nos alvos detectados por ambos)

| A vs B | McNemar alvos (so_A-so_B, p) | laterais | ESP | Wilcoxon tempo (n, A mais rapido/mais lento/igual, dif media A-B, p) |
|---|---|---|---|---|
| HMM-REAL vs bruto_referencia | 6-14, p=0.12 | 5-13, p=0.096 | 3-0, p=0.25 | n=232, 99/63/70, +1.6, p=0.9 |
| HMM-SINTETICO vs bruto_referencia | 5-16, p=0.027 | 15-10, p=0.42 | 3-0, p=0.25 | n=230, 79/85/66, +3.1, p=0.06 |
| bruto_otimizado_LOSO vs bruto_referencia | 1-3, p=0.62 | 2-0, p=0.5 | 0-0, p=1 | n=243, 15/6/222, -0.5, p=0.079 |
| HMM-REAL vs HMM-SINTETICO | 6-3, p=0.51 | 1-14, p=0.00098 | 0-0, p=1 | n=232, 72/42/118, -2.1, p=0.0022 |

## Por intensidade (det n/88 / FP lat n/88 / tempo medio s)

| Intensidade | HMM-REAL det / FP lat / tempo | HMM-SINTETICO det / FP lat / tempo | bruto_otimizado_LOSO det / FP lat / tempo | bruto_referencia det / FP lat / tempo |
|---|---|---|---|---|
| 30dB | 37/88 / 6/88 / 213.0 | 38/88 / 7/88 / 218.9 | 38/88 / 10/88 / 221.6 | 39/88 / 9/88 / 218.2 |
| 40dB | 57/88 / 8/88 / 174.6 | 58/88 / 14/88 / 177.9 | 63/88 / 10/88 / 181.9 | 63/88 / 10/88 / 182.5 |
| 50dB | 62/88 / 8/88 / 116.5 | 61/88 / 11/88 / 116.4 | 62/88 / 13/88 / 115.3 | 63/88 / 12/88 / 117.5 |
| 60dB | 47/88 / 9/88 / 67.9 | 47/88 / 9/88 / 74.0 | 48/88 / 8/88 / 77.3 | 47/88 / 8/88 / 81.5 |
| 70dB | 35/88 / 2/88 / 36.9 | 31/88 / 3/88 / 39.7 | 33/88 / 1/88 / 41.5 | 34/88 / 1/88 / 42.6 |

## Por participante held-out (det n/40 / FP lat n/48 / FP ESP n/8)

| Participante | HMM-REAL det/FP lat/FP ESP | HMM-SINTETICO det/FP lat/FP ESP | bruto_otimizado_LOSO det/FP lat/FP ESP | bruto_referencia det/FP lat/FP ESP |
|---|---|---|---|---|
| Ab | 25/40 / 4/48 / 0/8 | 22/40 / 7/48 / 0/8 | 24/40 / 5/48 / 0/8 | 24/40 / 5/48 / 0/8 |
| An | 26/40 / 8/48 / 0/8 | 26/40 / 9/48 / 0/8 | 27/40 / 8/48 / 0/8 | 27/40 / 8/48 / 0/8 |
| Bb | 18/40 / 4/48 / 0/8 | 18/40 / 3/48 / 0/8 | 19/40 / 2/48 / 0/8 | 19/40 / 2/48 / 0/8 |
| Er | 24/40 / 4/48 / 2/8 | 25/40 / 4/48 / 2/8 | 27/40 / 5/48 / 1/8 | 27/40 / 5/48 / 1/8 |
| Lu | 18/40 / 3/48 / 2/8 | 18/40 / 4/48 / 2/8 | 20/40 / 7/48 / 0/8 | 20/40 / 7/48 / 0/8 |
| Qu | 22/40 / 0/48 / 0/8 | 20/40 / 1/48 / 0/8 | 21/40 / 1/48 / 0/8 | 21/40 / 1/48 / 0/8 |
| Sa | 33/40 / 3/48 / 0/8 | 33/40 / 4/48 / 0/8 | 33/40 / 5/48 / 0/8 | 33/40 / 5/48 / 0/8 |
| So | 22/40 / 3/48 / 0/8 | 24/40 / 6/48 / 0/8 | 21/40 / 3/48 / 0/8 | 23/40 / 1/48 / 0/8 |
| Ti | 17/40 / 5/48 / 0/8 | 16/40 / 5/48 / 0/8 | 16/40 / 4/48 / 0/8 | 16/40 / 4/48 / 0/8 |
| Vi | 10/40 / 1/48 / 2/8 | 10/40 / 2/48 / 2/8 | 12/40 / 3/48 / 2/8 | 12/40 / 3/48 / 2/8 |
| Wr | 23/40 / 2/48 / 0/8 | 23/40 / 5/48 / 0/8 | 24/40 / 4/48 / 0/8 | 24/40 / 4/48 / 0/8 |

## Por fold: HMM

| Fold (h) | Braco | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | FP lat treino | FP lat held-out | Ausente k, mu | a_k, b_k | a_rho, b_rho | k medio | rho medio | borda/limite | convergiu | SNRs usadas (dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ab | REAL | 0.3 | 0.1 | 0.5 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 0.067 | 0.083 | 0.921, 0.891 | 1.41, 0.298 | 0.277, 8.78 | 4.73 | 0.0316 | True/False | True | - |
| Ab | SINTETICO | 0.001 | 0.001 | 0.01 | consecutivas=1 (consecutivas) | 0.083 | 0.146 | 0.921, 0.891 | 1.3, 0.257 | 0.434, 9.97 | 5.06 | 0.0436 | False/False | True | -24.4, -20.8, -17.6, -13.7, -11.5 |
| An | REAL | 0.3 | 0.1 | 0.5 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 0.058 | 0.167 | 0.843, 0.871 | 1.5, 0.347 | 0.279, 9.64 | 4.31 | 0.0289 | True/False | True | - |
| An | SINTETICO | 0.1 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.071 | 0.188 | 0.843, 0.871 | 1.34, 0.283 | 0.387, 9.72 | 4.74 | 0.0398 | False/False | True | -26.5, -20.9, -19.0, -13.9, -12.3 |
| Bb | REAL | 0.3 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.083 | 0.083 | 0.912, 0.878 | 1.47, 0.321 | 0.306, 10.2 | 4.6 | 0.0299 | True/False | True | - |
| Bb | SINTETICO | 0.05 | 0.01 | 0.3 | consecutivas=1 (consecutivas) | 0.090 | 0.062 | 0.912, 0.878 | 1.3, 0.248 | 0.469, 10.2 | 5.26 | 0.0461 | False/False | True | -24.5, -20.2, -18.0, -13.5, -11.8 |
| Er | REAL | 0.2 | 0.05 | 0.5 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.067 | 0.083 | 0.934, 0.825 | 1.44, 0.311 | 0.271, 8.68 | 4.62 | 0.0313 | True/False | True | - |
| Er | SINTETICO | 0.1 | 0.02 | 0.5 | consecutivas=1 (consecutivas) | 0.069 | 0.083 | 0.934, 0.825 | 1.44, 0.304 | 0.433, 9.92 | 4.75 | 0.0437 | False/False | True | -24.9, -20.5, -17.9, -13.8, -11.8 |
| Lu | REAL | 0.3 | 0.05 | 0.5 | fracao=0.1, consecutivas=1 (fracao_e_consecutivas) | 0.079 | 0.062 | 0.918, 0.844 | 1.46, 0.318 | 0.297, 9.36 | 4.59 | 0.0317 | True/False | True | - |
| Lu | SINTETICO | 0.2 | 0.02 | 0.5 | fracao=0.1, consecutivas=4 (fracao_ou_consecutivas) | 0.077 | 0.083 | 0.918, 0.844 | 1.34, 0.267 | 0.465, 11.8 | 5.02 | 0.0395 | False/False | True | -24.8, -20.1, -17.6, -13.9, -11.8 |
| Qu | REAL | 0.3 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.083 | 0.000 | 0.909, 0.876 | 1.46, 0.316 | 0.281, 8.82 | 4.62 | 0.0319 | True/False | True | - |
| Qu | SINTETICO | 0.0001 | 0.02 | 0.3 | consecutivas=1 (consecutivas) | 0.092 | 0.021 | 0.909, 0.876 | 1.4, 0.284 | 0.432, 9.62 | 4.93 | 0.0449 | False/False | True | -25.0, -20.3, -18.0, -13.7, -11.7 |
| Sa | REAL | 0.3 | 0.1 | 0.5 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 0.075 | 0.062 | 0.893, 0.888 | 1.58, 0.38 | 0.31, 13.8 | 4.16 | 0.0225 | True/False | True | - |
| Sa | SINTETICO | 0.3 | 0.05 | 0.5 | consecutivas=1 (consecutivas) | 0.083 | 0.083 | 0.893, 0.888 | 1.48, 0.346 | 0.431, 12.3 | 4.29 | 0.0351 | False/False | True | -24.7, -20.2, -17.3, -16.2, -13.5 |
| So | REAL | 0.3 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.085 | 0.062 | 0.908, 0.867 | 1.46, 0.316 | 0.28, 8.82 | 4.62 | 0.0318 | True/False | True | - |
| So | SINTETICO | 0.0001 | 0.001 | 0.001 | consecutivas=1 (consecutivas) | 0.090 | 0.125 | 0.908, 0.867 | 1.35, 0.248 | 0.475, 10.7 | 5.44 | 0.0445 | False/False | True | -24.9, -20.0, -17.7, -13.7, -11.8 |
| Ti | REAL | 0.2 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.085 | 0.104 | 0.929, 0.864 | 1.44, 0.302 | 0.308, 9.55 | 4.76 | 0.0323 | True/False | True | - |
| Ti | SINTETICO | 0.05 | 0.05 | 0.5 | consecutivas=1 (consecutivas) | 0.073 | 0.104 | 0.929, 0.864 | 1.27, 0.234 | 0.48, 10.2 | 5.42 | 0.0473 | False/False | True | -24.5, -20.2, -17.4, -13.6, -11.8 |
| Vi | REAL | 0.2 | 0.05 | 0.5 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 0.085 | 0.021 | 0.938, 0.828 | 1.42, 0.293 | 0.306, 9.29 | 4.84 | 0.033 | True/False | True | - |
| Vi | SINTETICO | 0.1 | 0.02 | 0.5 | consecutivas=1 (consecutivas) | 0.085 | 0.042 | 0.938, 0.828 | 1.28, 0.234 | 0.496, 11.5 | 5.47 | 0.0431 | False/False | True | -24.4, -19.9, -17.4, -13.6, -11.5 |
| Wr | REAL | 0.3 | 0.1 | 0.5 | consecutivas=1 (consecutivas) | 0.069 | 0.042 | 0.953, 0.899 | 1.45, 0.31 | 0.282, 9.08 | 4.68 | 0.0311 | True/False | True | - |
| Wr | SINTETICO | 0.0001 | 0.1 | 0.5 | fracao=0.3, consecutivas=2 (fracao_ou_consecutivas) | 0.085 | 0.104 | 0.953, 0.899 | 1.29, 0.238 | 0.48, 10.3 | 5.43 | 0.0466 | False/False | True | -25.2, -20.3, -17.3, -14.1, -11.6 |

## Por fold: bruto otimizado

| Fold (h) | alfa | Regra | FP lat treino | FP lat held-out |
|---|---|---|---|---|
| Ab | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.083 | 0.104 |
| An | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.077 | 0.167 |
| Bb | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.090 | 0.042 |
| Er | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.083 | 0.104 |
| Lu | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.079 | 0.146 |
| Qu | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.092 | 0.021 |
| Sa | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.083 | 0.104 |
| So | 0.02 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 0.085 | 0.062 |
| Ti | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.085 | 0.083 |
| Vi | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.087 | 0.062 |
| Wr | 0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 0.085 | 0.083 |

## FP lateral held-out acima de 7% (folds; informativo, o teto real e por fold)

- REAL: 5 de 11 (Ab, An, Bb, Er, Ti)
- SINTETICO: 8 de 11 (Ab, An, Er, Lu, Sa, So, Ti, Wr)
- bruto_otimizado_LOSO: 7 de 11 (Ab, An, Er, Lu, Sa, Ti, Wr)

SNR com os 11 participantes: -24.9, -20.3, -17.7, -13.9, -11.9 dB (documento: -23.5, -18.5, -15.0, -13.5, -11.5).

Tempo de execucao: 210 s.

## Teto de FP lateral por fold (bruto_referencia no treino)

| Held-out | Teto | Contagem/total |
|---|---|---|
| Ab | 0.0833 (8.3%) | 40/480 |
| An | 0.0771 (7.7%) | 37/480 |
| Bb | 0.0896 (9.0%) | 43/480 |
| Er | 0.0833 (8.3%) | 40/480 |
| Lu | 0.0792 (7.9%) | 38/480 |
| Qu | 0.0917 (9.2%) | 44/480 |
| Sa | 0.0833 (8.3%) | 40/480 |
| So | 0.0917 (9.2%) | 44/480 |
| Ti | 0.0854 (8.5%) | 41/480 |
| Vi | 0.0875 (8.8%) | 42/480 |
| Wr | 0.0854 (8.5%) | 41/480 |
