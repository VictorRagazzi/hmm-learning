# Busca exploratoria hmm_viterbi (in-sample, 11 participantes)

Candidatos avaliados: 186624; viaveis (FP lateral <= 0.07): 31124; com dif. pareada <= 0: 0; pontos de Pareto (dedupe): 9.

bruto_referencia: det 252/440, FP lat 35/528, FP ESP 4/88, tempo medio/mediano 134.7/100.0.

## Vencedores

| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | McNemar so_bruto-so_hmm (alvos / laterais / ESP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_grade_original | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 252/440 (57.3%) | 36/528 | 5/88 | 132.3 / 85.0 | 244 | 53/115/76 | +2.7 | 2.5e-06 | 8-8 p=1 / 7-8 p=1 / 0-1 p=1 |
| a_grade_estendida | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 252/440 (57.3%) | 36/528 | 5/88 | 132.3 / 85.0 | 244 | 53/115/76 | +2.7 | 2.5e-06 | 8-8 p=1 / 7-8 p=1 / 0-1 p=1 |
| b_vazio_menor_dif_det_ge_bruto | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 252/440 (57.3%) | 36/528 | 5/88 | 132.3 / 85.0 | 244 | 53/115/76 | +2.7 | 2.5e-06 | 8-8 p=1 / 7-8 p=1 / 0-1 p=1 |

## Fronteira de Pareto (deteccao x tempo medio)

| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | McNemar so_bruto-so_hmm (alvos / laterais / ESP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 252/440 (57.3%) | 36/528 | 5/88 | 132.3 / 85.0 | 244 | 53/115/76 | +2.7 | 2.5e-06 | 8-8 p=1 / 7-8 p=1 / 0-1 p=1 |
| P2 | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=3 (fracao_ou_consecutivas) | 250/440 (56.8%) | 36/528 | 4/88 | 132.2 / 80.0 | 242 | 51/120/71 | +4.5 | 5.1e-08 | 10-8 p=0.81 / 7-8 p=1 / 1-1 p=1 |
| P3 | 0.0001 | 0.005 | 0.3 | consecutivas=1 (consecutivas) | 249/440 (56.6%) | 36/528 | 5/88 | 132.2 / 100.0 | 243 | 58/99/86 | +2.1 | 0.0035 | 9-6 p=0.61 / 8-9 p=1 / 0-1 p=1 |
| P4 | 0.1 | 0.01 | 0.5 | consecutivas=1 (consecutivas) | 244/440 (55.5%) | 36/528 | 5/88 | 128.1 / 80.0 | 236 | 66/96/74 | +3.4 | 0.00087 | 16-8 p=0.15 / 8-9 p=1 / 1-2 p=1 |
| P5 | 0.2 | 0.001 | 0.5 | fracao=0.2 (fracao) | 215/440 (48.9%) | 34/528 | 8/88 | 125.2 / 70.0 | 206 | 74/98/34 | +18.7 | 2.6e-07 | 46-9 p=4.3e-07 / 12-11 p=1 / 1-5 p=0.22 |
| P6 | 0.2 | 0.0001 | 0.5 | fracao=0.2 (fracao) | 210/440 (47.7%) | 34/528 | 8/88 | 124.0 / 70.0 | 201 | 74/93/34 | +21.2 | 1.3e-06 | 51-9 p=3.1e-08 / 12-11 p=1 / 1-5 p=0.22 |
| P7 | 0.3 | 1e-05 | 0.5 | fracao=0.4, consecutivas=3 (fracao_e_consecutivas) | 186/440 (42.3%) | 35/528 | 10/88 | 121.0 / 60.0 | 177 | 18/110/49 | +34.1 | 6.3e-12 | 75-9 p=4.3e-14 / 15-15 p=1 / 1-7 p=0.07 |
| P8 | 0.3 | 0.0001 | 0.5 | fracao=0.5 (fracao) | 179/440 (40.7%) | 36/528 | 10/88 | 119.4 / 60.0 | 170 | 48/87/35 | +34.5 | 1.6e-07 | 82-9 p=7.1e-16 / 18-19 p=1 / 1-7 p=0.07 |
| P9 | 0.3 | 1e-05 | 0.5 | fracao=0.7 (fracao) | 118/440 (26.8%) | 23/528 | 5/88 | 118.6 / 100.0 | 114 | 9/86/19 | +50.5 | 5.7e-11 | 138-4 p=6e-36 / 21-9 p=0.043 / 2-3 p=1 |

### Por intensidade: a_grade_estendida vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 40/88 | 8/88 | 6/88 | 206.6 | 205.5 |
| 40dB | 66/88 | 65/88 | 6/88 | 6/88 | 188.9 | 186.2 |
| 50dB | 65/88 | 63/88 | 11/88 | 11/88 | 120.2 | 117.9 |
| 60dB | 49/88 | 50/88 | 5/88 | 9/88 | 79.8 | 81.2 |
| 70dB | 31/88 | 34/88 | 1/88 | 1/88 | 41.6 | 45.0 |

### Por intensidade: b_vazio_menor_dif_det_ge_bruto vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 40/88 | 8/88 | 6/88 | 206.6 | 205.5 |
| 40dB | 66/88 | 65/88 | 6/88 | 6/88 | 188.9 | 186.2 |
| 50dB | 65/88 | 63/88 | 11/88 | 11/88 | 120.2 | 117.9 |
| 60dB | 49/88 | 50/88 | 5/88 | 9/88 | 79.8 | 81.2 |
| 70dB | 31/88 | 34/88 | 1/88 | 1/88 | 41.6 | 45.0 |

Tempo de execucao: 107 s.
