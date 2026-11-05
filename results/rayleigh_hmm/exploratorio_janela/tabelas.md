# Exploratorio: HMM (Viterbi) com Rayleigh Z por janela (in-sample, 11 participantes)

## Sanidade

- Ausente (MLE, Z por janela, ESP alvo, n=1152): k=1.0315, mu=0.9446 (teoria 1, 1)
- Fracao de Z_ESP > -ln(0.05): 4.43%
- Janelas com n_w = 0 (todas as sequencias): 16; distribuicao de n_w nas ESP alvo: {0: 8, 3: 8, 7: 24, 8: 64, 9: 160, 10: 896}
- Presente sintetico: 440 sequencias, 5760 Z
- Hierarquico: k~Gamma(62.102, 53.137) media 1.169 dp 0.148; rho~Gamma(2.670, 98.3) media 0.0272 dp 0.0166; na_borda=False (hiper no limite=False, massa fora={'k': 1.0936854569466074e-103, 'rho': 1.0303449954059066e-22}, dispersao inicios=2.1e-06)

Candidatos: 186624; viaveis (FP lat <= 0.07): 102546; com dif. pareada <= 0: 2021; pontos Pareto (dedupe): 21.

bruto_referencia (Z acumulado): det 252/440, FP lat 35/528, FP ESP 4/88, tempo medio/mediano 134.7/100.0.

## Vencedores e referencia

| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | McNemar so_bruto-so_hmm (alvos / laterais / ESP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_chave_ordem | 0.2 | 0.2 | 0.2 | fracao=0.3, consecutivas=10 (fracao_ou_consecutivas) | 129/440 (29.3%) | 34/528 | 4/88 | 169.7 / 130.0 | 117 | 17/78/22 | +50.4 | 4.7e-10 | 135-12 p=1.6e-27 / 28-27 p=1 / 4-4 p=1 |
| b_nao_mais_lento | 0.5 | 0.001 | 0.1 | fracao=0.7, consecutivas=6 (fracao_e_consecutivas) | 89/440 (20.2%) | 35/528 | 7/88 | 70.7 / 60.0 | 83 | 15/61/7 | -2.8 | 0.0052 | 169-6 p=1.6e-42 / 29-29 p=1 / 3-6 p=0.51 |
| ref_parametros_vencedor_acumulado | 0.01 | 0.005 | 0.3 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 4/440 (0.9%) | 0/528 | 0/88 | 52.5 / 55.0 | 4 | 0/4/0 | +22.5 | 0.12 | 248-0 p=4.4e-75 / 35-0 p=5.8e-11 / 4-0 p=0.12 |

## Fronteira de Pareto (deteccao x tempo medio)

| Candidato | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | Deteccao | FP lat | FP ESP | Tempo medio/mediano (s) | Pares | HMM mais rapido/lento/igual | Dif. media (s) | Wilcoxon p | McNemar so_bruto-so_hmm (alvos / laterais / ESP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 0.2 | 0.2 | 0.2 | fracao=0.3, consecutivas=10 (fracao_ou_consecutivas) | 129/440 (29.3%) | 34/528 | 4/88 | 169.7 / 130.0 | 117 | 17/78/22 | +50.4 | 4.7e-10 | 135-12 p=1.6e-27 / 28-27 p=1 / 4-4 p=1 |
| P2 | 0.2 | 0.0005 | 0.001 | fracao=0.1, consecutivas=2 (fracao_ou_consecutivas) | 125/440 (28.4%) | 36/528 | 3/88 | 149.5 / 120.0 | 120 | 23/72/25 | +43.5 | 3.6e-07 | 132-5 p=4.5e-33 / 27-28 p=1 / 4-3 p=1 |
| P3 | 0.2 | 0.0005 | 0.001 | fracao=0.1, consecutivas=3 (fracao_ou_consecutivas) | 119/440 (27.0%) | 35/528 | 3/88 | 141.9 / 100.0 | 114 | 23/66/25 | +38.5 | 3.7e-06 | 138-5 p=8.6e-35 / 28-28 p=1 / 4-3 p=1 |
| P4 | 0.3 | 0.001 | 0.02 | fracao=0.4, consecutivas=6 (fracao_ou_consecutivas) | 118/440 (26.8%) | 36/528 | 4/88 | 111.8 / 50.0 | 109 | 23/54/32 | +28.2 | 0.00042 | 143-9 p=3.5e-32 / 29-30 p=1 / 3-3 p=1 |
| P5 | 0.3 | 0.005 | 0.02 | fracao=0.4, consecutivas=10 (fracao_ou_consecutivas) | 114/440 (25.9%) | 36/528 | 5/88 | 107.7 / 50.0 | 105 | 24/49/32 | +24.1 | 0.0016 | 147-9 p=2.8e-33 / 29-30 p=1 / 3-4 p=1 |
| P6 | 0.3 | 0.02 | 0.05 | fracao=0.4, consecutivas=4 (fracao_ou_consecutivas) | 113/440 (25.7%) | 34/528 | 3/88 | 106.8 / 70.0 | 105 | 22/51/32 | +22.9 | 0.00072 | 147-8 p=3.2e-34 / 29-28 p=1 / 4-3 p=1 |
| P7 | 0.3 | 0.02 | 0.05 | fracao=0.3, consecutivas=1 (fracao_e_consecutivas) | 112/440 (25.5%) | 34/528 | 4/88 | 91.6 / 60.0 | 104 | 34/40/30 | +11.6 | 0.075 | 148-8 p=1.7e-34 / 29-28 p=1 / 4-4 p=1 |
| P8 | 0.3 | 0.02 | 0.1 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 100/440 (22.7%) | 34/528 | 1/88 | 77.7 / 50.0 | 89 | 31/29/29 | +8.7 | 0.26 | 163-11 p=7.2e-36 / 32-31 p=1 / 4-1 p=0.38 |
| P9 | 0.3 | 0.02 | 0.1 | fracao=0.1, consecutivas=3 (fracao_ou_consecutivas) | 97/440 (22.0%) | 31/528 | 1/88 | 77.1 / 50.0 | 86 | 31/26/29 | +9.9 | 0.51 | 166-11 p=1.1e-36 / 32-28 p=0.7 / 4-1 p=0.38 |
| P10 | 0.3 | 0.02 | 0.1 | fracao=0.1 (fracao) | 96/440 (21.8%) | 31/528 | 1/88 | 74.8 / 45.0 | 85 | 31/25/29 | +7.5 | 0.64 | 167-11 p=5.8e-37 / 32-28 p=0.7 / 4-1 p=0.38 |
| P11 | 0.5 | 0.005 | 0.1 | fracao=0.8, consecutivas=6 (fracao_e_consecutivas) | 86/440 (19.5%) | 36/528 | 7/88 | 70.1 / 60.0 | 80 | 15/57/8 | -2.4 | 0.0039 | 172-6 p=2.2e-43 / 29-30 p=1 / 3-6 p=0.51 |
| P12 | 0.5 | 0.02 | 0.3 | consecutivas=1 (consecutivas) | 83/440 (18.9%) | 31/528 | 6/88 | 33.0 / 30.0 | 75 | 51/2/22 | -32.1 | 1.4e-08 | 177-8 p=1.2e-42 / 30-26 p=0.69 / 3-5 p=0.73 |
| P13 | 0.5 | 0.05 | 0.3 | fracao=0.6, consecutivas=1 (fracao_e_consecutivas) | 80/440 (18.2%) | 36/528 | 6/88 | 31.8 / 30.0 | 69 | 45/1/23 | -33.9 | 1.3e-08 | 183-11 p=2.3e-41 / 30-31 p=1 / 3-5 p=0.73 |
| P14 | 0.5 | 0.02 | 0.3 | fracao=0.6, consecutivas=1 (fracao_e_consecutivas) | 73/440 (16.6%) | 29/528 | 5/88 | 30.5 / 30.0 | 65 | 43/0/22 | -35.8 | 7.9e-09 | 187-8 p=1.9e-45 / 30-24 p=0.5 / 3-4 p=1 |
| P15 | 0.5 | 0.2 | 0.5 | fracao=0.6, consecutivas=1 (fracao_e_consecutivas) | 61/440 (13.9%) | 36/528 | 6/88 | 30.0 / 30.0 | 53 | 30/0/23 | -23.2 | 1.4e-06 | 199-8 p=7.4e-49 / 27-28 p=1 / 3-5 p=0.73 |

### Por intensidade: a_chave_ordem vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 18/88 | 8/88 | 7/88 | 206.6 | 275.6 |
| 40dB | 66/88 | 37/88 | 6/88 | 9/88 | 188.9 | 255.4 |
| 50dB | 65/88 | 33/88 | 11/88 | 9/88 | 120.2 | 144.2 |
| 60dB | 49/88 | 27/88 | 5/88 | 6/88 | 79.8 | 82.6 |
| 70dB | 31/88 | 14/88 | 1/88 | 1/88 | 41.6 | 35.0 |

### Por intensidade: b_nao_mais_lento vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 8/88 | 8/88 | 4/88 | 206.6 | 63.8 |
| 40dB | 66/88 | 12/88 | 6/88 | 2/88 | 188.9 | 88.3 |
| 50dB | 65/88 | 17/88 | 11/88 | 11/88 | 120.2 | 77.6 |
| 60dB | 49/88 | 29/88 | 5/88 | 7/88 | 79.8 | 69.7 |
| 70dB | 31/88 | 23/88 | 1/88 | 6/88 | 41.6 | 60.0 |

### Por intensidade: ref_parametros_vencedor_acumulado vs bruto_referencia

| Condicao | Det bruto | Det HMM | FP lat bruto | FP lat HMM | Tempo medio bruto | Tempo medio HMM |
|---|---|---|---|---|---|---|
| 30dB | 41/88 | 0/88 | 8/88 | 0/88 | 206.6 | - |
| 40dB | 66/88 | 0/88 | 6/88 | 0/88 | 188.9 | - |
| 50dB | 65/88 | 0/88 | 11/88 | 0/88 | 120.2 | - |
| 60dB | 49/88 | 2/88 | 5/88 | 0/88 | 79.8 | 55.0 |
| 70dB | 31/88 | 2/88 | 1/88 | 0/88 | 41.6 | 50.0 |

Tempo de execucao: 126 s.
