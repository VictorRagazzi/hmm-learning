# LOSO, sem herança; exploratório; cada intensidade decidida pela própria gravação; estado do HMM atravessa as gravações dentro do protocolo

Percentuais: detecção geral sobre 440; cada intensidade sobre 88; FP lateral sobre 528; FP ESP e protocolos sobre 88. Tudo ajustado/escolhido (emissões, M, parâmetros, regra) usa só os 10 participantes de treino; teto de FP lateral no treino = FP lateral do bruto_referencia nos laterais de T. Viterbi roda por protocolo (um participante × frequência), sem vazamento entre participantes.

## Resultado fora da amostra (11 folds)

| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| rayleigh_loso | 55,5% | 43,2% | 71,6% | 70,5% | 54,5% | 37,5% | 8,9% | 3,4% | 93,2% | 40,9% | 132 / 90 |
| hmm_loso | 65,5% | 39,8% | 65,9% | 76,1% | 78,4% | 67,0% | 10,6% | 1,1% | 92,0% | 34,1% | 92 / 30 |

Primeira intensidade detectada por protocolo-alvo (% de 88):

| Método | ESP | 30dB | 40dB | 50dB | 60dB | 70dB | nenhuma |
|---|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% |
| rayleigh_loso | 3,4% | 40,9% | 37,5% | 8,0% | 3,4% | 0,0% | 6,8% |
| hmm_loso | 1,1% | 38,6% | 33,0% | 14,8% | 3,4% | 1,1% | 8,0% |

## Testes pareados

McNemar exato (só_A / só_B, p) e Wilcoxon de tempo nos pares detectados por ambos (n, mediana A/B, p).

| A vs B | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |
|---|---|---|---|---|---|---|
| rayleigh_loso vs hmm_loso | 21/65, p=2.15e-06 | 18/27, p=0.233 | 2/0, p=0.5 | 1/0, p=1 | 11/5, p=0.21 | 223, 90/30, p=0.0043 |
| bruto_referencia vs rayleigh_loso | 3/1, p=0.625 | 0/2, p=0.5 | 0/0, p=1 | 0/0, p=1 | 0/1, p=1 | 243, 90/90, p=0.0795 |
| bruto_referencia vs hmm_loso | 22/64, p=6.54e-06 | 17/28, p=0.135 | 2/0, p=0.5 | 1/0, p=1 | 10/5, p=0.302 | 224, 90/30, p=0.00219 |

## Frequência de escolha de M (folds, de 11)

| Método | M=30 | M=60 | M=120 | M=240 | M=None |
|---|---:|---:|---:|---:|---:|
| rayleigh_loso | 0 | 0 | 0 | 0 | 11 |
| hmm_loso | 0 | 0 | 5 | 4 | 2 |

## Por fold (participante retirado)

Det alvos /40, FP lat /48, FP ESP /8 do participante retirado; treino = valores em T.

| Fold | Método | M | Parâmetros | Regra | Teto | Treino det / FP lat | Retirado det / FP lat / FP ESP |
|---|---|---|---|---|---:|---|---|
| Ab | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 40 | 222 / 40 | 24 / 5 / 0 |
| Ab | hmm_loso | None | π=0.0001, A01=1e-05, A10=0.1 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 40 | 267 / 38 | 30 / 8 / 0 |
| An | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 37 | 219 / 37 | 27 / 8 / 0 |
| An | hmm_loso | 240 | π=0.001, A01=0.0001, A10=0.01 | fracao=0.4, consecutivas=3 (fracao_e_consecutivas) | 37 | 264 / 37 | 31 / 13 / 0 |
| Bb | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 43 | 227 / 43 | 19 / 2 / 0 |
| Bb | hmm_loso | 240 | π=0.0001, A01=1e-05, A10=0.1 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 43 | 266 / 40 | 23 / 1 / 0 |
| Er | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 40 | 219 / 40 | 27 / 5 / 1 |
| Er | hmm_loso | 120 | π=0.0001, A01=0.0001, A10=0.01 | fracao=0.2, consecutivas=6 (fracao_ou_consecutivas) | 40 | 272 / 40 | 32 / 11 / 1 |
| Lu | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 38 | 226 / 38 | 20 / 7 / 0 |
| Lu | hmm_loso | 120 | π=0.0001, A01=1e-05, A10=0.005 | consecutivas=1 (consecutivas) | 38 | 262 / 38 | 24 / 4 / 0 |
| Qu | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 44 | 225 / 44 | 21 / 1 / 0 |
| Qu | hmm_loso | 120 | π=0.0005, A01=0.0001, A10=0.01 | fracao=0.1, consecutivas=3 (fracao_ou_consecutivas) | 44 | 262 / 44 | 27 / 0 / 0 |
| Sa | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 40 | 213 / 40 | 33 / 5 / 0 |
| Sa | hmm_loso | 240 | π=0.01, A01=0.001, A10=0.2 | fracao=0.1 (fracao) | 40 | 255 / 40 | 34 / 4 / 0 |
| So | rayleigh_loso | None | α=0.02 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 44 | 223 / 41 | 21 / 3 / 0 |
| So | hmm_loso | 120 | π=0.1, A01=0.0005, A10=0.005 | fracao=0.8, consecutivas=10 (fracao_ou_consecutivas) | 44 | 269 / 44 | 22 / 1 / 0 |
| Ti | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 41 | 230 / 41 | 16 / 4 / 0 |
| Ti | hmm_loso | 240 | π=0.0001, A01=1e-05, A10=0.1 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 41 | 269 / 39 | 19 / 4 / 0 |
| Vi | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 42 | 234 / 42 | 12 / 3 / 2 |
| Vi | hmm_loso | 120 | π=0.0001, A01=0.0001, A10=0.005 | fracao=0.6, consecutivas=6 (fracao_ou_consecutivas) | 42 | 278 / 42 | 17 / 5 / 0 |
| Wr | rayleigh_loso | None | α=0.05 | fracao=0.7, consecutivas=3 (fracao_ou_consecutivas) | 41 | 222 / 41 | 24 / 4 / 0 |
| Wr | hmm_loso | None | π=0.0001, A01=1e-05, A10=0.1 | fracao=0.1, consecutivas=1 (fracao_ou_consecutivas) | 41 | 268 / 41 | 29 / 5 / 0 |

## Avisos do ajuste hierárquico (Presente, por fold e M)

Folds (de 11) com dispersão relativa dos inícios > 1e-3, por M: M=30: 0, M=60: 0, M=120: 0, M=240: 0, M=None: 6.

Folds com na_borda ou hiper_no_limite, por M: M=30: 11, M=60: 11, M=120: 9, M=240: 11, M=None: 11.

| M | Fold | na_borda | hiper_no_limite | dispersão inícios | Ausente k, μ | a_k, b_k |
|---|---|---|---|---:|---|---|
| 120 | Ab | True | False | 2.9e-06 | 0.934, 0.925 | 1.75, 0.619 |
| 120 | An | True | False | 1.9e-06 | 0.887, 0.887 | 1.92, 0.729 |
| 120 | Bb | True | False | 1.9e-06 | 0.924, 0.912 | 1.87, 0.682 |
| 120 | Er | True | False | 2.0e-06 | 0.943, 0.859 | 1.81, 0.649 |
| 120 | Lu | True | False | 1.8e-06 | 0.927, 0.876 | 1.84, 0.664 |
| 120 | Qu | True | False | 1.6e-06 | 0.921, 0.910 | 1.79, 0.632 |
| 120 | Sa | False | False | 1.6e-06 | 0.905, 0.921 | 2.2, 0.893 |
| 120 | So | True | False | 1.7e-06 | 0.918, 0.899 | 1.76, 0.614 |
| 120 | Ti | True | False | 5.4e-07 | 0.933, 0.887 | 1.78, 0.62 |
| 120 | Vi | False | False | 2.4e-06 | 0.947, 0.860 | 1.69, 0.57 |
| 120 | Wr | True | False | 3.2e-07 | 0.926, 0.933 | 1.83, 0.656 |
| 240 | Ab | True | False | 1.8e-06 | 0.908, 0.898 | 1.46, 0.362 |
| 240 | An | True | False | 1.6e-06 | 0.843, 0.871 | 1.58, 0.431 |
| 240 | Bb | True | False | 7.8e-07 | 0.899, 0.884 | 1.55, 0.405 |
| 240 | Er | True | False | 3.0e-06 | 0.919, 0.831 | 1.49, 0.378 |
| 240 | Lu | True | False | 2.3e-06 | 0.905, 0.850 | 1.5, 0.384 |
| 240 | Qu | True | False | 1.1e-06 | 0.896, 0.882 | 1.51, 0.382 |
| 240 | Sa | True | False | 4.6e-07 | 0.881, 0.894 | 1.69, 0.485 |
| 240 | So | True | False | 1.5e-06 | 0.895, 0.873 | 1.5, 0.377 |
| 240 | Ti | True | False | 5.2e-07 | 0.914, 0.870 | 1.47, 0.361 |
| 240 | Vi | True | False | 1.5e-06 | 0.924, 0.834 | 1.44, 0.347 |
| 240 | Wr | True | False | 6.4e-07 | 0.937, 0.905 | 1.5, 0.378 |
| 30 | Ab | True | False | 1.6e-06 | 0.991, 0.984 | 17.5, 14.9 |
| 30 | An | True | False | 3.4e-06 | 0.934, 0.928 | 22.9, 20.1 |
| 30 | Bb | True | False | 7.9e-07 | 0.922, 0.958 | 22.9, 19.8 |
| 30 | Er | True | False | 8.3e-06 | 0.927, 0.941 | 18.5, 15.7 |
| 30 | Lu | True | False | 2.4e-06 | 0.943, 0.940 | 20, 17.1 |
| 30 | Qu | True | False | 2.1e-06 | 0.929, 0.961 | 19.6, 16.8 |
| 30 | Sa | True | False | 1.6e-06 | 0.928, 0.969 | 25.1, 22 |
| 30 | So | True | False | 1.3e-06 | 0.935, 0.954 | 19.1, 16.2 |
| 30 | Ti | True | False | 1.8e-06 | 0.950, 0.955 | 18, 15.2 |
| 30 | Vi | True | False | 1.5e-06 | 0.942, 0.927 | 18.7, 15.7 |
| 30 | Wr | True | False | 2.3e-06 | 0.950, 0.940 | 19.7, 16.8 |
| 60 | Ab | True | False | 3.0e-07 | 0.997, 0.971 | 3.89, 2.32 |
| 60 | An | True | False | 4.0e-07 | 0.934, 0.913 | 4.51, 2.84 |
| 60 | Bb | True | False | 1.8e-06 | 0.946, 0.952 | 4.38, 2.69 |
| 60 | Er | True | False | 3.2e-06 | 0.951, 0.921 | 4.2, 2.52 |
| 60 | Lu | True | False | 7.4e-07 | 0.961, 0.917 | 4.12, 2.49 |
| 60 | Qu | True | False | 1.8e-06 | 0.953, 0.946 | 3.8, 2.25 |
| 60 | Sa | True | False | 5.0e-06 | 0.931, 0.957 | 4.87, 3.13 |
| 60 | So | True | False | 1.8e-06 | 0.951, 0.939 | 3.87, 2.27 |
| 60 | Ti | True | False | 3.3e-06 | 0.982, 0.928 | 3.85, 2.26 |
| 60 | Vi | True | False | 5.2e-07 | 0.981, 0.901 | 3.67, 2.11 |
| 60 | Wr | True | False | 4.0e-06 | 0.943, 0.949 | 4.15, 2.49 |
| None | Ab | True | False | 1.9e+01 | 0.921, 0.891 | 1.41, 0.298 |
| None | An | True | False | 1.0e-06 | 0.843, 0.871 | 1.5, 0.347 |
| None | Bb | True | False | 1.8e+01 | 0.912, 0.878 | 1.47, 0.321 |
| None | Er | True | False | 2.7e-06 | 0.934, 0.825 | 1.44, 0.311 |
| None | Lu | True | False | 1.8e+01 | 0.918, 0.844 | 1.46, 0.318 |
| None | Qu | True | False | 1.6e-06 | 0.909, 0.876 | 1.46, 0.316 |
| None | Sa | True | False | 3.5e-07 | 0.893, 0.888 | 1.58, 0.38 |
| None | So | True | False | 2.4e-06 | 0.908, 0.867 | 1.46, 0.316 |
| None | Ti | True | False | 1.9e+01 | 0.929, 0.864 | 1.44, 0.302 |
| None | Vi | True | False | 1.9e+01 | 0.938, 0.828 | 1.42, 0.293 |
| None | Wr | True | False | 1.8e+01 | 0.953, 0.899 | 1.45, 0.31 |

Tempo total: 374 s.
