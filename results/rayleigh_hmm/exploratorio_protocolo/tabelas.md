# Exploratório: protocolo concatenado ESP→30→…→70 dB (IN-SAMPLE, vazamento direto)

Janelas de 10 épocas dentro de cada gravação; Z com cap de M épocas; HMM com Viterbi causal sobre o protocolo inteiro; regras aplicadas por segmento. Sem herança: cada intensidade é decidida pela sua própria gravação; o estado do HMM atravessa as gravações como memória, mas a decisão é o estado atual. Percentuais: detecção geral sobre 440; cada intensidade sobre 88; FP lateral sobre 528; FP ESP e protocolos sobre 88. Teto de FP lateral = 45/528 (bruto_referencia). Exploratório, não é validação.

bruto_referencia conferido contra o pipeline atual: True

## Diagnóstico por intensidade

Z final = fim da gravação (M=None) nos alvos; p = exp(−Z).

| Intensidade | Épocas/gravação (mín/med/máx) | Janelas de 10 (mín/med/máx) | Mediana Z final | Alvos com p≤0,05 no fim |
|---|---:|---:|---:|---:|
| ESP | 30/120/300 | 3/12/30 | 0,56 | 6,8% |
| 30dB | 480/480/500 | 48/48/50 | 1,69 | 30,7% |
| 40dB | 480/480/480 | 48/48/48 | 3,51 | 61,4% |
| 50dB | 300/300/301 | 30/30/30 | 5,42 | 69,3% |
| 60dB | 120/180/180 | 12/18/18 | 3,45 | 58,0% |
| 70dB | 60/60/60 | 6/6/6 | 2,94 | 47,7% |

## M = 30

| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| bruto_otimizado | 36,6% | 23,9% | 42,0% | 43,2% | 39,8% | 34,1% | 6,8% | 5,7% | 83,0% | 38,6% | 116 / 50 |
| hmm_estado | 57,7% | 29,5% | 47,7% | 64,8% | 70,5% | 76,1% | 8,5% | 5,7% | 85,2% | 29,5% | 85 / 30 |

Configurações escolhidas:

- bruto_otimizado: `{"alfa": 0.05, "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.4, "consecutivas": 3}}`
- hmm_estado: `{"pi": [0.9, 0.1], "A": [[0.98, 0.02], [0.01, 0.99]], "regra": {"familia": "fracao_e_consecutivas", "fracao": 0.1, "consecutivas": 3}}`

Ausente: k=0.940, μ=0.951, n=1160. Presente: k~Gamma(20.05,17.18) (média 1.17), ρ~Gamma(0.261,9.89) (média 0.0264), na_borda=True, hiper_no_limite=False, disp. inícios=3.8e-07.

McNemar pareado vs bruto_referencia (só_ref / só_método, p); tempo: Wilcoxon (pares, mediana ref/método, p):

| Método | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |
|---|---|---|---|---|---|---|
| bruto_otimizado | 100/15, p=1.35e-16 | 29/20, p=0.253 | 1/3, p=0.625 | 12/3, p=0.0352 | 14/13, p=1 | 146, 60/50, p=0.25 |
| hmm_estado | 67/75, p=0.557 | 33/33, p=1 | 1/3, p=0.625 | 9/2, p=0.0654 | 22/13, p=0.175 | 179, 70/30, p=0.0202 |

Primeira intensidade detectada por protocolo-alvo (% de 88), consistência e tempo de protocolo (desde início do 30 dB):

| Método | ESP | 30dB | 40dB | 50dB | 60dB | 70dB | nenhuma | consistentes/com det | tempo protocolo méd/med (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% | 20/82 | 523 / 530 |
| bruto_otimizado | 5,7% | 22,7% | 27,3% | 14,8% | 8,0% | 5,7% | 15,9% | 17/73 | 738 / 760 |
| hmm_estado | 5,7% | 23,9% | 30,7% | 15,9% | 6,8% | 2,3% | 14,8% | 56/75 | 676 / 660 |

Tempo de execução: 27 s.

## M = 60

| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| bruto_otimizado | 46,8% | 38,6% | 55,7% | 58,0% | 47,7% | 34,1% | 8,5% | 5,7% | 90,9% | 45,5% | 133 / 95 |
| hmm_estado | 63,6% | 37,5% | 61,4% | 69,3% | 71,6% | 78,4% | 8,5% | 4,5% | 89,8% | 30,7% | 86 / 30 |

Configurações escolhidas:

- bruto_otimizado: `{"alfa": 0.01, "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.4, "consecutivas": 1}}`
- hmm_estado: `{"pi": [0.99, 0.01], "A": [[0.995, 0.005], [0.005, 0.995]], "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.3, "consecutivas": 8}}`

Ausente: k=0.957, μ=0.936, n=1160. Presente: k~Gamma(4.10,2.47) (média 1.66), ρ~Gamma(0.304,10.40) (média 0.0292), na_borda=True, hiper_no_limite=False, disp. inícios=4.6e-06.

McNemar pareado vs bruto_referencia (só_ref / só_método, p); tempo: Wilcoxon (pares, mediana ref/método, p):

| Método | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |
|---|---|---|---|---|---|---|
| bruto_otimizado | 68/28, p=5.46e-05 | 25/25, p=1 | 1/3, p=0.625 | 5/3, p=0.727 | 17/22, p=0.522 | 178, 80/80, p=0.249 |
| hmm_estado | 35/69, p=0.00111 | 25/25, p=1 | 0/1, p=1 | 5/2, p=0.453 | 16/8, p=0.152 | 211, 90/30, p=0.0519 |

Primeira intensidade detectada por protocolo-alvo (% de 88), consistência e tempo de protocolo (desde início do 30 dB):

| Método | ESP | 30dB | 40dB | 50dB | 60dB | 70dB | nenhuma | consistentes/com det | tempo protocolo méd/med (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% | 20/82 | 523 / 530 |
| bruto_otimizado | 5,7% | 37,5% | 30,7% | 12,5% | 4,5% | 1,1% | 8,0% | 18/80 | 594 / 550 |
| hmm_estado | 4,5% | 33,0% | 35,2% | 11,4% | 3,4% | 2,3% | 10,2% | 55/79 | 586 / 510 |

Tempo de execução: 27 s.

## M = 120

| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| bruto_otimizado | 54,8% | 45,5% | 63,6% | 67,0% | 52,3% | 45,5% | 7,6% | 9,1% | 94,3% | 39,8% | 121 / 70 |
| hmm_estado | 66,1% | 44,3% | 68,2% | 73,9% | 77,3% | 67,0% | 8,5% | 0,0% | 93,2% | 30,7% | 114 / 40 |

Configurações escolhidas:

- bruto_otimizado: `{"alfa": 0.02, "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.1, "consecutivas": 2}}`
- hmm_estado: `{"pi": [0.9999, 0.0001], "A": [[0.9999, 0.0001], [0.01, 0.99]], "regra": {"familia": "consecutivas", "consecutivas": 4}}`

Ausente: k=0.924, μ=0.897, n=1160. Presente: k~Gamma(1.83,0.66) (média 2.77), ρ~Gamma(0.316,10.43) (média 0.0302), na_borda=True, hiper_no_limite=False, disp. inícios=1.4e-06.

McNemar pareado vs bruto_referencia (só_ref / só_método, p); tempo: Wilcoxon (pares, mediana ref/método, p):

| Método | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |
|---|---|---|---|---|---|---|
| bruto_otimizado | 36/31, p=0.625 | 21/16, p=0.511 | 0/5, p=0.0625 | 3/4, p=1 | 9/9, p=1 | 210, 80/65, p=0.00027 |
| hmm_estado | 21/66, p=1.41e-06 | 22/22, p=1 | 3/0, p=0.25 | 2/2, p=1 | 15/7, p=0.134 | 225, 90/40, p=0.714 |

Primeira intensidade detectada por protocolo-alvo (% de 88), consistência e tempo de protocolo (desde início do 30 dB):

| Método | ESP | 30dB | 40dB | 50dB | 60dB | 70dB | nenhuma | consistentes/com det | tempo protocolo méd/med (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% | 20/82 | 523 / 530 |
| bruto_otimizado | 9,1% | 40,9% | 25,0% | 15,9% | 3,4% | 0,0% | 5,7% | 24/83 | 546 / 520 |
| hmm_estado | 0,0% | 44,3% | 33,0% | 13,6% | 2,3% | 0,0% | 6,8% | 52/82 | 600 / 580 |

Tempo de execução: 28 s.

## M = 240

| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| bruto_otimizado | 53,6% | 43,2% | 59,1% | 68,2% | 52,3% | 45,5% | 8,0% | 6,8% | 93,2% | 42,0% | 117 / 70 |
| hmm_estado | 65,7% | 38,6% | 67,0% | 76,1% | 79,5% | 67,0% | 8,1% | 2,3% | 92,0% | 28,4% | 93 / 30 |

Configurações escolhidas:

- bruto_otimizado: `{"alfa": 0.02, "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.1, "consecutivas": 2}}`
- hmm_estado: `{"pi": [0.9999, 0.0001], "A": [[0.99999, 1e-05], [0.1, 0.9]], "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.1, "consecutivas": 1}}`

Ausente: k=0.901, μ=0.872, n=1160. Presente: k~Gamma(1.51,0.39) (média 3.90), ρ~Gamma(0.308,10.12) (média 0.0305), na_borda=True, hiper_no_limite=False, disp. inícios=1.4e-06.

McNemar pareado vs bruto_referencia (só_ref / só_método, p); tempo: Wilcoxon (pares, mediana ref/método, p):

| Método | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |
|---|---|---|---|---|---|---|
| bruto_otimizado | 28/18, p=0.184 | 19/16, p=0.736 | 0/3, p=0.25 | 3/3, p=1 | 8/10, p=0.815 | 218, 90/75, p=0.0769 |
| hmm_estado | 18/61, p=1.27e-06 | 19/17, p=0.868 | 1/0, p=1 | 2/1, p=1 | 12/2, p=0.0129 | 228, 90/30, p=0.00436 |

Primeira intensidade detectada por protocolo-alvo (% de 88), consistência e tempo de protocolo (desde início do 30 dB):

| Método | ESP | 30dB | 40dB | 50dB | 60dB | 70dB | nenhuma | consistentes/com det | tempo protocolo méd/med (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% | 20/82 | 523 / 530 |
| bruto_otimizado | 6,8% | 40,9% | 28,4% | 12,5% | 3,4% | 1,1% | 6,8% | 24/82 | 555 / 525 |
| hmm_estado | 2,3% | 36,4% | 35,2% | 12,5% | 4,5% | 1,1% | 8,0% | 53/81 | 607 / 610 |

Tempo de execução: 28 s.

## M = None

| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| bruto_otimizado | 55,9% | 44,3% | 71,6% | 71,6% | 53,4% | 38,6% | 8,5% | 3,4% | 93,2% | 39,8% | 133 / 95 |
| hmm_estado | 64,8% | 42,0% | 71,6% | 76,1% | 77,3% | 56,8% | 8,1% | 4,5% | 90,9% | 27,3% | 90 / 30 |

Configurações escolhidas:

- bruto_otimizado: `{"alfa": 0.05, "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.7, "consecutivas": 3}}`
- hmm_estado: `{"pi": [0.99, 0.01], "A": [[0.9999, 0.0001], [0.2, 0.8]], "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.1, "consecutivas": 1}}`

Ausente: k=0.914, μ=0.866, n=1160. Presente: k~Gamma(1.46,0.32) (média 4.59), ρ~Gamma(0.289,9.48) (média 0.0305), na_borda=True, hiper_no_limite=False, disp. inícios=1.8e+01.

McNemar pareado vs bruto_referencia (só_ref / só_método, p); tempo: Wilcoxon (pares, mediana ref/método, p):

| Método | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |
|---|---|---|---|---|---|---|
| bruto_otimizado | 0/0, p=1 | 0/0, p=1 | 0/0, p=1 | 0/0, p=1 | 0/0, p=1 | 246, 95/95, p=0.0339 |
| hmm_estado | 13/52, p=1.17e-06 | 19/17, p=0.868 | 0/1, p=1 | 2/0, p=0.5 | 11/0, p=0.000977 | 233, 90/30, p=0.00135 |

Primeira intensidade detectada por protocolo-alvo (% de 88), consistência e tempo de protocolo (desde início do 30 dB):

| Método | ESP | 30dB | 40dB | 50dB | 60dB | 70dB | nenhuma | consistentes/com det | tempo protocolo méd/med (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bruto_referencia | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% | 20/82 | 523 / 530 |
| bruto_otimizado | 3,4% | 42,0% | 36,4% | 8,0% | 3,4% | 0,0% | 6,8% | 20/82 | 523 / 530 |
| hmm_estado | 4,5% | 37,5% | 31,8% | 11,4% | 4,5% | 1,1% | 9,1% | 46/80 | 576 / 570 |

Tempo de execução: 27 s.

Tempo total: 168 s.
