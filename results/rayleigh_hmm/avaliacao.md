# Avaliação complementar — Rayleigh × HMM

Gerado por `src/rayleigh_hmm/avaliacao.py`. Usa as configurações escolhidas por `experimento.py` (sem refazer a busca) e as mesmas emissões. Validação in-sample.

Reprodução das gravações completas: conferida (detecção, FP lateral, FP alvo ESP e tempo médio idênticos a `resultados.json`).

## A. Detecção × FP lateral

Cada curva varre um único controle (α no bruto e na referência; limiar da posteriori no HMM), com o resto da configuração escolhida fixo. A curva parte de (0, 0); o FP é o lateral global (528 sequências) e a detecção, a dos 440 alvos de estímulo. Acima do maior FP alcançado, a detecção interpolada repete o último valor. A referência é um teste único com todas as épocas completas da gravação (sem regra sequencial); o tempo de exame é a gravação inteira.

| Método | Detecção a FP 5% | Detecção a FP 7% | Detecção a FP 10% | Área parcial (FP ≤ 10%) |
|---|---:|---:|---:|---:|
| Rayleigh, teste único no fim | 55.5% | 58.4% | 62.9% | 0.533 |
| Rayleigh otimizado | 49.7% | 54.5% | 57.2% | 0.471 |
| HMM (preditiva) | 49.9% | 54.3% | 55.5% | 0.477 |
| HMM (MAP) | 49.5% | 54.5% | 55.7% | 0.476 |

## B. Detecção acumulada × tempo de exame

Fração dos alvos (denominador = todos os alvos do painel, detectados ou não) já detectados até t segundos, com as configurações escolhidas. FP lateral: laterais de todas as condições. Durações máximas de gravação: 30dB = 500 s, 40dB = 480 s, 50dB = 300 s, 60dB = 180 s, 70dB = 60 s.

| Painel | Método | 30 s | 60 s | 120 s | 180 s | 300 s | 480 s |
|---|---|---:|---:|---:|---:|---:|---:|
| 30dB | Rayleigh otimizado | 1.1% | 3.4% | 11.4% | 14.8% | 36.4% | 44.3% |
| 30dB | HMM (preditiva) | 5.7% | 10.2% | 15.9% | 19.3% | 35.2% | 43.2% |
| 40dB | Rayleigh otimizado | 1.1% | 11.4% | 27.3% | 43.2% | 60.2% | 71.6% |
| 40dB | HMM (preditiva) | 9.1% | 13.6% | 29.5% | 40.9% | 54.5% | 67.0% |
| 50dB | Rayleigh otimizado | 2.3% | 22.7% | 45.5% | 61.4% | 71.6% | 71.6% |
| 50dB | HMM (preditiva) | 10.2% | 22.7% | 45.5% | 59.1% | 71.6% | 71.6% |
| 60dB | Rayleigh otimizado | 3.4% | 25.0% | 43.2% | 53.4% | 53.4% | 53.4% |
| 60dB | HMM (preditiva) | 20.5% | 27.3% | 45.5% | 54.5% | 54.5% | 54.5% |
| 70dB | Rayleigh otimizado | 14.8% | 38.6% | 38.6% | 38.6% | 38.6% | 38.6% |
| 70dB | HMM (preditiva) | 30.7% | 39.8% | 39.8% | 39.8% | 39.8% | 39.8% |
| Total (alvos) | Rayleigh otimizado | 4.5% | 20.2% | 33.2% | 42.3% | 52.0% | 55.9% |
| Total (alvos) | HMM (preditiva) | 15.2% | 22.7% | 35.2% | 42.7% | 51.1% | 55.2% |
| FP lateral | Rayleigh otimizado | 0.4% | 1.7% | 4.5% | 6.2% | 7.4% | 8.5% |
| FP lateral | HMM (preditiva) | 1.5% | 2.7% | 4.4% | 5.9% | 6.6% | 7.6% |

## C. Truncamento das gravações

Cada gravação é truncada em T s (primeiras ⌊T / 10⌋ avaliações), com as mesmas emissões e parâmetros. Só entram sequências com gravação de pelo menos T s, para que todas as condições tenham a mesma duração máxima. A coluna Arquivos conta arquivos (participante × condição).

### T = 60 s

Arquivos por condição: ESP = 10, 30dB = 11, 40dB = 11, 50dB = 11, 60dB = 11, 70dB = 11. Excluídos (gravação < 60 s): 1 de 66 (ESP = 1).

| Condição | Método | Arquivos | Detecção | FP lateral | FP alvo ESP | Tempo médio / mediano (s) |
|---|---|---:|---:|---:|---:|---:|
| ESP | Rayleigh otimizado | 10 | — | 1/80 (1.2%) | 2/80 (2.5%) | — |
| ESP | HMM (preditiva) | 10 | — | 1/80 (1.2%) | 3/80 (3.8%) | — |
| 30dB | Rayleigh otimizado | 11 | 3/88 (3.4%) | 1/88 (1.1%) | — | 47 / 50 |
| 30dB | HMM (preditiva) | 11 | 9/88 (10.2%) | 1/88 (1.1%) | — | 38 / 30 |
| 40dB | Rayleigh otimizado | 11 | 10/88 (11.4%) | 3/88 (3.4%) | — | 46 / 45 |
| 40dB | HMM (preditiva) | 11 | 12/88 (13.6%) | 5/88 (5.7%) | — | 37 / 30 |
| 50dB | Rayleigh otimizado | 11 | 20/88 (22.7%) | 2/88 (2.3%) | — | 46 / 40 |
| 50dB | HMM (preditiva) | 11 | 20/88 (22.7%) | 2/88 (2.3%) | — | 40 / 40 |
| 60dB | Rayleigh otimizado | 11 | 22/88 (25.0%) | 1/88 (1.1%) | — | 46 / 40 |
| 60dB | HMM (preditiva) | 11 | 24/88 (27.3%) | 3/88 (3.4%) | — | 35 / 30 |
| 70dB | Rayleigh otimizado | 11 | 34/88 (38.6%) | 1/88 (1.1%) | — | 43 / 40 |
| 70dB | HMM (preditiva) | 11 | 35/88 (39.8%) | 2/88 (2.3%) | — | 34 / 30 |

### T = 120 s

Arquivos por condição: ESP = 8, 30dB = 11, 40dB = 11, 50dB = 11, 60dB = 11, 70dB = 0. Excluídos (gravação < 120 s): 14 de 66 (ESP = 3, 70dB = 11).

| Condição | Método | Arquivos | Detecção | FP lateral | FP alvo ESP | Tempo médio / mediano (s) |
|---|---|---:|---:|---:|---:|---:|
| ESP | Rayleigh otimizado | 8 | — | 2/64 (3.1%) | 1/64 (1.6%) | — |
| ESP | HMM (preditiva) | 8 | — | 1/64 (1.6%) | 2/64 (3.1%) | — |
| 30dB | Rayleigh otimizado | 11 | 10/88 (11.4%) | 2/88 (2.3%) | — | 80 / 90 |
| 30dB | HMM (preditiva) | 11 | 14/88 (15.9%) | 2/88 (2.3%) | — | 58 / 45 |
| 40dB | Rayleigh otimizado | 11 | 24/88 (27.3%) | 6/88 (6.8%) | — | 72 / 70 |
| 40dB | HMM (preditiva) | 11 | 26/88 (29.5%) | 6/88 (6.8%) | — | 68 / 70 |
| 50dB | Rayleigh otimizado | 11 | 40/88 (45.5%) | 6/88 (6.8%) | — | 71 / 65 |
| 50dB | HMM (preditiva) | 11 | 40/88 (45.5%) | 4/88 (4.5%) | — | 68 / 65 |
| 60dB | Rayleigh otimizado | 11 | 38/88 (43.2%) | 6/88 (6.8%) | — | 64 / 60 |
| 60dB | HMM (preditiva) | 11 | 40/88 (45.5%) | 7/88 (8.0%) | — | 54 / 45 |
| 70dB | Rayleigh otimizado | 0 | — | — | — | — |
| 70dB | HMM (preditiva) | 0 | — | — | — | — |

### T = 180 s

Arquivos por condição: ESP = 3, 30dB = 11, 40dB = 11, 50dB = 11, 60dB = 9, 70dB = 0. Excluídos (gravação < 180 s): 21 de 66 (ESP = 8, 60dB = 2, 70dB = 11).

| Condição | Método | Arquivos | Detecção | FP lateral | FP alvo ESP | Tempo médio / mediano (s) |
|---|---|---:|---:|---:|---:|---:|
| ESP | Rayleigh otimizado | 3 | — | 2/24 (8.3%) | 0/24 (0.0%) | — |
| ESP | HMM (preditiva) | 3 | — | 1/24 (4.2%) | 0/24 (0.0%) | — |
| 30dB | Rayleigh otimizado | 11 | 13/88 (14.8%) | 4/88 (4.5%) | — | 95 / 100 |
| 30dB | HMM (preditiva) | 11 | 17/88 (19.3%) | 4/88 (4.5%) | — | 75 / 60 |
| 40dB | Rayleigh otimizado | 11 | 38/88 (43.2%) | 6/88 (6.8%) | — | 106 / 90 |
| 40dB | HMM (preditiva) | 11 | 36/88 (40.9%) | 6/88 (6.8%) | — | 94 / 80 |
| 50dB | Rayleigh otimizado | 11 | 54/88 (61.4%) | 9/88 (10.2%) | — | 94 / 90 |
| 50dB | HMM (preditiva) | 11 | 52/88 (59.1%) | 7/88 (8.0%) | — | 89 / 80 |
| 60dB | Rayleigh otimizado | 9 | 34/72 (47.2%) | 7/72 (9.7%) | — | 92 / 85 |
| 60dB | HMM (preditiva) | 9 | 35/72 (48.6%) | 8/72 (11.1%) | — | 81 / 70 |
| 70dB | Rayleigh otimizado | 0 | — | — | — | — |
| 70dB | HMM (preditiva) | 0 | — | — | — | — |

### T = 300 s

Arquivos por condição: ESP = 1, 30dB = 11, 40dB = 11, 50dB = 11, 60dB = 0, 70dB = 0. Excluídos (gravação < 300 s): 32 de 66 (ESP = 10, 60dB = 11, 70dB = 11).

| Condição | Método | Arquivos | Detecção | FP lateral | FP alvo ESP | Tempo médio / mediano (s) |
|---|---|---:|---:|---:|---:|---:|
| ESP | Rayleigh otimizado | 1 | — | 2/8 (25.0%) | 0/8 (0.0%) | — |
| ESP | HMM (preditiva) | 1 | — | 2/8 (25.0%) | 0/8 (0.0%) | — |
| 30dB | Rayleigh otimizado | 11 | 32/88 (36.4%) | 5/88 (5.7%) | — | 182 / 205 |
| 30dB | HMM (preditiva) | 11 | 31/88 (35.2%) | 5/88 (5.7%) | — | 159 / 150 |
| 40dB | Rayleigh otimizado | 11 | 53/88 (60.2%) | 8/88 (9.1%) | — | 144 / 160 |
| 40dB | HMM (preditiva) | 11 | 48/88 (54.5%) | 7/88 (8.0%) | — | 130 / 120 |
| 50dB | Rayleigh otimizado | 11 | 63/88 (71.6%) | 12/88 (13.6%) | — | 117 / 100 |
| 50dB | HMM (preditiva) | 11 | 63/88 (71.6%) | 8/88 (9.1%) | — | 117 / 100 |
| 60dB | Rayleigh otimizado | 0 | — | — | — | — |
| 60dB | HMM (preditiva) | 0 | — | — | — | — |
| 70dB | Rayleigh otimizado | 0 | — | — | — | — |
| 70dB | HMM (preditiva) | 0 | — | — | — | — |
