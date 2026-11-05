## Blocos de 30 epocas

Sequencias com >= 3 blocos: estimulo 352, ESP 64. Fortes por intensidade (p<0.001): {'30dB': 9, '40dB': 34, '50dB': 40, '60dB': 27, '70dB': 0}

### Resumo por grupo

| grupo | n | S med | S nulo-emb med | % deriva sig | % estavel sig | med abs(df) mHz | med S apos/antes | med Z apos/antes |
|---|---|---|---|---|---|---|---|---|
| forte p<0.001 | 110 | 0.831 | 0.824 | 6 | 100 | 0.31 | 1.01 | 1.02 |
| frouxo p<0.01 | 152 | 0.796 | 0.794 | 5 | 99 | 0.34 | 1.01 | 1.03 |
| todas estimulo | 352 | 0.582 | 0.593 | 4 | 55 | 0.92 | 1.09 | 1.14 |
| ESP (todas) | 64 | 0.39 | 0.434 | 5 | 6 | 8.77 | 1.97 | 3.03 |
| ESP p<0.01 (ESP 'fortes') | 1 | 0.982 | 0.874 | 0 | 100 | 0.61 | 1.01 | 0.973 |

### Fortes por intensidade

| grupo | n | S med | S nulo-emb med | % deriva sig | % estavel sig | med abs(df) mHz | med S apos/antes | med Z apos/antes |
|---|---|---|---|---|---|---|---|---|
| 30dB | 9 | 0.734 | 0.691 | 0 | 100 | 0.21 | 1.01 | 1.02 |
| 40dB | 34 | 0.739 | 0.738 | 9 | 100 | 0.21 | 1.01 | 1.02 |
| 50dB | 40 | 0.848 | 0.835 | 5 | 100 | 0.49 | 1.03 | 1.05 |
| 60dB | 27 | 0.938 | 0.933 | 7 | 100 | 0.57 | 1.01 | 1.03 |
| 70dB | 0 | | | | | | | |

### Primeira x segunda metade (diferenca de fase 2a - 1a, fortes / ESP)

- forte p<0.001: n=110 media=-0.07 rad R=0.86 dp=0.54 p_Rayleigh=2.77e-36 p_V(0)=1.31e-37
- ESP (todas): n=64 media=-2.86 rad R=0.11 dp=2.11 p_Rayleigh=0.477 p_V(0)=0.879

### Assinatura de relogio (fortes; gravacoes com >= 3 frequencias fortes)

| arquivo | Fs | n freq | eps=df/f mediana | MAD | dp | R2 (df ~ f) |
|---|---|---|---|---|---|---|
| Ab40dB.mat | 1000 | 5 | -1.58e-06 | 3.82e-06 | 3.84e-06 | -0.01 |
| Ab50dB.mat | 1000 | 3 | -6.74e-06 | 4.90e-07 | 2.71e-06 | 0.17 |
| An30dB.mat | 1000 | 3 | 1.95e-06 | 2.98e-07 | 1.03e-06 | -0.00 |
| An40dB.mat | 1000 | 4 | 1.71e-06 | 1.88e-06 | 7.85e-05 | -0.01 |
| An50dB.mat | 1000 | 5 | -4.59e-06 | 3.64e-06 | 3.11e-06 | -0.05 |
| An60dB.mat | 1000 | 4 | 1.83e-06 | 4.72e-06 | 5.97e-06 | -0.01 |
| Bb40dB.mat | 1750 | 3 | 1.20e-05 | 3.87e-07 | 3.19e-06 | 0.15 |
| Bb50dB.mat | 1750 | 6 | 1.35e-05 | 9.67e-07 | 1.67e-06 | -0.15 |
| Er40dB.mat | 1000 | 3 | -2.18e-06 | 4.30e-07 | 7.10e-07 | -0.09 |
| Er50dB.mat | 1000 | 6 | -4.03e-06 | 2.90e-06 | 3.21e-06 | 0.01 |
| Er60dB.mat | 1000 | 4 | -4.88e-06 | 7.29e-06 | 1.02e-05 | -0.04 |
| Lu60dB.mat | 1750 | 3 | -3.45e-06 | 1.21e-06 | 1.55e-06 | -0.18 |
| Qu40dB.mat | 1000 | 4 | -1.87e-07 | 1.08e-06 | 1.65e-06 | 0.01 |
| Qu50dB.mat | 1000 | 3 | -3.02e-06 | 2.94e-06 | 3.60e-06 | -0.06 |
| Sa40dB.mat | 1000 | 5 | -1.46e-06 | 2.58e-06 | 2.74e-06 | 0.03 |
| Sa50dB.mat | 1000 | 8 | -5.25e-07 | 1.06e-06 | 3.42e-06 | 0.01 |
| Sa60dB.mat | 1000 | 8 | -5.78e-07 | 8.52e-06 | 1.30e-05 | -0.03 |
| So40dB.mat | 1750 | 3 | -1.15e-06 | 1.08e-06 | 1.97e-06 | -0.08 |
| So50dB.mat | 1750 | 3 | 2.94e-06 | 5.95e-06 | 7.49e-06 | -0.00 |
| Ti40dB.mat | 1000 | 3 | 1.15e-06 | 4.62e-07 | 1.01e-06 | -0.01 |
| Wr40dB.mat | 1000 | 3 | -4.03e-06 | 7.32e-07 | 1.34e-06 | -0.32 |
| Wr50dB.mat | 1000 | 5 | -3.84e-06 | 7.61e-07 | 2.81e-06 | -0.07 |
| Wr60dB.mat | 1000 | 3 | -2.60e-06 | 3.82e-06 | 8.15e-06 | -0.02 |

Mediana de |eps| = 2.60e-06; mediana do dp intra-gravacao de eps = 3.11e-06; mediana R2 = -0.01.
Mediana de eps por gravacao, agrupada por participante: Ab: -1.6e-06, -6.7e-06; An: 1.9e-06, 1.7e-06, -4.6e-06, 1.8e-06; Bb: 1.2e-05, 1.3e-05; Er: -2.2e-06, -4.0e-06, -4.9e-06; Lu: -3.4e-06; Qu: -1.9e-07, -3.0e-06; Sa: -1.5e-06, -5.3e-07, -5.8e-07; So: -1.1e-06, 2.9e-06; Ti: 1.2e-06; Wr: -4.0e-06, -3.8e-06, -2.6e-06

Ganho de Z por correcao em TODAS as sequencias: estimulo mediana 1.14, ESP (acaso) 3.03

## Blocos de 60 epocas

Sequencias com >= 3 blocos: estimulo 336, ESP 24. Fortes por intensidade (p<0.001): {'30dB': 9, '40dB': 34, '50dB': 40, '60dB': 16, '70dB': 0}

### Resumo por grupo

| grupo | n | S med | S nulo-emb med | % deriva sig | % estavel sig | med abs(df) mHz | med S apos/antes | med Z apos/antes |
|---|---|---|---|---|---|---|---|---|
| forte p<0.001 | 99 | 0.901 | 0.91 | 9 | 87 | 0.29 | 1.01 | 1.02 |
| frouxo p<0.01 | 140 | 0.887 | 0.9 | 6 | 85 | 0.33 | 1.01 | 1.03 |
| todas estimulo | 336 | 0.739 | 0.755 | 7 | 45 | 0.81 | 1.09 | 1.13 |
| ESP (todas) | 24 | 0.424 | 0.414 | 4 | 4 | 6.04 | 1.95 | 2.55 |
| ESP p<0.01 (ESP 'fortes') | 0 | | | | | | | |

### Fortes por intensidade

| grupo | n | S med | S nulo-emb med | % deriva sig | % estavel sig | med abs(df) mHz | med S apos/antes | med Z apos/antes |
|---|---|---|---|---|---|---|---|---|
| 30dB | 9 | 0.9 | 0.829 | 0 | 100 | 0.25 | 1.01 | 1.02 |
| 40dB | 34 | 0.856 | 0.867 | 6 | 94 | 0.17 | 1.01 | 1.02 |
| 50dB | 40 | 0.921 | 0.929 | 15 | 82 | 0.45 | 1.03 | 1.06 |
| 60dB | 16 | 0.982 | 0.965 | 6 | 75 | 0.40 | 1.01 | 1.03 |
| 70dB | 0 | | | | | | | |

### Primeira x segunda metade (diferenca de fase 2a - 1a, fortes / ESP)

- forte p<0.001: n=99 media=-0.06 rad R=0.86 dp=0.56 p_Rayleigh=3.38e-32 p_V(0)=1.47e-33
- ESP (todas): n=24 media=2.90 rad R=0.31 dp=1.53 p_Rayleigh=0.101 p_V(0)=0.981

### Assinatura de relogio (fortes; gravacoes com >= 3 frequencias fortes)

| arquivo | Fs | n freq | eps=df/f mediana | MAD | dp | R2 (df ~ f) |
|---|---|---|---|---|---|---|
| Ab40dB.mat | 1000 | 5 | -1.43e-07 | 2.13e-06 | 3.63e-06 | -0.00 |
| Ab50dB.mat | 1000 | 3 | -5.52e-06 | 5.15e-07 | 2.99e-06 | 0.12 |
| An30dB.mat | 1000 | 3 | 1.75e-06 | 1.47e-06 | 1.29e-06 | -0.05 |
| An40dB.mat | 1000 | 4 | 2.76e-07 | 1.17e-06 | 2.11e-06 | -0.01 |
| An50dB.mat | 1000 | 5 | -4.66e-06 | 2.17e-06 | 1.98e-06 | -0.08 |
| An60dB.mat | 1000 | 4 | 2.51e-06 | 3.48e-06 | 4.46e-06 | -0.03 |
| Bb40dB.mat | 1750 | 3 | 1.22e-05 | 8.14e-07 | 3.09e-06 | 0.16 |
| Bb50dB.mat | 1750 | 6 | 1.38e-05 | 1.41e-06 | 1.85e-06 | -0.45 |
| Er40dB.mat | 1000 | 3 | -1.40e-06 | 1.85e-07 | 8.66e-07 | -0.18 |
| Er50dB.mat | 1000 | 6 | -3.78e-06 | 2.51e-06 | 3.20e-06 | 0.02 |
| Er60dB.mat | 1000 | 4 | -4.61e-06 | 4.78e-06 | 9.30e-06 | -0.05 |
| Lu60dB.mat | 1750 | 3 | 8.03e-08 | 5.41e-09 | 1.92e-06 | 0.06 |
| Qu40dB.mat | 1000 | 4 | -2.39e-08 | 1.67e-06 | 2.18e-06 | 0.02 |
| Qu50dB.mat | 1000 | 3 | -3.51e-06 | 3.13e-06 | 4.02e-06 | -0.07 |
| Sa40dB.mat | 1000 | 5 | -1.33e-06 | 3.02e-06 | 2.60e-06 | 0.03 |
| Sa50dB.mat | 1000 | 8 | 8.22e-08 | 1.53e-06 | 3.35e-06 | 0.00 |
| So40dB.mat | 1750 | 3 | -1.43e-07 | 4.24e-07 | 1.95e-06 | -0.05 |
| So50dB.mat | 1750 | 3 | 2.65e-06 | 5.93e-06 | 7.11e-06 | -0.00 |
| Ti40dB.mat | 1000 | 3 | 1.07e-06 | 8.95e-07 | 1.05e-06 | 0.00 |
| Wr40dB.mat | 1000 | 3 | -4.32e-06 | 9.53e-07 | 1.00e-06 | -0.44 |
| Wr50dB.mat | 1000 | 5 | -5.10e-06 | 2.80e-06 | 2.97e-06 | -0.09 |

Mediana de |eps| = 2.51e-06; mediana do dp intra-gravacao de eps = 2.60e-06; mediana R2 = -0.01.
Mediana de eps por gravacao, agrupada por participante: Ab: -1.4e-07, -5.5e-06; An: 1.8e-06, 2.8e-07, -4.7e-06, 2.5e-06; Bb: 1.2e-05, 1.4e-05; Er: -1.4e-06, -3.8e-06, -4.6e-06; Lu: 8.0e-08; Qu: -2.4e-08, -3.5e-06; Sa: -1.3e-06, 8.2e-08; So: -1.4e-07, 2.7e-06; Ti: 1.1e-06; Wr: -4.3e-06, -5.1e-06

Ganho de Z por correcao em TODAS as sequencias: estimulo mediana 1.13, ESP (acaso) 2.55


Tempo de execucao: 45.8 s
