# Modelo hierarquico Gamma com JAGS (rjags)

Config: 20000 iteracoes, burn-in 5000 (adaptacao 1000), thin 10, 4 cadeias, 440 sequencias; tempo 1137 s; JAGS 5.5.

## Hiperparametros (escala natural; R-hat e ESS calculados em log)

Colunas `ref` = amostrador manual (resumo_r.md).

| Parametro | media | dp | R-hat | ESS | ref media | ref dp | dif/dp ref |
|---|---|---|---|---|---|---|---|
| a_k | 1.462 | 0.105 | 1.0011 | 3565 | 1.463 | 0.106 | -0.01 |
| b_k | 0.319 | 0.030 | 1.0004 | 3670 | 0.319 | 0.030 | -0.01 |
| a_rho | 0.290 | 0.020 | 1.0025 | 1677 | 0.290 | 0.020 | 0.00 |
| b_rho | 9.475 | 1.089 | 1.0007 | 3994 | 9.495 | 1.100 | -0.02 |

## Latentes (440 sequencias, em log)

| Latente | R-hat max | R-hat mediano | ESS min | ESS mediano |
|---|---|---|---|---|
| log k_s | 1.0059 | 1.0002 | 3782 | 6000 |
| log rho_s | 1.1169 | 1.0006 | 937 | 5819 |

## Amostradores escolhidos pelo JAGS

| No | Amostrador | Quantidade |
|---|---|---|
| hiper (log_*) | base::RealSlicer | 4 |
| k | base::RealSlicer | 440 |
| rho | base::RealSlicer | 440 |

