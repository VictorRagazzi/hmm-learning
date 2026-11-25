# MCMC completo em R (Metropolis-dentro-de-Gibbs)

Config: 20000 iteracoes, burn-in 5000, thin 10, 4 cadeias, 440 sequencias; tempo 30 s.

## Hiperparametros (escala natural, pos burn-in com thin)

| Parametro | media | dp | R-hat | ESS |
|---|---|---|---|---|
| a_k | 1.463 | 0.106 | 0.9999 | 2040 |
| b_k | 0.319 | 0.030 | 1.0002 | 2045 |
| a_rho | 0.290 | 0.020 | 1.0000 | 2963 |
| b_rho | 9.495 | 1.100 | 0.9996 | 4119 |

## Aceitacao media (pos burn-in)

- hiperparametros (a_k, a_rho): 0.418
- latentes k_s: 0.440; rho_s: 0.441

## Latentes (440 sequencias)

| Latente | R-hat max | R-hat mediano | ESS min | ESS mediano |
|---|---|---|---|---|
| log k_s | 1.0013 | 1.0000 | 3719 | 5692 |
| log rho_s | 1.0149 | 1.0001 | 688 | 5355 |
