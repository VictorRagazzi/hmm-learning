# Seleção do sinal — Rayleigh de teste único no fim

Gerado por `src/rayleigh_hmm/selecao_sinal.py`. Um teste de Rayleigh por (arquivo, frequência) com todas as épocas válidas, α = 0.05. Detecção = alvos em 30–70 dB (n = 440); FP lateral = laterais em todas as condições (n = 528); FP alvo ESP (n = 88); KS = p-valores dos alvos ESP contra Uniforme(0,1) (checagem da nula); colunas de dB = detecção por intensidade.

**Vazamento aceito:** o ranking de canais e a escolha usam os mesmos dados da avaliação (in-sample). Os números não são estimativa de generalização nem clínica.

**Regra de escolha:** maior detecção entre as opções com FP lateral ≤ 7% e KS ESP com p > 0.05; desempate pelo menor FP lateral. O ranking de canais para o top-k usa a mesma regra (elegíveis primeiro, depois detecção).

## Opção escolhida

`media:top4|rej2.0`: combinação `media`, canais FC,Fz,Cz,F4, rejeição `2.0`. Detecção 57.5%, FP lateral 3.6%, FP alvo ESP 4.5%, KS ESP p = 0.443, épocas rejeitadas 2.6%.

## Avisos

- media:todos: 1 arquivos com variância do sinal combinado < 5% da média dos canais (mínimo 0.0437)
- media:top4: 1 arquivos com variância do sinal combinado < 5% da média dos canais (mínimo 0.047)
- media:top6: 1 arquivos com variância do sinal combinado < 5% da média dos canais (mínimo 0.0476)
- media:top4|rej1.5: 1 arquivos com variância do sinal combinado < 5% da média dos canais (mínimo 0.047)
- media:top4|rej2.0: 1 arquivos com variância do sinal combinado < 5% da média dos canais (mínimo 0.047)
- media:top4|rej3.0: 1 arquivos com variância do sinal combinado < 5% da média dos canais (mínimo 0.047)

## (a) Canais isolados

| Opção | Canais | Detecção /440 | FP lateral /528 | FP alvo ESP /88 | KS ESP p | Rejeitadas | 30dB | 40dB | 50dB | 60dB | 70dB | Elegível |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canal:FC | FC | 53.4% | 4.0% | 6.8% | 0.236 | 0.0% | 31% | 61% | 69% | 58% | 48% | sim |
| canal:Fz | Fz | 50.9% | 3.2% | 3.4% | 0.311 | 0.0% | 24% | 65% | 64% | 62% | 40% | sim |
| canal:Cz | Cz | 49.5% | 3.2% | 4.5% | 0.794 | 0.0% | 27% | 59% | 64% | 55% | 43% | sim |
| canal:F4 | F4 | 49.3% | 4.9% | 3.4% | 0.237 | 0.0% | 33% | 58% | 62% | 50% | 43% | sim |
| canal:C4 | C4 | 45.5% | 2.3% | 5.7% | 0.469 | 0.0% | 27% | 53% | 60% | 49% | 38% | sim |
| canal:P3 | P3 | 43.4% | 3.8% | 8.0% | 0.320 | 0.0% | 22% | 52% | 55% | 49% | 40% | sim |
| canal:C3 | C3 | 42.7% | 4.0% | 10.2% | 0.647 | 0.0% | 25% | 47% | 53% | 44% | 44% | sim |
| canal:Pz | Pz | 40.7% | 2.7% | 8.0% | 0.831 | 0.0% | 28% | 49% | 53% | 40% | 33% | sim |
| canal:P4 | P4 | 40.2% | 5.7% | 3.4% | 0.034 | 0.0% | 24% | 45% | 59% | 44% | 28% | não |
| canal:F3 | F3 | 38.6% | 3.2% | 5.7% | 0.270 | 0.0% | 22% | 51% | 44% | 38% | 39% | sim |
| canal:F7 | F7 | 30.2% | 3.4% | 6.8% | 0.712 | 0.0% | 19% | 32% | 35% | 35% | 30% | sim |
| canal:Oz | Oz | 28.2% | 4.2% | 6.8% | 0.602 | 0.0% | 17% | 35% | 38% | 32% | 19% | sim |
| canal:T3 | T3 | 26.4% | 3.2% | 4.5% | 0.478 | 0.0% | 22% | 31% | 30% | 28% | 22% | sim |
| canal:T5 | T5 | 26.1% | 4.4% | 8.0% | 0.509 | 0.0% | 19% | 32% | 28% | 25% | 26% | sim |
| canal:T6 | T6 | 21.1% | 4.4% | 3.4% | 0.004 | 0.0% | 17% | 25% | 24% | 17% | 23% | não |
| canal:T4 | T4 | 20.7% | 3.4% | 1.1% | 0.696 | 0.0% | 10% | 25% | 18% | 30% | 20% | sim |

## (b)/(c) Combinações

| Opção | Canais | Detecção /440 | FP lateral /528 | FP alvo ESP /88 | KS ESP p | Rejeitadas | 30dB | 40dB | 50dB | 60dB | 70dB | Elegível |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| media:top4 | FC,Fz,Cz,F4 | 57.0% | 3.0% | 3.4% | 0.229 | 0.0% | 30% | 73% | 70% | 64% | 49% | sim |
| media:top2 | FC,Fz | 55.9% | 3.4% | 4.5% | 0.035 | 0.0% | 26% | 66% | 73% | 65% | 50% | não |
| media:top3 | FC,Fz,Cz | 55.5% | 2.8% | 6.8% | 0.690 | 0.0% | 31% | 68% | 69% | 60% | 49% | sim |
| media:top6 | FC,Fz,Cz,F4,C4,P3 | 55.5% | 3.2% | 5.7% | 0.816 | 0.0% | 27% | 70% | 73% | 58% | 49% | sim |
| pca:top4 | FC,Fz,Cz,F4 | 53.9% | 2.7% | 3.4% | 0.701 | 0.0% | 27% | 64% | 68% | 62% | 48% | sim |
| pca:top3 | FC,Fz,Cz | 53.9% | 3.0% | 4.5% | 0.457 | 0.0% | 32% | 65% | 67% | 59% | 47% | sim |
| pca:top2 | FC,Fz | 53.6% | 3.6% | 4.5% | 0.189 | 0.0% | 26% | 61% | 69% | 61% | 50% | sim |
| pca:top6 | FC,Fz,Cz,F4,C4,P3 | 52.0% | 2.7% | 5.7% | 0.831 | 0.0% | 25% | 61% | 67% | 58% | 49% | sim |
| media:todos | todos | 45.5% | 3.8% | 6.8% | 0.820 | 0.0% | 27% | 53% | 56% | 49% | 42% | sim |
| pca:todos | todos | 42.5% | 3.0% | 6.8% | 0.516 | 0.0% | 24% | 45% | 52% | 50% | 41% | sim |

## (d) Rejeição de artefatos

| Opção | Canais | Fator | Detecção /440 | FP lateral /528 | FP alvo ESP /88 | KS ESP p | Rejeitadas | 30dB | 40dB | 50dB | 60dB | 70dB | Elegível |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| media:top4|rej2.0 | FC,Fz,Cz,F4 | 2.0 | 57.5% | 3.6% | 4.5% | 0.443 | 2.6% | 30% | 72% | 70% | 66% | 50% | sim |
| media:top4|rej3.0 | FC,Fz,Cz,F4 | 3.0 | 57.3% | 3.4% | 4.5% | 0.475 | 1.0% | 30% | 74% | 70% | 64% | 49% | sim |
| media:top4 | FC,Fz,Cz,F4 | — | 57.0% | 3.0% | 3.4% | 0.229 | 0.0% | 30% | 73% | 70% | 64% | 49% | sim |
| media:top4|rej1.5 | FC,Fz,Cz,F4 | 1.5 | 55.9% | 4.0% | 5.7% | 0.117 | 6.4% | 28% | 73% | 67% | 62% | 49% | sim |
| canal:FC | FC | — | 53.4% | 4.0% | 6.8% | 0.236 | 0.0% | 31% | 61% | 69% | 58% | 48% | sim |
| canal:FC|rej3.0 | FC | 3.0 | 52.7% | 4.2% | 6.8% | 0.163 | 0.6% | 28% | 60% | 69% | 58% | 48% | sim |
| canal:FC|rej2.0 | FC | 2.0 | 51.6% | 3.8% | 6.8% | 0.628 | 2.2% | 27% | 57% | 69% | 57% | 48% | sim |
| canal:FC|rej1.5 | FC | 1.5 | 51.6% | 4.2% | 6.8% | 0.464 | 5.8% | 28% | 58% | 68% | 56% | 48% | sim |
