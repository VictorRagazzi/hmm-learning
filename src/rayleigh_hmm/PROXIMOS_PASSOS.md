# Próximos passos — anotados em 22/10/2026 para fazer depois

Itens combinados com o usuário, a implementar **depois** de (1) melhor
canal/combinação de canais + rejeição de artefatos e (2) curvas de
avaliação + truncamento das gravações.

## Outros ajustes baratos lembrados na mesma conversa

- Buscar `LIMIAR_POSTERIORI` (hoje fixo em 0,5).
- Fixar Presente→Ausente = 0 (estado absorvente).
- Passo de 5 épocas, com correção para testes sequenciais (gasto de α).
- Intervalos de confiança por bootstrap de pacientes.

## C. SNR sintética ligada às intensidades reais (decidido e IMPLEMENTADO em 04/11/2026)

**Implementado em 04/11/2026:** `SNR_SINTETICA_DB` agora tem os 5 valores fixos, sem
sorteio e com fase 0; resultados em `results/rayleigh_hmm/resultados.json` e `README.md`.
O texto abaixo é o registro da decisão (o "Hoje" descreve o estado anterior).

Hoje (`experimento.py`): `SNR_SINTETICA_DB = (-25, -10)` sorteada uniforme em dB,
`REPETICOES_SINTETICAS = 5` sorteios por sequencia ESP (440 sequencias).

Problema: a faixa e o sorteio sao arbitrarios, e a Gamma de ρ ajustada reflete
em boa parte a distribuicao que nos escolhemos.

Decisao: nada sorteado. Cada sequencia ESP recebe 5 SNRs fixas, uma por
intensidade de estimulacao, medidas nos proprios arquivos com estimulo
(88 × 5 = 440 sequencias, mesmo volume de hoje).

SNR medida no bin-alvo (sinal `SINAL` do experimento, epocas validas, 11
participantes × 8 frequencias = 88 por intensidade). Estimador sem vies:
S = |media(X)|² − media(|X|²)/n, SNR = S / (P − S), com P = media(|X|²); media
de S/P entre as 88 sequencias, convertida para dB depois da media.

**Tabela original (média de 4 canais FC,Fz,Cz,F4, com rejeição 2,0×):**

| Intensidade | S/P medio (± erro padrao) | SNR | Valor a usar |
|---|---|---|---|
| ESP (controle) | 0,0000 ± 0,0015 | ~0 | — |
| 30 dB | 0,0043 ± 0,0008 | −23,6 dB | −23,5 dB |
| 40 dB | 0,0136 ± 0,0015 | −18,6 dB | −18,5 dB |
| 50 dB | 0,0304 ± 0,0042 | −15,0 dB | −15,0 dB |
| 60 dB | 0,0443 ± 0,0078 | −13,3 dB | −13,5 dB |
| 70 dB | 0,0655 ± 0,0109 | −11,5 dB | −11,5 dB |

**Nota de 04/11/2026 (sinal atual: canal FC, sem rejeição).** Por decisão do usuário
a média de 4 canais e a rejeição foram abandonadas; a SNR foi remedida com o mesmo
estimador (88 sequências por intensidade) e `SNR_SINTETICA_DB` passou a
`(-25.0, -20.5, -17.5, -14.0, -12.0)`:

| Intensidade | S/P medio (± erro padrao) | SNR | Valor a usar |
|---|---|---|---|
| ESP (controle) | 0,0031 ± 0,0020 | ~0 (−25,1 dB, ≈ 1,5 EP de 0) | — |
| 30 dB | 0,0032 ± 0,0007 | −24,9 dB | −25,0 dB |
| 40 dB | 0,0092 ± 0,0013 | −20,3 dB | −20,5 dB |
| 50 dB | 0,0166 ± 0,0029 | −17,7 dB | −17,5 dB |
| 60 dB | 0,0390 ± 0,0070 | −13,9 dB | −14,0 dB |
| 70 dB | 0,0608 ± 0,0099 | −11,9 dB | −12,0 dB |

Notas (tabela original):
- O ESP da ~0, o que confirma que o estimador nao tem vies relevante.
- A SNR e media populacional: a variacao entre participantes/frequencias e
  grande (em 30 dB, so 72% das sequencias tem S > 0). A variacao de ρ entre
  exames passa a vir do ruido de cada ESP, nao de um sorteio.
- Medir a SNR nos estimulos e usar esses mesmos estimulos na avaliacao e
  vazamento leve (5 numeros medios, nao limiares). Declarar isso; nao tratar
  como validacao independente.
- Alternativa a considerar: alem da media, usar a dispersao medida em cada
  intensidade (ex.: quartis) em vez de um unico valor por intensidade.

Ao implementar: trocar `SNR_SINTETICA_DB` por uma lista de 5 valores em
`experimento.py`, remover o sorteio em `gerar_presente_sintetico` (a fase da
senoide nao altera a distribuicao de Z), refazer o ajuste dos hiperparametros
e os resultados. O slide 15 ja descreve este desenho; os numeros dos slides
17 (resultado do argmax) e 18 (integral por extenso) ainda vem do sorteio e
precisam ser refeitos.

## D. Testar se as diferenças HMM × Rayleigh são significativas (anotado e IMPLEMENTADO em 04/11/2026)

**Implementado em 04/11/2026:** `comparacao_significancia` em `resultados.json`, tabela em
`tabelas.md` e `README.md` (McNemar de alvos/laterais/ESP e Wilcoxon do tempo).

Hoje só a detecção nos alvos tem teste (McNemar, p = 0,65). Fazer o básico
para as demais métricas do slide 22:

- McNemar também para alarme falso (laterais e ESP).
- Wilcoxon pareado no tempo até detectar (80 s × 100 s), nos casos
  detectados pelos dois métodos.

## Nota de 04/11/2026: calibração do Presente passou a usar gravações reais

Decisão do usuário: o pipeline principal (`experimento.py`, `CALIBRACAO_PRESENTE = "real"`)
ajusta o Presente nos Z e N acumulados de todas as 440 sequências-alvo com estímulo
(todo participante ouviu todos os estímulos). Motivo: no LOSO com FC
(`results/rayleigh_hmm/exploratorio_loso_fc/`) a calibração real teve a mesma detecção
que a sintética (6 × 3, p = 0,51) e menos FP lateral (37 contra 50/528, McNemar 1 × 14,
p = 0,001). A estimativa a reportar é a do LOSO (`explorar_loso_fc.py`); a avaliação
de `experimento.py` passa a ser in-sample com vazamento direto.

- A seção C (SNR sintética) fica como histórico e alternativa (`CALIBRACAO_PRESENTE =
  "sintetica"`). A alternativa dos "quartis" por intensidade perdeu o sentido.
- Pontos em aberto:
  - A(Pres→Aus) escolhida bate no máximo da grade (0,5) em 11 dos 11 folds LOSO do braço
    REAL (e em 7 de 11 do sintético): ampliar a grade ou fixar um valor.
  - O ajuste marginal sinaliza massa fora da grade (ρ: 0,27% > 0,1%, `na_borda`); um dos 3 inícios
    do otimizador não saiu do ponto inicial. Ampliar `GRADE_HIER_*` e revisar os inícios.
