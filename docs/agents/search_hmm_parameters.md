# `src/search_hmm_parameters.py`

## Responsabilidade

Faz busca em grade sobre a parametrização do HMM e da regra de decisão,
reutilizando o detector, a calibração de B e o Viterbi dos módulos principais.
O objetivo implementado é maximizar acurácia balanceada, sujeito a limite de
falso positivo lateral do HMM.

## Espaço atual

- `MIN_CONSECUTIVE`: 1 a 12.
- `MIN_PERCENT`: 15 valores entre 0,01 e 1,00.
- modo: OR ou AND.
- `PI_INICIAL`: 13 tuplas declaradas; `(0.40, 0.60)` aparece duas vezes.
- `K_SINTETICO`: 0,005; 0,01; 0,02; 0,05; 0,1.
- falso positivo máximo do HMM: 12,5%.

São 360 regras por par `(K, PI)` e, contando a duplicata de PI, 23.400
avaliações do HMM. O detector bruto é otimizado separadamente apenas sobre as
360 regras.

## Estratégia de execução

1. Calcula valores/p-valores/labels dos 55 arquivos estimulados.
2. Para cada K, altera temporariamente `gom.K_SINTETICO`, recalibra B nos 11
   ESP e restaura a configuração em `finally`.
3. Para cada PI, roda Viterbi reutilizando labels.
4. Avalia todas as regras depois do Viterbi, sem repetir FFT.
5. Descarta combinações HMM cujo falso positivo exceda 12,5%.
6. Ordena por acurácia balanceada, depois menor falso positivo e maior detecção.
7. Imprime o melhor HMM e o melhor detector bruto.

A acurácia balanceada implementada é:

```text
(taxa_deteccao + (1 - taxa_falso_positivo_lateral)) / 2
```

## Efeitos e saída

O script não atualiza `get_obs_matrix.py`, não grava a B vencedora e não salva
um JSON da busca. A alteração de K existe apenas durante cada chamada. Para
adotar um resultado é necessário modificar configurações conscientemente,
regenerar artefatos e documentar a decisão.

## Riscos metodológicos e técnicos

- Seleção e avaliação usam o mesmo conjunto de arquivos `*dB.mat`; a melhor
  métrica é otimista e não representa generalização.
- O teto do detector bruto não recebe o corte de falso positivo aplicado ao
  HMM, então a comparação deve mencionar essa assimetria.
- Há uma PI duplicada e a função de pré-cálculo extrai cada sequência duas
  vezes (primeiro p-valores, depois labels), apesar do comentário sobre reuso.
- A busca depende do detector ativo em `gom`, mas seu `__main__` não chama
  explicitamente `selecionar_detector`; use `ASSR_DETECTOR` para tornar a
  escolha inequívoca e compatível com o artefato analisado.
- A matriz A é carregada do disco e mantém a heurística/duração do artefato
  existente; a busca não recalibra A por configuração de janelamento.
