# Status do projeto

Atualizado em 25/09/2026 a partir do código e dos JSON presentes na árvore de
trabalho. Este arquivo descreve o estado observado, não uma validação clínica.

## O que está implementado

- Leitura de EEG MATLAB v7.3 (`x`, `Fs`, `freqEstim` e `binsM`) com `h5py`.
- Extração, por época, do coeficiente FFT complexo no bin mais próximo de cada
  frequência.
- Registro plugável com quatro detectores: MSC, MMSC split-half, Rayleigh e CSM.
- Calibração de uma matriz de emissão global B, com diagnósticos separados por
  frequência, a partir de 11 arquivos ESP.
- Construção da linha `Ausente` com dados ESP reais e da linha `Presente` com
  senoide sintética coerente em fase injetada nos mesmos dados.
- Construção de uma matriz de transição A por heurística de duração, usando 11
  arquivos ESP e 55 arquivos com estímulo.
- Inferência Viterbi independente por arquivo e frequência, em log-espaço.
- Comparação da decisão do HMM com o detector bruto, tanto nas frequências de
  estímulo quanto nas laterais de `binsM`.
- Agregações por nível em dB e participante e persistência da análise em JSON.
- Busca em grade de `K_SINTETICO`, distribuição inicial e regra de decisão,
  com restrição máxima de falso positivo lateral.
- Exemplo didático separado de Viterbi em `src/viterbi.py`; ele não participa
  do pipeline ASSR.

## Configuração efetiva atual

| Item | Valor no código/artefato atual |
|---|---|
| Detector padrão de B e inferência | `csm` |
| Canal | `0` |
| Estados | `Ausente`, `Presente` |
| Labels | `muito_baixo`, `baixo`, `medio`, `alto`, `muito_alto` |
| Fronteiras de p-valor | `0.50`, `0.10`, `0.05`, `0.01` |
| Alpha bruto | `0.05` |
| Janela/passo de B e inferência | `10/10` épocas |
| Janela/passo de A | `10/5` épocas |
| `PI_INICIAL` da inferência | `[0.99, 0.01]` |
| Regra da inferência | 1 consecutiva OU 5% em `Presente` |
| `K_SINTETICO` | `0.01 * std(época)` |
| Suavização de B | `0.5` por célula |

O detector também pode ser escolhido por `ASSR_DETECTOR` ou, ao executar
`get_obs_matrix.py`/`hmm_inference.py` interativamente, pelo menu. O mesmo
detector deve ser usado para gerar B e fazer inferência.

## Artefatos presentes

### `results/observation_matrix.json`

- Detector: CSM.
- 11 arquivos, 8 frequências e 1.160 janelas por estado.
- Falso positivo global ESP: 2,7586%.
- Matriz B:

```text
Ausente  [0.545806, 0.384086, 0.041720, 0.026237, 0.002151]
Presente [0.149247, 0.342796, 0.119140, 0.162151, 0.226667]
```

### `results/transition_matrix.json`

- Ausente: 279 janelas, 268 autopassagens, 11 arquivos.
- Presente: 3.225 janelas, 3.170 autopassagens, 55 arquivos.
- Matriz A:

```text
Ausente  [0.960573, 0.039427]
Presente [0.017054, 0.982946]
```

### `results/inference_analysis.json`

O relatório foi gerado com CSM, janela/passo 10/10 e regra 1 consecutiva OU
5%. Ele contém resumo global, agrupamentos por dB e participante e detalhes por
arquivo. Consulte o próprio JSON para números de desempenho: ele é um
diagnóstico exploratório e pode mudar quando B ou as regras forem regeneradas.

## Estado de verificação

Em 25/09/2026:

- `py_compile` passou para todos os módulos de `src/`;
- os três JSON foram lidos com sucesso;
- todas as linhas das matrizes A e B somam 1;
- há 11 arquivos `*ESP.mat` e 55 arquivos `*dB.mat` em `data/`.

Os pipelines completos não foram reexecutados durante a criação desta pasta,
para não sobrescrever resultados que já fazem parte de alterações locais.

## Pendências e inconsistências conhecidas

1. `AGENTS.md` e `CONTEXTO_GET_MATRIX.md` determinam janela 10/passo 5
   consistente entre A e B, mas `get_obs_matrix.py` está em 10/10 e
   `get_tran_matrix.py` em 10/5. A duração implícita de A, portanto, não está na
   mesma escala temporal das observações usadas pela inferência.
2. O contexto técnico ainda descreve MSC como detector vigente e contém
   resultados antigos de MSC/10/5; o código e os artefatos atuais usam CSM/10/10.
3. O JSON de B registra `distribuicao_nula` como `Beta(1, 9)` mesmo quando o
   detector ativo é CSM. Os thresholds atuais coincidem numericamente com os de
   MSC pela transformação implementada, mas o metadado é semanticamente errado.
4. Várias chaves e funções ainda contêm `msc` no nome (`valor_msc`,
   `thresholds_msc`, `calcular_msc_janelas`) embora possam representar qualquer
   detector. Isso é compatibilidade histórica, não garantia de semântica MSC.
5. MMSC usa uma aproximação Beta para a média de duas MSC; o próprio código
   reconhece que essa não é a distribuição exata.
6. A busca em grade usa os mesmos 55 arquivos com estímulo para selecionar os
   parâmetros e medir detecção/falso positivo. Não há separação treino/validação
   nem validação cruzada, logo a métrica otimizada não é desempenho fora da
   amostra.
7. `PI_INICIAL_VALUES` contém `(0.40, 0.60)` duas vezes, causando trabalho
   duplicado. A extração de labels na busca também é repetida por frequência.
8. `search_hmm_parameters.py` só imprime o melhor resultado; não salva um
   artefato reproduzível com espaço de busca, ranking ou versão dos dados.
9. Não existe suíte automatizada de testes. As validações atuais estão nos
   próprios scripts e na compilação sintática.
10. A árvore de trabalho já possuía alterações não commitadas ao criar esta
    documentação, inclusive a renomeação de `hhm_*` para `hmm_*`. Preserve-as.

## Limites de interpretação

A não é Baum–Welch nem uma medição de persistência fisiológica. B(Presente) é
sintética, e `K_SINTETICO` não equivale diretamente a dB acústicos. Frequências
laterais e resultados da busca são controles exploratórios. Nada disso deve ser
apresentado como estimativa clínica validada.
