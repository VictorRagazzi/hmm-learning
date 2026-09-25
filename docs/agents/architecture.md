# Arquitetura

## Objetivo

O projeto explora detecção de resposta auditiva de estado estável (ASSR) em EEG
com um HMM discreto de dois estados. A observação contínua de um detector
espectral é convertida em cinco labels; as matrizes A e B e uma distribuição
inicial alimentam Viterbi. A decisão final por frequência é uma regra adicional
sobre o caminho decodificado, não parte do HMM.

## Fluxo principal

```text
data/*.mat
   |
   +-- *ESP.mat -------------------------------+
   |      |                                    |
   |      v                                    v
   |  get_obs_matrix.py                  get_tran_matrix.py
   |  detector + discretização           duração dos arquivos
   |      |                                    |
   |      v                                    v
   |  observation_matrix.json            transition_matrix.json
   |                \                         /
   |                 \                       /
   +-- *dB.mat --------> hmm_inference.py <---+
                           |       |
                           |       +-- baseline do detector bruto
                           v
                    inference_analysis.json

search_hmm_parameters.py reutiliza get_obs_matrix.py e hmm_inference.py para
recalibrar B e avaliar uma grade de parâmetros; atualmente só imprime a melhor.
```

## Modelo de dados

Cada `.mat` deve fornecer:

- `x`: `(canais, épocas, amostras)`;
- `Fs`: taxa de amostragem do próprio arquivo;
- `freqEstim`: frequências de estímulo;
- `binsM`: frequências laterais de controle, pareadas por posição.

Não se deve fixar `Fs` nem quantidade de épocas. A calibração valida que
`freqEstim` e `binsM` sejam consistentes entre arquivos ESP. O código localiza o
bin da FFT mais próximo em vez de assumir que o alvo coincide implicitamente
com um índice.

## Fronteiras entre módulos

- `detectors.py` conhece a estatística, sua cauda nula e seus thresholds.
- `get_obs_matrix.py` conhece dados EEG, janelamento, discretização e calibração
  de B. Ele também centraliza configurações importadas pela inferência.
- `get_tran_matrix.py` estima A isoladamente a partir do número de janelas.
- `hmm_inference.py` consome A/B, gera sequências de labels, roda Viterbi,
  decide e agrega resultados.
- `search_hmm_parameters.py` orquestra avaliações repetidas reutilizando as
  funções anteriores.
- `viterbi.py` é apenas didático e não é importado pelo pipeline.

Há acoplamento importante: `hmm_inference.py` importa constantes e funções de
`get_obs_matrix.py`. Trocar detector, canal, labels, janela ou passo exige
regenerar B e validar compatibilidade com A. Atualmente A não compartilha essas
constantes por importação, o que permitiu a divergência 10/10 versus 10/5.

## Semântica das matrizes

`A[i,j] = P(S_t=j | S_{t-1}=i)` é hoje uma heurística. Para cada arquivo com N
janelas são contadas N-1 autopassagens e uma saída implícita. Não existem
estados ocultos rotulados nem treinamento Baum–Welch.

`B[i,k] = P(O_t=label_k | S_t=i)` é uma matriz global compartilhada pelas oito
frequências. `Ausente` vem de histogramas ESP; `Presente`, dos mesmos sinais
após injeção sintética. Somam-se contagens de todas as frequências e só então se
normaliza, com pseudocontagem 0,5.

As sequências temporais continuam independentes por arquivo e frequência. Uma
B global não autoriza concatenar participantes, condições ou frequências.

## Inferência e decisão

O Viterbi opera em log-espaço e produz o caminho de máxima probabilidade. Em
paralelo, a baseline marca como positiva cada janela cujo p-valor bruto seja
menor ou igual a alpha. A mesma regra de percentual/consecutividade é aplicada
a ambas as sequências binárias.

Cada frequência é a unidade de avaliação. `binsM` não entra como denominador do
detector: suas frequências são processadas separadamente como controle lateral.

## Persistência e efeitos colaterais

Importar `get_obs_matrix.py` e `get_tran_matrix.py` não escreve arquivos. Seus
pontos de entrada criam ou sobrescrevem os JSON em `results/`. A inferência
também sobrescreve seu relatório. A busca altera temporariamente
`gom.K_SINTETICO` em memória, restaura-o em `finally` e não grava saída.

## Como executar

Na raiz, prefira definir o detector explicitamente para evitar menu e garantir
consistência:

```bash
ASSR_DETECTOR=csm .venv/bin/python src/get_obs_matrix.py
.venv/bin/python src/get_tran_matrix.py
ASSR_DETECTOR=csm .venv/bin/python src/hmm_inference.py
ASSR_DETECTOR=csm .venv/bin/python src/search_hmm_parameters.py
```

Antes de regenerar artefatos, resolva conscientemente a divergência de passo
documentada em `status.md`. Depois confira schema, somas de linha, contagens e
falso positivo global/por frequência, conforme `AGENTS.md`.
