# Contexto técnico — parametrização do HMM para ASSR

Este documento apresenta o contexto do projeto e explica os programas que
parametrizam e aplicam o HMM:

- `src/get_obs_matrix.py`: matriz de emissão `B`;
- `src/get_tran_matrix.py`: matriz de transição `A`;
- `src/hhm_inference.py`: inferência Viterbi e decisão exploratória de ASSR.

Leia este arquivo antes de modificar qualquer um dos programas. Atualize-o na
mesma tarefa sempre que houver mudança de comportamento, configuração, formato
de saída, hipótese estatística ou resultado de referência.

## 1. Objetivo do projeto

O projeto investiga a detecção de resposta auditiva de estado estável (ASSR) em
EEG com um HMM de dois estados:

```text
ESTADOS = ["Ausente", "Presente"]
```

As tabelas tratadas nesta etapa são:

- `A[i, j] = P(estado seguinte = j | estado atual = i)`;
- `B[i, k] = P(label observado = k | estado = i)`.

Os dois scripts de parametrização estimam as tabelas offline e gravam artefatos
JSON em `results/`. O script `hhm_inference.py` consome esses artefatos, executa
Viterbi e aplica regras heurísticas de decisão por frequência e por arquivo.
Isso não constitui treinamento adicional nem decisão clínica final.

As estimativas atuais incluem hipóteses heurísticas e uma resposta sintética.
Não devem ser apresentadas como parâmetros clínicos validados.

## 2. Dados e geometria

Os dados ficam em `data/` e usam nomes como:

```text
AbESP.mat
Ab30dB.mat
Ab40dB.mat
...
```

- `ESP` indica ausência de estímulo;
- `30dB`, `40dB`, etc. indicam gravações com estímulo;
- o prefixo, como `Ab`, identifica o participante.

Cada arquivo MATLAB v7.3 contém:

- `x`: EEG no formato `(canais, épocas, amostras)`;
- `Fs`: frequência de amostragem;
- `freqEstim`: frequências-alvo;
- `binsM`: frequências de controle/ruído associadas aos alvos.

Nos arquivos atuais existem 16 canais, mas a quantidade de épocas e `Fs` varia.
Os arquivos ESP inspecionados têm entre 30 e 300 épocas e `Fs` de 1000 ou
1750 Hz. Não fixe 180 épocas nem uma única taxa de amostragem.

Os metadados atuais indicam:

```text
freqEstim = [81, 83, 85, 87, 89, 91, 93, 95] Hz
binsM     = [82, 84, 86, 88, 90, 92, 94, 96] Hz
```

Esses vetores devem continuar sendo lidos e validados nos arquivos, e não
substituídos por constantes implícitas.

## 3. Configuração compartilhada

Os dois programas usam atualmente:

```text
WINDOW_SIZE_EPOCHS = 10
WINDOW_STEP_EPOCHS = 5
```

Uma gravação com `N` épocas gera:

```text
floor((N - WINDOW_SIZE_EPOCHS) / WINDOW_STEP_EPOCHS) + 1
```

janelas completas, quando `N` é suficiente.

Essas constantes estão declaradas separadamente nos dois arquivos de
parametrização. `hhm_inference.py` importa as configurações e as funções do
detector diretamente de `get_obs_matrix.py`. Qualquer alteração de tamanho ou
passo ainda deve ser aplicada conscientemente a `get_obs_matrix.py` e
`get_tran_matrix.py`, seguida de nova geração de `A` e `B`.

As janelas se sobrepõem porque o passo é menor que o tamanho. Consequentemente,
janelas consecutivas não são estatisticamente independentes.

## 4. Matriz de emissão B — `get_obs_matrix.py`

### 4.1 Finalidade

O programa estima a distribuição dos labels de evidência condicionada a cada
estado. A matriz global possui duas linhas e cinco colunas:

| Estado | muito_baixo | baixo | medio | alto | muito_alto |
|---|---:|---:|---:|---:|---:|
| Ausente | ... | ... | ... | ... | ... |
| Presente | ... | ... | ... | ... | ... |

O script usa somente os arquivos `*ESP.mat` para a calibração das duas linhas.
A linha `Presente` é exploratória e resulta de injeção sintética sobre esses
mesmos sinais.

### 4.2 Detector espectral

Para cada época `m`, canal configurado e frequência, são extraídas as potências:

```text
P_alvo(m)  = |X_m(f_alvo)|²
P_ruido(m) = |X_m(f_ruido)|²
```

`f_alvo` vem de `freqEstim` e `f_ruido` do elemento correspondente de `binsM`.
Para uma janela com `M` épocas, calcula-se:

```text
F = média(P_alvo) / média(P_ruido)
```

Sob ausência de resposta, coeficientes aproximadamente gaussianos e
independentes e piso espectral localmente plano, o código usa a aproximação:

```text
F ~ F(2M, 2M)
```

Com 10 épocas, os graus de liberdade são `(20, 20)`. A estatística é uma razão
local de potências com referência F. Ela não é MSC nem CSM e não deve receber
esses nomes.

Se forem alterados número de bins de ruído, canais, agregação ou detector, os
graus de liberdade e a distribuição nula precisam ser derivados novamente.

### 4.3 Discretização

Configuração atual:

```python
NIVEIS_OBSERVACAO = [
    "muito_baixo", "baixo", "medio", "alto", "muito_alto"
]
P_VALUE_BOUNDARIES = [0.50, 0.10, 0.05, 0.01]
SIGNIFICANCE_LEVEL = 0.05
```

Os thresholds são quantis teóricos da distribuição F, e não quantis empíricos
dos dados ESP. Para janelas de 10 épocas:

```text
Thresholds F = [1.00000000, 1.79384331, 2.12415521, 2.93773528]
F crítico para alpha=0.05 = 2.124155
```

| Label | Faixa de p-valor |
|---|---:|
| `muito_baixo` | `p > 0.50` |
| `baixo` | `0.10 < p <= 0.50` |
| `medio` | `0.05 < p <= 0.10` |
| `alto` | `0.01 < p <= 0.05` |
| `muito_alto` | `p <= 0.01` |

Se a quantidade de labels mudar, deve existir exatamente uma fronteira a menos
que o número de labels.

### 4.4 Linha Ausente

O programa:

1. lê todos os arquivos `*ESP.mat`;
2. calcula o valor F de cada janela e frequência;
3. converte cada valor em um label;
4. soma as contagens das oito frequências;
5. aplica pseudocontagem `SMOOTHING = 0.5` por célula;
6. normaliza o histograma global.

As frequências são agrupadas para estimar uma única matriz B. Resultados por
frequência continuam preservados como diagnóstico. Agrupar B não autoriza
concatenar sequências temporais de frequências, participantes ou condições
diferentes durante a futura inferência do HMM.

Participantes com mais janelas contribuem mais para o histograma. Caso se queira
peso igual por participante, essa regra deverá ser implementada explicitamente.

### 4.5 Linha Presente

Em cada época ESP, o programa injeta uma senoide:

```text
tom(t) = K_SINTETICO * std(época) * sin(2*pi*f_alvo*t)
```

O valor atual é `K_SINTETICO = 0.01`. Depois da injeção, aplica-se exatamente o
mesmo detector e os mesmos thresholds.

Não há equivalência estabelecida entre `K_SINTETICO` e nível acústico em dB.
Essa linha não é uma distribuição fisiológica validada e não deve ser usada
como validação independente, pois parte dos mesmos arquivos usados em Ausente.

### 4.6 Saída da matriz B

Ao executar o script, o arquivo abaixo é criado ou sobrescrito:

```text
results/observation_matrix.json
```

O JSON contém:

- `estados_linhas` e `labels_colunas`;
- `matriz`: lista numérica 2 × 5, pronta para consumo;
- `probabilidades`: a mesma tabela indexada por nomes;
- contagens antes da normalização;
- configuração do detector e da injeção;
- thresholds, valor crítico e quantidade de dados;
- falso positivo global em ESP;
- diagnóstico e contagens por frequência;
- aviso sobre a natureza sintética de `B(Presente)`.

Os milhares de registros intermediários auditáveis permanecem no retorno da
função Python, mas não são escritos no JSON da tabela para evitar duplicação.

## 5. Matriz de transição A — `get_tran_matrix.py`

### 5.1 Finalidade e interpretação

O programa constrói uma matriz 2 × 2:

| Estado atual | próximo Ausente | próximo Presente |
|---|---:|---:|
| Ausente | `p_self_ausente` | `1 - p_self_ausente` |
| Presente | `1 - p_self_presente` | `p_self_presente` |

Arquivos `*ESP.mat` são associados ao estado Ausente e arquivos `*dB.mat` ao
estado Presente.

### 5.2 Heurística de persistência

Para cada arquivo que produz `N` janelas, o código conta:

```text
N - 1 autopassagens
1 saída implícita para o outro estado
```

Depois soma todos os arquivos da condição:

```text
p_self = total de autopassagens / total de janelas
```

Assim, cada gravação contribui implicitamente com uma saída, embora essa
transição não tenha sido observada nos dados. A matriz mede principalmente a
persistência induzida pela duração das gravações e pelo janelamento.

Esta é uma heurística inicial. Ela não é uma estimativa supervisionada de
transições reais, não decorre de estados ocultos inferidos e não é resultado de
Baum–Welch. Arquivos mais longos produzem maior permanência e têm maior peso.

### 5.3 Validação de entrada

O programa confirma que:

- `x` e `Fs` existem;
- `x` possui três dimensões;
- o primeiro eixo possui 16 canais nos dados atuais;
- o número de amostras por época coincide com `int(Fs)`.

Se o formato dos dados mudar, essa validação deve ser revista com base nos
arquivos reais, não simplesmente removida.

### 5.4 Saída da matriz A

Ao executar o script, o arquivo abaixo é criado ou sobrescrito:

```text
results/transition_matrix.json
```

O JSON contém:

- `estados_linhas` e `estados_colunas`;
- `matriz`: lista numérica 2 × 2;
- `probabilidades`: tabela indexada por estado de origem e destino;
- configuração de janelamento e padrões de arquivos;
- totais de arquivos, épocas, janelas e autopassagens;
- detalhamento por arquivo;
- descrição explícita da heurística e aviso metodológico.

## 6. Resultados de referência atuais

### 6.1 Matriz de emissão B

Com 11 arquivos ESP, oito frequências, janela 10, passo 5 e pseudocontagem 0.5:

```text
2232 janelas por estado
FP global observado em ESP = 5.06%
```

| Estado | muito_baixo | baixo | medio | alto | muito_alto |
|---|---:|---:|---:|---:|---:|
| Ausente | 0.51399 | 0.37659 | 0.05840 | 0.04095 | 0.01007 |
| Presente | 0.31305 | 0.37525 | 0.09420 | 0.09734 | 0.12016 |

O falso positivo por frequência varia; 83 Hz apresentou 10.04%, embora a taxa
global tenha ficado próxima de 5%. Preserve esse diagnóstico.

### 6.2 Matriz de transição A

Com 279 janelas ESP e 3225 janelas dos arquivos com estímulo:

| Estado atual | Ausente | Presente |
|---|---:|---:|
| Ausente | 0.960573 | 0.039427 |
| Presente | 0.017054 | 0.982946 |

Esses valores refletem a regra de uma saída implícita por arquivo. Não os
interprete como duração fisiológica validada dos estados.

## 7. API Python

Matriz B:

```python
from src.get_obs_matrix import construir_matrizes_observacao

resultados = construir_matrizes_observacao("data")
matriz_global = resultados["global"]
diagnosticos = resultados["por_frequencia"]
```

Matriz A:

```python
from src.get_tran_matrix import construir_matriz_transicao

resultado = construir_matriz_transicao("data")
matriz = resultado["matriz"]
```

Importar os dois módulos de parametrização não grava arquivos. A escrita ocorre
no bloco principal ou quando as funções `salvar_tabela_observacao` e
`salvar_tabela_transicao` são chamadas explicitamente. A inferência é executada
atualmente pelo ponto de entrada de linha de comando descrito na próxima seção.

## 8. Execução e regeneração das tabelas

Na raiz do projeto:

```bash
.venv/bin/python src/get_obs_matrix.py
.venv/bin/python src/get_tran_matrix.py
.venv/bin/python src/hhm_inference.py
```

Os comandos imprimem os resumos e atualizam:

```text
results/observation_matrix.json
results/transition_matrix.json
results/inference_analysis.json
```

Verificação sintática:

```bash
.venv/bin/python -m py_compile \
  src/get_obs_matrix.py \
  src/get_tran_matrix.py \
  src/hhm_inference.py
```

## 9. Inferência e decisão — `hhm_inference.py`

### 9.1 Entrada e unidade de inferência

O script lê a matriz A de `results/transition_matrix.json` e a matriz B de
`results/observation_matrix.json`. A ordem dos estados e labels e a soma das
linhas são validadas antes do processamento. A distribuição inicial é
`PI_INICIAL = [0.99, 0.01]`, importada de `get_obs_matrix.py`.

São processados os 55 arquivos `*dB.mat`. Para cada arquivo, as oito
frequências de `freqEstim` geram oito sequências Viterbi independentes. Não há
concatenação entre frequências, participantes, arquivos ou níveis de estímulo.
O Viterbi opera em log-espaço para evitar underflow e retorna o caminho mais
provável entre os estados `Ausente` e `Presente`.

O detector não é reimplementado. O programa importa de `get_obs_matrix.py` a
leitura dos `.mat`, extração de potência, cálculo da razão F por janela,
thresholds e discretização. Assim, cada sequência usa:

```text
F = média(P_freqEstim[i]) / média(P_binsM[i])
```

com janela de 10 épocas, passo 5 e os cinco labels da matriz B.

### 9.2 Decisão por frequência e por arquivo

A configuração atual é:

```python
MIN_CONSECUTIVE = 3
MIN_PERCENT = 0.30
MODO_REGRA_DECISAO = "OR"
K_DE_N = 2
K_DE_N_LATERAL = 2
```

Uma frequência é detectada quando o caminho Viterbi possui pelo menos 30% das
janelas em `Presente` **ou** pelo menos três janelas `Presente` consecutivas.
O valor vigente é três, não cinco. Um arquivo é detectado quando pelo menos
duas de suas oito frequências satisfazem essa regra (`2-de-8`).

Como as janelas se sobrepõem, três janelas consecutivas, com tamanho 10 e passo
5, cobrem 20 épocas únicas e não são três observações independentes. Os
limiares de 30%, três consecutivas e 2-de-8 são escolhas heurísticas ainda não
validadas.

### 9.3 Falso positivo pelas frequências laterais

O controle lateral usa somente os valores de `binsM` e cria pares circulares:

```text
(82, 84), (84, 86), (86, 88), (88, 90),
(90, 92), (92, 94), (94, 96), (96, 82) Hz
```

Em cada par, o primeiro valor é o pseudo-alvo e o segundo é o pseudo-ruído. A
mesma inferência, regra por frequência e regra `2-de-8` são aplicadas. Uma
decisão positiva no conjunto lateral é contabilizada como falso positivo do
arquivo.

O controle é calculado nos próprios arquivos com estímulo para incluir
artefatos da condição de aquisição. Porém, esses pares laterais não foram usados
para estimar a linha empírica `B(Ausente)`, que foi ajustada com
`freqEstim[i] / binsM[i]` nos arquivos ESP. Logo, a taxa lateral é um diagnóstico
exploratório, não uma estimativa clínica validada de especificidade.

### 9.4 Agregação e saída

O arquivo `results/inference_analysis.json` contém:

- configuração das regras de decisão;
- taxas globais por arquivo;
- detecção agrupada por nível de estímulo;
- detecção e falso positivo por participante;
- decisão e número de frequências positivas para cada arquivo;
- por frequência, percentual em `Presente`, maior sequência consecutiva,
  decisão e caminho Viterbi completo.

A taxa por nível é a fração dos 11 arquivos daquele nível classificados como
positivos. A taxa por participante é a fração dos cinco arquivos desse
participante classificados como positivos.

O relatório persistido não contém os valores F, p-valores, labels ou intervalos
de épocas de cada janela. Os labels existem durante o processamento, mas são
omitidos por `construir_relatorio`, e os valores F retornados pelo detector não
são incorporados ao resultado. Essa é uma limitação de auditabilidade da
implementação atual.

### 9.5 Resultados de referência atuais

Com a configuração acima, foram processados 55 arquivos, 11 participantes,
cinco níveis e oito frequências por arquivo. O resultado global foi:

```text
Detecção nos arquivos com estímulo: 18/55 = 32.73%
Falso positivo lateral:             24/55 = 43.64%
```

| Nível | Detectados | Taxa de detecção |
|---:|---:|---:|
| 30 dB | 5/11 | 45.45% |
| 40 dB | 2/11 | 18.18% |
| 50 dB | 7/11 | 63.64% |
| 60 dB | 2/11 | 18.18% |
| 70 dB | 2/11 | 18.18% |

| Participante | Detecção estímulo | Falso positivo lateral |
|---|---:|---:|
| Ab | 40% | 60% |
| An | 20% | 40% |
| Bb | 20% | 0% |
| Er | 40% | 60% |
| Lu | 40% | 20% |
| Qu | 40% | 80% |
| Sa | 60% | 60% |
| So | 0% | 40% |
| Ti | 20% | 20% |
| Vi | 40% | 60% |
| Wr | 40% | 40% |

A taxa lateral global supera a taxa de detecção e a detecção não cresce
monotonicamente com o nível. Esses resultados indicam que as regras e os
parâmetros atuais ainda não separam adequadamente alvo e controle; não devem ser
interpretados como desempenho clínico.

## 10. Regras para alterações futuras

1. Inspecione os `.mat` autênticos antes de mudar leitores ou eixos.
2. Mantenha o janelamento de A e B consistente ou documente por que divergem.
3. Não fixe `Fs`, número de épocas, `freqEstim` ou `binsM` a partir de um único
   participante.
4. Preserve participante, condição, canal, frequência, janela e `Fs` quando o
   processamento depender deles.
5. Mantenha B global e os diagnósticos por frequência, salvo decisão explícita
   em contrário.
6. Não chame a observação atual de MSC ou CSM.
7. Reavalie a distribuição nula ao mudar o detector ou sua agregação.
8. Não converta `K_SINTETICO` em dB acústicos sem calibração experimental.
9. Trate A como heurística de persistência enquanto não houver uma sequência de
   estados observada ou um método de estimação do HMM.
10. Não apresente A ou B como parâmetros clínicos validados.
11. Ao mudar código ou configuração, regenere os dois JSON quando a alteração
    afetar ambas as tabelas.
12. Atualize este documento com novos resultados de referência.
13. Ao alterar regras de inferência, regenere `inference_analysis.json` e
    atualize a Seção 9.

Depois de alterações, confirme que:

- todas as linhas de A e B somam 1;
- os JSON são válidos e reproduzem as matrizes impressas;
- as contagens globais de B equivalem à soma por frequência;
- os totais de A correspondem aos arquivos e janelas listados;
- os scripts podem ser importados sem executar processamento ou escrita;
- nenhum arquivo de dados foi renomeado ou sobrescrito.
