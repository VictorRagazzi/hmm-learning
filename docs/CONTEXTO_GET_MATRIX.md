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
Viterbi e aplica uma regra heurística de decisão a cada frequência.
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

Para cada época `m`, canal configurado e frequência-alvo, extrai-se o
coeficiente FFT complexo `X_m(f_alvo)`. Para uma janela com `M` épocas, a
magnitude quadrática da coerência (MSC) é:

```text
MSC = |soma_m X_m(f_alvo)|² / (M * soma_m |X_m(f_alvo)|²)
```

A MSC pertence a `[0, 1]` e usa magnitude e consistência de fase entre épocas.
Ela não é CSM, que normalizaria cada coeficiente para módulo unitário antes da
soma, nem é o teste F espectral local usado anteriormente.

Sob ausência de resposta, coeficientes complexos gaussianos circulares e
épocas independentes, o código usa:

```text
MSC ~ Beta(1, M - 1)
```

Com 10 épocas, a distribuição nula é `Beta(1, 9)`. Isso pressupõe que as épocas
preservem uma referência de fase comum em relação ao estímulo. `binsM` continua
sendo lido, validado e preservado como metadado e controle lateral, mas não é
denominador da MSC.

Se forem alterados número de épocas, canais, agregação ou detector, a
distribuição nula precisa ser derivada novamente.

### 4.3 Discretização

Configuração atual:

```python
NIVEIS_OBSERVACAO = [
    "muito_baixo", "baixo", "medio", "alto", "muito_alto"
]
P_VALUE_BOUNDARIES = [0.50, 0.10, 0.05, 0.01]
SIGNIFICANCE_LEVEL = 0.05
```

Os thresholds são quantis teóricos da distribuição Beta, e não quantis empíricos
dos dados ESP. Para janelas de 10 épocas:

```text
Thresholds MSC = [0.07412529, 0.22573632, 0.28312884, 0.40051575]
MSC crítica para alpha=0.05 = 0.283129
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
2. calcula o valor MSC de cada janela e frequência;
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

A senoide reinicia com a mesma fase em todas as épocas e, portanto, representa
uma resposta sintética coerente em fase. Ela não modela jitter de fase,
variabilidade fisiológica de latência nem perda de sincronismo entre épocas;
isso pode tornar `B(Presente)` otimista para um detector de coerência.

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
- thresholds MSC, MSC crítica e quantidade de dados;
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

Com 11 arquivos ESP, oito frequências, janela 10, passo 5 e pseudocontagem 0,5,
foram obtidas 2232 janelas por estado e falso positivo global de 4,08%:

| Estado | muito_baixo | baixo | medio | alto | muito_alto |
|---|---:|---:|---:|---:|---:|
| Ausente | 0,503692 | 0,414634 | 0,040501 | 0,035578 | 0,005594 |
| Presente | 0,135825 | 0,321996 | 0,110316 | 0,180577 | 0,251287 |

O falso positivo por frequência foi: 81 Hz 2,51%; 83 Hz 5,38%; 85 Hz 4,66%;
87 Hz 5,38%; 89 Hz 4,30%; 91 Hz 2,15%; 93 Hz 2,87%; e 95 Hz 5,38%.

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
leitura dos `.mat`, extração dos coeficientes FFT complexos, cálculo da MSC por
janela, thresholds e discretização. Assim, cada sequência usa:

```text
MSC = |soma_m X_m(freqEstim[i])|² /
      (M * soma_m |X_m(freqEstim[i])|²)
```

com janela de 10 épocas, passo 5 e os cinco labels da matriz B.

### 9.2 Decisão e agregação por frequência

A configuração atual é:

```python
MIN_CONSECUTIVE = 4
MIN_PERCENT = 0.10
MODO_REGRA_DECISAO = "OR"
```

Uma frequência é detectada quando o caminho Viterbi possui pelo menos 10% das
janelas em `Presente` **ou** pelo menos quatro janelas `Presente` consecutivas.
Cada frequência é um
experimento independente na agregação: se uma das oito frequências de um
arquivo for detectada, esse arquivo contribui com `1/8 = 12,5%` para a taxa,
sem uma decisão intermediária `k-de-N` por arquivo.

Como as janelas se sobrepõem, três janelas consecutivas, com tamanho 10 e passo
5, cobrem 20 épocas únicas e não são três observações independentes. Os
limiares por frequência são escolhas heurísticas ainda não validadas.

### 9.3 Falso positivo pelas frequências laterais

O controle lateral aplica a MSC diretamente em cada frequência de `binsM`:

```text
82, 84, 86, 88, 90, 92, 94 e 96 Hz
```

Como a MSC não usa denominador espectral, não há mais pares circulares de
pseudo-alvo/pseudo-ruído. A mesma inferência e regra por frequência são
aplicadas às oito frequências laterais. Cada lateral é um experimento
independente no cálculo da taxa de falso positivo.

O controle é calculado nos próprios arquivos com estímulo para incluir
artefatos da condição de aquisição. Porém, essas frequências laterais não foram
usadas para estimar a linha empírica `B(Ausente)`, que foi ajustada nas
frequências de `freqEstim` dos arquivos ESP. Logo, a taxa lateral é um
diagnóstico exploratório, não uma estimativa clínica validada de especificidade.

### 9.4 Agregação e saída

O arquivo `results/inference_analysis.json` contém:

- configuração das regras de decisão;
- taxas globais por frequência;
- detecção e falso positivo lateral agrupados por nível de estímulo;
- detecção e falso positivo por participante;
- número de frequências positivas para cada arquivo;
- por frequência, percentual em `Presente`, maior sequência consecutiva,
  decisão e caminho Viterbi completo.

As taxas de detecção e de falso positivo lateral usam como denominador o total
de frequências avaliadas no agrupamento (global, nível de estímulo ou
participante), e não o número de arquivos. Com oito frequências por arquivo,
um grupo de 11 arquivos possui 88 experimentos de estímulo e 88 laterais.

Os resultados intermediários em memória preservam valores MSC, p-valores,
labels e intervalos de épocas. O relatório persistido ainda omite esses campos e
mantém apenas os caminhos e resumos por frequência; essa continua sendo uma
limitação de auditabilidade do JSON de inferência.

### 9.5 Resultados de referência atuais

Com 55 arquivos, 8 frequências por arquivo e a configuração descrita acima,
foram avaliados 440 experimentos de estímulo e 440 experimentos laterais:

```text
Detecção nas frequências de estímulo: 98/440 = 22,27%
Falso positivo nas laterais:          18/440 =  4,09%
```

| Nível | Frequências detectadas | Taxa de detecção | Laterais positivas | Falso positivo |
|---:|---:|---:|---:|---:|
| 30 dB | 5/88 | 5,68% | 3/88 | 3,41% |
| 40 dB | 19/88 | 21,59% | 9/88 | 10,23% |
| 50 dB | 17/88 | 19,32% | 2/88 | 2,27% |
| 60 dB | 30/88 | 34,09% | 3/88 | 3,41% |
| 70 dB | 27/88 | 30,68% | 1/88 | 1,14% |

Essas taxas são diagnósticos exploratórios por frequência, não desempenho
clínico validado. O código valida o detector, o janelamento e as fronteiras de
p-valor da matriz B antes de executar a inferência.

## 10. Regras para alterações futuras

1. Inspecione os `.mat` autênticos antes de mudar leitores ou eixos.
2. Mantenha o janelamento de A e B consistente ou documente por que divergem.
3. Não fixe `Fs`, número de épocas, `freqEstim` ou `binsM` a partir de um único
   participante.
4. Preserve participante, condição, canal, frequência, janela e `Fs` quando o
   processamento depender deles.
5. Mantenha B global e os diagnósticos por frequência, salvo decisão explícita
   em contrário.
6. A observação atual é MSC; não a chame de CSM nem de teste F espectral local.
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
