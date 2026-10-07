# Contexto técnico — parametrização do HMM para ASSR

**Nova etapa de 05/10/2026:** emissão contínua com uma única Beta por MAP para
Presente, prioris configuráveis de μ e κ e comparação LOSO com MLE e detector
bruto. Calibração Presente usa estímulos reais ≥50 dB; Ausente permanece MLE
em ESP. **Decisão posterior vigente:** explorar in-sample com todos os pacientes
no ajuste e avaliação, sem LOSO, conforme **Seção 24**. Método fixo histórico
na Seção 22; search LOSO histórico na Seção 23.
A referência histórica MLE está na Seção 20; a mistura preditiva MCMC da
Seção 21 e os histogramas das Seções 18–19 continuam como experimentos separados.

Este documento apresenta o contexto do projeto e explica os programas que
parametrizam e aplicam o HMM:

- `src/get_obs_matrix.py`: matriz de emissão `B`;
- `src/get_tran_matrix.py`: matriz de transição `A`;
- `src/hmm_inference.py`: inferência Viterbi e decisão exploratória de ASSR.

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
JSON em `results/`. O script `hmm_inference.py` consome esses artefatos, executa
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
parametrização. `hmm_inference.py` importa as configurações e as funções do
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
O detector `csm` registrado no código é o T² circular de Hotelling, escrito
como um F. Sob as mesmas hipóteses gaussianas e de variância circular, a
transformação usada no código o torna algebricamente equivalente à MSC; são
duas formulações do mesmo teste, não evidências independentes. Isso não é o
teste F espectral local usado anteriormente, que estima ruído em bins vizinhos.

Sob ausência de resposta, coeficientes complexos gaussianos circulares e
épocas independentes, o código usa:

```text
MSC ~ Beta(1, M - 1)
```

Com 10 épocas, a distribuição nula é `Beta(1, 9)`. Isso pressupõe que as épocas
preservem uma referência de fase comum em relação ao estímulo. `binsM` continua
sendo lido, validado e preservado como metadado e controle lateral, mas não é
denominador da MSC.

Rayleigh testa concentração de fase e usa atualmente a cauda aproximada
`exp(-M * PLV²)`. `MMSC` neste repositório é uma estatística split-half própria:
calcula MSC em duas metades e tira a média. Como a média de duas variáveis
`Beta(1, floor(M/2)-1)` não é Beta, sua cauda é obtida pela convolução numérica
dessas distribuições, com tabela interpolada. Em janelas ímpares, a última
época não entra no split-half. Esse detector continua exploratório.

`hotelling` usa o vetor bidimensional `[Re(X), Im(X)]`, estima sua covariância
2×2 completa e testa média zero. Para `M > 2`, a forma escalada do T² segue
`F(2, M-2)` sob normalidade multivariada. Diferentemente do CSM circular, ele
permite variâncias diferentes e correlação entre real e imaginário.

`spectral_f` usa uma matriz com o coeficiente alvo e dois bins locais de ruído:

```text
F_local = M * |media(X_alvo)|² / media(|X_ruido|²)
```

Sob coeficientes complexos gaussianos circulares, bins independentes e mesma
variância, usa `F(2, 4M)`. Os bins de ruído são os mais próximos que não
coincidem com o alvo nem com qualquer frequência de `freqEstim`. Para laterais,
isso impede que uma frequência estimulada adjacente entre no denominador. A
regra é determinística, mas igualdade de variância e independência espectral
ainda precisam ser verificadas nos dados reais.

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
.venv/bin/python src/hmm_inference.py
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
  src/hmm_inference.py
```

## 9. Inferência e decisão — `hmm_inference.py`

### 9.1 Entrada e unidade de inferência

O script lê a matriz A de `results/transition_matrix.json`. Para cada detector
selecionado, calibra em memória uma matriz B própria a partir dos arquivos ESP,
usando o mesmo fluxo de `get_obs_matrix.py`. A distribuição inicial é
`PI_INICIAL = [0.99, 0.01]`, importada de `get_obs_matrix.py`.

As configurações a avaliar ficam no array do topo do arquivo:

```python
DETECTOR_COMBINATIONS = [("rayleigh", "csm")]
```

Cada tupla é uma execução: `("rayleigh",)` seleciona somente Rayleigh e
`("rayleigh", "csm")` combina ambos. As chaves disponíveis no registro são
`msc`, `mmsc`, `rayleigh` e `csm`.

São processados os 55 arquivos `*dB.mat`. Para cada arquivo, as oito
frequências de `freqEstim` geram oito sequências Viterbi independentes. Não há
concatenação entre frequências, participantes, arquivos ou níveis de estímulo.
O Viterbi opera em log-espaço para evitar underflow e retorna o caminho mais
provável entre os estados `Ausente` e `Presente`.

Os detectores não são reimplementados. O programa reutiliza o registro de
`detectors.py`, a leitura dos `.mat` e a extração dos coeficientes FFT de
`get_obs_matrix.py`. Cada detector produz seu próprio valor, p-valor, threshold
teórico e label em cada janela.

Quando há mais de um detector, o Viterbi combina as emissões sincronizadas por
produto:

```text
P(o_t | estado) = produto_d B_d[estado, label_d(t)]
```

No código, o produto é calculado como soma de log-probabilidades. Isso assume
independência condicional entre detectores dado o estado. Como os testes são
calculados sobre os mesmos coeficientes de EEG, eles podem ser correlacionados;
a fusão é uma aproximação exploratória e pode supercontar evidência.

### 9.2 Decisão e agregação por frequência

A configuração atual é:

```python
MIN_CONSECUTIVE = 1
MIN_PERCENT = 0.05
MODO_REGRA_DECISAO = "OR"
```

Uma frequência é detectada quando o caminho Viterbi possui pelo menos 5% das
janelas em `Presente` **ou** pelo menos uma janela `Presente`.
Cada frequência é um
experimento independente na agregação: se uma das oito frequências de um
arquivo for detectada, esse arquivo contribui com `1/8 = 12,5%` para a taxa,
sem uma decisão intermediária `k-de-N` por arquivo.

Como as janelas se sobrepõem, três janelas consecutivas, com tamanho 10 e passo
5, cobrem 20 épocas únicas e não são três observações independentes. Os
limiares por frequência são escolhas heurísticas ainda não validadas.

### 9.3 Falso positivo pelas frequências laterais

O controle lateral aplica cada detector diretamente em cada frequência de
`binsM`:

```text
82, 84, 86, 88, 90, 92, 94 e 96 Hz
```

A mesma inferência e regra por frequência são aplicadas às oito frequências
laterais. Cada lateral é um experimento independente no cálculo da taxa de
falso positivo.

O controle é calculado nos próprios arquivos com estímulo para incluir
artefatos da condição de aquisição. Porém, essas frequências laterais não foram
usadas para estimar a linha empírica `B(Ausente)`, que foi ajustada nas
frequências de `freqEstim` dos arquivos ESP. Logo, a taxa lateral é um
diagnóstico exploratório, não uma estimativa clínica validada de especificidade.

### 9.4 Agregação e saída

O arquivo `results/inference_analysis.json` contém:

- lista de combinações de detectores e regra de fusão;
- acurácia balanceada, detecção e falso positivo do HMM por combinação;
- as mesmas métricas do detector bruto separadas e identificadas por teste;
- por combinação, arquivo e frequência: labels de cada detector, caminho
  Viterbi, valores, p-valores e decisão bruta de cada teste.

As taxas de detecção e de falso positivo lateral usam como denominador o total
de frequências avaliadas no agrupamento global ou por nível de estímulo, e não
o número de arquivos. Com oito frequências por arquivo, 11 gravações no mesmo
nível correspondem a 88 experimentos de estímulo e 88 laterais. O relatório
agrupa resultados por intensidade (dB); não imprime taxas por participante.

Além da análise por janelas, o relatório aplica cada detector uma vez a todas
as épocas de cada gravação. Essa é uma baseline descritiva: mantém a mesma
unidade de decisão por frequência, mas não atravessa o Viterbi nem a regra
temporal das janelas. O alfa por frequência não garante 5% de erro por
gravação, pois cada gravação contém vários alvos e controles.

O JSON preserva os valores, p-valores, labels e intervalos por detector. Isso
pode torná-lo consideravelmente maior ao selecionar várias combinações.

A saída no terminal imprime resumo global e, por intensidade, detecção e falso
positivo lateral para HMM e detectores brutos, além da baseline de gravação
completa. Participante, arquivo e frequência seguem preservados nos registros
detalhados para rastreabilidade, sem agregar ou priorizar métricas por
participante.

### 9.5 Busca de parâmetros

`src/search_hmm_parameters.py` possui seu próprio array
`DETECTOR_COMBINATIONS`. A busca varia detector ou combinação juntamente com
`K_SINTETICO`, `PI_INICIAL`, a matriz de transição `A` e a regra de decisão.
`PI_INICIAL` usa `(p_Ausente, 1-p_Ausente)` com `p_Ausente` em
`0.99, 0.95, 0.90, ..., 0.50`. Para `A`, usa os mesmos valores nas grades de
autopermanência de Ausente e Presente e inclui também a matriz heurística
salva em `transition_matrix.json`. A busca rejeita combinações que fundem MSC
e CSM, aplica `FP_MAXIMO_HMM = 0.05` e imprime o melhor resultado por detector
ou combinação e por intensidade, incluindo a melhor `A` e a baseline de
gravação completa. Os resultados são gravados atomicamente em
`results/search_hmm_parameters.json`; a opção `--resume` retoma as combinações
completas após interrupção. Na busca completa atual, foram avaliadas 73.810
configurações de alto nível e 26.571.600 regras temporais, com 8.571.044
regras excluídas por exceder o limite de FP. A melhor configuração foi
Rayleigh+CSM, `K_SINTETICO=0.01`, `PI_INICIAL=[0.5, 0.5]`,
`A=[[0.5, 0.5], [0.15, 0.85]]` e decisão por 5 consecutivas AND 30% das
janelas. Ela detectou 155/440 frequências de estímulo (35,23%) e marcou
21/440 frequências laterais como positivas (4,77%), com acurácia balanceada de
65,23%. Essa otimização usa os mesmos arquivos na seleção e na avaliação,
portanto é ajuste exploratório in-sample: não é Baum–Welch,
não corresponde a transições ocultas observadas e não constitui validação
independente. Um grid sem candidatos sob 5% também é salvo e reportado.

Os resultados detalhados por combinação, nível em dB e detector bruto estão
no JSON. As taxas permanecem diagnósticos exploratórios por frequência, não
desempenho clínico validado.

## 10. Experimento Rayleigh com emissão contínua real

`src/continuous_rayleigh_hmm.py` é um experimento separado do pipeline de B
discreta. Ele não altera `observation_matrix.json` nem substitui a calibração
sintética vigente. Seu objetivo é testar, de forma deliberadamente in-sample,
se a discretização em cinco labels e a senoide sintética limitam o HMM.

O experimento usa somente Rayleigh e avalia janelas de 10, 20, 30 e 60 épocas,
sempre com passo igual à metade da janela. Para cada tamanho, calcula o p-valor
de todas as janelas e ajusta duas densidades Beta contínuas:

```text
Ausente: p-valores de freqEstim nos arquivos ESP
Presente: p-valores de freqEstim nos arquivos com estímulo
```

O Viterbi recebe diretamente `log f(p | estado)`. Não há thresholds nem labels
intermediários: dois p-valores numericamente diferentes mantêm evidências
diferentes, mesmo que pertencessem à mesma faixa da B discreta. `binsM` dos
arquivos estimulados não entra no ajuste das emissões e continua reservado para
medir falso positivo lateral.

### Vazamento deliberado

`B(Presente)` usa todos os 55 arquivos estimulados, incluindo os mesmos
participantes e frequências usados na avaliação. Além disso, A, distribuição
inicial e regra temporal são escolhidas no mesmo conjunto. Isso é vazamento de
dados intencional solicitado para um teste de viabilidade; os números são um
limite exploratório in-sample e não estimam generalização. Uma etapa futura
deve separar participantes, preferencialmente com validação leave-one-subject-
out aninhada.

### Resultados de referência

O artefato `results/continuous_rayleigh_hmm.json` foi gerado com 11 arquivos
ESP e 55 estimulados. Todas as configurações abaixo respeitam FP lateral
global máximo de 5%:

| Janela | HMM: detecção | HMM: FP | Rayleigh janelado: detecção | Rayleigh janelado: FP |
|---:|---:|---:|---:|---:|
| 10 | 28,18% | 4,09% | 21,82% | 3,86% |
| 20 | 35,23% | 4,77% | 25,91% | 4,55% |
| 30 | 35,68% | 4,77% | 30,91% | 5,00% |
| 60 | 45,91% | 5,00% | 40,23% | 3,86% |

O Rayleigh aplicado uma vez à gravação completa permaneceu superior: 53,41%
de detecção, 3,86% de FP e 74,77% de acurácia balanceada, contra 45,91%, 5,00%
e 70,45% do melhor HMM contínuo. Na janela de 60 épocas, somente 10 dos 11
arquivos ESP contribuíram para `Ausente`, pois `SoESP.mat` possui 30 épocas.

O FP é limitado apenas na agregação global. Na melhor janela, os FPs por nível
foram 5,68%, 4,55%, 7,95%, 6,82% e 0,00% para 30, 40, 50, 60 e 70 dB. Portanto,
o teto global não implica controle por intensidade, frequência, participante
ou gravação.

Execução:

```bash
.venv/bin/python src/continuous_rayleigh_hmm.py
```

## 11. Detecção precoce causal

`src/early_detection_rayleigh_hmm.py` compara o HMM contínuo com Rayleigh
acumulado e grava o instante da primeira decisão. Nos dados atuais cada época
tem `Fs` amostras e dura um segundo; portanto o índice final da janela em
épocas também representa segundos de exame.

O caminho Viterbi completo não é usado para atribuir tempo, pois seu
backtracking incorpora observações futuras. Em seu lugar, a cada janela o
programa atualiza o delta do Viterbi e usa somente o melhor estado terminal
disponível naquele instante. O exame para na primeira vez em que a sequência
causal satisfaz `MIN_CONSECUTIVE` e/ou `MIN_PERCENT`.

Rayleigh é recalculado sobre todas as épocas acumuladas até os mesmos pontos de
decisão. Como consultar repetidamente `p <= 0,05` inflaria FP, o programa busca
alpha sequencial em `0,05` até `0,0001` e aceita somente resultados com FP
lateral global de no máximo 5%.

Foram testadas janelas de 10, 20, 30, 60, 90, 120 e 180 épocas, com passo de
meia janela e sem sobreposição. Arquivos mais curtos que a janela permanecem no
denominador, mas não podem ser detectados nessa configuração.

### Resultados precoces de referência

O melhor resultado global de ambos ocorreu em janela/passo 60/60:

| Método | Detecção | FP | Mediana até detecção | Duração média do exame |
|---|---:|---:|---:|---:|
| HMM contínuo causal | 43,64% | 4,77% | 120 s | 234,78 s |
| Rayleigh acumulado (`alpha=0,025`) | 51,59% | 4,77% | 120 s | 219,64 s |

Na comparação pareada 60/60, ambos detectaram 174 frequências: o HMM foi
anterior em 6, empatou em 133 e Rayleigh foi anterior em 35. O HMM ainda teve
18 detecções exclusivas contra 53 do Rayleigh. Assim, essa configuração não
mostra antecipação global pelo HMM.

A configuração 180/90 foi a única vantagem exploratória clara do HMM:
41,14% de detecção e 4,77% de FP contra 36,36% e 2,73% do Rayleigh. Entre 156
detecções comuns, o HMM foi anterior em 14, empatou em 140 e foi posterior em
2; teve 25 detecções exclusivas contra 4. A duração média do exame foi 239,76 s
para HMM e 248,62 s para Rayleigh. Os FPs não são idênticos, apenas ambos estão
abaixo do teto de 5%, e a configuração foi escolhida in-sample.

O resultado completo fica em `results/early_detection_rayleigh_hmm.json`, com
taxas, tempos, parâmetros, resultados por dB e comparações pareadas para as 14
combinações. O mesmo vazamento deliberado da emissão real continua presente.

Execução:

```bash
.venv/bin/python src/early_detection_rayleigh_hmm.py
```

## 12. Comparação individual entre detectores

`src/compare_detectors_early_hmm.py` generaliza o experimento causal para
Rayleigh, MSC, MMSC, Hotelling T² geral e F espectral local. Cada HMM usa
somente a emissão contínua do detector em questão e é comparado com o mesmo
detector aplicado cumulativamente. CSM não é repetido porque é equivalente à
MSC na implementação atual.

O protocolo permanece igual: sete tamanhos de janela, com e sem sobreposição,
busca de A, π e regra temporal, alpha sequencial do detector bruto e FP lateral
global máximo de 5%. O resultado é salvo em
`results/early_detection_detectors_hmm.json` e continua deliberadamente
in-sample.

| Detector | Melhor janela/passo HMM | HMM det./FP | Melhor janela/passo bruto | Bruto det./FP | Ganho HMM em detecção |
|---|---:|---:|---:|---:|---:|
| Rayleigh | 60/60 | 43,64% / 4,77% | 60/60 | 51,59% / 4,77% | −7,95 p.p. |
| MSC | 60/30 | 46,59% / 5,00% | 60/30 | 53,64% / 5,00% | −7,05 p.p. |
| MMSC | 60/60 | 37,95% / 4,77% | 20/10 | 44,55% / 5,00% | −6,59 p.p. |
| Hotelling geral | 60/60 | 44,55% / 4,09% | 20/20 | 48,18% / 3,64% | −3,64 p.p. |
| F espectral local | 120/60 | 43,64% / 4,55% | 20/10 | 47,05% / 5,00% | −3,41 p.p. |

MSC foi o melhor HMM individual e o melhor detector acumulado. Mesmo assim, o
HMM não aumentou detecção, acurácia balanceada ou velocidade global. Na
comparação pareada MSC 60/30, ambos detectaram 191 frequências: HMM foi
anterior em 19, empatou em 142 e MSC foi anterior em 30; houve 14 detecções
exclusivas do HMM contra 45 da MSC. Ambos tiveram mediana de 90 s até detecção,
mas a duração média incluindo não detectados foi 220,17 s no HMM e 210,58 s na
MSC.

Os dois detectores adicionais também não inverteram a conclusão. Hotelling
geral chegou mais perto do próprio bruto, mas perdeu 3,64 pontos percentuais e
teve mediana de detecção 120 s contra 100 s. O F espectral perdeu 3,41 pontos e
teve mediana 180 s contra 90 s. Antes da execução, uma simulação nula com 5.000
réplicas e `M=20` produziu FP de 4,94% para Hotelling e 5,22% para F espectral
em alpha 5%, compatível com suas distribuições teóricas.

Execução:

```bash
.venv/bin/python src/compare_detectors_early_hmm.py
```

## 13. Regras para alterações futuras

1. Inspecione os `.mat` autênticos antes de mudar leitores ou eixos.
2. Mantenha o janelamento de A e B consistente ou documente por que divergem.
3. Não fixe `Fs`, número de épocas, `freqEstim` ou `binsM` a partir de um único
   participante.
4. Preserve participante, condição, canal, frequência, janela e `Fs` quando o
   processamento depender deles.
5. Mantenha B global e os diagnósticos por frequência, salvo decisão explícita
   em contrário.
6. Identifique sempre o detector de cada resultado; não chame MSC de CSM nem
   trate combinações como um novo teste com distribuição nula própria.
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

## 14. Ponto de retomada histórico — 27/09/2026

O método mantido após a decisão final está na Seção 20. Esta seção preserva a intenção
naquela data, não autoriza automaticamente retomar fusão ou busca.

Estado ao encerrar em 27/09/2026:

- cinco detectores não redundantes foram comparados individualmente;
- nenhum HMM contínuo causal superou o mesmo detector acumulado;
- MSC acumulada foi o melhor resultado: 53,64% de detecção com 5,00% de FP;
- Hotelling geral e F espectral local reduziram a distância HMM/bruto, mas não
  produziram ganho do HMM;
- todos os 70 vencedores por detector/configuração respeitam FP global de 5%;
- o vazamento de participantes permanece deliberadamente ativo.

Próxima tarefa: testar fusão **Rayleigh + F espectral local** com emissão
bivariada conjunta e comparar:

1. HMM causal com emissão conjunta;
2. fusão acumulada conjunta sem HMM;
3. Rayleigh acumulado e F espectral acumulado isolados.

A emissão conjunta deve modelar a correlação entre detectores, não usar produto
de marginais. Manter o protocolo atual de janelas 10/20/30/60/90/120/180,
passos meia janela e janela inteira, decisão causal, tempo de exame, comparação
pareada e FP lateral global máximo de 5%. A validação sem vazamento fica para a
etapa posterior, conforme decisão explícita desta sessão.

## 15. HMM como detector de transição entre fases

`src/phase_transition_hmm.py` testa uma unidade de análise diferente dos
experimentos anteriores. Para cada participante e frequência, concatena as
observações na ordem:

```text
ESP -> 30 dB -> 40 dB -> 50 dB -> 60 dB -> 70 dB
```

As janelas são calculadas separadamente dentro de cada arquivo e nunca cruzam
uma fronteira. Apenas as sequências de p-valores resultantes são concatenadas;
participantes e frequências continuam independentes. O estado terminal online
do Viterbi é usado em cada instante, sem backtracking futuro.

O experimento compara cada HMM contínuo com o mesmo detector bruto janelado e
causal. Ambos procuram o primeiro disparo após o começo de 30 dB. O relatório
guarda a primeira intensidade, o tempo total do protocolo e o tempo desde o
início da estimulação, incluindo a duração máxima para não detectados. A taxa
de detecção usa 88 trajetórias participante x frequência como denominador.

O controle de FP combina dois tipos de trajetória negativa: disparo da
frequência-alvo durante ESP e disparo da lateral pareada durante o protocolo.
São 176 oportunidades negativas, e a seleção exige FP combinado global de no
máximo 5%. O JSON também separa FP em ESP e lateral e apresenta detecção e FP
por participante e por frequência.

Este primeiro teste é deliberadamente **in-sample**: emissões Beta, A, pi,
janelamento e regra de disparo são selecionados e avaliados nos mesmos 11
participantes. Os resultados medem viabilidade e não generalização. Conforme
decisão explícita, leave-one-subject-out só será executado se esta etapa for
promissora.

Resultados de referência em `results/phase_transition_hmm.json`:

| Detector | Melhor HMM | HMM det./FP | Melhor bruto | Bruto det./FP | Tempo total médio HMM/bruto |
|---|---:|---:|---:|---:|---:|
| Rayleigh | 120/60 | 79,55% / 4,55% | 90/45 | 69,32% / 4,55% | 1021,5 / 1159,7 s |
| MSC | 180/90 | 79,55% / 3,98% | 180/180 | 77,27% / 3,41% | 973,4 / 1088,1 s |
| MMSC | 180/180 | 70,45% / 2,84% | 120/60 | 69,32% / 3,98% | 1176,0 / 1184,0 s |
| Hotelling geral | 120/60 | 80,68% / 3,98% | 180/180 | 78,41% / 4,55% | 984,0 / 1085,8 s |
| F espectral local | 60/60 | 79,55% / 4,55% | 180/180 | 75,00% / 3,98% | 1083,1 / 1139,2 s |

O HMM superou o detector bruto janelado nos cinco detectores neste ajuste
in-sample. Hotelling obteve a maior detecção (71/88), enquanto MSC teve o menor
tempo total médio entre os HMMs vencedores. A vantagem ainda não demonstra
generalização e pode refletir a otimização no próprio conjunto.

Os tempos próximos de 1000 segundos na tabela anterior não são tempo de
execução do programa nem latência somente entre os casos detectados. Eles são
a duração média simulada do exame desde o começo de ESP e atribuem a duração
máxima do protocolo às trajetórias nunca detectadas. Como uma época dura um
segundo nos dados atuais e as gravações de 30 e 40 dB têm em geral 480 épocas
cada, o começo de 50 dB ocorre aproximadamente 960 segundos após o começo da
estimulação.

Excluindo ESP e considerando somente trajetórias efetivamente detectadas:

| Detector | HMM média/mediana desde 30 dB | Bruto média/mediana desde 30 dB |
|---|---:|---:|
| Rayleigh | 734 / 730 s | 822 / 795 s |
| MSC | 674 / 660 s | 799 / 840 s |
| MMSC | 855 / 840 s | 855 / 900 s |
| Hotelling geral | 699 / 720 s | 804 / 840 s |
| F espectral local | 811 / 840 s | 846 / 840 s |

No MSC-HMM, 28 trajetórias foram detectadas primeiro em 30 dB, 30 em 40 dB,
7 em 50 dB, 5 em 60 dB, nenhuma em 70 dB e 18 nunca foram detectadas. Portanto,
58/88 foram detectadas até 40 dB; a mediana de 660 segundos corresponde
aproximadamente a 480 segundos de 30 dB mais 180 segundos de 40 dB, e não a
uma detecção média somente em 50 dB.

Execução:

```bash
.venv/bin/python src/phase_transition_hmm.py
```

## 16. Validação leave-one-subject-out do protocolo concatenado

`src/phase_transition_hmm_loso.py` repete o experimento da Seção 15 com 11
dobras leave-one-subject-out. Em cada dobra, um participante é completamente
retido. Somente os outros dez participantes são usados para:

- ajustar as duas emissões Beta contínuas, Ausente e Presente;
- escolher tamanho e passo da janela;
- escolher a matriz A e a distribuição inicial pi;
- escolher a consecutividade mínima do HMM;
- escolher alpha e consecutividade do detector bruto.

O participante retido é aplicado somente depois que essas escolhas terminam.
O limite de FP combinado de 5% é uma restrição no treino; o FP do participante
externo é reportado sem descarte ou reajuste. As 11 predições externas são
então reunidas, produzindo novamente 88 trajetórias-alvo e 176 oportunidades
negativas. Cada participante aparece exatamente uma vez no teste.

Resultados em `results/phase_transition_hmm_loso.json`:

| Detector | HMM det./FP externo | Bruto det./FP externo | Tempo total médio HMM/bruto |
|---|---:|---:|---:|
| Rayleigh | 77,27% / 6,82% | 67,05% / 6,25% | 1019,6 / 1201,0 s |
| MSC | 73,86% / 9,66% | 77,27% / 3,41% | 1061,6 / 1088,1 s |
| MMSC | 71,59% / 6,82% | 61,36% / 2,84% | 1157,9 / 1243,3 s |
| Hotelling geral | 76,14% / 5,11% | 78,41% / 4,55% | 1085,8 / 1085,8 s |
| F espectral local | 72,73% / 10,23% | 75,00% / 3,98% | 1101,0 / 1139,2 s |

Entre as trajetórias detectadas, média/mediana desde o começo de 30 dB:

| Detector | HMM | Bruto |
|---|---:|---:|
| Rayleigh | 709 / 685 s | 859 / 840 s |
| MSC | 730 / 680 s | 799 / 840 s |
| MMSC | 840 / 840 s | 868 / 880 s |
| Hotelling geral | 785 / 840 s | 804 / 840 s |
| F espectral local | 772 / 700 s | 846 / 840 s |

O resultado externo não confirma a conclusão otimista in-sample. Rayleigh e
MMSC conservaram maior detecção e menor tempo que seus detectores brutos, mas
ambos tiveram FP externo de 6,82%, acima do limite pretendido. MSC e F
espectral perderam detecção e apresentaram FP de 9,66% e 10,23%. Hotelling foi
o mais próximo do limite, com FP de 5,11%, mas detectou menos que o bruto.
Assim, nenhum HMM demonstrou simultaneamente vantagem de detecção e controle
externo de FP em até 5%.

As emissões e matrizes A são específicas por dobra e estão preservadas no JSON
junto com a janela, pi, regra, métricas de treino, métricas de teste e registros
por participante e frequência. Elas não substituem `observation_matrix.json`
ou `transition_matrix.json`, que continuam descrevendo o pipeline discreto
global e a heurística de duração, respectivamente.

Execução completa:

```bash
.venv/bin/python src/phase_transition_hmm_loso.py
```

O programa também aceita `--detector` para execução independente e
`--merge-inputs` para consolidar artefatos calculados em processos separados.

## 17. HMM de transição com observações discretas

`src/phase_transition_discrete_hmm.py` repete o protocolo de fases com cinco
labels categóricos. Ele usa o registro de `detectors.py` para Rayleigh, MSC,
MMSC, Hotelling geral e F espectral local; CSM não é tratado como detector
independente da MSC. O p-valor de cada janela é convertido usando exatamente
as fronteiras teóricas de `get_obs_matrix.py`:

| Label | Faixa de p-valor |
|---|---:|
| muito_baixo | `p > 0,50` |
| baixo | `0,10 < p <= 0,50` |
| medio | `0,05 < p <= 0,10` |
| alto | `0,01 < p <= 0,05` |
| muito_alto | `p <= 0,01` |

Cada protocolo é uma única combinação participante × frequência e mantém
separadas as frequências-alvo e laterais. As janelas são calculadas dentro de
cada arquivo nas fases ESP, 30, 40, 50, 60 e 70 dB; só os labels são
concatenados na ordem das fases. Os registros mantêm participante, condição,
canal, alvo, lateral, intervalo local e acumulado, `Fs`, estatística, p-valor e
label. O forward/Viterbi causal atualiza `delta` com observações passadas e a
observação corrente; não faz backtracking nem usa observações futuras para
atribuir o disparo. O detector bruto de comparação é janelado e causal.

### Emissões discretas e seleção

`B` tem duas linhas e cinco colunas. A linha Ausente conta labels de alvos nos
arquivos ESP; a linha Presente conta os mesmos alvos nas gravações estimuladas
reais. Não há senoide sintética. Cada célula recebe pseudocontagem 0,5 e cada
linha é normalizada uma vez. As contagens, `B`, `A`, `pi`, janela, passo,
alpha/consecutividade e métricas ficam no JSON de cada configuração ou dobra.

Foram testadas janelas de 10, 20, 30, 60, 90, 120 e 180 épocas, com passo de
meia janela e passo de janela inteira. O conjunto de candidatos de `A`, `pi`,
consecutividade e alpha é o mesmo do experimento contínuo. O treino exige FP
combinado <=5%; na validação externa, o FP é relatado sem reajustar ou descartar
uma dobra. A unidade é 88 trajetórias-alvo e 176 oportunidades negativas: 88
alvos ESP e 88 trajetórias laterais.

### Vazamento e protocolo de validação

No in-sample, `B`, `A`, `pi`, janela/passo e regra são ajustados e avaliados nos
mesmos 11 participantes. É um teste exploratório com vazamento deliberado.
No LOSO, cada uma das 11 dobras retém uma pessoa completamente: B e todos os
parâmetros são selecionados somente nos outros dez; uma predição externa é
emitida para cada uma das oito frequências da pessoa retida. O JSON registra
os dez sujeitos de treino e `B`/contagens por dobra para auditoria. A pessoa
retida não participa da B ou da seleção da dobra. O FP externo não é usado para
reajuste.

### Resultados in-sample

Resultados em `results/phase_transition_discrete_hmm.json`. A tabela mostra
detecção, FP combinado e duração média desde o começo de 30 dB, atribuindo a
duração máxima a não detectados:

| Detector | HMM discreto | Bruto janelado | HMM contínuo |
|---|---:|---:|---:|
| Rayleigh | 71,59% / 4,55% / 860 s | 56,82% / 1,14% / 1112 s | 79,55% / 4,55% / 890 s |
| MSC | 78,41% / 2,27% / 963 s | 77,27% / 3,41% / 956 s | 79,55% / 3,98% / 841 s |
| MMSC | 64,77% / 3,98% / 1020 s | 68,18% / 3,41% / 1059 s | 70,45% / 2,84% / 1044 s |
| Hotelling | 69,32% / 3,98% / 969 s | 78,41% / 4,55% / 954 s | 80,68% / 3,98% / 852 s |
| F espectral | 68,18% / 3,41% / 974 s | 75,00% / 3,98% / 1007 s | 79,55% / 4,55% / 951 s |

In-sample, o discreto só aumenta a detecção sobre o mesmo detector bruto em
MSC e Rayleigh. Sua detecção fica abaixo da emissão contínua nos cinco casos.

As matrizes abaixo correspondem à configuração vencedora do HMM discreto por
detector (ordem de colunas: muito_baixo, baixo, medio, alto, muito_alto). As
contagens completas para as 14 configurações estão no JSON.

| Detector | Estado | Contagens | Probabilidades de B |
|---|---|---|---|
| MSC | Ausente | 20, 11, 1, 0, 0 | 0,5942; 0,3333; 0,0435; 0,0145; 0,0145 |
| MSC | Presente | 222, 329, 77, 125, 199 | 0,2331; 0,3452; 0,0812; 0,1315; 0,2090 |
| Rayleigh | Ausente | 13, 11, 0, 0, 0 | 0,5094; 0,4340; 0,0189; 0,0189; 0,0189 |
| Rayleigh | Presente | 98, 179, 50, 78, 107 | 0,1914; 0,3489; 0,0982; 0,1526; 0,2089 |
| MMSC | Ausente | 16, 7, 1, 0, 0 | 0,6226; 0,2830; 0,0566; 0,0189; 0,0189 |
| MMSC | Presente | 131, 186, 37, 69, 89 | 0,2556; 0,3625; 0,0729; 0,1351; 0,1740 |
| Hotelling | Ausente | 13, 10, 1, 0, 0 | 0,5094; 0,3962; 0,0566; 0,0189; 0,0189 |
| Hotelling | Presente | 105, 180, 45, 57, 125 | 0,2051; 0,3508; 0,0884; 0,1118; 0,2439 |
| F espectral | Ausente | 13, 10, 1, 0, 0 | 0,5094; 0,3962; 0,0566; 0,0189; 0,0189 |
| F espectral | Presente | 115, 172, 48, 65, 112 | 0,2245; 0,3353; 0,0943; 0,1273; 0,2187 |

### Resultados LOSO e comparação

Resultados completos, contagens de B e predições por dobra estão em
`results/phase_transition_discrete_hmm_loso.json`. Cada célula abaixo informa
detecção / FP combinado / duração média desde 30 dB incluindo não detectados:

| Detector | HMM discreto LOSO | Bruto janelado LOSO | HMM contínuo LOSO |
|---|---:|---:|---:|
| Rayleigh | 68,18% / 7,95% / 1019 s | 67,05% / 6,25% / 1069 s | 77,27% / 6,82% / 888 s |
| MSC | 72,73% / 3,98% / 993 s | 77,27% / 3,41% / 956 s | 73,86% / 9,66% / 930 s |
| MMSC | 64,77% / 4,55% / 1057 s | 61,36% / 2,84% / 1111 s | 71,59% / 6,82% / 1026 s |
| Hotelling | 78,41% / 5,68% / 895 s | 78,41% / 4,55% / 954 s | 76,14% / 5,11% / 954 s |
| F espectral | 75,00% / 4,55% / 973 s | 75,00% / 3,98% / 1007 s | 72,73% / 10,23% / 969 s |

Exemplo de `B` LOSO, dobra Ab (treino em An, Bb, Er, Lu, Qu, Sa, So, Ti, Vi e
Wr); cada dobra tem sua própria matriz e contagens no JSON:

| Detector | Ausente: contagens | Presente: contagens |
|---|---|---|
| MSC | 20, 11, 1, 0, 0 | 198, 310, 68, 107, 181 |
| Rayleigh | 18, 12, 2, 0, 0 | 183, 304, 87, 126, 164 |
| MMSC | 35, 27, 2, 0, 0 | 268, 322, 83, 109, 98 |
| Hotelling | 52, 41, 0, 2, 1 | 418, 581, 133, 209, 243 |
| F espectral | 55, 36, 3, 2, 0 | 438, 599, 133, 178, 236 |

Entre detectados, o tempo médio/mediano desde 30 dB para HMM discreto LOSO foi
Rayleigh 799/760 s, MSC 806/770 s, MMSC 819/750 s, Hotelling 731/720 s e F
espectral 800/840 s. O JSON traz também tempo dentro da intensidade de primeira
detecção, médias por participante e por frequência e pareamento por trajetória.

MSC, MMSC e F espectral ficaram com FP externo <=5%. MSC teve quatro detecções
menos que o bruto (64 contra 68) e foi em média mais lento; MMSC detectou três
a mais e teve média desde 30 dB 54 s menor, mas FP 1,71 ponto percentual maior;
F espectral empatou em detecção, foi 34 s mais rápido em média e teve FP 0,57
ponto maior. Portanto, MMSC/F espectral mostram troca entre tempo e FP, mas não
há ganho robusto que justifique preferir a emissão discreta ao bruto ou ao HMM
contínuo. Rayleigh e Hotelling excederam o teto de FP externo. A conclusão é
exploratória e não representa validação clínica.

Execução completa:

```bash
.venv/bin/python src/phase_transition_discrete_hmm.py
.venv/bin/python src/phase_transition_discrete_hmm_loso.py
```

Os comandos aceitam `--detector`, `--window` e `--step-mode` para execução por
blocos; os artefatos `--merge-inputs` consolidam os resultados por detector.

## 18. Emissão por histograma da estatística ORD — 30/09/2026

`src/histogram_hmm.py` implementa a nova alternativa à emissão Beta de
p-valores. Os experimentos das Seções 10–17 e seus resultados históricos
permanecem disponíveis; executar seus comandos ainda usa seus métodos originais.
Para usar a nova emissão, execute o comando desta seção.

A observação é o valor da estatística retornada pelo detector, sem transformar
em p-valor para alimentar o HMM. O padrão é Rayleigh (PLV²), janela 10, passo 5,
canal 0 e K=0,01. Também suporta MSC, MMSC, CSM, Hotelling e F espectral local.
Hotelling/F local usam a estatística normalizada F/(1+F) do registro atual,
não F bruto. Todos esses valores pertencem a [0,1].

As duas emissões são calibradas exclusivamente em ESP autêntico: Ausente usa
EEG original; Presente usa as mesmas épocas após senoide de amplitude
`K * std(época)`, coerente em fase. Há um modelo global por detector, somando
contagens das oito frequências antes de normalizar. Cada estado usa os mesmos
20 intervalos uniformes em [0,1], com pseudocontagem 0,5 por célula:

```text
q[j,k] = (contagem[j,k] + 0,5) / (N[j] + 20*0,5)
f[j,k] = q[j,k] / largura[k]
```

A massa q de cada linha soma 1; a densidade f integra 1. O Viterbi recebe
`log f[j,bin(x_t)]`. A observação continua sendo x_t, não a altura do histograma.
Valores inéditos usam o intervalo correspondente; bins vazios continuam com
densidade positiva pela suavização. O limite 1 pertence ao último intervalo;
valores não finitos ou fora de [0,1] são rejeitados.

É uma densidade constante por partes, com resolução limitada pelos bins.
Com os mesmos intervalos, massa e densidade produzem o mesmo caminho porque a
largura cancela na comparação entre estados. Isso não garante equivalência ao
modelo Beta anterior: mudaram a estimativa das emissões e a origem de Presente.

O programa lê A de `transition_matrix.json`, exige janelamento compatível,
usa pi de `get_obs_matrix.py` e a regra vigente de `hmm_inference.py` (4
consecutivas AND 10%). Executa Viterbi completo com backtracking por arquivo e
frequência independente; não é detecção causal precoce nem concatena fases.

```bash
.venv/bin/python src/histogram_hmm.py --detector rayleigh --bins 20 --k 0.01
.venv/bin/python -m unittest discover -s tests -p test_histogram_hmm.py
```

Saída: `results/histogram_hmm.json`, com bordas, contagens, massas B 2×20,
densidades, A, pi, diagnósticos por frequência, registros de calibração e
sequências de inferência com estatística, p-valor diagnóstico, épocas, Fs,
participante, intensidade, canal, alvo e controle. Os labels da calibração
herdada são somente metadados; não entram na emissão. Os artefatos discretos
`observation_matrix.json` e `transition_matrix.json` preservam seus formatos.

Verificação inicial: 11 arquivos ESP, oito frequências, 2232 janelas por estado;
55 arquivos estimulados, 440 sequências-alvo e 440 laterais. FP do Rayleigh
bruto nas janelas ESP em alpha=5%: global 4,12%; por frequência 81 Hz 1,43%,
83 Hz 3,23%, 85 Hz 7,53%, 87 Hz 5,38%, 89 Hz 3,58%, 91 Hz 3,23%, 93 Hz 3,58%,
95 Hz 5,02%. Isso é diagnóstico do detector, não FP da decisão HMM.
Os testes verificam normalização, extremos, observações inéditas, bins vazios,
rejeição fora do suporte e Viterbi contra enumeração exaustiva de caminhos.
A/B discretas foram regeneradas e reproduziram a Seção 6. Presente sintético
não é calibração acústica em dB nem validação clínica.

### Comparação com parâmetros históricos fixos

**Histórico por arquivo:** esta comparação não implementou o protocolo
concatenado solicitado pelo usuário. A correção do teste está na Seção 19;
a decisão final de manter Beta está na Seção 20. Não usar as taxas desta subseção como resultados por
paciente × frequência ao longo de ESP e intensidades.

`src/compare_histogram_rayleigh.py` lê quatro vencedores de
`continuous_rayleigh_hmm.json` (10/5, 20/10, 30/15, 60/30) e o vencedor
Rayleigh isolado de `search_hmm_parameters.json` (10/5, K=0,005). Não executa
search nem seleciona parâmetros novos. Mantém A, pi e regra de cada vencedor;
as quatro configurações contínuas usam K=0,01. Recalibra somente os histogramas
ESP/sintéticos para cada janela/K, com 20 bins e pseudocontagem 0,5.

```bash
.venv/bin/python src/compare_histogram_rayleigh.py
```

Saída: `results/compare_histogram_rayleigh.json`, com origem dos parâmetros,
A, pi, regras, contagens/densidades B, métricas globais/por dB/por frequência e
sequências individuais. Cada configuração avalia os mesmos 55 arquivos e 440
alvos/440 laterais; arquivos curtos sem janela permanecem no denominador como
não detectados. A calibração 60/30 só usa dez ESP, pois um possui 30 épocas.

Cada célula abaixo é detecção / FP lateral (%):

| Origem e janela/passo | HMM histograma | Rayleigh bruto, mesma regra HMM | Rayleigh bruto, regra própria histórica |
|---|---:|---:|---:|
| Contínuo 10/5 | 9,77 / 0,68 | 2,27 / 0,00 | 21,82 / 3,86 |
| Contínuo 20/10 | 11,59 / 0,45 | 6,36 / 0,00 | 25,91 / 4,55 |
| Contínuo 30/15 | 16,36 / 0,91 | 21,14 / 2,05 | 30,91 / 5,00 |
| Contínuo 60/30 | 19,09 / 0,23 | 19,77 / 0,23 | 40,23 / 3,86 |
| Search antigo Rayleigh 10/5, K=0,005 | 25,68 / 6,14 | 1,36 / 0,00 | 63,64 / 35,23 |

O bruto é Rayleigh janelado com alpha=0,05, não acumulado nem de gravação
completa. São reportadas duas regras para separar emissão e decisão temporal.
As contagens do bruto com regra histórica reproduziram exatamente os artefatos
de origem. O bruto histórico da busca antiga não tinha restrição de FP e seu
FP de 35,23% impede interpretá-lo como um resultado sob teto de 5%.

O novo HMM perde detecção para o bruto com regra histórica nas quatro runs
contínuas, com FP também menor. Com a mesma regra, supera o bruto em 10/5 e
20/10, mas perde em 30/15 e 60/30. A configuração da busca antiga excede FP 5%
e foi reportada sem reajuste. Esses testes não demonstram equivalência ao HMM
Beta anterior: além da densidade, Presente passou de estímulo real para
sintético. Os parâmetros históricos já foram selecionados neste conjunto;
este teste fixo não é validação externa.

## 19. Experimento histórico: histograma ORD com fases concatenadas — 30/09/2026

Esta seção registra o teste e seus diagnósticos. A decisão posterior de manter
o método Beta está na Seção 20 e substitui as referências a método ativo abaixo.

Após esclarecimento do usuário, a unidade correta da comparação é uma
trajetória **paciente × frequência**, com a ordem:

```text
ESP -> 30 dB -> 40 dB -> 50 dB -> 60 dB -> 70 dB
```

`src/compare_histogram_phase_rayleigh.py` implementa essa comparação. As janelas
são calculadas separadamente em cada arquivo; somente os valores observados
são concatenados. Nenhuma janela cruza uma fronteira, e pacientes/frequências
nunca são concatenados entre si. Há 11 pacientes, oito alvos e oito controles
laterais: 88 trajetórias-alvo e 88 laterais. Delta do HMM e a contagem de
consecutivas continuam entre intensidades; não reinicializam a cada arquivo.
Essa concatenação entre condições é uma decisão explícita do usuário para
este protocolo, diferente do pipeline original por gravação.

### Calibração e inferência

- Observação HMM: estatística Rayleigh PLV², sem conversão para p-valor na
  emissão. Densidade constante por bin, obtida com `histogram_hmm.py`.
- Ausente: histograma dos alvos ESP originais. Presente: mesmos ESP com
  senoide de amplitude `K=0,01 * std(época)`, mesma fase entre épocas.
- Modelo global por configuração: 20 bins comuns uniformes em [0,1],
  pseudocontagem 0,5, soma das contagens antes de normalizar. As massas B
  somam 1 e as densidades integram 1. Estímulos reais não ajustam a emissão.
- Inferência HMM: estado terminal Viterbi a cada prefixo, usando apenas passado
  e observação atual, sem backtracking futuro. Disparo pela consecutividade
  histórica fixada.
- Baseline: Rayleigh **janelado**, com alpha e consecutividade históricos;
  não é detector acumulado nem estatística de toda a gravação.
- Tempos: fim da janela local + duração das fases anteriores. O código verifica
  em todos os arquivos que `amostras/Fs = 1 segundo por época`; não fixa Fs.
  Frequências correspondentes são verificadas entre fases.

### Parâmetros fixos; nenhuma busca nova

O script lê as 14 configurações Rayleigh já calculadas em
`results/phase_transition_hmm.json` e o vencedor Rayleigh discreto de
`results/phase_transition_discrete_hmm.json`, totalizando 15 testes fixos.
Mantém A, pi, janela/passo e consecutividade do HMM, além de alpha e
consecutividade do bruto de cada run. Reestima somente as emissões pelo método
novo ESP/sintético. Não chama os métodos de search, não ajusta regras por FP
observado e não seleciona novo vencedor.

Os destaques foram definidos nos artefatos anteriores, antes do novo teste:

| Método | Janela/passo | Parâmetros |
|---|---|---|
| HMM, vencedor contínuo anterior | 120/60 | A=[[0,99;0,01],[0,15;0,85]], pi=[0,95;0,05], 1 consecutiva |
| Rayleigh, vencedor bruto anterior | 90/45 | alpha=0,005, 1 consecutiva |
| HMM, vencedor discreto anterior | 180/180 | A=[[0,95;0,05],[0,15;0,85]], pi=[0,8;0,2], 1 consecutiva |

O JSON guarda resultados pareados com mesmo janelamento e também o pareamento
entre os dois destaques históricos, cujas janelas diferem. Não confundir essa
comparação com um teste mantendo todas as escolhas temporais iguais.

### Métricas solicitadas pelo usuário

Por paciente:

```text
detecção = frequências-alvo detectadas após começo de 30 dB / 8 alvos
FP lateral = laterais com qualquer disparo em ESP ou intensidades / 8 laterais
```

Globalmente, os denominadores são 88 alvos e 88 laterais. Um alvo que dispara
em ESP gera FP ESP separado; um disparo após o início de 30 dB conta como
detecção, conforme a convenção histórica. Consecutivas podem atravessar a
fronteira ESP/30; não se exige um reset para detecção posterior.

Tempo de detecção médio/mediano desde 30 dB usa **somente trajetórias
detectadas**. Para não detectados, tempo de detecção é `null`; uma métrica
separada de duração média censurada atribui a duração máxima do protocolo.
Também são registrados tempo total desde ESP, intensidade de primeira
detecção e tempo dentro dessa intensidade. Comparação pareada de velocidade
usa apenas trajetórias detectadas por ambos; exclusividades são separadas.

FP combinado dos experimentos antigos permanece um diagnóstico:
`(FP alvos ESP + FP laterais após 30) / 176`. Ele não é o FP lateral solicitado.
Por exemplo, oito laterais positivas e zero FP ESP representam 4,55% combinado
e **9,09% lateral**. A acurácia balanceada atual usa
`(detecção + 1 - FP lateral) / 2`; não comparar diretamente com a versão antiga
sem identificar o denominador.

### Resultados dos destaques fixos

| Método | Detecção | FP lateral | Acurácia balanceada lateral | Tempo médio/mediano desde 30, detectados |
|---|---:|---:|---:|---:|
| HMM Beta antigo recalculado, 120/60, Presente real | 70/88 = 79,55% | 8/88 = 9,09% | 85,23% | 734 / 730 s |
| Histograma 120/60, Presente real (controle diagnóstico) | 31/88 = 35,23% | 0/88 = 0,00% | 67,61% | 1044 / 1140 s |
| HMM histograma, parâmetros contínuos 120/60 | 26/88 = 29,55% | 0/88 = 0,00% | 64,77% | 1038 / 1140 s |
| Rayleigh bruto, parâmetros anteriores 90/45 | 61/88 = 69,32% | 8/88 = 9,09% | 80,11% | 822 / 795 s |
| HMM histograma, parâmetros discretos 180/180 | 6/88 = 6,82% | 0/88 = 0,00% | 53,41% | 1190 / 1140 s |

Na comparação pareada do HMM histograma sintético 120/60 com Rayleigh 90/45:
ambos detectam 26 trajetórias;
HMM é anterior em uma e Rayleigh em 25; zero exclusivas HMM e 35 exclusivas
Rayleigh. A duração média desde 30 incluindo não detectados é 1359 s no HMM
e 1028 s no bruto. Não chamar essas durações de latência média de detecção.

Com mesmo janelamento 120/60, o bruto detecta 65,91% com FP lateral 3,41%.
Em 60/60, o HMM fixo detecta 53,41% com FP 2,27%, contra bruto 63,64% e
4,55%. Janelas curtas podem aumentar detecção com FP elevado: HMM 10/5 chega
a 84,09% de detecção e 55,68% de FP lateral. Todas as 15 configurações são
reportadas, inclusive as que excedem FP 5%, sem descartá-las nem reajustar.

Conclusão deste teste: os parâmetros vencedores antigos não preservaram seu
desempenho ao trocar emissão real/Beta por histograma ESP/sintético. Mudaram
tanto a estimativa de densidade quanto a origem de Presente; não atribuir
toda a diferença ao histograma. Os parâmetros históricos foram selecionados
nos próprios participantes: o teste atual não é uma nova validação LOSO nem
validação clínica. K não equivale a dB acústicos.

### Execução, artefatos e retomada

```bash
.venv/bin/python src/compare_histogram_phase_rayleigh.py
.venv/bin/python -m unittest discover -s tests -p 'test_histogram*.py'
```

- `results/compare_histogram_phase_rayleigh.json`: fonte e parâmetros de cada
  configuração, contagens/massas/densidades B, fases por paciente/frequência,
  Fs, canal, arquivos, intervalos, estatística/p-valor, detecção/FP/tempos,
  resumo por paciente/frequência e pareamento.
- `results/compare_histogram_phase_rayleigh.md`: tabelas por paciente dos
  destaques anteriores e todas as configurações com detecção e FP lateral.
- `docs/agents/compare_histogram_phase_rayleigh.md`: contrato do módulo.
- As baselines reproduziram as taxas históricas e os quatro testes numéricos
  passaram: normalização, extremos/inéditos, Viterbi exaustivo, causalidade,
  relógio e FP lateral que ocorre somente em ESP.

Este é o ponto de retomada atual. Preservar a concatenação e as definições de
métrica acima. Não substituir essa entrada pelo teste por arquivo da Seção 18,
nem retomar buscas ou fusão automaticamente. O próximo experimento depende de
nova instrução do usuário; a tarefa atual autorizou somente testes fixos e
atualização dos docs.

### Verificação da forma antiga e diagnóstico de resolução

Após o usuário questionar a queda, o mesmo script passou a reaplicar as
emissões Beta antigas armazenadas no JSON original, com os mesmos p-valores,
A, pi, janela/passo e consecutividade, e a reaplicar B categórica na
configuração discreta. Não executa ajuste Beta nem search para esse controle.
As 15 taxas de detecção e FP combinado do HMM antigo reproduziram os artefatos
originais, verificadas por assertivas. Portanto a queda não surgiu de uma
troca inadvertida do protocolo de fases ou do Viterbi causal.

Um segundo controle ajusta histogramas da estatística com **Presente real**
nos mesmos dados utilizados pelo experimento antigo. É apenas diagnóstico
in-sample para separar efeitos de modelagem; não substitui a calibração
ESP/sintética escolhida pelo usuário. Todos os métodos mantêm parâmetros
temporais fixos. As métricas e emissões desse controle são identificadas como
`histograma_presente_real_diagnostico` no JSON.

No destaque 120/60: Beta real 79,55% de detecção → histograma real 35,23% →
histograma sintético 29,55%. A troca de representação/densidade já reduz a
detecção mesmo mantendo a origem de Presente. A diferença adicional para
29,55% acontece ao substituir Presente real por sintético, nessa configuração.
Esses controles não isolam individualmente todos os efeitos de família,
transformação, suavização e bins; não extrapolar uma causa única para todas
as configurações.

Há uma limitação concreta dos bins uniformes escolhidos: em 120/60, as 104
observações Ausente estão no primeiro bin [0;0,05), assim como 1592/1744
(91,28%) das observações Presente reais. O limiar Rayleigh bruto em alpha=0,05
é PLV² ≈0,02496, dentro desse mesmo bin. Valores pouco e muito significativos
nessa região recebem a mesma emissão histogramática. A transformação
p=exp(-M*PLV²) era invertível para M fixo; removê-la não deveria perder
informação por si só, mas o agrupamento em 20 bins perde resolução.

Não houve ajuste automático de bins, K, A ou regras para recuperar detecção.
Qualquer experimento posterior sobre resolução dos bins ou emissão sintética
deve ser explicitamente definido, preservando o protocolo causal concatenado.

## 20. Decisão final: manter Beta contínua e preparar visualização didática

O usuário decidiu **manter o método antigo**, após comparar as emissões. Aqui,
"antigo" significa o HMM contínuo com densidades Beta dos **p-valores**, usando
ESP para Ausente e gravações estimuladas reais para Presente. Não significa
retomar cinco labels nem manter Presente sintético no modelo contínuo.
Os códigos Beta foram preservados durante os experimentos; não houve mudança
de detector, dados, parâmetros ou código de inferência nesta decisão documental.

### Caminho implementado, do EEG à decisão

1. Leia `x`, `Fs`, `freqEstim` e `binsM` de cada arquivo real. Use canal
   configurado, um coeficiente FFT complexo por época na frequência analisada.
2. Em uma janela de M épocas, Rayleigh usa
   `u_m = X_m / (abs(X_m) + EPS)` e
   `R² = abs(sum(u_m)/M)²`, limitado a [0,1]. Normalizar a fase evita usar
   magnitude como evidência nesse teste.
3. Converta para p-valor pela aproximação nula `p = exp(-M * R²)`. Esse p-valor
   não é probabilidade de estado Ausente nem a emissão do HMM.
4. Agrupe os p-valores de alvos ESP e, separadamente, alvos reais de todas as
   intensidades. Ajuste duas Betas por máxima verossimilhança com loc=0 e
   scale=1 (`continuous_rayleigh_hmm._fit_beta`). Limite os valores ao intervalo
   [1e-9;1-1e-9] para estabilidade. As frequências compartilham um modelo global
   por configuração; laterais não entram no ajuste.
5. Em cada janela, calcule `log f_A(p_t)` e `log f_P(p_t)` com `beta.logpdf`.
   Não há labels nem lookup de histograma. Um p-valor inédito é substituído na
   fórmula Beta ajustada. Densidade pode exceder 1; a integral é 1. Não confundir
   PDF com CDF, probabilidade de um ponto ou probabilidade posterior do estado.
6. Inicialize `delta_1(j) = log(pi_j) + log f_j(p_1)`. Atualize
   `delta_t(j) = log f_j(p_t) + max_i(delta_(t-1)(i) + log A_ij)`.
   Para detecção causal, use `argmax(delta_t)` a cada prefixo, sem backtracking
   futuro. Delta é uma pontuação de caminho, não posterior normalizada.
7. Mantenha a trajetória independente por paciente e frequência, na ordem
   ESP → 30 → 40 → 50 → 60 → 70 dB. Janelas ficam dentro dos arquivos;
   delta e consecutivas continuam entre intensidades. Conte detecção alvo
   após começo de 30 dB, FP alvo ESP separado e FP lateral no protocolo todo.

`phase_transition_hmm.py` implementa os passos 5–7 e o protocolo; utiliza os
utilitários de FFT/leitura e ajuste de `continuous_rayleigh_hmm.py` e
`get_obs_matrix.py`. Seu ponto de entrada principal também executa busca:
**não rodar esse main para uma mera reprodução/visualização**. Para parâmetros
fixos, leia os modelos e escolhas de `results/phase_transition_hmm.json` e
reutilize as funções de inferência, sem chamar `_melhor_hmm`/`_melhor_bruto`.
O campo `hmm_antigo_recalculado` de
`results/compare_histogram_phase_rayleigh.json` já contém a reprodução validada
dessas emissões no protocolo, com as métricas laterais solicitadas.

### Configuração de referência e resultados reproduzidos

Rayleigh HMM contínuo vencedor histórico, janela/passo 120/60:

```text
A = [[0.99, 0.01], [0.15, 0.85]]
pi = [0.95, 0.05]
min_consecutive = 1
Beta Ausente:  a=1.184151213734531, b=0.8862635970365302, n=104
Beta Presente: a=0.39052284005446136, b=0.9567359610170701, n=1744
```

São os valores salvos, não reajustados nesta tarefa. A foi selecionada pela
busca histórica; não é a A heurística de `transition_matrix.json` nem uma
estimativa de transições fisiológicas observadas. Com M=120, arquivos ESP
curtos não contribuem para o ajuste, mas os pacientes continuam na avaliação.

- HMM Beta: 70/88 = 79,55% de detecção, 8/88 = 9,09% FP lateral; acurácia
  balanceada usando FP lateral 85,23%; média/mediana desde 30 entre detectados
  734/730 s. FP combinado histórico = 4,55%, acurácia correspondente = 87,50%.
- Rayleigh bruto vencedor histórico, janela/passo 90/45, alpha=0,005,
  consecutivas=1: 61/88 = 69,32% detecção, 8/88 = 9,09% FP lateral,
  acurácia balanceada lateral 80,11%, média/mediana 822/795 s.
- No mesmo janelamento 120/60, bruto: 58/88 = 65,91% detecção,
  3/88 = 3,41% FP lateral, tempo médio entre detectados 825 s.

Por paciente, divida alvos detectados pelos oito alvos e laterais com disparo
pelas oito laterais. Mantenha as definições de tempo e censura da Seção 19.
Não comparar FP combinado e FP lateral como se tivessem o mesmo denominador.
Os resultados acima são in-sample: emissões Presente real e parâmetros foram
ajustados/selecionados nos próprios pacientes. As runs LOSO históricas têm
artefatos próprios; essa decisão não converte o resultado em validação externa.

### Por que o histograma não foi adotado

Na comparação fixa 120/60: Beta real 79,55% → histograma real 35,23% →
histograma sintético 29,55% de detecção. Os 20 bins uniformes agrupavam todas
as 104 observações ESP e 91,28% das 1744 observações estimuladas no primeiro
intervalo [0;0,05), perdendo resolução inclusive ao redor do limiar Rayleigh
de 5% (R²≈0,02496). Não é prova de que todo histograma é inferior, nem de que
transformar em p-valor cria informação. Essa implementação/configuração não
preservou o desempenho; a troca de Presente real por sintético também mudou
a hipótese de emissão. Preservar códigos e artefatos como histórico, sem
adotá-los automaticamente ou executar nova busca de bins/K.

### Solicitação histórica do prompt da apresentação

Preparar uma apresentação com plots detalhados dos dados reais, FFT/fases,
Rayleigh, conversão em p-valor, ajuste Beta, avaliação de densidades em uma
janela real e atualização numérica de Viterbi, seguida da trajetória de fases
e métricas por paciente. Não alterar o método, executar search nem transformar
essa tarefa em vídeo automaticamente. O prompt completo está em
`docs/prompts/apresentacao_metodo_hmm_beta.md`. A frase de apresentação ainda
não implementada correspondia à etapa do prompt; o registro da entrega está
abaixo. Para o estado atual dos arquivos e a apresentação pronta da comparação
bayesiana, consulte a Seção 21 e `docs/agents/status.md`.

### Apresentação entregue — 30/09/2026

A visualização didática autorizada foi implementada em
`outputs/apresentacao_metodo_beta/index.html`, com versão independente
`apresentacao.pdf`, 13 figuras em PNG/SVG/PDF, tabelas CSV, parâmetros,
cálculos numéricos e manifesto SHA-256 das fontes. Reprodução:

```bash
.venv/bin/python outputs/apresentacao_metodo_beta/gerar.py
```

O script usa os parâmetros salvos, sem search e sem sobrescrever `results/`.
Confere todos os p-valores/intervalos dos 66 arquivos do protocolo com o EEG,
os 176 percursos causais e seus prefixos com a função original, e eventos e
tempos com a reprodução salva. Confirma 70/88 alvos, 8/88 laterais, 734/730 s,
FP combinado 4,55% e acurácia balanceada lateral 85,23%. O MLE diagnóstico
reproduz os parâmetros salvos, sem substituir o modelo usado na avaliação.

Exemplos ilustrativos: Ab alvo 81 Hz, Ab alvo 89 Hz e Lu lateral 82 Hz.
A unidade física do EEG não está declarada nos metadados disponíveis;
os eixos usam unidade armazenada, sem atribuição de µV. A entrega explicita
proxy Presente real e seleção in-sample. Não houve alteração metodológica,
novos parâmetros, execução de LOSO ou modificação dos artefatos originais.

## 21. Alternativa experimental: emissão preditiva Beta bayesiana

**Etapa concluída:** implementação, inferência posterior, comparação fixa,
verificações, documentação e apresentação com 46 figuras PNG/SVG/PDF e PDF
consolidado de 46 páginas estão prontos. Não há pendências para concluir essa
entrega. Os artefatos foram conferidos e estão em `outputs/bayesian_beta_hmm/`.
O método ativo permanece sendo a emissão Beta por MLE da Seção 20.

`src/bayesian_beta_hmm.py` implementa uma alternativa separada ao MLE da
Seção 20; não o substitui nem a adota automaticamente. Para cada estado:

```text
p_i | μ_s,κ_s ~ Beta(μ_s κ_s, (1−μ_s) κ_s)
μ_s ~ Beta(u_s,v_s)
κ_s ~ Gamma(shape_s,rate_s)
```

A Beta dos dados é distinta da priori Beta da média μ. κ controla
concentração. PyMC Gamma usa alpha=shape e beta=rate; NumPy/SciPy usam
scale=1/rate. μ e κ são independentes a priori, mas não necessariamente na
posterior. As prioris configuráveis no topo são predefinidas por argumentos
de média/forma e simulações preditivas com seed fixa, sem selecionar por
detecção/FP:

| Priori | Ausente: u,v,shape,rate | Presente: u,v,shape,rate |
|---|---|---|
| principal | 8,8,2,1 | 2,4,2,1 |
| ampla | 2,2,2,0.5 | 1,2,2,0.5 |
| simétrica | 2,2,2,1 | 2,2,2,1 |

A principal centra μ Ausente em 0,5, coerente com média nula ideal, sem
impor p uniforme. μ Presente tem média 1/3, permitindo p moderados pois
estímulo real é proxy. κ tem média 2 e dispersão ampla. A priori implica
informação sobre formas: mesmo média centrada em 0,5 pode produzir curvas
em U e massa nos extremos. As simulações de 10.000 pares por estado/priori
e as curvas da mistura são salvas e plotadas. As outras prioris verificam
sensibilidade, sem eleger vencedora na avaliação.

Inferência conjunta com PyMC NUTS, quatro cadeias, 2000 passos de aquecimento
e 2000 amostras por cadeia, target_accept=0,95, seed base 20260930.
ArviZ calcula R-hat rank-normalized, ESS bulk/tail, MCSE, BFMI e divergências.
Salvam-se amostras completas em NetCDF e resumos/intervalos 95% de μ,κ,a,b.
O cálculo usa estatísticas suficientes da likelihood Beta exata. Critérios
diagnósticos exigidos: zero divergências, R-hat<1,01, ESS>400 e BFMI>0,3.

Emissão alternativa:

```text
log f_s(p | calibração) = logsumexp_r Beta.logpdf(p,a_s[r],b_s[r]) − log(S)
```

S=2000 amostras conjuntas fixas, com índices salvos, usadas para todas as
janelas. É média de densidades, não média dos logs nem PDF nos parâmetros
médios. A PDF original integra 1; densidades podem exceder 1. O clipping de p
em [1e−9,1−1e−9] é preservado do MLE, torna extremos indistinguíveis e altera
a likelihood desses pontos; quantidades alteradas e massa de cauda são
registradas. O wrapper de clipping não é uma nova PDF normalizada no suporte
fechado; verifica-se a mistura original por CDF e quadratura nas duas bordas.

Observações e parâmetros históricos são reutilizados, sem search:
janela/passo 120/60, canal 0, Rayleigh, A e pi da Seção 20, uma consecutiva.
P-valores, intervalos e R² salvos são conferidos com os 66 EEG reais; manifesto
SHA-256 e metadados preservam proveniência. Ausente usa 104 janelas-alvo ESP;
Presente usa 1744 janelas-alvo reais estimuladas; laterais não ajustam.
Uma emissão global por estado é compartilhada entre frequências. Nenhum
histograma, label ou dado sintético entra no experimento.

Baseline MLE é reproduzida antes de amostrar: 70/88 alvos detectados,
0/88 FP alvo ESP, 8/88 FP lateral e média/mediana 734/730 s entre detectados.
Todos os eventos e parâmetros MLE conferem com os artefatos salvos.
A trajetória causal continua por participante × frequência na ordem
ESP→30→40→50→60→70, sem janelas cruzando arquivos. Todos os prefixos são
verificados. Detecção após início de 30, FP alvo ESP e FP lateral no protocolo
inteiro são separados. Tempos null de não detectados e duração censurada são
preservados; comparação pareada usa apenas detectados por ambos.

O teste é exploratório in-sample, não validação independente nem clínica.
Janelas sobrepostas e observações de um participante têm dependência; a
likelihood fatorizada pode subestimar incerteza. n_janelas não é número
comprovado de observações independentes. Usar marginais preditivas em cada
janela não integra conjuntamente parâmetros compartilhados por toda a
trajetória, portanto não torna o HMM inteiro bayesiano conjunto. A/pi fixas.
Uma avaliação LOSO própria poderá ser feita posteriormente; não foi misturada
com este teste fixo ou executada nesta tarefa.

Artefatos novos: `results/bayesian_beta_hmm.json`,
`results/bayesian_beta_samples/`, galeria comentada
`outputs/bayesian_beta_hmm/index.html`, figuras PNG/SVG/PDF e PDF consolidado.
Plots abrangem prioris, preditivas a priori, dados por estado/fase/participante,
posteriores marginais/conjuntas, cadeias/ranks, checagens preditivas, MLE versus
preditiva, log-razão das emissões, delta causal, todos os participantes,
taxas/tempos/pareamentos e recuperação de parâmetros simulados.
Documentação completa em `docs/agents/bayesian_beta_hmm.md`.

```bash
uv pip install --python .venv/bin/python -r requirements-bayesian.txt
.venv/bin/python src/bayesian_beta_hmm.py
.venv/bin/python src/bayesian_beta_hmm.py --reuse-samples
.venv/bin/python src/bayesian_beta_hmm.py --plots-only
.venv/bin/python -m unittest discover -s tests -p test_bayesian_beta_hmm.py
```

Os módulos históricos de EEG/emissão/transição não foram alterados; seus
JSON e resultados permanecem preservados. Métodos e artefatos novos estão
separados, com versões e seeds registradas.

### Resultados da alternativa fixa e verificações

| Emissão | Detecção /88 | FP alvo ESP /88 | FP lateral /88 | Média/mediana desde 30, detectados (s) |
|---|---:|---:|---:|---:|
| MLE histórico reproduzido | 70 | 0 | 8 | 734/730 |
| Preditiva principal | 68 | 0 | 8 | 745,0/730 |
| Preditiva ampla | 68 | 0 | 8 | 736,2/730 |
| Preditiva simétrica | 68 | 0 | 8 | 744,1/730 |

Os três conjuntos não foram classificados ou selecionados por essas taxas.
Principal versus MLE: 68 detecções comuns, 59 empates e nove atrasos Bayes,
zero antecipações; duas exclusivas MLE (Ti 89 Hz e Vi 89 Hz), nenhuma exclusiva
Bayes. Diferença média pareada Bayes−MLE = +18,54 s. As médias não pareadas
usam conjuntos de detectados diferentes e não substituem esse pareamento.

Na principal, posterior média [intervalo equal-tail 95%]:

| Estado | μ | κ | a | b |
|---|---|---|---|---|
| Ausente | 0,5691 [0,5181;0,6196] | 2,0593 [1,6039;2,5735] | 1,1727 [0,8854;1,4921] | 0,8865 [0,6845;1,1200] |
| Presente | 0,2898 [0,2770;0,3028] | 1,3491 [1,2699;1,4317] | 0,3909 [0,3690;0,4132] | 0,9582 [0,8932;1,0270] |

Embora parâmetros médios sejam próximos do MLE, a mistura é diferente:
em p=1e−9, logpdf Ausente passa de −3,7812 (MLE) para +0,0152 (preditiva
principal), enquanto Presente passa de 11,6676 para 11,6786. A posterior
Ausente inclui a<1, aumentando a cauda perto de zero; não confundir isso com
uma mudança no detector. A log-razão local muda antes de atualizar delta;
os efeitos nas decisões são os da tabela e dos percursos completos salvos.

Nas seis posteriores de calibração: zero divergências, R-hat máximo 1,0028,
ESS bulk mínimo 2986 e tail mínimo 3468; BFMI mínimo >0,97. Nos dois conjuntos
simulados, os quatro parâmetros verdadeiros ficaram nos intervalos de 95%,
também com diagnósticos aprovados. Seis testes passaram: likelihood comprimida
versus Beta original, média de densidades versus alternativas incorretas,
extremos/underflow, integração, Viterbi versus enumeração/prefixos e baseline
salva com tempos null preservados.

Na calibração contribuem oito dos onze ESP e 44 dos 55 estimulados; todos os
onze participantes continuam na avaliação. Clipping alterou zero p ESP e
onze p estimulados. Na principal a massa preditiva fora de [1e−9,1−1e−9] é
aproximadamente 8,77e−8 Ausente e 3,04e−4 Presente. As densidades originais
integraram 1 por CDF e quadratura, sem somar alturas numa grade.

As simulações a priori produziram massas p<0,05 Ausente/Presente de
11,84%/30,20% (principal), 10,77%/27,30% (ampla) e 16,70%/16,57% (simétrica).
Isso mostra informação sobre formas/caudas e não corresponde ao FP do HMM.
Em particular, a priori Ausente principal não impõe p uniforme.
Os resultados não indicam melhora nesta comparação; o MLE continua mantido.

### Conclusão para apresentação e retomada

Não houve ganho de detecção, FP **nem tempo** neste teste: a mediana empatou
em 730 s, e as três médias bayesianas foram maiores que 734 s da MLE.
A comparação pareada principal também não teve antecipações, somente empates
e atrasos. Não descrever a diferença entre médias como melhora de velocidade
ou ignorar que o conjunto de detectados mudou.

A contribuição concluída é a inferência da incerteza dos parâmetros e sua
propagação para uma emissão preditiva marginal. Ela não demonstra vantagem
de desempenho nem torna a trajetória inteira um HMM bayesiano conjunto.
Documentação de entrada/retomada: `docs/agents/README.md`, `status.md`,
`architecture.md`, `phase_transition_hmm.md` e `bayesian_beta_hmm.md`.

Uma nova avaliação LOSO da alternativa ou modelagem da dependência pode ser
proposta posteriormente. Não executar automaticamente nova busca, seleção de
priori, alteração de regras ou substituição do MLE. A entrega atual está pronta
para revisão e uso na disciplina, com limites in-sample explicitados.

## 22. Beta única por MAP e comparação LOSO das emissões — 05/10/2026

O usuário autorizou uma nova implementação com **uma única distribuição Beta**,
MAP em vez de mistura posterior, validação leave-one-subject-out e prioris
facilmente alteráveis. Implementação: `src/map_beta_hmm_loso.py`.
Não há senoide, histograma ou labels na emissão. Os módulos históricos e seus
JSON não foram sobrescritos; executar `hmm_inference.py` ainda usa o fluxo
discreto histórico. Para esta tarefa, execute o novo módulo explicitamente.

### Modelo, parâmetros e unidade de comparação

Para Presente, p_i ~ Beta(μκ,(1−μ)κ), μ ~ Beta(2,4),
κ ~ Gamma(shape=2,rate=1), independentes a priori. Configurações no topo:
`MU_PRIOR_ALPHA`, `MU_PRIOR_BETA`, `KAPPA_PRIOR_SHAPE`,
`KAPPA_PRIOR_RATE`, `PRESENT_MIN_DB=50`. Gamma usa scale=1/rate no SciPy.
O MAP maximiza log-verossimilhança + log-priori **nas coordenadas μ,κ**.
A busca usa logit(μ),log(κ) sem adicionar Jacobiano à função objetivo;
um Jacobiano na otimização mudaria o modo estimado. Há gradiente analítico,
três inicializações e verificações de convergência/borda numérica.

Somente Presente recebe priori; Ausente usa exatamente a mesma Beta MLE nos
dois HMMs, isolando a alteração solicitada. A emissão MAP é
BetaPDF(p;μ_MAP κ_MAP,(1−μ_MAP)κ_MAP), uma única Beta. A emissão MLE é outra
Beta ajustada exclusivamente aos dados da dobra, sem priori. Não se usam
alturas de histograma como probabilidades nem uma média de PDFs posteriores.
As densidades originais integram 1, podem exceder 1 e não são posteriores de
estado. Clipping p∈[1e−9,1−1e−9] preservado e quantidades alteradas registradas.

Onze dobras: retém todos os arquivos/fases/frequências de uma pessoa; os outros
dez ajustam ambas as emissões. Presente usa somente alvos em **50,60,70 dB**;
Ausente usa alvos ESP; laterais nunca calibram. Corte idêntico para MLE e MAP.
Avaliação inclui ESP e **todas** as intensidades 30–70 dB. ≥50 dB é proxy
exploratório de resposta mais forte, não confirmação fisiológica por janela;
a distribuição pode representar pior respostas fracas em 30–40 dB.

Rayleigh, canal 0, janela/passo 120/60, A=[[0.99,0.01],[0.15,0.85]],
pi=[0.95,0.05], uma consecutiva; bruto janelado p≤0.005 e uma consecutiva.
Mesma janela e consecutividade para os três métodos. A/pi/janela/alpha vêm
da referência histórica, não são pesquisados nesta tarefa. **LOSO somente
das emissões**: as escolhas temporais históricas foram feitas nestes pacientes;
não apresentar esta comparação como validação externa de toda a seleção.
Não foi executada busca nova nem imposto teto pós-hoc de FP.

O protocolo permanece paciente × frequência, ESP→30→40→50→60→70, causal,
sem backtracking; janelas não cruzam arquivos. Delta e consecutivas persistem
entre intensidades. FP lateral = qualquer disparo inclusive ESP /88 laterais;
FP alvo ESP é separado /88 alvos. Tempos desde 30 só entre detectados,
não detectados têm null e duração censurada separada. Avaliação por pessoa e
por frequência e pareamento MAP/MLE são preservados no JSON.

### Resultados e verificações

| Método | Detecção /88 | FP lateral /88 | FP alvo ESP /88 | Acurácia balanceada lateral | Média/mediana desde 30, detectados (s) |
|---|---:|---:|---:|---:|---:|
| Detector bruto | 58 (65,91%) | 3 (3,41%) | 0 | 81,25% | 824,5/810 |
| HMM sem Bayes (MLE) | 70 (79,55%) | 11 (12,50%) | 1 | 83,52% | 744,6/730 |
| HMM com Bayes (MAP) | 70 (79,55%) | 11 (12,50%) | 1 | 83,52% | 744,6/730 |

MAP/MLE: 70 detecções comuns, 70 empates de tempo e nenhuma exclusividade.
Não houve melhora por MAP com estas prioris; maior detecção dos HMMs sobre
o bruto acompanha FP lateral maior, acima de 5%. Não comparar diretamente
com o resultado in-sample de 70/88 e 8/88 da Seção 20.

São 66 arquivos reais (11 ESP e 55 estimulados), 11 participantes, oito alvos
e oito laterais, 176 trajetórias. O conjunto total elegível possui 104 janelas
Ausente e 512 Presente ≥50 dB; cada dobra remove as contribuições do retido.
Na dobra Ab: 96 ESP/464 Presente; Ausente a=1.1501807728,b=0.8678080186;
Presente MLE a=0.2507164967,b=1.1012103494; MAP μ=0.1858367418,
κ=1.3484708983,a=0.2505954382,b=1.0978754601. Uma emissão global por estado
compartilhada entre frequências, sem estado temporal compartilhado.

Quatro testes passam: MAP versus otimização independente da PDF em μ,κ,
influência da priori/extremos inválidos, exclusão de pessoa/laterais/fases
abaixo do corte, normalização da posterior por quadratura. Posteriores para
plots usam grade adaptativa de μ,log(κ) com Jacobiano κ na **integração**, massa
de borda <1e−6 e refinamento 401→801 pontos. São aproximações numéricas locais
para visualização, não emissão nem nova estimação MCMC. Likelihood iid pode
subestimar incerteza por dependência de janelas/pessoa. MAP depende da
parametrização e prioris extremas/singulares podem não admitir modo interior.

### Reprodução e artefatos

```bash
.venv/bin/python src/map_beta_hmm_loso.py
.venv/bin/python src/map_beta_hmm_loso.py --mu-prior 2 4 --kappa-prior 2 1 --present-min-db 50
.venv/bin/python src/map_beta_hmm_loso.py --plots-only
.venv/bin/python -m unittest discover -s tests -p test_map_beta_hmm_loso.py
```

`--plots-only` usa a configuração e os dados do JSON salvo; não reaplica
valores novos do topo/CLI. Para alterar priori ou corte, execute sem essa flag.
Para preservar variantes use `--output results/map_beta_variante.json` e
`--output-dir outputs/map_beta_variante`. Apagar imagens não impede regeneração.

- `results/map_beta_hmm_loso.json`: configuração, manifesto SHA-256/metadata
  reais, todos os protocolos, calibração auditável por dobra, parâmetros A/B/pi,
  resultados por pessoa/frequência, logpdf em Presente retido e pareamento.
- `outputs/map_beta_hmm_loso/index.html`: tabela dos três métodos e figuras.
- `comparacao.csv` e `README.md` na mesma pasta: tabela exportável e comandos.
- `parametros_por_dobra.csv` e `posterior_parametros_<pessoa>.csv` na mesma
  pasta: parâmetros MLE/MAP e curvas numéricas das prioris/posteriores.
- Onze figuras, cada uma PNG/SVG/PDF: priori/posterior marginal de μ,
  priori/posterior marginal de κ, dados/Betas MLE e MAP e posterior conjunta.
  Cada figura usa somente treinamento da dobra identificada.
- `docs/agents/map_beta_hmm_loso.md`: contrato e ponto de entrada.

Explorar prioris após observar resultados externos é análise de sensibilidade,
não seleção validada independente. Não escolher automaticamente a melhor
priori, executar busca ou apagar materiais históricos. Scripts antigos de
parametrização não precisam ser regenerados: A/B discretas não foram alteradas.

### Extensão: consecutivas e porcentagem mínima configuráveis

`map_beta_hmm_loso.py` agora tem `MIN_CONSECUTIVE=1`, `MIN_PERCENT=None` e
`MODO_REGRA_DECISAO="OR"`. Padrão mantém exatamente os eventos/tempos anteriores.
CLI: `--min-consecutive 4 --min-percent 0.10 --decision-mode AND`.
`--min-consecutive none`/`--min-percent none` desativam um limiar; ambos
desativados são inválidos. Fração deve estar em (0,1], consecutivas inteiro ≥1.
AND exige todos os critérios ativos; OR aceita qualquer um. A mesma regra
vale para bruto e ambos os HMMs, sem retreinar B a partir dos eventos de teste.

A porcentagem é acumulada sobre **janelas observadas desde ESP até a atual**,
causal, e não reinicia entre fases; não usa o total futuro do protocolo.
Consecutivas também continuam entre fases. O primeiro disparo é preservado,
mesmo se a porcentagem cair depois. Detecção alvo começa em 30 dB; um prefixo
com evidência ESP ainda pode contribuir à decisão nessa fronteira, como na
regra histórica de consecutivas. FP alvo ESP fica separado e FP lateral cobre
o protocolo inteiro. Não aplicar porcentagem sobre o caminho completo para
atribuir tempo precoce, pois isso introduziria observações futuras.

JSON/HTML/README registram `consecutivas`, `min_percent`,
`modo_regra_decisao` e definição do denominador. Regeneração por `--plots-only`
usa os critérios salvos, não os valores atuais do código/CLI. A/B discretas e
métodos históricos não foram modificados. Sete testes passaram: os quatro
anteriores, AND/OR/desativação/validação, causalidade/continuidade da porcentagem
e reprodução de todos os eventos padrão nos 11 participantes e três métodos.

## 23. Search compatível com Beta única MAP e LOSO

O usuário autorizou pesquisar MIN_PERCENT, MIN_CONSECUTIVE, pi, A, tamanho e
passo das janelas para o método contínuo. `search_hmm_parameters.py` era
discreto/sintético e mantinha janelamento fixo; não funcionava com as alterações
recentes. Também tinha uma linha isolada `j` que causava NameError, removida.
Nova implementação: `src/search_map_beta_hmm.py`, delegada pela flag
`--continuous-map` do search original. Sem a flag, o método antigo permanece.

```bash
.venv/bin/python src/search_hmm_parameters.py --continuous-map
.venv/bin/python src/search_hmm_parameters.py --continuous-map --resume
# equivalente:
.venv/bin/python src/search_map_beta_hmm.py --resume
```

Prioris/corte vêm do módulo MAP ou de `--mu-prior`, `--kappa-prior`,
`--present-min-db` e permanecem fixos durante a busca; não são hiperparâmetros
pesquisados. Rayleigh, canal configurado e alpha bruto .005 preservados.
Não há senoide, labels nem média de PDFs. Ausente MLE compartilhado; Presente
MLE ou MAP em alvos reais acima do corte. Laterais controlam FP, não ajustam B.

### Grade e seleção externa

Janelas 10/20/30/60/90/120/180; passos metade da janela e janela inteira.
Pi: P(Ausente) em .99,.95,.90,.85,.80,.75,.70,.65,.60,.55,.50.
As mesmas probabilidades nas duas autopermanências de A geram 121 matrizes;
inclui a A numérica heurística salva e a A atual do módulo MAP sem duplicatas.
Atualmente são 122 A, 1.342 pares A/pi, 14 janelamentos.

Consecutivas 1–12, percentuais .01,.025,.05,.075,.09,.10,.20,.30,.40,.50,
.60,.70,.80,.90,1, modos OR/AND. Inclui limiares individuais desativados:
387 regras. São 7.270.956 candidatos por HMM/dobra; bruto tem 5.418.
MLE e MAP pesquisados separadamente, não restringidos aos valores fixos
individuais atualmente no topo do módulo MAP. Grades no topo do novo search.

Maximiza BA lateral sob **FP lateral de treino ≤5%**, inclusive para bruto.
Desempate: detecção maior, FP menor e ordem determinística da grade, sem
minimizar tempo. FP alvo ESP reportado separado. Não usar FP combinado para
substituir lateral. Só se calcula regra AND nos prefixos que satisfazem ambos
os critérios simultaneamente; máximos separados em instantes distintos não
podem ser fundidos por AND.

Cada um dos onze pacientes fica completamente fora da calibração e da busca
da sua dobra. Nos outros dez ajusta-se B por janelamento, selecionam-se A/pi,
janela/passo e regra e só então aplica-se ao retido. Controle de 5% é apenas
no treino; FP externo reportado sem descarte/reajuste. Busca em todos os
pacientes depois das dobras salva configuração final para uso/deployment,
com métricas de treino explícitas. Não existe único vencedor temporal que
tenha sido aplicado a todas as dobras após ver os testes.

Delta causal, contadores e percentual acumulado desde ESP continuam entre
fases; janelas nunca cruzam arquivos. Estatística/p-valor e metadados reais
recalculados por tamanho, sem alterar fórmula Rayleigh ou controle binsM.
Buscar A não estima transições fisiológicas nem Baum–Welch.

### Execução e validação

EEG/FFT carregados uma vez; observações em cache por janela/passo. Betas
ajustadas uma vez por treino/janelamento; Viterbi causal em lotes A/pi;
regras reaproveitam estados. Máscaras excluem padding. Vencedores reaplicados
pelo avaliador escalar e contagens de treino conferidas por assertivas.
Onze testes MAP/search passaram, incluindo lote/prefixos versus scalar,
regras AND/OR versus disparo causal, sequências vazias/padding e validade
das grades. Import do search original passou sem processar dados.

Checkpoint atômico por dobra em `results/search_map_beta_hmm.json`; `--resume`
exige coincidência de grade, dados/hashes, fontes, prioris, corte e alpha.
Interrupção em uma dobra a repete; dobras completas não são refeitas.
`--quick` é somente diagnóstico reduzido, já executado com dados reais e
saídas em `/tmp/search_map_beta_quick`; não usar suas taxas como busca completa.
`--windows`/`--step-modes` restringem a grade; `--output`/`--output-dir` permitem
variantes. O resumo completo vai para `outputs/search_map_beta_hmm/`:
HTML, README e tabela CSV dos três métodos, parâmetros finais e por dobra.
Não altera automaticamente constantes de inferência nem tabelas discretas.

LOSO agora seleciona emissões e hiperparâmetros temporais dentro do treino,
superando a limitação do teste fixo da Seção 22. Ainda é exploração desta
base: prioris/corte/desenho discutidos com esses dados, Presente proxy,
dependência temporal e somente onze pessoas. Não é validação clínica nem
ótimo global fora da grade. Modelos MLE/MAP com configurações próprias não
isolam o efeito da priori; o teste fixo anterior mantém essa comparação.
Documentação detalhada: `docs/agents/search_map_beta_hmm.md`.

### Resultados da busca completa e conclusão

Execução concluída: 11 dobras LOSO e reajuste final em todos; prioris/corte
fixos Beta(2,4), Gamma(2,1) shape/rate e ≥50 dB. Foram 174.567.960 avaliações
de configuração/regra no total: 87.251.472 por HMM e 65.016 bruto.

| Método | Detecção /88 | FP lateral /88 | FP alvo ESP /88 | BA lateral | Média/mediana desde 30 entre detectados (s) |
|---|---:|---:|---:|---:|---:|
| Bruto | 55 (62,50%) | 3 (3,41%) | 0 | 79,55% | 860,4/900 |
| HMM MLE | 61 (69,32%) | 15 (17,05%) | 0 | 76,14% | 732,9/705 |
| HMM MAP | 61 (69,32%) | 15 (17,05%) | 0 | 76,14% | 732,9/705 |

MLE/MAP tiveram eventos/tempos externos idênticos, sem ganho demonstrado por
MAP. Os HMMs detectaram mais que bruto, mas FP externo 17,05% excede 5% e
BA foi menor. Não usar teto de treino como promessa de especificidade no teste.
Médias de tempo usam conjuntos de detectados diferentes; não são antecipações
pareadas. Todos os resultados externos vêm dos modelos específicos de cada
dobra, não de uma configuração escolhida após ver os onze testes.

Configuração final MAP/MLE, reajustada nos 11 pacientes para uso:

```text
WINDOW_SIZE_EPOCHS = 90
WINDOW_STEP_EPOCHS = 45
MIN_CONSECUTIVE = 3
MIN_PERCENT = 0.60
MODO_REGRA_DECISAO = OR
PI = [0.90,0.10]
MATRIX_A = [[0.80,0.20],[0.45,0.55]]
```

Essa run final teve 66/88 alvos e 4/88 FP lateral no próprio treino (75%,
4,55%, BA=85,23%); não corresponde às métricas LOSO acima. Bruto final usa
120/60, somente porcentagem .025, consecutivas desativadas, alpha=.005.
Betas MAP finais: Ausente a=1.1698079195,b=1.0134264024,n=128; Presente
a=.3035875494,b=.9561090720,μ=.2410005268,κ=1.2596966214,n=672.
Parâmetros completos por dobra e ajuste final em `results/search_map_beta_hmm.json`.

Foram conferidos hashes/metadados dos 66 EEG reais, exclusão do sujeito em
todas as fases, reajuste independente das Betas vencedoras e reprodução de
todos os eventos/métricas externas. Retomada por checkpoint testada na grade
reduzida. Arquivos antigos e constantes alteradas manualmente pelo usuário
no módulo MAP (4 consecutivas/.1/OR) preservados; não adotados automaticamente
os vencedores. JSON completo/HTML/README/CSV gravados nas saídas da Seção 23.

## 24. Viabilidade exploratória in-sample, todos os pacientes

O usuário decidiu retirar LOSO nesta etapa e permitir deliberadamente
vazamento de dados para explorar um nicho de funcionamento em trabalho de
disciplina. A implementação suporta `--validation in-sample` (agora padrão)
e `--validation loso` (histórico explícito), com saídas/checkpoints separados.

```bash
.venv/bin/python src/search_hmm_parameters.py --continuous-map
.venv/bin/python src/search_hmm_parameters.py --continuous-map --validation in-sample --resume
# Opcional, protocolo histórico; não executar automaticamente:
.venv/bin/python src/search_hmm_parameters.py --continuous-map --validation loso
```

Nenhuma pessoa é retida no modo in-sample. B Ausente usa todas as janelas
elegíveis dos alvos ESP; B Presente MLE/MAP usa alvos reais ≥50 dB de todos
os sujeitos elegíveis. Uma Beta por estado/método/janelamento, global entre
frequências. Não há parâmetros específicos por paciente. Laterais não
calibram B; servem para selecionar e medir FP nos mesmos dados. Corte,
prioris e detector preservados: μ~Beta(2,4), κ~Gamma(2,1), Rayleigh.
Avaliação continua ESP e 30–70 dB, trajetória causal paciente×frequência.

Mesma grade da Seção 23: 14 janelamentos, 1.342 A/pi, 387 regras. Executada
uma única busca global para cada método: 14.547.330 avaliações no total,
7.270.956 por HMM e 5.418 bruto. Maximiza BA lateral sob FP na própria amostra
≤5%. B, regras, janelamento, A/pi e avaliação compartilham a amostra.
Não são predições externas ou validação independente. Search histórico LOSO
e seus resultados foram preservados, sem novas dobras nesta execução.

| Método | Detecção /88 | FP lateral /88 | FP alvo ESP /88 | BA lateral | Média/mediana desde 30, detectados (s) |
|---|---:|---:|---:|---:|---:|
| Bruto | 58 (65,91%) | 3 (3,41%) | 0 | 81,25% | 824,5/810 |
| HMM MLE | 66 (75,00%) | 4 (4,55%) | 0 | 85,23% | 784,6/737,5 |
| HMM MAP | 66 (75,00%) | 4 (4,55%) | 0 | 85,23% | 784,6/737,5 |

HMM final: janela/passo 90/45, A=[[.8,.2],[.45,.55]], pi=[.9,.1], três
consecutivas OR fração .60 no prefixo desde ESP. Bruto final: 120/60,
consecutivas desativadas, fração .025, alpha=.005. B MAP: Ausente
a=1.1698079195,b=1.0134264024,n=128; Presente a=.3035875494,b=.9561090720,
μ=.2410005268,κ=1.2596966214,n=672. No vencedor, oito ESP têm épocas
suficientes para M=90; os outros três não contribuem para Ausente, mas os
onze sujeitos permanecem na avaliação. Não truncar/remover pessoas do teste.

Resultado reproduz exatamente o ajuste global `deployment_fit` da Seção 23.
MAP e MLE têm mesmos eventos/tempos; não houve vantagem da priori demonstrada.
FP HMM caiu de 17,05% LOSO para 4,55% in-sample, com mudança do protocolo de
avaliação: não atribuir essa diferença a melhora de generalização. É um
exemplo de viabilidade aparente no conjunto usado no ajuste, adequado para
apresentar como experimento com vazamento deliberado, sem afirmação clínica.

Artefatos vigentes: `results/search_map_beta_hmm_in_sample.json`,
`outputs/search_map_beta_hmm_in_sample/index.html`, `README.md`, `comparacao.csv`.
JSON tem `specification.validation=in-sample`, `folds={}`, todos os onze em
`deployment_fit.train_subjects`, `test_subject=null`, modelos globais e
`in_sample_evaluation`/`comparison` com eventos e métricas por pessoa/frequência.
Títulos e textos identificam in-sample. Opções de prioris/corte/saídas da
Seção 23 continuam válidas. `--resume` verifica também o modo e os hashes;
checkpoint anterior a mudança de código não deve ser reaproveitado.

Doze testes MAP/search passaram; inclusão de todos, ausência de dobras,
avaliação na mesma amostra e rejeição de checkpoint de outro modo testadas.
Reprodução do ajuste global anterior e métricas/eventos auditados. Módulo
fixo `map_beta_hmm_loso.py` continua histórico LOSO; o comando vigente é o
search acima. Constantes individuais e arquivos antigos preservados.
