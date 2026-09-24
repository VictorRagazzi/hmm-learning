# Instruções para agentes — parametrização do HMM para ASSR

## Leitura obrigatória

Antes de alterar o processamento de EEG, `src/get_obs_matrix.py` ou
`src/get_tran_matrix.py`, leia por
completo [docs/CONTEXTO_GET_MATRIX.md](docs/CONTEXTO_GET_MATRIX.md).
Esse documento é a referência técnica do estado atual, fórmulas, dados,
resultados, limitações e comandos de verificação.

Sempre que uma alteração mudar detector, configuração, estrutura retornada,
resultados de referência ou decisões metodológicas, atualize também esse
documento na mesma tarefa. Não deixe a documentação descrever uma versão antiga.

## Escopo atual

O projeto estima a matriz de emissão `B` e uma matriz de transição heurística
`A` para um HMM de detecção de ASSR em EEG. Treinamento, inferência e decisão
final do HMM ainda não fazem parte desta etapa.

Não apresente probabilidades obtidas por injeção sintética como estimativas
clínicas validadas.

## Implementação vigente

- Matriz B: `src/get_obs_matrix.py`.
- Matriz A: `src/get_tran_matrix.py`.
- Entrada de calibração: todos os arquivos `data/*ESP.mat`.
- EEG: `x` no formato `(canais, épocas, amostras)`; leia `Fs`, `freqEstim` e
  `binsM` de cada arquivo.
- Canal configurável: `CHANNEL_INDEX`.
- Detector: magnitude quadrática da coerência (MSC) entre épocas em
  `freqEstim[i]`; `binsM` é controle lateral, não denominador.
- Com `M` épocas por janela, usa-se `Beta(1, M-1)` sob H0.
- Configuração atual: janela de 10 épocas, passo 5, cinco labels e `alpha=0.05`.
- Thresholds: quantis teóricos da distribuição Beta definidos por faixas de
  p-valor; não são quantis empíricos dos dados ESP.
- `B(Ausente)`: histograma das janelas ESP.
- `B(Presente)`: mesmas janelas ESP após senoide com amplitude
  `K_SINTETICO * std(época)`.
- Suavização atual: pseudocontagem 0.5 por célula.
- A matriz B final é global: some contagens das oito frequências e normalize
  uma única vez.
- Preserve resultados por frequência como diagnóstico. Não concatene
  sequências temporais de frequências, participantes ou condições diferentes.
- A matriz A usa uma heurística de duração: cada arquivo com N janelas
  contribui com N-1 autopassagens e uma saída implícita. Ela não foi estimada
  a partir de transições ocultas observadas.
- Ao executar os scripts, grave as tabelas em `results/observation_matrix.json`
  e `results/transition_matrix.json`.

A estatística atual é MSC; ela não é CSM nem teste F espectral local.

## Decisões que não devem ser revertidas silenciosamente

- Use uma única matriz B compartilhada pelas frequências.
- Mantenha a taxa de falso positivo separada por frequência, pois 83 Hz mostrou
  desvio relevante mesmo com falso positivo global próximo de 5%.
- Não fixe 180 épocas: os arquivos reais ESP possuem entre 30 e 300.
- Não fixe `Fs`: os dados atuais incluem 1000 e 1750 Hz.
- Não associe `K_SINTETICO` diretamente a dB acústicos.
- Não use estímulos reais para ajustar thresholds e avaliar o mesmo ajuste como
  se fosse validação independente.
- Se o detector, número de épocas, canais ou forma de agregação mudar, reveja
  a distribuição nula da MSC.
- Não descreva a matriz A atual como resultado de Baum–Welch, transições reais
  observadas ou persistência fisiológica validada.
- Mantenha tamanho e passo das janelas consistentes nos dois scripts, salvo
  decisão explícita e documentada em contrário.

## Metadados e validação

Preserve nos resultados intermediários participante, condição, canal,
frequência, bin de controle, intervalo de épocas, `Fs`, valor MSC, p-valor e
label.

Depois de alterações, execute:

```bash
.venv/bin/python -m py_compile src/get_obs_matrix.py
.venv/bin/python src/get_obs_matrix.py
.venv/bin/python src/get_tran_matrix.py
```

Relate as matrizes A e B, quantidade de arquivos/frequências/janelas, falso
positivo global e falso positivo por frequência. Confirme que os JSON foram
gravados e que todas as linhas somam 1. Use arquivos ESP autênticos; um arquivo
de estímulo apenas renomeado não serve para estimar `B(Ausente)`.

## Estilo

Mantenha configurações relevantes no topo do arquivo principal. Prefira poucas
funções claras e fluxo explícito. Antes de mudanças substanciais, inspecione o
código e os metadados reais. Preserve alterações não relacionadas já existentes
na árvore de trabalho.
