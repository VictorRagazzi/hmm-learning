# `src/search_hmm_parameters.py`

## Entrada contínua MAP/MLE (05/10/2026)

Use `.venv/bin/python src/search_hmm_parameters.py --continuous-map` para a
busca compatível com as alterações recentes: Beta única Presente por MAP,
MLE e bruto, regra percentual causal e seleção de janelas/passos/A/pi/regra
em todos os pacientes (in-sample, padrão atual) ou nos treinos LOSO com
`--validation loso`. A flag delega a `search_map_beta_hmm.py`; documentação em
[`search_map_beta_hmm.md`](search_map_beta_hmm.md). Sem a flag, continua o
fluxo discreto/sintético histórico descrito abaixo. Uma linha isolada `j`,
que causava NameError no import/execução, foi removida.

## Responsabilidade

Busca em grade sobre detectores e suas combinações, `K_SINTETICO`, `PI_INICIAL`,
matriz A e regras temporais. A seleção maximiza acurácia balanceada, sujeita a
FP lateral máximo de 5% no conjunto avaliado.

## Espaço atual

- `MIN_CONSECUTIVE`: 1 a 12.
- `MIN_PERCENT`: 15 valores entre 0,01 e 1,00.
- modo: OR ou AND.
- `PI_INICIAL`: 11 valores para P(Ausente), de 0,99 a 0,50 em passos de 0,05
  após 0,99.
- `K_SINTETICO`: 0,005; 0,01; 0,02; 0,05; 0,1.
- cada autopermanência de A usa os mesmos 11 valores, mais a A heurística
  atualmente salva.
- 11 combinações de detectores; combinações MSC+CSM são excluídas porque as
  estatísticas são equivalentes sob as hipóteses do código.
- FP máximo do HMM: 5%.

O grid completo avalia 73.810 combinações de alto nível e 26.571.600 regras
temporais. O detector bruto por janelas e o detector de gravação completa são
referências separadas, não candidatos HMM.

## Estratégia de execução

1. Calcula estatísticas, p-valores e labels para os 55 arquivos estimulados;
   guarda também p-valores calculados uma vez sobre todas as épocas.
2. Para cada K, recalibra B nos 11 arquivos ESP.
3. Executa Viterbi em lote para sequências de comprimentos iguais.
4. Avalia as 360 regras temporais e descarta as que excedem 5% de FP lateral.
5. Reporta resultados globais e por nível de estímulo (dB), sem sumarização
   de desempenho por participante.
6. Salva checkpoint atômico depois de cada combinação de detectores.

`--resume` reutiliza combinações completas no checkpoint, desde que a
configuração coincida. Uma interrupção dentro de uma combinação repete essa
combinação ao retomar.

## Efeitos e saída

O resultado é gravado em `results/search_hmm_parameters.json`. A busca não
substitui `transition_matrix.json` nem `observation_matrix.json`; adotar uma
configuração campeã exige uma decisão explícita e regeneração consistente.

## Limites metodológicos

- Seleção e avaliação usam os mesmos arquivos `*dB.mat`; métricas são
  exploratórias in-sample e não estimam generalização.
- O detector bruto não recebe a regra de seleção do HMM; compare as métricas
  considerando essa diferença.
- FP de 5% é calculado sobre frequências laterais agrupadas, não controla
  necessariamente 5% por gravação.
- A continua sendo uma heurística de duração; buscar uma A que maximize a
  mesma métrica de avaliação não a transforma em Baum–Welch nem em transições
  ocultas observadas.
- A baseline por gravação completa aplica o detector às épocas sem sobrepor
  janelas, mas ainda executa vários alvos e laterais por gravação.
