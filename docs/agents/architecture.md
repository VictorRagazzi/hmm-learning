# Arquitetura

## Modo vigente: busca global in-sample

Por decisão posterior do usuário, o search contínuo agora tem padrão
`--validation in-sample`: B, seleção e avaliação nos mesmos onze pacientes.
Sem dobras ou paciente retido; modelos Beta globais por estado/configuração.
Saídas `_in_sample` próprias e identificação explícita do vazamento deliberado.
`--validation loso` mantém o fluxo histórico abaixo. Seção 24 do contexto e
`search_map_beta_hmm.md` documentam comandos/resultados; não retomar LOSO
automaticamente. Fórmulas do MAP, detector e causalidade permanecem.

## Extensão de 05/10/2026: Beta única MAP / MLE / bruto em LOSO

Busca dos parâmetros temporais: `search_map_beta_hmm.py`, também acessível por
`search_hmm_parameters.py --continuous-map`, reaproveita MAP/leitores/protocolo,
cacheia janelamentos e executa Viterbi causal em lotes A/pi. Cada método
seleciona seus parâmetros no treino LOSO sob FP lateral ≤5%; retido somente
avalia. Um ajuste extra em todos fornece configuração final para uso, com
métricas explicitamente de treino. Ver `search_map_beta_hmm.md`.

`map_beta_hmm_loso.py` é o novo ponto de entrada autorizado. Reutiliza
leitura/FFT/Rayleigh de `continuous_rayleigh_hmm.py`, protocolos auditáveis e
métricas de `compare_histogram_phase_rayleigh.py` e Viterbi causal de
`phase_transition_hmm.py`. Não chama mains/buscas ou injeção sintética.

Por pessoa retida: alvos ESP de treino → Beta Ausente MLE; alvos de treino
≥50 dB → Beta Presente MLE e MAP. MAP usa prioris configuráveis Beta em μ e
Gamma em κ, maximiza posterior nas coordenadas μ,κ e usa uma única PDF Beta.
Os dois HMMs compartilham Ausente, A/pi/janelas/regra. Avalia retido em ESP e
30–70 dB; bruto usa mesmas janelas com alpha=.005. A/pi/janela/alpha históricos
fixos limitam a validação a LOSO das emissões. Ver Seção 22 e
`map_beta_hmm_loso.md`. JSON/galeria próprios; posterior por quadratura serve
somente para visualização. O histórico abaixo continua descrevendo cada módulo
original; `hmm_inference.py` não passou a usar MAP automaticamente.

## Objetivo

O projeto explora detecção de resposta auditiva de estado estável (ASSR) em EEG
com um HMM de dois estados. Existem emissões categóricas, Beta por MLE,
histograma da estatística ORD e a alternativa preditiva Beta bayesiana.
O protocolo ativo
após a decisão final de 30/09/2026 usa Beta dos p-valores Rayleigh, calibrada
com alvos ESP e alvos das gravações estimuladas reais,
e inferência causal por paciente × frequência nas fases ESP, 30, 40, 50, 60 e
70 dB. A e pi são reutilizados de execuções anteriores, sem nova busca.
O disparo depende de consecutividade; o estado terminal do Viterbi não é uma
probabilidade posterior normalizada.

## Protocolo compartilhado: fases concatenadas

`phase_transition_hmm.py` reúne um protocolo por paciente e por
frequência, para alvos e laterais separadamente. Calcula janelas dentro de cada
arquivo e concatena somente as observações na ordem das fases. As janelas não
atravessam arquivos; delta e consecutividade persistem entre intensidades.
Pacientes e frequências nunca compartilham estado temporal.

`continuous_rayleigh_hmm.py` fornece o ajuste Beta global por estado: Ausente
usa p-valores dos alvos ESP; Presente usa alvos reais de todas as intensidades.
FFT → PLV² → p=exp(-M*PLV²) → duas log-densidades Beta → delta causal do Viterbi.
O Rayleigh bruto compara p com alpha histórico. Histograma ESP/sintético não é
a emissão mantida; seus módulos e artefatos permanecem como experimentos.

Para reprodução/plots, leia parâmetros/emissões salvos em
`results/phase_transition_hmm.json` e chame as funções com esses valores.
O main de `phase_transition_hmm.py` executa busca; não executá-lo para uma
visualização. A e pi escolhidos historicamente são diferentes da A heurística
dos scripts originais de parametrização.

A inferência atualiza o melhor estado terminal sem backtracking futuro.
Detecção alvo começa em 30 dB; FP lateral considera qualquer disparo no
protocolo inteiro, incluindo ESP. O FP do alvo durante ESP é separado.
Cada paciente tem oito alvos e oito laterais: taxa de detecção = detectados/8;
taxa de FP = laterais com disparo/8. Tempos desde 30 dB são reportados entre
detectados; a duração máxima dos não detectados aparece em uma métrica censurada
separada. Consulte a Seção 20 do contexto técnico para parâmetros e resultados.

## Alternativa bayesiana concluída, mantida separadamente

`bayesian_beta_hmm.py` lê observações e parâmetros históricos, reproduz MLE,
confere EEG/metadados e ajusta a posterior de μ/κ de cada estado. A emissão
é a média das densidades Beta posteriores em log-espaço, calculada com
logsumexp. A e pi ficam fixas; a atualização causal é a mesma do protocolo
ativo. Não há histogramas, labels, injeção sintética ou busca nessa alternativa.

As saídas são `results/bayesian_beta_hmm.json` e
`results/bayesian_beta_samples/`, sem sobrescrever resultados históricos.
`plot_bayesian_beta_hmm.py` gera 46 figuras PNG/SVG/PDF, PDF consolidado,
galeria comentada e CSVs em `outputs/bayesian_beta_hmm/`.

A implementação, comparação fixa e verificações estão concluídas. Não houve
ganho de detecção, FP ou tempo; por isso a MLE continua sendo o método ativo.
O ganho é a representação da incerteza dos parâmetros. A likelihood fatorizada
ignora dependência entre janelas/participantes, e marginais preditivas por
janela não integram conjuntamente parâmetros compartilhados pela trajetória.
Leia `bayesian_beta_hmm.md` e a Seção 21 do contexto antes de retomar o módulo.

## Pipelines históricos por arquivo

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
```

search_hmm_parameters.py reutiliza get_obs_matrix.py e hmm_inference.py para
recalibrar B e avaliar uma grade de parâmetros, persistida em JSON.

O fluxo experimental contínuo é separado do pipeline discreto:

```text
coeficientes FFT / bins locais
       |
       +--> continuous_rayleigh_hmm.py
       +--> early_detection_rayleigh_hmm.py
       +--> compare_detectors_early_hmm.py
                    |
                    v
       early_detection_detectors_hmm.json
```

Esse fluxo ajusta emissões Beta contínuas em dados reais, roda inferência
causal e compara cada HMM com o mesmo detector acumulado. Atualmente inclui
Rayleigh, MSC, MMSC, Hotelling geral e F espectral local; CSM é omitido por ser
equivalente à MSC.

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
- `continuous_rayleigh_hmm.py` fornece calibração contínua real e utilitários
  compartilhados pelos experimentos causais.
- `phase_transition_hmm.py` constrói trajetórias concatenadas e implementa
  Viterbi causal; seu main executa busca histórica.
- `bayesian_beta_hmm.py` implementa a alternativa experimental e a comparação
  fixa com MLE, com amostras/diagnósticos próprios.
- `plot_bayesian_beta_hmm.py` gera a apresentação e tabelas da alternativa.
- `early_detection_rayleigh_hmm.py` mede tempo causal e compara com detector
  acumulado.
- `compare_detectors_early_hmm.py` executa o protocolo para os cinco detectores
  não redundantes e mantém checkpoint por detector.
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

Depois confira schema, somas de linha, contagens e falso positivo global/por
frequência, conforme `AGENTS.md`. O próximo trabalho está descrito na seção
"Ponto de retomada" de `status.md`.
