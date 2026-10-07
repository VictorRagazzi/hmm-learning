# Status do projeto

## Etapa vigente — in-sample sem LOSO, concluída

Usuário autorizou vazamento deliberado para explorar viabilidade na disciplina.
`search_hmm_parameters.py --continuous-map` agora usa in-sample por padrão:
todos os sujeitos ajustam B, selecionam hiperparâmetros e são avaliados.
`--validation loso` preserva o protocolo anterior como opção explícita.
Novas saídas: `results/search_map_beta_hmm_in_sample.json` e
`outputs/search_map_beta_hmm_in_sample/`. Ver Seção 24 e `search_map_beta_hmm.md`.

Busca completa concluída, 14.547.330 avaliações: bruto 58/88 detecções e
3/88 FP lateral (BA 81,25%); MLE/MAP 66/88 e 4/88 (BA 85,23%), zero FP
alvo ESP e eventos/tempos MLE/MAP iguais. Vencedor HMM 90/45,
A=[[.8,.2],[.45,.55]], pi=[.9,.1], 3 consecutivas OR 60%; prioris/corte mantidos.
FP 4,55% é in-sample, não validação externa. Nenhum ganho MAP sobre MLE.
Doze testes passam; modo/checkpoint/inclusão de todos conferidos e resultado
reproduz ajuste global anterior. Saídas LOSO e constantes históricas preservadas.

## Busca compatível com MAP/MLE — 05/10/2026

Novo motor `search_map_beta_hmm.py`, acessível por
`.venv/bin/python src/search_hmm_parameters.py --continuous-map`.
O search original era discreto/sintético, com janela fixa; uma linha isolada
`j` que bloqueava import/execução foi removida. A flag mantém separados os
dois fluxos. Documentação em `search_map_beta_hmm.md` e Seção 23 do contexto.

Busca janela/passo, A, pi, consecutivas, porcentagem e AND/OR nos dez sujeitos
de treino de cada dobra LOSO. Prioris/corte do módulo MAP permanecem fixos.
Controle de FP **lateral** de treino ≤5% nos três métodos; teste externo nunca
é filtrado. Grade atual: 14 janelamentos, 1.342 pares A/pi e 387 regras.
MLE/MAP pesquisados separadamente. Configuração final com todos os pacientes
salva à parte como ajuste para uso, sem transformar seu treino em teste.

Artefatos: `results/search_map_beta_hmm.json` e
`outputs/search_map_beta_hmm/` (HTML/README/CSV). Checkpoint atômico por dobra,
`--resume` verifica fontes, dados e configuração. Teste reduzido e retomada
passaram; onze testes MAP/search passaram. Não aplicar automaticamente os
vencedores às constantes de `map_beta_hmm_loso.py`; suas alterações manuais
MIN_CONSECUTIVE=4/MIN_PERCENT=.1 foram preservadas.

Busca completa concluída (11 dobras + reajuste final): 174.567.960 avaliações.
LOSO: bruto 55/88 detecções e 3/88 FP lateral (BA=79,55%); MLE/MAP 61/88 e
15/88 (BA=76,14%), eventos externos idênticos, zero FP alvo ESP. Não houve
ganho MAP; FP HMM externo 17,05% excedeu 5%. Reajuste final MAP/MLE:
90/45, A=[[.8,.2],[.45,.55]], pi=[.9,.1], 3 consecutivas OR 60%; treino
66/88 e 4/88, BA=85,23%, sem interpretar como LOSO. Auditoria final recalculou
Betas vencedoras com exclusões completas e reproduziu eventos/métricas.

## Nova etapa concluída — 05/10/2026: Beta única MAP em LOSO

Extensão dos critérios: `MIN_CONSECUTIVE`, `MIN_PERCENT` e
`MODO_REGRA_DECISAO` no topo do módulo MAP, também por CLI
`--min-consecutive 4 --min-percent 0.10 --decision-mode AND`.
Porcentagem causal no prefixo desde ESP; contadores continuam entre fases.
`none` desativa um limiar; pelo menos um ativo. Padrão 1/None/OR preserva
resultados. Sete testes passam, incluindo reprodução dos eventos históricos.
Leia `map_beta_hmm_loso.md` para a definição exata do denominador.

O usuário solicitou uma única Beta Presente por MAP com μ~Beta(2,4),
κ~Gamma(2,1), prioris facilmente alteráveis e corte de treino ≥50 dB.
Implementação/execução/documentação prontas em `src/map_beta_hmm_loso.py`;
ver `map_beta_hmm_loso.md` e Seção 22. Ausente MLE compartilhado; avaliação
ESP e todas as intensidades, com onze dobras excluindo uma pessoa inteira.
Comparação com bruto e MLE, PNG/SVG/PDF por dobra e tabela exportável em
`outputs/map_beta_hmm_loso/`; JSON em `results/map_beta_hmm_loso.json`.
`--plots-only` permite recuperar todas as imagens após apagá-las.

Resultados: bruto 58/88 detecções, 3/88 FP lateral; MLE/MAP 70/88, 11/88,
um FP alvo ESP. Tempos médios/medianos desde 30: bruto 824,5/810 s;
HMMs 744,6/730 s. MAP não melhorou eventos/tempos frente ao MLE; FP HMM
12,5% excede 5%. Quatro testes passaram, MAP foi conferido com otimização
independente e quadraturas posteriores com refinamento/bordas.

A/pi/janela/alpha históricos fixos: LOSO somente das emissões, sem busca nova
nem validação independente de toda a seleção. ≥50 dB é proxy de resposta.
Módulos e resultados históricos abaixo foram preservados. O ponto de entrada
`hmm_inference.py` continua discreto; o novo método usa o módulo MAP dedicado.

Referência histórica atualizada em 30/09/2026. As seções abaixo preservam
o contexto de cada etapa, não uma validação clínica.

## Decisão final e ponto de retomada — manter método Beta

O usuário decidiu manter o HMM contínuo antigo, após testar histogramas.
"Antigo" = densidades Beta dos p-valores Rayleigh; Ausente ajustado com alvos
ESP, Presente com alvos reais das gravações estimuladas. Não é a B de cinco
labels nem a emissão sintética. Leia a Seção 20 de
`docs/CONTEXTO_GET_MATRIX.md` e `phase_transition_hmm.md`.

Preservar a unidade paciente × frequência e a concatenação
ESP → 30 → 40 → 50 → 60 → 70 dB, com janelas dentro dos arquivos. Delta e
consecutivas continuam entre fases, com estado terminal Viterbi causal sem
backtracking futuro. FP lateral solicitado conta qualquer disparo no protocolo
e divide pelas laterais medidas. Tempos entre detectados e duração censurada
continuam separados. Histogramas ficam somente como histórico comparativo.

Referência fixa Rayleigh 120/60: A=[[0,99;0,01],[0,15;0,85]], pi=[0,95;0,05],
uma consecutiva. Reprodução do antigo: 70/88 alvos (79,55%), 8/88 laterais
(9,09%), acurácia balanceada lateral 85,23%, média/mediana desde 30 dB
734/730 s entre detectados. FP combinado histórico 4,55% não é FP lateral.
Histograma real/sintético nessa configuração caiu para 35,23%/29,55%; os bins
uniformes perderam resolução. Não adotar histogramas nem executar busca nova
automaticamente. As runs LOSO existentes continuam separadas do in-sample.

## Última etapa concluída — emissão Beta bayesiana e apresentação

A implementação experimental está pronta em `src/bayesian_beta_hmm.py`, com
figuras em `src/plot_bayesian_beta_hmm.py`. A posterior conjunta de μ e κ foi
estimada por NUTS para três prioris predefinidas, e a média das PDFs posteriores
foi usada como emissão alternativa no mesmo Viterbi causal. Detector, canal,
dados, A, pi, janela/passo 120/60 e regra ficaram fixos; nenhuma busca foi feita.

Concluído e verificado:

- Baseline MLE reproduzida exatamente antes da comparação.
- Observações/intervalos conferidos com os 66 arquivos EEG reais; 104 janelas
  ESP e 1744 estimuladas na calibração global, sem laterais no ajuste.
- Amostras completas NetCDF, amostras preditivas fixas NPZ e resumos com
  intervalos de credibilidade de μ,κ,a,b salvos em arquivos próprios.
- Seis posteriores de calibração convergiram pelos critérios registrados:
  zero divergências, R-hat máximo 1,0028, ESS bulk/tail mínimos 2986/3468.
- Recuperação em dois conjuntos simulados, normalização das densidades,
  extremos/underflow, logsumexp e causalidade por prefixos verificados.
- Seis testes automatizados passaram; hashes, denominadores e exportações
  foram conferidos.
- **46 figuras em PNG/SVG/PDF**, PDF consolidado de 46 páginas, galeria
  comentada, relatório e CSVs globais/por participante/por frequência prontos.

| Emissão | Detectados /88 | FP alvo ESP /88 | FP lateral /88 | Média/mediana entre detectados (s) |
|---|---:|---:|---:|---:|
| MLE reproduzida | 70 | 0 | 8 | 734/730 |
| Bayes principal | 68 | 0 | 8 | 745,0/730 |
| Bayes ampla | 68 | 0 | 8 | 736,2/730 |
| Bayes simétrica | 68 | 0 | 8 | 744,1/730 |

**Conclusão:** não houve melhora de detecção, FP ou tempo. A mediana empatou;
as médias foram maiores. Na comparação pareada da principal, 68 detecções
comuns tiveram 59 empates e nove atrasos Bayes, sem antecipações; duas foram
exclusivas MLE, com diferença média pareada +18,54 s. O ganho metodológico é
quantificar incerteza dos parâmetros, não desempenho demonstrado do detector.
Nenhuma priori foi selecionada pelo desempenho; a MLE permanece ativa.

Entrega e reprodução:

- [Documentação técnica](bayesian_beta_hmm.md), Seção 21 do contexto.
- [Resultados](../../results/bayesian_beta_hmm.json) e
  `results/bayesian_beta_samples/`.
- [Galeria comentada](../../outputs/bayesian_beta_hmm/index.html),
  [PDF](../../outputs/bayesian_beta_hmm/apresentacao_figuras.pdf) e
  [relatório](../../outputs/bayesian_beta_hmm/README.md).

```bash
.venv/bin/python src/bayesian_beta_hmm.py --reuse-samples
.venv/bin/python src/bayesian_beta_hmm.py --plots-only
.venv/bin/python -m unittest discover -s tests -p test_bayesian_beta_hmm.py
```

Não há trabalho pendente para concluir esta entrega. LOSO da alternativa,
modelagem da dependência entre janelas/participantes ou inferência conjunta
dos parâmetros compartilhados na trajetória são possibilidades futuras;
não foram executadas nem estão autorizadas automaticamente. O teste atual é
in-sample, não validação independente ou clínica. Não rodar o main de
`phase_transition_hmm.py` para reprodução: ele executa search.

O prompt antigo de apresentação foi preservado como histórico. O registro da
apresentação anterior cita `outputs/apresentacao_metodo_beta/index.html`, mas
essa pasta não está disponível na árvore inspecionada; `apresentacao.pdf`
está presente. A galeria bayesiana acima foi conferida e está disponível.

## Implementação histórica de referência — 27/09/2026

- Leitura de EEG MATLAB v7.3 (`x`, `Fs`, `freqEstim` e `binsM`) com `h5py`.
- Extração, por época, do coeficiente FFT complexo no bin mais próximo de cada
  frequência.
- Registro plugável com seis detectores: MSC, MMSC split-half, Rayleigh, CSM,
  Hotelling T² geral e F espectral local.
- Calibração de uma matriz de emissão global B, com diagnósticos separados por
  frequência, a partir de 11 arquivos ESP.
- Construção da linha `Ausente` com dados ESP reais e da linha `Presente` com
  senoide sintética coerente em fase injetada nos mesmos dados.
- Construção de uma matriz de transição A por heurística de duração, usando 11
  arquivos ESP e 55 arquivos com estímulo.
- Inferência Viterbi independente por arquivo e frequência, em log-espaço.
- Comparação da decisão do HMM com o detector bruto, tanto nas frequências de
  estímulo quanto nas laterais de `binsM`.
- Agregações por nível em dB, baseline de gravação completa e persistência da
  análise em JSON.
- Busca em grade de detectores, `K_SINTETICO`, distribuição inicial, matriz A
  e regra de decisão, com restrição de FP lateral máximo de 5% e checkpoint.
- Experimento separado com Rayleigh, janelas de 10/20/30/60 épocas e emissões
  Beta contínuas ajustadas em dados reais, deliberadamente in-sample.
- Comparação causal de tempo de exame entre HMM contínuo e Rayleigh acumulado,
  com sete janelas, passos com/sem sobreposição e FP global máximo de 5%.
- Comparação individual HMM versus detector acumulado para Rayleigh, MSC,
  MMSC, Hotelling geral e F espectral local, sem fusão.
- Exemplo didático separado de Viterbi em `src/viterbi.py`; ele não participa
  do pipeline ASSR.

## Configuração histórica do pipeline categórico por arquivo

| Item | Valor no código/artefato atual |
|---|---|
| Detector para B | `msc` |
| Combinação padrão da inferência | `rayleigh+mmsc+csm` |
| Canal | `0` |
| Estados | `Ausente`, `Presente` |
| Labels | `muito_baixo`, `baixo`, `medio`, `alto`, `muito_alto` |
| Fronteiras de p-valor | `0.50`, `0.10`, `0.05`, `0.01` |
| Alpha bruto | `0.05` |
| Janela/passo de B, A e inferência | `10/5` épocas |
| `PI_INICIAL` da inferência | `[0.99, 0.01]` |
| Regra da inferência | 4 consecutivas OU 10% em `Presente` |
| `K_SINTETICO` | `0.01 * std(época)` |
| Suavização de B | `0.5` por célula |

O detector também pode ser escolhido por `ASSR_DETECTOR` ou, ao executar
`get_obs_matrix.py`/`hmm_inference.py` interativamente, pelo menu. O mesmo
detector deve ser usado para gerar B e fazer inferência.

## Artefatos históricos preservados

### `results/observation_matrix.json`

- Detector: MSC.
- 11 arquivos, 8 frequências e 2.232 janelas por estado.
- Falso positivo global ESP: 4,08%.
- Matriz B:

```text
Ausente  [0.503692, 0.414634, 0.040501, 0.035578, 0.005594]
Presente [0.135825, 0.321996, 0.110316, 0.180577, 0.251287]
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

O relatório foi gerado com Rayleigh+MMSC+CSM, janela/passo 10/5 e regra de
decisão 4 consecutivas OU 10% das janelas. A detecção foi 47,50% e o FP
lateral 20,45%, mostrando que a configuração de inferência padrão não atende
ao alvo de FP inferior a 5%.

### `results/search_hmm_parameters.json`

A busca completa avaliou 73.810 configurações de alto nível e 26.571.600 regras
temporais, com limite de FP lateral de 5%. A melhor configuração foi
Rayleigh+CSM, com `K_SINTETICO=0.01`, `PI_INICIAL=[0.5, 0.5]`, matriz A
`[[0.5, 0.5], [0.15, 0.85]]` e regra `5 consecutivas AND 30%`. No conjunto
in-sample, detectou 35,23% (155/440) e teve FP lateral de 4,77% (21/440), com
acurácia balanceada de 65,23%. O resultado não é validação independente.

### `results/continuous_rayleigh_hmm.json`

O experimento contínuo ajusta `Ausente` com ESP real e `Presente` com todas as
frequências estimuladas reais, usando os mesmos participantes na avaliação por
decisão explícita desta etapa. O melhor tamanho foi 60 épocas, com 45,91% de
detecção, 5,00% de FP lateral global e 70,45% de acurácia balanceada. Rayleigh
na gravação completa ainda foi superior: 53,41%, 3,86% e 74,77%. Esses números
são in-sample e não constituem validação independente.

### `results/early_detection_rayleigh_hmm.json`

Na melhor configuração global, janela/passo 60/60, o HMM causal detectou
43,64% com 4,77% de FP e o Rayleigh acumulado detectou 51,59% com os mesmos
4,77% de FP. Ambos tiveram mediana de 120 s até detecção, e Rayleigh encerrou
os exames em menor tempo médio. Em 180/90 o HMM obteve uma vantagem
exploratória (41,14% contra 36,36%), mas com FP maior (4,77% contra 2,73%). A
seleção e avaliação continuam deliberadamente in-sample.

### `results/early_detection_detectors_hmm.json`

Nenhum HMM individual superou o mesmo detector acumulado. Rayleigh obteve
43,64%/4,77% no HMM contra 51,59%/4,77% bruto; MSC 46,59%/5,00% contra
53,64%/5,00%; e MMSC 37,95%/4,77% contra 44,55%/5,00%. Os pares representam
detecção/FP global e continuam in-sample. MSC foi o melhor detector nos dois
lados, mas o HMM reduziu sua detecção em 7,05 pontos percentuais.
Hotelling geral obteve 44,55%/4,09% no HMM contra 48,18%/3,64% bruto; F
espectral local obteve 43,64%/4,55% contra 47,05%/5,00%. Eles reduziram a
distância, mas também não mostraram ganho do HMM.

## Estado de verificação

Em 27/09/2026:

- `py_compile` passou para os módulos alterados de `src/`;
- os JSON de B, A, inferência discreta e experimento contínuo foram lidos com
  sucesso;
- todas as linhas das matrizes A e B somam 1;
- há 11 arquivos `*ESP.mat` e 55 arquivos `*dB.mat` em `data/`.

Os pipelines de B e A foram reexecutados nesta tarefa com os dados autênticos.
As buscas contínuas e causais foram concluídas, incluindo cinco detectores em
14 configurações de janela/passo cada; seus JSON estão com `status=complete`.
A busca discreta ampliada permanece concluída para as 11 combinações.

## Pendências e inconsistências conhecidas

1. Várias chaves e funções ainda contêm `msc` no nome (`valor_msc`,
   `thresholds_msc`, `calcular_msc_janelas`) embora possam representar qualquer
   detector. Isso é compatibilidade histórica, não garantia de semântica MSC.
2. MMSC é um split-half experimental; seus p-valores agora vêm da convolução
   das distribuições Beta das duas metades, não de uma aproximação Beta.
3. A busca em grade usa os mesmos 55 arquivos com estímulo para selecionar os
   parâmetros e medir detecção/falso positivo. Não há separação treino/validação
   nem validação cruzada, logo a métrica otimizada não é desempenho fora da
   amostra.
4. MSC e CSM circular são equivalentes sob as hipóteses estatísticas atuais;
   combinações que as tratam como independentes são rejeitadas.
5. `search_hmm_parameters.py` salva o estado por combinação em
   `results/search_hmm_parameters.json` e pode retomar com `--resume`.
6. A ausência de suíte automatizada era o estado histórico de 27/09/2026.
   Hoje existem testes em `tests/`, incluindo os seis da emissão bayesiana e
   os testes de histograma. Os resultados recentes estão registrados acima.

## Limites de interpretação

A heurística do pipeline discreto não é Baum–Welch nem uma medição de
persistência fisiológica. B(Presente) desse pipeline é sintética, e
`K_SINTETICO` não equivale diretamente a dB acústicos. A emissão contínua
mantida na decisão final usa Presente real, conforme o início deste arquivo. Frequências
laterais e resultados da busca são controles exploratórios. Nada disso deve ser
apresentado como estimativa clínica validada.

## Ponto de retomada histórico — 27/09/2026

O trabalho parou depois da comparação causal individual dos cinco detectores
não redundantes. Nenhum HMM superou sua própria baseline acumulada; MSC bruto
permanece o melhor resultado global. Hotelling geral e F espectral local foram
implementados, calibrados em simulação nula e incluídos no artefato comparativo.

Próximo experimento acordado: combinar **Rayleigh + F espectral local** usando
uma emissão conjunta aprendida nos dados reais. Não multiplicar as duas
emissões marginais como se fossem independentes, pois ambas vêm do mesmo EEG.
Uma implementação inicial apropriada é ajustar, por estado, uma densidade
bivariada sobre os dois p-valores transformados (por exemplo,
`[-log10(p_rayleigh), -log10(p_spectral_f)]`) e usar sua log-densidade no HMM
causal. Deve haver uma baseline de fusão acumulada sem HMM, sob o mesmo teto de
FP global de 5%, para separar ganho de fusão de ganho temporal.

Manter nesta próxima etapa o vazamento deliberado atual: todos os participantes
podem ajustar e avaliar a emissão. Somente depois de concluir seleção de
estrutura/detector será feita validação sem vazamento. Preservar as sete
janelas, os dois modos de passo, tempos de exame, comparação pareada e
diagnósticos por dB.

## Experimento histórico do histograma — 30/09/2026, antes da decisão final

A decisão final no início deste arquivo substitui as referências a método
ativo nesta seção. As instruções abaixo documentam o experimento realizado.

Na etapa do experimento, esta seção substituiu o roteiro anterior. Leia a Seção 19 de
`docs/CONTEXTO_GET_MATRIX.md` e `compare_histogram_phase_rayleigh.md` antes de
continuar. O usuário definiu explicitamente:

1. Emissão por histograma da estatística ORD, não Beta dos p-valores.
2. Ausente calibrado com ESP; Presente com os mesmos ESP + senoide de amplitude
   K × std(época), sem gravações estimuladas no ajuste da emissão.
3. Teste por paciente × frequência: ESP → 30 → 40 → 50 → 60 → 70 dB.
4. Parâmetros das runs anteriores fixos; **não executar search** nesta tarefa.
5. Detecção e FP lateral por paciente, tempos e comparação com Rayleigh bruto.

Entrada atual: `src/compare_histogram_phase_rayleigh.py`. Reutiliza 14
configurações Rayleigh contínuas de transição e o vencedor discreto (15 testes
fixos), com K=0,01, 20 bins comuns e pseudocontagem 0,5. Janelas não cruzam
arquivos; delta e consecutividade continuam entre intensidades. Estado terminal
Viterbi causal, sem backtracking futuro. Baseline Rayleigh janelada com alpha
e consecutividade históricos. Há 88 alvos e 88 laterais de 11 pacientes.

FP solicitado usa laterais com qualquer disparo no protocolo, inclusive ESP,
dividido por laterais medidas; não é FP combinado de ESP/laterais sobre 176.
Cada paciente tem oito frequências de cada tipo. Tempos desde 30 dB são
calculados entre detectados; não detectados têm tempo nulo e duração censurada
separada. Pares detectados por ambos têm comparação de velocidade separada.

Resultados com vencedores **predefinidos nos artefatos históricos**:

- HMM histograma com A/pi/regra antigos 120/60: 26/88 alvos (29,55%), FP lateral
  0/88, acurácia balanceada 64,77%, tempo médio/mediano detectados 1038/1140 s.
- Rayleigh bruto com parâmetros antigos 90/45: 61/88 (69,32%), FP lateral
  8/88 (9,09%), acurácia balanceada 80,11%, tempo 822/795 s.
- Pareamento: 26 detectados por ambos, HMM antes em um, bruto antes em 25,
  zero detecções exclusivas HMM e 35 exclusivas bruto.
- No mesmo janelamento 120/60, bruto 65,91% detecção / 3,41% FP lateral.

Saídas completas: `results/compare_histogram_phase_rayleigh.json` e `.md`, com
taxas e tempos por paciente/frequência e todos os parâmetros. O teste anterior
`compare_histogram_rayleigh.py` era por arquivo e não atendeu à unidade
solicitada; mantê-lo apenas como histórico, sem usar suas taxas neste protocolo.
As baselines brutas reproduziram os artefatos antigos. Há agora quatro testes
numéricos em `tests/test_histogram*.py`; a informação histórica de ausência de
suíte acima não se aplica a esses módulos.

As runs LOSO históricas continuam nos JSON próprios; o novo teste não executou
LOSO. Parâmetros antigos foram selecionados neste conjunto. Comparação muda
emissão e origem de Presente, não apenas remove p-valores. Não há validação
clínica, equivalência K/dB ou novo vencedor selecionado. Próximo trabalho ainda
não definido: aguardar instrução do usuário, sem retomar fusão/search do roteiro
antigo automaticamente. Preserve as alterações preexistentes na árvore.

Após questionamento da queda, o HMM antigo também foi recalculado nas 15
configurações, usando as emissões armazenadas e parâmetros originais. As taxas
de detecção/FP combinado reproduziram os artefatos. No destaque 120/60, Beta
real antigo: 79,55% detecção / 9,09% FP lateral / 734 s entre detectados.
Um controle com histograma e Presente real: 35,23% / 0% / 1044 s. Histograma
sintético: 29,55% / 0% / 1038 s. Esses controles estão no mesmo JSON e relatório,
identificados como diagnóstico in-sample, sem substituir a emissão ativa.
As 104 observações ESP e 91,28% das 1744 observações estimuladas reais caem no
primeiro bin [0;0,05): há perda de resolução nos 20 bins uniformes. O limiar
bruto de 5% para M=120, PLV²≈0,02496, fica nesse bin. Não afirmar equivalência
ao ajuste Beta nem atribuir toda a queda à senoide. Nenhuma resolução foi
ajustada automaticamente para melhorar o resultado.

## Entrega didática — 30/09/2026

Apresentação do método mantido entregue em
`outputs/apresentacao_metodo_beta/index.html` e `apresentacao.pdf`.
Reprodução: `.venv/bin/python outputs/apresentacao_metodo_beta/gerar.py`.
Inclui 13 figuras PNG/SVG/PDF, exemplos reais, metadados, eventos, cálculos e
manifesto das fontes. Validação do EEG e dos 176 percursos/prefixos reproduz
70/88 alvos, 8/88 laterais, média/mediana 734/730 s; sem search, mudança de
método ou sobrescrita dos resultados históricos. Consulte o README da entrega.
