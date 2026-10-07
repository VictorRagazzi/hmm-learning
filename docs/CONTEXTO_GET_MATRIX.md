# Contexto técnico — MVP ASSR / ELT610

Versão simplificada em 07/10/2026, na branch `elt610`, por solicitação do usuário.
O escopo é calcular A/B, comparar emissão Beta por máxima verossimilhança (MLE)
e por máxima a posteriori (MAP), aplicar Viterbi às intensidades concatenadas
e otimizar parâmetros. Todos os pacientes participam do ajuste e da avaliação.
Os experimentos históricos foram removidos do diretório de trabalho.
Por pedido posterior, os seis detectores foram recuperados em `detectors.py`,
sem recuperar pipelines históricos. Rayleigh continua sendo o padrão.

## 1. Configuração e dados

Todas as configurações ficam em `src/config.py`. Referência escolhida:

```text
canal = 0
detector = rayleigh
janela/passo = 90/45 épocas
método = MAP
A aplicada = [[0.80,0.20],[0.45,0.55]]
pi = [0.90,0.10]
regra = 3 consecutivas OR 60% de positivos no prefixo
Presente: intensidade >=50 dB
mu ~ Beta(2,4); kappa ~ Gamma(shape=2,rate=1)
```

Arquivos MATLAB v7.3 reais em `data/`: 11 ESP e 55 estimulados, 11 pacientes.
Cada paciente tem fases ESP, 30, 40, 50, 60 e 70 dB. `dados.py` lê:

- `x`: (16 canais, épocas, amostras); apenas o canal escolhido é carregado;
- `Fs`: 1000 ou 1750 Hz nos dados atuais;
- `freqEstim`: oito alvos, atualmente 81,83,85,87,89,91,93,95 Hz;
- `binsM`: oito controles laterais, atualmente 82,84,86,88,90,92,94,96 Hz.

Os valores vêm dos arquivos. Valida-se a geometria, frequências, EEG finito,
correspondência entre fases e amostras/Fs=1 segundo por época. A duração de um
segundo é validada, não inferida de uma Fs fixa. Números de épocas variam.

## 2. EEG → observação do detector

Para cada época, calcula-se FFT no canal escolhido e extrai-se o coeficiente
complexo X no bin mais próximo da frequência analisada. Não há filtragem,
remoção automática de artefatos ou seleção de canais adicional.

Para M épocas de uma janela:

```text
u_m = X_m / (|X_m| + 1e-12)
PLV² = |soma(u_m)/M|²
p = exp(-M * PLV²)
```

Essa é a aproximação de cauda Rayleigh sob ausência de concentração de fase,
com épocas independentes e fases uniformes sob H0. Mantém-se M>=10.
O p-valor não é probabilidade do estado Ausente. B é ajustada aos p-valores
reais; a Beta ajustada para B é distinta da distribuição nula de cada detector.

`DETECTOR_NAME` seleciona um dos seis detectores para toda a execução:

| Chave | Estatística e p-valor sob H0 |
|---|---|
| rayleigh | PLV²; cauda aproximada exp(-M·PLV²) |
| msc | \|soma X\|²/(M·soma \|X\|²); cauda Beta(1,M-1) |
| csm | T² circular de Hotelling, expresso como F(2,2(M-1)); estatística transformada equivalente à MSC |
| mmsc | Média das MSCs de duas metades; cauda por convolução numérica das duas Betas |
| hotelling | T² com covariância 2×2 de Re/Im; forma escalada F(2,M-2) |
| spectral_f | M·\|média X_alvo\|² / média \|X_ruído\|²; F(2,4M) |

MSC/CSM pressupõem coeficientes gaussianos circulares e épocas independentes;
são formulações equivalentes, não evidências independentes. MMSC é o split-half
experimental do projeto; uma época final ímpar não entra nas metades e a cauda
usa o tamanho efetivo. Hotelling pressupõe normalidade multivariada e permite
covariância real/imaginária não circular. Mantêm-se as fórmulas históricas.

F espectral requer dois bins locais de ruído, distintos do alvo, de DC e de
todas as frequências de estímulo. Seleção determinística por distância ao alvo,
com desempate pelo índice menor. `dados.py` guarda os coeficientes necessários e
as frequências dos bins, preservadas nas observações. A distribuição F pressupõe
bins gaussianos circulares, independentes e com mesma variância; essas hipóteses
não são garantidas nos EEG reais. Hotelling e F espectral são reportados na
escala F/(1+F). A emissão usa somente seu p-valor, não essa transformação.

Cada arquivo de N épocas gera `max(0, floor((N-M)/passo)+1)` janelas.
Janelas ficam dentro dos arquivos. Não há senoide sintética, labels ou
histogramas como emissão. `binsM` define controles, não denominador Rayleigh.
Não há fusão de detectores: seleciona-se um por execução. B MLE/MAP recebe os
p-valores de qualquer um; deve ser reajustada quando o detector mudar. A
heurística depende somente de duração/janelamento e continua igual.

## 3. A calculada por duração — get_tran_matrix.py

Cada arquivo com n janelas contribui com n-1 autopassagens e uma saída
implícita. Arquivos sem janela contribuem com zero. ESP corresponde a Ausente;
todas as intensidades correspondem a Presente para esta heurística:

```text
p_self = soma(max(0,n-1)) / soma(n)
A = [[p_self_A, 1-p_self_A], [1-p_self_P, p_self_P]]
```

Com janela/passo 90/45: ESP tem 16 janelas, oito autopassagens e oito arquivos
contribuintes; estímulos têm 283 janelas, 239 autopassagens e 44 contribuintes.

```text
A calculada = [[0.500000,0.500000], [0.155477,0.844523]]
```

A depende da duração das gravações. Não representa transições ocultas
observadas nem Baum–Welch. O JSON `transition_matrix.json` guarda essa A.
A aplicada à inferência padrão é a A escolhida pela busca histórica e declarada
em config.py. A busca inclui a heurística recalculada para seu próprio
janelamento e também pesquisa outras probabilidades. Não confundir as duas.

## 4. B global contínua — get_obs_matrix.py

Para cada estado, B é uma função de densidade:

```text
B_s(p) = BetaPDF(p; a_s,b_s)
       = p^(a_s-1) (1-p)^(b_s-1) / BetaFunction(a_s,b_s)
```

Calibração global, compartilhada entre frequências e pacientes:

- Ausente: p-valores dos alvos ESP, ajustados por MLE nos dois métodos;
- Presente: alvos reais >=50 dB, ajustados por MLE ou MAP;
- laterais não calibram B; medem FP;
- dados de 30/40 dB são avaliados, mas não ajustam Presente.

Pacientes com mais janelas contribuem mais. Todos os pacientes são elegíveis;
nenhum é retido para teste. Para M=90, três ESP não têm janela, e nenhum arquivo
70 dB tem janela. Essas fases permanecem no relógio e todos os pacientes na
avaliação. Há 128 janelas de calibração Ausente e 672 Presente (50/60 dB).

### Máxima verossimilhança e priori

MLE usa `scipy.stats.beta.fit`, suporte fixo (0,1). MAP usa:

```text
a=mu*kappa; b=(1-mu)*kappa
mu ~ Beta(u,v); kappa ~ Gamma(shape,rate)
objetivo = soma(log BetaPDF(p_i;a,b)) + log priori(mu) + log priori(kappa)
```

MAP maximiza nas coordenadas físicas mu,kappa. Logit(mu) e log(kappa) mantêm
parâmetros válidos na otimização; não se adiciona Jacobiano ao objetivo.
SciPy Gamma recebe scale=1/rate. Três inicializações reduzem sensibilidade à
inicialização; falha de convergência/borda numérica gera erro.
A priori influencia somente Presente. Não há amostragem MCMC ou mistura posterior.

Clipping p para [1e-9,1-1e-9] evita log(0); contagens alteradas são registradas
(zero Ausente e quatro Presente nesta configuração). Isso altera a likelihood
dos extremos. A normalização refere-se à Beta original, não a uma nova PDF
obtida por clipping da variável.

| Emissão | a | b |
|---|---:|---:|
| Ausente MLE, compartilhada | 1.1698079195 | 1.0134264024 |
| Presente MLE | 0.3036932088 | 0.9569621148 |
| Presente MAP | 0.30358755 | 0.95610907 |

### PDF, CDF e tabela B para apresentação

O Viterbi recebe `log B_s(p_t)`. Uma densidade pode exceder 1, mas sua integral
no suporte é 1. Não é posterior de estado nem probabilidade de um ponto.
`BetaCDF(p;a,b)` é a probabilidade acumulada.

O JSON inclui uma tabela 2×5, calculada por diferenças de CDF nos intervalos
[0,.01], [.01,.05], [.05,.10], [.10,.50], [.50,1]. Ela serve para explicar B;
não alimenta a inferência e não usa contagens de histograma. Linhas somam 1.
MAP, ordem dos estados Ausente/Presente:

```text
[[0.004642,0.025856,0.038092,0.380476,0.550934],
 [0.242544,0.152978,0.092896,0.311756,0.199826]]
```

Gráficos: `outputs/distribuicoes_B.png` e `.pdf` (PDF/CDF por estado),
`outputs/prioris.png` (prioris de mu/kappa para Presente). Não representam
posteriores marginais de mu/kappa.

## 5. Concatenação e Viterbi — dados.py / viterbi.py

Cada trajetória é paciente × frequência × grupo (alvo/lateral), na ordem
ESP→30→40→50→60→70. Há 88 alvos e 88 controles, 176 trajetórias. Concatena-se
somente a sequência de observações; janelas não cruzam arquivos. Pacientes e
frequências não compartilham estado temporal. O relógio inclui fases sem janela.

```text
delta_1(j) = log(pi_j) + log B_j(p_1)
delta_t(j) = log B_j(p_t) + max_i(delta_(t-1)(i) + log A_ij)
```

`viterbi()` retorna o caminho completo por backtracking e os melhores estados
terminais de cada prefixo. A primeira detecção usa apenas estados causais;
backtracking completo incorpora futuro e não é usado para atribuir o tempo.
Delta é pontuação de caminho, não posterior normalizada. Probabilidade zero
em A/pi continua proibindo caminhos; empate escolhe Ausente.

## 6. Decisão e avaliação — hmm_inference.py

Consecutivas contam estados Presente seguidos. Percentual é a fração de estados
Presente nas janelas já observadas desde ESP. `None` desativa um critério;
pelo menos um deve ficar ativo. OR aceita qualquer critério; AND exige ambos
no mesmo instante. Delta e contadores não reiniciam entre intensidades.

- Detecção: alvo que dispara após começo de 30 dB /88 alvos.
- FP lateral: lateral com qualquer disparo, incluindo ESP /88 laterais.
- FP alvo ESP: reportado separado /88 alvos.
- BA = (taxa de detecção + 1 - taxa de FP lateral)/2.
- Tempo médio/mediano de detecção: somente detectados, desde começo de 30 dB.
  Não detectados têm tempo null e duração censurada separada.

Há métricas globais, por paciente e por frequência, pareando alvos e controles
conforme os metadados. Resultado escolhido MLE e MAP: 66/88 detecções (75%),
4/88 FP lateral (4,55%), zero FP alvo ESP, BA=85,23%, média/mediana
784,6/737,5 s desde 30 dB entre detectados. Duração média incluindo não
detectados: 962,6 s. Nenhum ganho de decisões/tempos MAP sobre MLE nessa execução.

São 4.784 janelas observadas no protocolo completo, incluindo alvos e laterais.
A inferência preserva observações, intervalos, arquivo, canal, Fs, nível,
frequência, controle, detector, estatística, bins de ruído quando aplicáveis,
p-valor e ambos os caminhos.

FP lateral HMM por alvo pareado: 81/83/89/91 Hz =0%; 85/87/93/95 Hz =9,09%.
O diagnóstico bruto Rayleigh nos alvos ESP, alpha=.05, é distinto:
global 4,69%; 81:0%; 83:12,5%; 85:18,75%; 87:0%; 89:6,25%; 91/93/95:0%.
Cada frequência tem somente 16 janelas ESP nesta configuração.

## 7. Busca — search_hmm_parameters.py

Só existe busca global in-sample; LOSO e baselines externas foram removidos.
Pesquisa separadamente MLE/MAP, com as mesmas prioris/corte fixos.
O detector vem de config.py e permanece fixo durante a busca. Trocar o detector
exige uma nova busca; --usar-busca verifica o detector do resultado salvo.
Parâmetros pesquisados:

- janelas 10/20/30/60/90/120/180, passo metade ou janela inteira;
- pi Ausente .99,.95,.90,.85,.80,.75,.70,.65,.60,.55,.50;
- as mesmas probabilidades nas duas autopermanências de A;
- A heurística do janelamento e A configurada, sem duplicatas;
- consecutivas 1–12; percentuais .01,.025,.05,.075,.09,.10,.20,.30,.40,.50,
  .60,.70,.80,.90,1; OR/AND e um limiar desativado (387 regras).

Maximiza BA sob FP lateral <=5% na própria amostra. Desempata por detecção
maior, FP menor e primeiro candidato na ordem da grade. Não minimiza tempo.
A heurística agora acompanha cada janelamento; a versão histórica incluía
uma A heurística fixa de 10/5 em toda a grade. A recorrência e as regras são
as mesmas da versão escolhida.

FFTs são carregadas uma vez; para cada janela, B é ajustada uma vez e a
recorrência é calculada em lotes de 32 A/pi. Regras reutilizam estados. Padding
não entra nas contagens. Vencedores são conferidos pelo avaliador individual.
Não há checkpoints, LOSO, exportações HTML ou código de compatibilidade antigo.
A saída completa é `results/search_hmm_parameters.json`; `--rapida` usa grade
reduzida e salva `_rapida.json`, sem substituir o vencedor da busca completa.

A busca completa simplificada terminou: 14.541.912 avaliações, janela/passo
90/45, A=[[.8,.2],[.45,.55]], três consecutivas OR 60% para MLE e MAP.
O desempate determinístico escolheu pi=[.99,.01], enquanto a configuração
de referência em config.py mantém pi=[.9,.1]. As duas produzem os mesmos
eventos e tempos nesta base: 66 alvos, quatro FP laterais, zero FP alvo ESP.
Não houve ganho de MAP. O vencedor foi reaplicado pelo comando --usar-busca.

## 8. Reprodução e verificações

```bash
.venv/bin/python -m py_compile src/*.py
.venv/bin/python src/get_obs_matrix.py
.venv/bin/python src/get_tran_matrix.py
.venv/bin/python src/hmm_inference.py
.venv/bin/python src/search_hmm_parameters.py --rapida
.venv/bin/python src/search_hmm_parameters.py
.venv/bin/python src/hmm_inference.py --usar-busca
.venv/bin/python -m unittest discover -s tests
```

Saídas A/B/inferência em `results/transition_matrix.json`,
`results/observation_matrix.json`, `results/inference_analysis.json`.
O comando de inferência sem flag usa config.py; com `--usar-busca`, usa a
configuração vencedora salva para o método escolhido e recalibra B global.
Importar módulos não processa EEG, não grava arquivos e não inicia busca.

Dez testes matemáticos/integrados passaram: Viterbi contra enumeração,
causalidade e zeros, MAP contra otimização independente, influência da priori,
regras temporais, lote versus sequência individual com padding, normalização,
metadados reais, calibração e resultado escolhido. Todos os eventos e tempos
das 176 trajetórias coincidiram com o artefato anterior à simplificação.
Os testes adicionais exercitam os seis detectores nos EEG reais, ajuste MLE/MAP,
Viterbi e seleção em lote, exclusão de alvos dos bins de ruído, equivalência
MSC/CSM e taxas nulas simuladas. A rejeição de vencedor de outro detector
também foi conferida. Os resultados numéricos de referência acima continuam
sendo do Rayleigh; os demais detectores precisam de calibração/busca próprias.

### Auditoria da referência anterior à simplificação

Recalcularam-se os três métodos com os 66 EEG reais; seus hashes SHA-256
coincidem com o manifesto da execução histórica. Compararam-se 176 trajetórias
por método, incluindo detecção, FP ESP, disparo lateral, intensidade, tempo
desde 30 dB e duração com censura. Todos esses campos coincidiram exatamente.
MLE/MAP também tiveram eventos idênticos entre si. O resultado está em
`results/verificacao_referencia.json`.

| Método | Detectados /88 | FP lateral /88 | BA | Média / mediana desde 30 dB, detectados (s) |
|---|---:|---:|---:|---:|
| Bruto Rayleigh | 58 | 3 | 81,25% | 824,5 / 810 |
| HMM MLE | 66 | 4 | 85,23% | 784,6 / 737,5 |
| HMM MAP | 66 | 4 | 85,23% | 784,6 / 737,5 |

Bruto histórico: janela/passo 120/60, alpha=.005, consecutivas desativadas,
fração .025 no prefixo. Foi removido da busca na simplificação, mas sua
reprodução agora fica no teste `test_referencia_detector_bruto`. Não precisa
de emissão Beta ou Viterbi. A auditoria dos HMMs usou 90/45, A escolhida,
regra 3 OR .60 e pi=[.9,.1], e reaplicou também os vencedores atuais com
pi=[.99,.01]: os eventos e tempos permaneceram iguais nos dois casos.

O teste de referência dos HMMs verifica também BA e mediana. Os tempos usam
somente detectados; os conjuntos bruto/HMM diferem. A diferença de médias
não é uma antecipação pareada de cada detecção. Zero FP alvo ESP nos três.
Os relatórios HTML e artefatos LOSO antigos saíram da árvore do projeto na
simplificação. O comando vigente de busca é `python src/search_hmm_parameters.py`,
sem as flags históricas `--continuous-map --validation in-sample`.

## 9. Limitações

Presente >=50 dB é um proxy de resposta, não confirmação fisiológica por janela.
Janelas sobrepostas e observações de um paciente têm dependência. A likelihood
Beta fatorizada ignora essa dependência. Seleção e avaliação usam os mesmos
pacientes e laterais; teto de FP na amostra não garante desempenho externo.
A calculada/otimizada não é persistência fisiológica validada. Nenhum resultado
é apresentado como validação clínica ou independente.
