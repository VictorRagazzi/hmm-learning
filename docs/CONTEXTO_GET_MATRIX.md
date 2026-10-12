# Contexto técnico — MVP ASSR / ELT610

Versão simplificada em 07/10/2026, na branch `elt610`, por solicitação do usuário.
O escopo é calcular A/B, comparar emissão Beta por máxima verossimilhança (MLE)
e por máxima a posteriori (MAP), aplicar Viterbi separadamente em cada intensidade
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
regra = 3 consecutivas OR 60% de positivos no prefixo do arquivo
critério móvel opcional = K estados Presente nas últimas N janelas completas
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
70 dB tem janela. Esses arquivos permanecem na avaliação, com relógio local, e todos os pacientes
são preservados. Há 128 janelas de calibração Ausente e 672 Presente (50/60 dB).

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

## 5. Sequências independentes e Viterbi — dados.py / viterbi.py

A pedido do usuário, não se concatenam mais intensidades. Cada trajetória é
paciente × intensidade × frequência × grupo (alvo/lateral): 1.056 sequências,
528 alvos (88 ESP e 440 estimulados) e 528 controles. Há 66 arquivos e 11 pacientes.
As janelas permanecem dentro dos arquivos. Cada arquivo reinicia delta com pi,
os contadores e o relógio, cujo zero é o começo da gravação daquela intensidade.
Arquivos sem janela mantêm sua duração e metadados e não são descartados.

```text
delta_1(j) = log(pi_j) + log B_j(p_1)
delta_t(j) = log B_j(p_t) + max_i(delta_(t-1)(i) + log A_ij)
```

`viterbi()` retorna caminho por backtracking e estados terminais de cada prefixo.
O tempo de detecção usa somente os estados causais. Delta é pontuação de caminho,
não posterior normalizada. Zero em A/pi proíbe caminhos; empate escolhe Ausente.
B continua global, com a mesma calibração e priori. A heurística não muda.

## 6. Decisão e avaliação — hmm_inference.py

Há três critérios opcionais, combinados por OR ou AND no mesmo instante:

- `consecutivas`: estados Presente seguidos no arquivo;
- `percentual`: fração de estados Presente no prefixo do arquivo;
- `ultimas_janelas=N`, `min_deteccoes_ultimas=K`: pelo menos K estados Presente
  nas N janelas mais recentes, em janela móvel causal.

`None` desativa um critério; pelo menos um precisa ficar ativo. Os dois campos
do critério móvel precisam ser inteiros com 1 <= K <= N, ou ambos None.
Exemplo: N=5, K=2. Esse critério só passa depois de cinco janelas completas;
reavalia-se a cada nova janela e registra-se o primeiro disparo. Não significa
avaliar apenas o final da gravação, nem contar p-valores brutos como detecções.
Para usar somente essa regra em `config.py`, defina `MIN_CONSECUTIVE=None`,
`MIN_PERCENT=None`, `ULTIMAS_JANELAS=5`, `MIN_DETECCOES_ULTIMAS=2`.
O padrão mantém 3 consecutivas OR 60%, agora dentro de cada arquivo.

O JSON fornece as três métricas por paciente/intensidade em
`metricas.por_paciente_intensidade`, além de resumos globais, por paciente,
por intensidade e por frequência:

- Detecções: número/taxa de alvos estimulados que disparam naquela gravação.
- Tempo: média/mediana entre detectados, em segundos desde o início do arquivo.
  Cada resultado guarda `tempo_deteccao_s`, null se não detectado, e
  `duracao_avaliada_s`, que usa a duração da gravação nos não detectados.
- FP: número/taxa de laterais que disparam na mesma intensidade. FP dos alvos
  ESP tem número/taxa próprios, sem misturar os dois tipos de falso positivo.

ESP não entra no denominador de detecção: seu tempo/taxa de detecção e BA são
null. Alvos ESP entram somente em FP alvo ESP. Laterais de todas as intensidades,
incluindo ESP, entram no FP lateral global. BA=(detecção + 1 - FP lateral)/2.
Nenhum arquivo sem janela é descartado: os alvos estimulados continuam no
denominador como não detectados e os laterais como sem disparo. `n_sem_janelas`
explicita a falta de observações. Portanto, zero FP sem janelas não demonstra
especificidade naquele arquivo. Por paciente/intensidade há oito alvos e oito
controles pareados; em ESP, oito alvos ESP e oito controles.

São 4.784 janelas no janelamento padrão 90/45, incluindo alvos e laterais.
Observações preservam intervalos, arquivo, canal, Fs, nível, detector, estatística,
p-valor e bins locais quando aplicáveis. Os caminhos são guardados por arquivo.
Os antigos resultados/tempos das sequências concatenadas não são referências
para o protocolo atual.

Com a configuração padrão MAP 90/45 (3 consecutivas OR 60%), a execução
independente teve 139/440 detecções, 10/528 FP laterais, zero FP alvo ESP,
BA=64,85%, média/mediana 162,19/90 s entre detectados e 224 sequências sem janela.
Por intensidade (detecções / tempo médio em segundos / FP lateral):

| Nível | Detecções /88 | Tempo médio (s) | FP lateral /88 |
|---|---:|---:|---:|
| ESP | não aplicável | null | 0 |
| 30 | 15 | 201,00 | 2 |
| 40 | 38 | 222,63 | 4 |
| 50 | 48 | 140,63 | 2 |
| 60 | 38 | 113,68 | 2 |
| 70 | 0 | null | 0 |

70 dB não tem janelas completas para M=90 em nenhum paciente. Esses resultados
são os de config.py, não os do vencedor da nova busca.

## 7. Busca — search_hmm_parameters.py

A busca global continua in-sample, com todos os pacientes no ajuste e avaliação.
Cada arquivo é uma sequência independente, tanto no lote quanto na inferência
individual. Pesquisa MLE/MAP separadamente; detector, priori e corte vêm de config.
Parâmetros:

- janelas 10/20/30/60/90/120/180, passo metade ou janela inteira;
- pi Ausente .99,.95,.90,.85,.80,.75,.70,.65,.60,.55,.50;
- mesmas probabilidades nas duas autopermanências de A;
- A heurística recalculada por janelamento e A configurada, sem duplicatas;
- consecutivas 1–12; percentuais .01,.025,.05,.075,.09,.10,.20,.30,.40,.50,
  .60,.70,.80,.90,1; OR/AND e um limiar desativado (387 regras);
- regras móveis isoladas: N em (3,5,10), K em (1,2,3), apenas K <= N (9 regras).
  As grades móveis ficam em `ULTIMAS_JANELAS_VALUES` e
  `MIN_DETECCOES_ULTIMAS_VALUES`. A busca rápida inclui N=5, K=2.

A inferência aceita combinar o critério móvel com os demais por OR/AND; a grade
padrão pesquisa as regras antigas e as móveis isoladas, sem produto entre elas.
Maximiza BA sob FP lateral global <=5%, usando 440 alvos estimulados e 528
laterais. ESP alvo não vira detecção nem controle lateral. A busca desempata
por maior detecção, menor FP e primeiro candidato. Não minimiza tempo.

FFTs são carregadas uma vez, B é ajustada por janelamento e Viterbi calcula
lotes de 32 A/pi. Padding nunca entra nos critérios, inclusive na soma móvel.
Vencedores são verificados pelo avaliador individual e guardam as mesmas
métricas por paciente/intensidade. Não há LOSO ou checkpoints.

Saída completa: `results/search_hmm_parameters.json`; rápida: `_rapida.json`.
`--usar-busca` rejeita resultados salvos do protocolo concatenado, além de
rejeitar detector incompatível. `configuracao.protocolo` identifica
`intensidades_independentes`. Trocar detector exige recalibrar B e refazer busca.

A busca completa do protocolo independente terminou com 14.880.096 avaliações.
MLE e MAP escolheram janela/passo 60/30, A=[[.85,.15],[.20,.80]], pi=[.80,.20],
regra 3 consecutivas OR 70%. Ambos tiveram 203/440 detecções (46,14%), 22/528
FP laterais (4,17%), três FP alvo ESP (3,41%), BA=70,98%, tempo médio/mediano
160,34/150 s entre detectados e 16 sequências sem janela. As nove regras móveis
participaram da busca; nenhuma foi a vencedora global. Config.py mantém os
parâmetros padrão anteriores para permitir comparação explícita por `--usar-busca`.

O vencedor Rayleigh MAP foi aplicado novamente por `hmm_inference.py --usar-busca`,
reproduzindo exatamente as métricas salvas na busca. Os JSONs A/B/inferência e
figuras agora correspondem ao vencedor 60/30; os valores padrão 90/45 acima
continuam como referência explícita de config.py. Foram exportados
`results/rayleigh_melhor_resultado.md` e
`results/rayleigh_melhor_por_paciente_intensidade.csv`, com as três métricas
por paciente/intensidade. O teto de FP é global: 30 e 40 dB têm FP lateral
individual de 7,95% e 10,23%, respectivamente.

### Comparação atual com Rayleigh bruto

A pedido do usuário, `src/comparar_rayleigh.py` compara os vencedores atuais
HMM MAP/MLE com o detector bruto, no protocolo independente. Trata-se de uma
comparação atual, não recuperação dos experimentos históricos. O bruto define
positivo por p <= alpha, sem emissão Beta ou Viterbi, e aplica as mesmas regras
causais de decisão temporal. A busca bruta usa os 14 janelamentos e 396 regras
salvos pela busca HMM e `RAW_ALPHA_VALUES=(.001,.0025,.005,.01,.025,.05,.10)` em
config.py: 38.808 candidatos. Maximiza a mesma BA sob FP lateral global <=5%,
com os mesmos desempates e denominadores; ESP alvo permanece separado.

O vencedor bruto foi 180/90 épocas, alpha=.10, uma consecutiva AND 70% do prefixo:
176/440 detecções (40%), 25/528 FP laterais (4,73%), zero FP alvo ESP,
BA=67,63%, média/mediana 190,74/180 s. HMM MAP/MLE reproduziram exatamente a
busca salva: 203/440 (46,14%), 22/528 (4,17%), FP alvo ESP=3/88 (3,41%),
BA=70,98%, média/mediana 160,34/150 s. MAP e MLE não diferem nos eventos/tempos.

Os métodos usam seus melhores janelamentos próprios. O bruto tem 336 sequências
sem janela (oito ESP, dois arquivos de 60 dB e todos os 70 dB); HMM tem 16
sequências sem janela (um ESP). Não se descarta nenhum paciente/arquivo.
Assim, diferenças incluem efeito do janelamento e cobertura das gravações;
a diferença de tempo médio não é antecipação pareada nem efeito isolado do HMM.
Alpha=.10 é um parâmetro escolhido na amostra; não significa FP de 10% por
arquivo nem garantia externa de FP <=5% após seleção.

Reprodução: `.venv/bin/python src/comparar_rayleigh.py`. Esse comando verifica
que a busca HMM é completa, Rayleigh e independente, reaplica seus vencedores,
confere todas as métricas e busca o bruto com os EEG atuais. Salva
`results/comparacao_rayleigh.json`, `results/comparacao_rayleigh.md` e
`results/comparacao_rayleigh_por_paciente_intensidade.csv`, incluindo métricas
por paciente/intensidade e frequência, parâmetros e eventos. Não substitui
os artefatos individuais A/B/inferência. FP aparece também em porcentagem.

## 8. Reprodução e verificações

```bash
.venv/bin/python -m py_compile src/*.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python src/get_obs_matrix.py
.venv/bin/python src/get_tran_matrix.py
.venv/bin/python src/hmm_inference.py
.venv/bin/python src/search_hmm_parameters.py --rapida
.venv/bin/python src/search_hmm_parameters.py
.venv/bin/python src/hmm_inference.py --usar-busca
```

Saídas A/B/inferência: `results/transition_matrix.json`,
`results/observation_matrix.json`, `results/inference_analysis.json`.
Sem flag, a inferência usa config.py; com `--usar-busca`, aplica o vencedor e
recalibra B. Importar módulos não processa EEG, grava arquivos ou inicia busca.

Os testes verificam Viterbi contra enumeração, zeros e causalidade, MAP contra
otimização independente, priori, critérios móveis/combinados, lote versus
individual com padding e sequências vazias, 66 EEG reais, calibração e
normalização de A/B, independência por intensidade, relógio local, preservação
dos arquivos sem janela e os seis detectores. A auditoria histórica não
representa o protocolo independente e não é mais exigida como referência.

## 9. Limitações

Presente >=50 dB é um proxy de resposta, não confirmação fisiológica por janela.
Janelas sobrepostas e observações de um paciente têm dependência. A likelihood
Beta fatorizada ignora essa dependência. Seleção e avaliação usam os mesmos
pacientes e laterais; teto de FP na amostra não garante desempenho externo.
A calculada/otimizada não é persistência fisiológica validada. Nenhum resultado
é apresentado como validação clínica ou independente.
