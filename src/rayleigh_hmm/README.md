# Experimento enxuto: Rayleigh bruto × HMM com emissão Gamma bayesiana

Pergunta: usar o teste de Rayleigh como observação de um HMM de 2 estados
(Ausente/Presente) melhora a detecção de ASSR, ou encurta o exame, em
relação ao teste sozinho, sem mais falso positivo lateral que o teste de
referência (Rayleigh α = 0,05, 3 consecutivas)?

```bash
.venv/bin/python src/rayleigh_hmm/selecao_sinal.py    # ~1,5 min, escolhe o sinal
.venv/bin/python src/rayleigh_hmm/experimento.py      # ~2 min
.venv/bin/python src/rayleigh_hmm/avaliacao.py        # ~20 s, usa resultados.json
.venv/bin/python src/rayleigh_hmm/explorar_loso_fc.py # validação oficial (LOSO, fora da amostra)
.venv/bin/python -m unittest tests.test_rayleigh_hmm
```

`explorar_loso_fc.py` é a **etapa oficial de validação**: leave-one-subject-out com
o sinal FC e o mesmo teto de FP; em cada fold tudo o que é ajustado ou escolhido
(Ausente, Presente, π, A, regra) usa só os outros 10 participantes. Ele dá a
estimativa **fora da amostra**, que é a que se reporta (braço `HMM-REAL`,
`results/rayleigh_hmm/exploratorio_loso_fc/`). Os números de `experimento.py` e
`avaliacao.py` com calibração real são in-sample com vazamento direto (abaixo).

Toda a configuração fica no topo de `experimento.py` (bloco `SINAL`, `FP_IGUAL_REFERENCIA`) e de `selecao_sinal.py`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `experimento.py` | configuração, exame sequencial, busca em grade, tabelas |
| `selecao_sinal.py` | varredura barata (sem HMM) de canal, média, PCA e rejeição de artefatos |
| `dados.py` | leitura dos `.mat`, sinal combinado (canal/média/PCA), rejeição, FFT por época, Rayleigh acumulado com máscara, senoide sintética |
| `emissao.py` | Gamma: MLE, priori, posteriori em grade e preditiva posterior |
| `hmm.py` | posteriori filtrada (causal) e forward-backward, 2 estados, vetorizado |
| `graficos.py` | figuras em `outputs/rayleigh_hmm/` |
| `avaliacao.py` | avaliação complementar sem refazer a busca: curva detecção × FP, detecção acumulada × tempo e truncamento das gravações |

Saídas em `results/rayleigh_hmm/`:

| Arquivo | Conteúdo |
|---|---|
| `tabelas.md` | tabelas geral, por intensidade e por paciente, prontas para o relatório |
| `tabela_geral.csv`, `tabela_por_intensidade.csv`, `tabela_por_paciente.csv` | as mesmas tabelas, com contagens, taxas e tempos médio/mediano/mín./máx. |
| `resultados.json` | configuração, emissões, parâmetros escolhidos, melhor por família de regra, tabelas e McNemar |
| `por_sequencia.csv` | detecção e tempo de exame de cada participante × condição × frequência |
| `por_paciente_intensidade.csv` | detectados/8 por participante × condição (mapa de calor) |
| `avaliacao.md`, `avaliacao.json` | tabelas e números da avaliação complementar (figuras `curva_deteccao_fp.png`, `curva_deteccao_tempo.png`, `truncamento.png`) |
| `selecao_sinal.md`, `selecao_sinal.json` | tabelas da varredura de sinal e opção escolhida |

A pasta `results/rayleigh_hmm_baseline_canal0/` guarda uma saída anterior (canal 0 = FC, sem rejeição, com a SNR sintética sorteada antiga), apenas para comparação histórica.

**Aviso:** as pastas exploratórias (`results/rayleigh_hmm/exploratorio_*` e `diagnostico_fase`, geradas por `explorar_*.py` e `diagnostico_fase.py`; `explorar_janela.py`, `explorar_pvalor.py` e `diagnostico_fase.py` ficam em `exploracoes/`, desatualizados) foram rodadas com o sinal antigo (média de FC, Fz, Cz, F4 com rejeição 2,0) e as SNRs antigas; não foram refeitas.

## Desenho

- **Dados:** 11 participantes × (ESP, 30, 40, 50, 60, 70 dB), épocas de 1 s (resolução de 1 Hz). Alvos `freqEstim` = 81…95 Hz e laterais
  `binsM` = 82…96 Hz.
- **Sinal (`SINAL` em `experimento.py`):** antes da FFT, os canais listados
  viram um sinal por época: `canal` (um canal), `media` (média no tempo) ou
  `pca` (1ª componente principal por arquivo, épocas concatenadas, canais
  centrados; não usa as frequências-alvo). Nomes dos eletrodos em
  `data/eletrodos.mat`: FC, F4, T6, P4, T4, Oz, C4, T5, P3, F7, F3, T3, C3,
  Fz, Pz, Cz (canal 0 = FC). Há aviso se a variância do sinal combinado for
  desprezível (ex.: média de todos em referência média comum).
  **Sinal atual:** só o canal FC (índice 0), sem rejeição de épocas
  (`{"combinacao": "canal", "canais": [0], "rejeicao_fator": None}`). Decisão do
  usuário em 04/11/2026: abandonar a média de 4 canais e a rejeição. FC é o melhor
  canal isolado em `selecao_sinal.md` (53,4%, in-sample); a média de FC, Fz, Cz, F4
  com rejeição 2,0 (57,5%) é o que `selecao_sinal.py` escolheria.
- **Rejeição de artefatos** (desligada no sinal atual; `rejeicao_fator = None`): com `rejeicao_fator`, a época é rejeitada se o
  desvio padrão do sinal combinado passar de fator × mediana dos desvios do
  arquivo. Épocas rejeitadas não entram nos fasores nem em N (N = épocas
  **válidas** acumuladas, usado na média da emissão Presente e na
  injeção da senoide). O tempo de exame continua contando épocas
  **gravadas**. A fração rejeitada por sequência vai em `por_sequencia.csv`.
- **Rayleigh acumulado:** a cada 10 épocas novas (sem sobreposição), o teste
  é refeito com **todas** as N épocas desde o início: Z = N·PLV², p = exp(−Z) (N conta só épocas válidas).
  O bruto e o HMM usam essa mesma sequência.
- **Z ou p-valor:** p = exp(−Z) é monotônica, então as decisões são
  idênticas. No HMM, o jacobiano 1/p aparece nas duas emissões e se cancela.
  A conta é feita em Z = −ln p, e as figuras mostram p.
- **Sem concatenação:** cada participante × condição × frequência é uma
  sequência independente.
- **Exame sequencial:** a cada avaliação, usando só o que já foi gravado,
  a avaliação "detecta" se p ≤ α (bruto) ou P(Presente | obs₁..t) ≥ 0,5
  (HMM, posteriori **filtrada**, causal). O exame para na primeira avaliação
  em que a regra dispara.
  - **Tempo de exame** = épocas acumuladas × 1 s. Média e mediana são
    calculadas só entre os alvos detectados.
  - Se a regra nunca dispara, não há detecção.
- **Regras de parada** (nenhuma dispara antes de 3 avaliações = 30 s):
  - fração mínima f ∈ {0,1 … 0,8} das avaliações até agora;
  - c ∈ {1 … 10} detecções consecutivas;
  - f E c e f OU c.

  São 144 regras. "3 últimas" equivale a "3 consecutivas". A referência
  ingênua é Rayleigh α = 0,05 com 3 consecutivas.
- **Ausente:** sob H0, Z ~ Exp(1) (p ~ Uniforme) em qualquer N. Gamma por
  MLE nos Z do ESP; KS contra Exp(1) só no último passo de cada sequência.
- **Presente (bayesiano):** Gamma(k, média 1 + (N−1)ρ), em que ρ é o PLV²
  verdadeiro da resposta.
  - Priori: k ~ Gamma(2, 1) e ρ ~ Gamma(2, 200).
  - **Calibração (`CALIBRACAO_PRESENTE`, decisão do usuário em 04/11/2026):**
    `"real"` (atual) ajusta o Presente em Z e N acumulados de **todas as 440
    sequências-alvo das gravações com estímulo** (30–70 dB × 8 frequências × 11
    participantes; 13 120 pares (Z, N)); todo participante ouviu todos os
    estímulos, então todas contam como "resposta presente". Vale para a posteriori
    em grade (preditiva e MAP) e para o ajuste hierárquico (marginal e Viterbi).
    Motivo: no LOSO com FC, a calibração real teve a mesma detecção que a
    sintética (6 × 3, p = 0,51) e bem menos alarmes falsos laterais (37 contra
    50/528, McNemar 1 × 14, p = 0,001). `"sintetica"` mantém o comportamento
    anterior, descrito a seguir.
  - **Vazamento direto com calibração real:** o Presente é ajustado nos próprios alvos
    que depois são avaliados, então a avaliação de `experimento.py` é **in-sample com
    vazamento direto** e não é a estimativa de desempenho. O número a reportar é o
    do LOSO (`explorar_loso_fc.py`, braço HMM-REAL).
  - (Só com `"sintetica"`) Verossimilhança: ESP + senoide de fase 0, sem sorteio: cada sequência ESP
    recebe 5 SNRs fixas no bin (−25,0; −20,5; −17,5; −14,0; −12,0 dB, uma por
    intensidade de 30 a 70 dB), 88 × 5 = 440 sequências sintéticas. Os valores
    são a S/P média medida nos arquivos de estímulo com o sinal FC, sem rejeição
    (`PROXIMOS_PASSOS.md`, C; com a média de 4 canais eram −23,5 a −11,5 dB).
    **Vazamento leve:** as 5 médias vêm dos mesmos arquivos usados na avaliação
    (números médios, não limiares); isso não é validação independente.
  - Posteriori em grade; emissão = **preditiva posterior**. A variante MAP
    também é rodada.
- **Presente marginal (`hmm_marginal`):** pdf fixa
  p(Z | N) = ∬ Gamma(Z | k, μ(N, ρ)) · Gamma(k | a_k, b_k) · Gamma(ρ | a_ρ, b_ρ) dk dρ,
  usada como emissão Presente de um HMM comum, com o filtro de sempre. Nada é
  aprendido durante o exame.
  - Hiperparâmetros por máxima verossimilhança com um par (k, ρ) por sequência
    sintética (arquivo × frequência × SNR), integral em grade log-espaçada
    fixa (`GRADE_HIER_*`), 3 inícios; `na_borda` em `resultados.json` indica massa
    fora da grade > 0,1% ou hiperparâmetro no limite do otimizador.
  - A emissão é a média em J = 10 × 24 nós de pesos iguais, nos quantis médios das
    Gammas ajustadas.
- **Busca:** α (bruto) ou π, A (8 × 9 × 9, HMM) × regra. Maximiza a
  detecção dos alvos de estímulo com FP lateral ≤ **teto**. Desempata pelo menor
  FP e depois pelo menor tempo médio.
  - **Teto de FP lateral (decisão do usuário, 04/11/2026):** igual ao FP lateral do
    `bruto_referencia` (Rayleigh fixo, α = 0,05, 3 consecutivas), hoje 45/528
    (8,52%), calculado em `main` antes das buscas (`FP_IGUAL_REFERENCIA = True`).
    Vale para o bruto otimizado e as quatro variantes HMM: podem passar de 7%,
    nunca do FP da referência. Motivo: escolher o HMM na mesma faixa de detecção
    da referência, sem mais alarmes falsos que ela (com 7% fixo, a referência
    ficava fora do limite e a comparação de detecção não era a FP igual). A
    comparação usa a contagem exata (folga de 1e-12 na fração n/528). O teto e a
    regra ficam em `resultados.json["configuracao"]` (`fp_maximo`, `fp_maximo_laterais`,
    `fp_maximo_regra`). Com `FP_IGUAL_REFERENCIA = False` volta o teto fixo
    `FP_MAXIMO_FIXO = 0,07` (alias `FP_MAXIMO`, usado pelos scripts exploratórios).
- **Validação:** `experimento.py` é in-sample (mesmos pacientes no ajuste e na
  avaliação) e, com calibração real, com vazamento direto no Presente. Não é
  estimativa clínica nem de generalização. A estimativa fora da amostra vem do
  LOSO (`explorar_loso_fc.py`).
- **Limitações:**
  - O Z acumulado de passos vizinhos compartilha épocas, o que viola a
    independência condicional das observações do HMM.
  - O tempo de exame é limitado pela duração da gravação: 480 s em 30–40 dB,
    300 s em 50 dB, 120–180 s em 60 dB e 60 s em 70 dB. Por isso, tempos
    menores em intensidades altas refletem em parte gravações mais curtas.

## Seleção do sinal

`selecao_sinal.py` faz um único teste de Rayleigh por (arquivo, frequência)
com todas as épocas válidas (α = 0,05), sem HMM. Regra: maior detecção com FP
lateral ≤ 7% (teto próprio deste script, independente do teto do experimento) e KS dos p-valores ESP contra Uniforme com p > 0,05; desempate
pelo menor FP lateral. O ranking de canais para o top-k e a escolha usam os
mesmos dados da avaliação (vazamento aceito, in-sample).

Top 5 canais isolados (detecção /440 | FP lateral | KS p): FC 53,4% | 4,0% |
0,236; Fz 50,9% | 3,2% | 0,311; Cz 49,5% | 3,2% | 0,794; F4 49,3% | 4,9% |
0,237; C4 45,5% | 2,3% | 0,469. Melhor combinação: média de FC, Fz, Cz, F4
(57,0% | 3,0% | 0,229); média ou PCA de todos os 16 canais fica abaixo do
melhor canal (45,5% e 42,5%). Rejeição de artefatos tem efeito pequeno
(+0,5 ponto com fator 2,0, dentro do ruído). **Escolhido:** média de FC, Fz,
Cz, F4 com `rejeicao_fator = 2,0` (57,5% | 3,6% | KS p = 0,443; 2,6% das
épocas rejeitadas). Tabelas completas em `results/rayleigh_hmm/selecao_sinal.md`.
Aviso: em 1 arquivo, a média dos canais teve variância < 5% da dos canais.

**Decisão posterior (04/11/2026):** o pipeline principal usa só o canal FC, sem
rejeição, em vez do escolhido acima (a escolha do canal é in-sample). `selecao_sinal.py`
não foi refeito e continua indicando a média de 4 canais.

## Resultado oficial: LOSO com calibração real (fora da amostra)

`explorar_loso_fc.py`, 11 folds, sinal FC, teto de FP por fold = FP lateral do
`bruto_referencia` no treino (`results/rayleigh_hmm/exploratorio_loso_fc/tabelas.md`).
Geral (detecção /440, FP lateral /528, FP ESP /88):

| Método | Detecção | FP lateral | FP ESP | Tempo médio / mediano (s) |
|---|---:|---:|---:|---:|
| **HMM-REAL (Viterbi, calibração real)** | 238, 54,1% | 37, 7,0% | 6, 6,8% | 124 / 80 |
| HMM-SINTÉTICO (Viterbi, calibração sintética) | 235, 53,4% | 50, 9,5% | 6, 6,8% | 130 / 80 |
| Rayleigh otimizado (LOSO) | 244, 55,5% | 47, 8,9% | 3, 3,4% | 132 / 90 |
| Rayleigh α=0,05, 3 consecutivas (referência) | 246, 55,9% | 45, 8,5% | 3, 3,4% | 133 / 95 |

HMM-REAL × HMM-SINTÉTICO: detecção 6 × 3 (p = 0,51), laterais 1 × 14 (p = 0,001).
HMM-REAL × referência: detecção 6 × 14 (p = 0,12), laterais 5 × 13 (p = 0,096).
O HMM não supera a referência em detecção; a vantagem da calibração real é o
menor FP lateral. Em todos os 11 folds do braço REAL, A(Pres→Aus) escolhida é 0,5,
o máximo da grade (ver `PROXIMOS_PASSOS.md`).

## Resultado in-sample de `experimento.py` (sinal FC, sem rejeição, teto de FP = FP da referência)

**Calibração real do Presente: avaliação in-sample com vazamento direto; não é o
desempenho esperado (ver LOSO acima).** Geral, detecção /440, FP lateral /528,
FP ESP /88 (entre parênteses, a versão anterior com calibração sintética):

| Método | Detecção | FP lateral | FP ESP | Tempo médio / mediano (s) |
|---|---:|---:|---:|---:|
| Rayleigh α=0,05, 3 consecutivas (referência) | 246 (246) | 45 (45) | 3 (3) | 133 / 95 |
| HMM preditiva | 243 (239) | 40 (45) | 5 (6) | 123 / 80 |
| HMM MAP | 243 (238) | 41 (44) | 5 (6) | 123 / 80 |
| HMM marginal | 243 (244) | 42 (44) | 6 (5) | 123 / 80 |
| HMM Viterbi | 244 (241) | 44 (45) | 7 (7) | 121 / 80 |

Ausente inalterado (k = 0,914; μ = 0,866). Calibração real: 440 sequências, 13 120 pares
(Z, N). Hiperparâmetros: k ~ Gamma(1,460; 0,3181), média 4,59; ρ ~ Gamma(0,2895;
9,483), média 0,0305; média posteriori da grade ρ = 0,0123, k = 0,834. `na_borda =
True` (massa de ρ fora da grade 0,27% > 0,1%) e um dos 3 inícios do otimizador parou
no ponto inicial, com verossimilhança pior (os outros dois coincidem no melhor), o
que eleva `dispersao_relativa_inicios`. O texto abaixo (referência de 04/11/2026, calibração
sintética) é histórico.

### Resultado anterior (calibração sintética)

Geral (detecção /440, FP lateral /528, FP ESP /88; entre parênteses, a versão
anterior com FC e teto fixo de 7%):

| Método | Detecção | FP lateral | FP ESP | Tempo médio / mediano (s) |
|---|---:|---:|---:|---:|
| Rayleigh α=0,05, 3 consecutivas (referência) | 246, 55,9% | 45, 8,5% | 3, 3,4% | 133 / 95 |
| Rayleigh otimizado (α=0,05; fração ≥ 0,7 OU 3 consecutivas) | 246 (236), 55,9% | 45 (36), 8,5% | 3 (6), 3,4% | 133 / 95 (118 / 70) |
| HMM preditiva | 239 (230), 54,3% | 45 (33), 8,5% | 6 (4), 6,8% | 126 / 80 (123 / 80) |
| HMM MAP | 238 (231), 54,1% | 44 (34), 8,3% | 6 (4), 6,8% | 124 / 80 (124 / 80) |
| HMM marginal | 244 (236), 55,5% | 44 (36), 8,3% | 5 (6), 5,7% | 123 / 80 (123 / 80) |
| HMM Viterbi | 241 (233), 54,8% | 45 (36), 8,5% | 7 (6), 8,0% | 121 / 70 (131 / 90) |

Com o teto igual ao FP da referência (45/528), o bruto otimizado cai na própria
referência (mesmo α e mesma detecção; só a regra muda, com tempo médio 132,5 s) e
os HMM recuperam 5 a 8 alvos em relação ao teto de 7%. A detecção continua in-sample
e a diferença contra a referência é pequena. Configurações escolhidas (π = (Aus, Pres)):
HMM preditiva π = (0,99; 0,01), A = [[0,995; 0,005], [0,3; 0,7]], 1 consecutiva;
HMM MAP π = (0,9; 0,1), A = [[0,98; 0,02], [0,5; 0,5]], 1 consecutiva; HMM marginal
π = (0,7; 0,3), A = [[0,95; 0,05], [0,5; 0,5]], fração ≥ 0,3 OU 2 consecutivas;
Viterbi π = (0,8; 0,2), A = [[0,95; 0,05], [0,5; 0,5]], 1 consecutiva. Tabelas por
intensidade e paciente em `results/rayleigh_hmm/tabelas.md`.

### Significância HMM × Rayleigh (`comparacao_significancia` em `resultados.json`)

McNemar exato (binomial) nos pares discordantes e Wilcoxon pareado (padrão do
scipy) no tempo de exame, só nos alvos detectados pelos dois métodos. O bruto
otimizado difere da referência só nos empates de tempo, então as detecções dos
dois são iguais e os McNemar coincidem. Sem correção para múltiplas comparações;
in-sample.

| HMM | Alvos (só bruto × só HMM, p) | Laterais | ESP | Tempo contra a referência: n pares, mediana bruto / HMM, diferenças nulas, p |
|---|---|---|---|---|
| preditiva | 10 × 3, p = 0,092 | 8 × 8, p = 1 | 0 × 3, p = 0,25 | 236, 90 / 80 s, 83, p = 0,33 |
| MAP | 12 × 4, p = 0,077 | 7 × 6, p = 1 | 0 × 3, p = 0,25 | 234, 90 / 80 s, 72, p = 0,024 |
| marginal | 11 × 9, p = 0,82 | 11 × 10, p = 1 | 0 × 2, p = 0,5 | 235, 90 / 80 s, 100, p = 0,54 |
| Viterbi | 13 × 8, p = 0,38 | 11 × 11, p = 1 | 0 × 4, p = 0,125 | 233, 90 / 80 s, 70, p = 0,13 |

Nenhuma diferença de detecção ou de FP lateral é significativa. No tempo, só o MAP
tem p < 0,05 contra a referência (p = 0,055 contra o bruto otimizado). Viterbi
contra a referência: dos 233 pares, o HMM é mais rápido em 106, mais lento em 57 e
igual em 70 (diferença média pareada −1,4 s); contra o bruto otimizado, 102 / 58 / 73,
−1,0 s, p = 0,24. Os tempos têm muitos empates (passos de 10 s), o que o Wilcoxon só aproxima.

### Emissão marginal (in-sample, dados sintéticos no ajuste)

Ausente: MLE nos 1160 Z do ESP, k = 0,914, μ = 0,866 (inalterado pelo teto de FP).
Hiperparâmetros: k ~ Gamma(1,344; 0,2672), média 5,03, dp 4,34; ρ ~ Gamma(0,4532;
10,64), média 0,0426, dp 0,0633. Sem borda; 3 inícios convergem. A grade precisou
descer a k = 0,01 e ρ = 1e−10, pois as formas baixas põem massa perto de zero.
(Estabilidade a dobrar a grade e sensibilidade a J = 60/240/600 foram medidas com a SNR
sorteada anterior e não foram refeitas.)

`hmm_marginal` (J = 240): 244/440 (55,5%) | FP lateral 44/528 (8,3%) | FP ESP 5/88
(5,7%) | 123/80 s, contra o bruto de referência 246/440 (55,9%) | 45/528 (8,5%) |
3/88 (3,4%) | 133/95 s (McNemar 11 × 9, p = 0,82). Sem ganho sobre o bruto.
`avaliacao.py` e `graficos.py` não incluem esse método.

### Detecção por Viterbi (`hmm_viterbi`, in-sample, dados sintéticos no ajuste)

Mesma emissão de `hmm_marginal` (J = 240). A cada avaliação t, Viterbi usa só os Z de
1 a t; t é detectada se o caminho mais provável de 1..t tiver algum estado Presente.
Esse indicador substitui `posteriori >= LIMIAR_POSTERIORI` nas mesmas regras, grades e
limite de FP. Empates ficam com Ausente: detectar exige Presente estritamente mais
provável. A avaliação t pode perder a detecção de t−1, pois o melhor caminho muda.

Escolhido: π(Presente) = 0,2; A(Aus→Pres) = 0,05; A(Pres→Aus) = 0,5; 1 consecutiva.
Resultado: 241/440 (54,8%) | FP lateral 45/528 (8,5%) | FP ESP 7/88 (8,0%) | 121/70 s,
contra o bruto de referência 246/440 | 45/528 | 3/88 | 133/95 s (McNemar 13 × 8,
p = 0,38; Wilcoxon do tempo p = 0,13) e contra o bruto otimizado 246/440 | 45/528 |
3/88 | 133/95 s (mesmo McNemar; tempo p = 0,24). Por intensidade (detecção | FP
lateral | tempo médio), referência x Viterbi: 30 dB 44,3% | 10,2% | 218 s x 42,0% |
8,0% | 211 s; 40 dB 71,6% | 11,4% | 183 s x 64,8% | 11,4% | 169 s; 50 dB 71,6% | 13,6% |
117 s x 71,6% | 10,2% | 117 s; 60 dB 53,4% | 9,1% | 81 s x 53,4% | 10,2% | 64 s; 70 dB
38,6% | 1,1% | 43 s x 42,0% | 3,4% | 38 s. Sem ganho de detecção; a regra vem de
busca nos mesmos dados.
`avaliacao.py` e `graficos.py` também não incluem esse método.

Parâmetros naturais para explorar: `SNR_SINTETICA_DB` e `PRIORI_PRESENTE` (o efeito da priori aparece com
poucos dados sintéticos), `EPOCAS_POR_JANELA`, `MINIMO_AVALIACOES`, as
grades de regra e o teto de FP (`FP_IGUAL_REFERENCIA`, `FP_MAXIMO_FIXO`).

## Avaliação complementar (`avaliacao.py`)

Usa os parâmetros de `resultados.json` e as mesmas emissões (mesma semente);
antes de tudo confere que as métricas das gravações completas reproduzem as de
`resultados.json` (reproduziram exatamente).

- **A. Detecção × FP lateral:** varre α (bruto) ou o limiar da posteriori
  (HMM), com o resto fixo. Detecção a FP lateral de 5% | 7% | 10% | área parcial
  (FP ≤ 10%): teste único no fim (teto) 55,5% | 58,4% | 62,9% | 0,533; Rayleigh
  otimizado 49,7% | 54,5% | 57,2% | 0,471; HMM preditiva 48,5% | 52,2% | 54,8% |
  0,469; HMM MAP 48,0% | 52,2% | 56,1% | 0,465. A linha tracejada da figura
  `curva_deteccao_fp.png` marca o teto da busca (FP da referência, 8,5%, lido de
  `resultados.json`), não mais 7%. A curva do HMM fica um pouco abaixo
  da do bruto; o teste único com todas as épocas fica ~4 a 8 pontos acima.
- **B. Detecção acumulada × tempo:** o denominador é o total de alvos; as
  gravações curtas (60 dB: 180 s, 70 dB: 60 s) param de crescer na duração máxima.
- **C. Truncamento** (T = 60, 120, 180, 300 s, só gravações com pelo menos T s):
  em T = 60 s entram todas as condições (só 1 arquivo ESP, com 30 épocas, sai) e a
  detecção sobe com a intensidade (3,4% em 30 dB a 38,6% em 70 dB no bruto otimizado), ou
  seja, o declínio em 60–70 dB da gravação completa vem em grande parte da
  duração menor. Para T ≥ 120 s, 70 dB sai (gravação de 60 s) e, para T = 300 s,
  também 60 dB. Tabelas completas em `results/rayleigh_hmm/avaliacao.md`.
