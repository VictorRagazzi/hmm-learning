# Prompt para implementar a apresentação do método mantido

Você trabalha no projeto `/home/torugo/projetos/hmm-learning`. Implemente uma
apresentação didática detalhada, em português, com plots e exemplos numéricos
reais que expliquem precisamente o HMM de detecção de ASSR que decidimos manter.
Quero entender cada etapa, do EEG à detecção, sem saltos conceituais.

Leia integralmente `AGENTS.md` e `docs/CONTEXTO_GET_MATRIX.md`, especialmente a
Seção 20. Leia também `docs/agents/status.md`, `architecture.md` e
`phase_transition_hmm.md`; inspecione o código antes de produzir os gráficos.
Preserve as alterações preexistentes na árvore.

## Decisão e limites da tarefa

O método mantido é a emissão contínua **Beta dos p-valores Rayleigh**:
Ausente ajustado aos alvos ESP, Presente aos alvos das gravações estimuladas
reais. Não é B de cinco labels, histograma da estatística ou Presente sintético.

A unidade é paciente × frequência, com ESP → 30 → 40 → 50 → 60 → 70 dB.
Janelas são calculadas dentro de cada arquivo e não cruzam fronteiras; apenas
observações são concatenadas. Delta e consecutivas continuam entre fases.
Pacientes, alvos e laterais têm trajetórias independentes. A decisão é causal,
usando o melhor estado terminal do Viterbi por prefixo, sem backtracking futuro.

Não execute search, otimize parâmetros nem altere o método para melhorar os
plots. O main de `src/phase_transition_hmm.py` executa busca: não o rode.
Reutilize funções com modelos/parâmetros já salvos em
`results/phase_transition_hmm.json`. O arquivo
`results/compare_histogram_phase_rayleigh.json` fornece fases, observações e a
reprodução antiga em `hmm_antigo_recalculado` e `detalhes_hmm_antigo`.

Use como exemplo principal Rayleigh 120/60:
`A=[[0.99,0.01],[0.15,0.85]]`, `pi=[0.95,0.05]`, uma consecutiva.
Leia os parâmetros Beta diretamente do artefato, sem inventá-los. Explique que
A foi escolhida na run histórica e difere da A heurística de duração.

## Conteúdo visual obrigatório

1. **Dados e protocolo.** Mostre arquivos, paciente, canal, Fs, frequência-alvo
   e lateral, dimensões de x e duração das épocas. Desenhe o protocolo completo
   e o janelamento 120/60 com sobreposição e limites de cada intensidade.
   Explique quais arquivos ESP não têm janela completa e por que seus pacientes
   continuam na avaliação. Diferencie estado oculto e condição experimental.

2. **EEG e FFT.** Para exemplos reais de ESP e de estímulo, plote trechos de EEG
   com unidades confirmadas, espectros e o bin analisado. Mostre os coeficientes
   complexos de épocas de uma janela no plano real/imaginário e suas fases em
   gráfico circular. Documente normalização da FFT, unidades e bin efetivamente
   utilizado; não fixe Fs nem atribua unidades desconhecidas.

3. **Teste Rayleigh passo a passo.** Mostre os vetores
   `u_m=X_m/(abs(X_m)+EPS)`, sua soma/média e
   `R²=abs(sum(u_m)/M)²`. Use valores reais de uma janela e compare com a função
   original. Distinga PLV, PLV² e Z=M*PLV². Explique o efeito de fases dispersas
   versus concentradas e que Rayleigh usa fase, não magnitude como evidência.

4. **Conversão em p-valor.** Plote p=exp(-M*R²) para M=120, marque a janela real
   escolhida e mostre a conta numérica. Identifique a aproximação nula e seus
   pressupostos. Explique por que p não é P(Ausente|EEG), não é a emissão e não
   é probabilidade posterior. Mostre o clipping [1e-9,1-1e-9] do ajuste/avaliação.

5. **Construção das emissões.** Mostre distribuições empíricas de R² e de
   p-valores, separadas por ESP e estímulos reais, e as curvas Beta ajustadas
   sobre os p-valores. Histogramas aqui são somente visualização. Exiba a, b,
   número de observações, origem dos dados e ajuste por máxima verossimilhança,
   loc=0, scale=1. Explique que frequências compartilham o modelo global e
   laterais não ajustam a emissão. Não confunda essa Beta ajustada aos p-valores
   com uma distribuição nula da estatística MSC.

6. **Um exemplo completo de densidade.** Escolha uma janela real, identifique
   paciente/frequência/intensidade/épocas e obtenha seu p. Marque esse p nas duas
   curvas; exiba f_A(p), f_P(p), log f_A(p), log f_P(p) e f_P(p)/f_A(p), todos
   calculados de verdade. Explique PDF, CDF, probabilidade de intervalo e por
   que densidade pode exceder 1. Mostre uma integral em um pequeno intervalo
   como ilustração, deixando claro que o código usa logpdf no ponto. Explique
   que valores inéditos são avaliados pela fórmula contínua, sem lookup.

7. **Viterbi numericamente.** Com a mesma trajetória, apresente a inicialização
   `delta_1=log(pi)+log f(p_1)` e uma atualização real:
   `delta_t(j)=log f_j(p_t)+max_i(delta_(t-1)(i)+log A_ij)`.
   Mostre candidatos das duas origens para cada destino, transições, emissão,
   vencedor e estado terminal. Plote delta_A, delta_P ou sua diferença ao longo
   do tempo. Não rotule delta como posterior. Demonstre que uma futura
   observação não muda o estado causal já atribuído e que delta não reinicia
   quando muda a intensidade. Esclareça por que as operações são em log.

8. **Trajetória completa e comparação.** Para uma trajetória detectada, outra
   não detectada e uma lateral com FP, monte painéis sincronizados: intensidade,
   R², p em escala adequada, densidades/log-razão, estado causal HMM, disparos
   Rayleigh bruto e primeiro instante detectado. Marque fronteiras e relógios
   desde ESP e desde 30 dB. Use parâmetros brutos salvos; diferencie comparação
   no mesmo janelamento 120/60 e vencedores anteriores HMM120/60 versus
   Rayleigh90/45. Não confunda bruto janelado com acumulado ou gravação completa.

9. **Resultados por paciente.** Plote e tabule detecção=alvos detectados/8 e
   FP=laterais com qualquer disparo no protocolo/8, incluindo ESP. Mostre
   contagens, taxas, primeira intensidade e tempos. Separe FP do alvo em ESP,
   FP lateral e FP combinado histórico sobre 176 oportunidades. Exiba média e
   mediana desde 30 somente entre detectados; não detectados têm tempo null.
   Duração censurada é outra métrica. Compare velocidade nos pares detectados
   por ambos e separe detecções exclusivas. Não chame detecção de acurácia.

10. **Decisão de manter Beta.** Inclua um painel histórico enxuto mostrando,
    em 120/60, Beta real 79,55%, histograma real 35,23% e histograma sintético
    29,55% de detecção. Mostre por que os 20 bins uniformes colapsavam 100% dos
    ESP e 91,28% dos estímulos reais em [0,0.05). Não afirme que todo histograma
    é inferior nem que converter em p-valor cria informação. Explique que
    calibração real e sintética também diferem.

## Entrega e verificação

Entregue um relatório/apresentação HTML navegável por seções, com textos,
fórmulas legíveis, plots e tabelas, e uma versão PDF. Use ferramentas padrão de
plotagem para exportar figuras independentes em PNG e SVG/PDF, além de um
script reproduzível. Organize em `outputs/apresentacao_metodo_beta/`, com README
de execução, fontes, parâmetros, seleção dos exemplos e arquivos gerados.
Pode usar plots interativos quando ajudam, mantendo equivalentes exportáveis.
Não gere vídeo nesta tarefa.

Todas as etapas devem informar o que entra, qual cálculo é feito, o que sai,
como interpretar e como isso alimenta a próxima etapa. Use cores consistentes
para Ausente/Presente e alvos/laterais, eixos com unidades, legendas e escalas
que mostrem o comportamento perto de p=0. Justifique a seleção dos exemplos,
sem apresentá-los como desempenho médio.

Valide os cálculos contra as funções originais e os registros salvos. A
referência antiga 120/60 é 70/88 alvos, 8/88 laterais, tempo médio/mediano
734/730 s entre detectados, acurácia balanceada lateral 85,23%. O FP combinado
histórico é 4,55%, não 9,09%; explique os denominadores. Se encontrar
divergência, investigue antes de publicar o gráfico. Inspecione visualmente
as figuras e o PDF para evitar cortes, sobreposições e rótulos ilegíveis.

Explique que usar condição estimulada como Presente é uma hipótese/proxy;
nem toda janela estimulada tem resposta comprovada. A seleção histórica é
in-sample, com reutilização dos participantes no ajuste e na avaliação;
não apresente como validação clínica ou LOSO. Preserve os artefatos originais.
Atualize a documentação apenas com a apresentação entregue e sua reprodução.
