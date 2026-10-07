# Emissão Beta bayesiana experimental para ASSR

`src/bayesian_beta_hmm.py` estima parâmetros e compara emissões; não substitui
o método MLE mantido na Seção 20. `src/plot_bayesian_beta_hmm.py` gera figuras
para a apresentação. Ambos são importáveis sem executar o experimento.

## Estado da entrega e conclusão

**Implementação, execução, verificações e apresentação concluídas.** Estão
prontos o JSON de resultados, as amostras posteriores, os diagnósticos,
relatório, CSVs e **46 figuras em PNG/SVG/PDF**, com PDF consolidado de
46 páginas e galeria comentada em `outputs/bayesian_beta_hmm/`.

Não houve melhora de detecção, falso positivo ou tempo na comparação fixa:
70→68 detecções, FPs iguais, mediana igual e médias maiores nas três prioris.
O ganho metodológico é quantificar incerteza de μ,κ,a,b. Isso não constitui
ganho demonstrado de desempenho. A MLE permanece como método ativo;
a alternativa foi preservada como experimento, sem escolher uma priori pelo
desempenho e sem adoção automática. Não há pendências desta entrega.

LOSO próprio, dependência por participante/janela e integração conjunta da
trajetória são possíveis trabalhos futuros, não executados nem autorizados
automaticamente. A comparação atual é exploratória in-sample.

## Modelo e distinção entre as duas Betas

Por estado s, a observação p segue Beta(a,b), onde a=μκ e b=(1−μ)κ.
A priori de μ é outra distribuição Beta(u,v); κ tem priori Gamma(shape,rate).
μ representa a média dos p-valores e κ controla concentração: a variância
condicional é μ(1−μ)/(κ+1). μ e κ são independentes a priori por estado,
mas a posterior estima sua dependência conjunta. Escolher Beta como família
dos dados não era inferência bayesiana no método antigo.

PyMC recebe `Gamma(alpha=shape, beta=rate)`. NumPy e SciPy recebem
`scale=1/rate` na simulação e nos gráficos. Confundir rate e scale mudaria
a priori. As distribuições têm suporte μ∈(0,1), κ>0.

## Prioris predefinidas e justificativa

As constantes ficam no topo do módulo principal. Cada tupla é
(u,v,shape,rate); não houve escolha de priori por detecção ou FP observado.

| Conjunto | Ausente | Presente |
|---|---|---|
| principal | (8,8,2,1) | (2,4,2,1) |
| ampla | (2,2,2,0.5) | (1,2,2,0.5) |
| simétrica | (2,2,2,1) | (2,2,2,1) |

A principal centra μ Ausente em 0,5: sob uma calibração nula ideal,
p-valores têm média 0,5. Beta(8,8) permite desvio da uniformidade sem fixar
a=b=1. Presente centra μ em 1/3, permitindo tanto p pequenos quanto valores
moderados, pois estímulo real é somente proxy e não confirmação de resposta
em toda janela. Gamma(2,1) tem média 2 e desvio padrão √2: inclui curvas
dispersas, formas em U e curvas concentradas sem fixar uma forma. Esta é
informação explícita sobre média e forma, não uma priori não informativa.

A ampla reduz a concentração da priori de μ e aumenta a média de κ para 4;
essa alteração permite investigar sensibilidade tanto da média quanto da
forma. A simétrica remove a preferência prévia por p menores em Presente.
É um controle plausível quando não se quer distinguir estados antes dos
dados. Não usa priori uniforme de μ; mesmo Beta(1,1) não significaria
ausência de informação sobre as PDFs condicionais.

São sorteados 10.000 pares μ,κ por estado/conjunto, com seed fixa, e um p por
par. Salvam-se parâmetros, p, quantis preditivos, massa p<0,05 e proporção de
curvas em U (a<1 e b<1). Os plots mostram as prioris de μ e κ, curvas Beta
condicionais sorteadas e a mistura preditiva. Isso verifica o significado
das prioris antes de qualquer uso das métricas do HMM. A mistura preditiva
Ausente pode ter massa considerável nas bordas mesmo com μ centrada em 0,5;
ela não é uma imposição de p uniforme. Os três conjuntos são reportados,
sem escolher um vencedor pelo conjunto avaliado.

## Posterior e emissão

PyMC 5.28.5 foi escolhido por implementar NUTS com transformações para
parâmetros restritos, amostras conjuntas e diagnósticos de divergência.
ArviZ 0.23.4 fornece R-hat rank-normalized, ESS bulk/tail, MCSE e BFMI.
Quatro cadeias, 2000 passos de aquecimento e 2000 amostras retidas por cadeia,
target_accept=0,95, execução sequencial e seeds derivadas de 20260930.
Versões efetivas estão no JSON; dependências em `requirements-bayesian.txt`.
O cálculo usa a likelihood Beta exata comprimida por n, soma(log p) e
soma(log(1−p)), via `pm.Potential`; isso evita repetir operações vetoriais
sem alterar a likelihood fatorizada. Não é aproximação MAP/MLE.

NetCDF preserva todas as cadeias e sample_stats. Resumos incluem média, desvio,
HDI 95%, intervalo equal-tail 95%, ESS, R-hat e MCSE para μ,κ,a,b.
O programa exige zero divergências, R-hat<1,01, ESS bulk/tail>400 e BFMI>0,3
em cada cadeia antes de avaliar uma alternativa. São critérios diagnósticos,
não prova de correção do modelo. Trace, rank e gráficos da posterior conjunta
complementam os números. Duas simulações iid de 1500 p-valores verificam
recuperação de μ,κ,a,b conhecidos e cobertura do intervalo 95%; dois exemplos
não constituem estudo de cobertura repetida.

A emissão usa um subconjunto fixo de 2000 amostras conjuntas, escolhido com
seed fixa e índices salvos. Calcula
`logsumexp(beta.logpdf(p,a[r],b[r])) − log(S)`. É log da média das densidades,
nunca média dos logs nem PDF nos parâmetros médios. O mesmo subconjunto é
usado em todas as janelas, participantes e frequências na execução.

p é limitado a [1e−9,1−1e−9], como no MLE. Essa estabilização torna valores
extremos indistinguíveis e altera a likelihood dos p fora desse intervalo.
O JSON registra quantos p de calibração foram alterados e a massa preditiva
nas caudas excluídas. A mistura Beta original integra 1 e pode exceder 1;
o wrapper que aplica clipping ao argumento não é uma nova PDF normalizada
no suporte fechado. A normalização testada é a PDF original, por CDF e por
quadratura independente nas duas metades com transformação x=exp(−t).
As alturas das PDFs não precisam somar 1 em uma grade.

## Comparação fixa e proveniência

Lê modelos/parâmetros de `results/phase_transition_hmm.json` e somente
observações/metadados da configuração `continuo_window_120_step_60` de
`results/compare_histogram_phase_rayleigh.json`. Nenhum histograma, label ou
dado sintético desse artefato entra no cálculo. Os helpers `avaliar/resumir`
são reutilizados apenas para eventos e métricas do protocolo.

Configuração: Rayleigh, canal 0, janela/passo 120/60,
A=[[0,99;0,01],[0,15;0,85]], pi=[0,95;0,05], min_consecutive=1.
Os valores são conferidos nos artefatos. A/pi permanecem fixas e não são
estimadas por esta tarefa. O programa não chama search nem executa o main
de `phase_transition_hmm.py`.

Antes de amostrar, reaplica a baseline MLE salva, compara todos os eventos
com os eventos salvos e confirma 70/88 alvos, zero FP alvo ESP, 8/88 laterais,
734/730 s. Também confere o MLE recalculado como diagnóstico, sem substituir
os parâmetros utilizados. P-valores salvos e intervalos são conferidos contra
EEG real dos 66 arquivos; FFT e Rayleigh usam os utilitários existentes.
Fs, eixos, canal, frequência e duração de um segundo por época são validados.
Um manifesto contém SHA-256 de cada arquivo e dos artefatos de origem.
Protocolos preservam participante, fase, arquivo, canal, Fs, frequência,
intervalos locais, offsets globais, R² e p. Cada alvo tem uma lateral pareada
pelo mesmo índice, registrada em trajetória separada. A calibração global
usa 104 janelas-alvo ESP e 1744 janelas-alvo estimuladas; laterais não ajustam.
Arquivos curtos sem janela continuam no denominador de 88 trajetórias.

Cada trajetória participante × frequência segue ESP→30→40→50→60→70.
Janelas não cruzam arquivos; delta e consecutividade continuam entre fases.
Usa argmax do delta terminal a cada janela, sem backtracking. Todos os
prefixos são verificados, e o cálculo MLE é confrontado com a função antiga.

Detecção alvo é qualquer disparo após começo de 30. FP alvo ESP é separado;
FP lateral é qualquer disparo inclusive em ESP. Denominadores são 88 alvos
e 88 laterais globalmente, oito de cada por participante. Tempos null de
não detectados são preservados; duração censurada histórica fica em campo
separado. Comparações de tempo pareadas usam somente alvos detectados pelos
dois métodos, com exclusividades separadas. FP combinado histórico é somente
diagnóstico identificado, não substitui FP lateral.

## Limitações

Esta comparação é exploratória in-sample: os mesmos participantes ajustam
emissões e são avaliados; parâmetros históricos já foram selecionados aqui.
Nenhuma melhora é prometida. LOSO da alternativa é trabalho posterior, sem
misturar os resultados históricos de LOSO com esta comparação fixa.

Janelas se sobrepõem 50% e observações do mesmo participante/frequência têm
dependência. A likelihood fatorizada considera-as independentes como uma
aproximação e pode subestimar incerteza. Contagem de janelas não é contagem
comprovada de observações independentes. O modelo não ajusta efeitos de
participante nem correlação temporal.

Usar marginais preditivas por janela no HMM não integra conjuntamente a
incerteza do mesmo parâmetro compartilhado pela trajetória: produto de
integrais difere da integral do produto. O HMM inteiro não é um modelo
bayesiano conjunto. A/pi ficam fixas; delta não é probabilidade posterior.
Não há validação clínica nem adoção automática.

## Reprodução e apresentação

Na raiz do projeto:

```bash
uv pip install --python .venv/bin/python -r requirements-bayesian.txt
.venv/bin/python src/bayesian_beta_hmm.py
.venv/bin/python src/bayesian_beta_hmm.py --reuse-samples
.venv/bin/python src/bayesian_beta_hmm.py --plots-only
.venv/bin/python -m unittest discover -s tests -p test_bayesian_beta_hmm.py
.venv/bin/python -m py_compile src/bayesian_beta_hmm.py src/plot_bayesian_beta_hmm.py
```

`--reuse-samples` confere dados, seed, priori e configurações registrados no
NetCDF antes de reutilizar; `--plots-only` lê resultados sem amostrar nem
refazer comparação. PyTensor/Matplotlib/ArviZ usam caches em `/tmp`.
Não foram alterados scripts de emissão/transição históricos; não é necessário
executá-los ou sobrescrever seus resultados para esta alternativa.

Entregas:

- `results/bayesian_beta_hmm.json`: proveniência, diagnósticos, posteriores,
  observações, trajetórias, eventos, métricas globais/por participante e
  frequência, sensibilidade e recuperação simulada.
- `results/bayesian_beta_samples/`: NetCDF completos, amostras fixas da
  preditiva e simulações a priori/recuperação em NPZ.
- `outputs/bayesian_beta_hmm/index.html`: galeria comentada para apresentação.
- `outputs/bayesian_beta_hmm/apresentacao_figuras.pdf`: todas as figuras;
  cada uma também em PNG, SVG e PDF separado.
- CSVs de métricas globais/por participante/por frequência, resumos posteriores,
  relatório `README.md` e manifesto de figuras na mesma pasta.

Ordem sugerida para explicar na disciplina: dados e pipeline → duas Betas
distintas → prioris e preditivas a priori → posteriores e diagnósticos →
PDFs MLE/preditivas e evidência → delta causal → trajetórias de todos os
participantes → taxas separadas e tempos pareados → limites metodológicos.

Referências de implementação:
[PyMC sample](https://www.pymc.io/projects/docs/en/v5.16.0/api/generated/pymc.sample.html),
[ArviZ diagnostics](https://python.arviz.org/en/v0.22.0/_modules/arviz/stats/diagnostics.html).

## Resultado observado nesta execução

Baseline: 70/88 alvos, zero FP alvo ESP, 8/88 laterais, 734/730 s.
Principal: 68/88, zero FP alvo ESP, 8/88 laterais, 745,0/730 s.
Ampla: 68/88 e 736,2/730 s; simétrica: 68/88 e 744,1/730 s, com mesmos FPs.
Não se escolheu uma priori por desempenho. Pareamento principal: 68 comuns,
59 empates, nove atrasos Bayes, nenhum adiantamento, duas exclusivas MLE
(Ti 89 Hz, Vi 89 Hz); diferença média pareada +18,54 s.

Posteriores calibradas: zero divergências, R-hat máximo 1,0028,
ESS bulk mínimo 2986, tail mínimo 3468, BFMI mínimo >0,97. Dois exemplos
simulados recuperaram os quatro parâmetros dentro dos intervalos de 95%.
Seis testes numéricos e de regressão passaram. Os parâmetros, intervalos e
diagnósticos completos estão no JSON e relatório gerado da galeria.

Oito ESP e 44 estimulados contribuem com janelas ao ajuste 120/60; todos os
66 arquivos fazem parte do protocolo e todos os participantes permanecem no
denominador. Clipping mudou onze p estimulados e zero ESP. As simulações a
priori principal implicam massa p<0,05 de 11,84% Ausente e 30,20% Presente,
explicitando que centrar μ Ausente em 0,5 não impõe distribuição uniforme.

Em p=1e−9 a logpdf Ausente muda de −3,7812 MLE para +0,0152 preditiva
principal, pois a posterior admite a<1; Presente muda de 11,6676 para 11,6786.
Isso ilustra alteração da emissão por mistura, mesmo com parâmetros médios
próximos do MLE. As decisões mudam somente ao passar essas densidades pelo
mesmo Viterbi causal. A alternativa não demonstrou melhora nesta comparação.
