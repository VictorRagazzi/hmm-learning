# Método mantido: HMM Beta causal com intensidades concatenadas

**Etapa nova, 05/10/2026:** comparação Beta única MAP Presente versus MLE e
bruto em LOSO está em [`map_beta_hmm_loso.md`](map_beta_hmm_loso.md) e Seção 22.
Use `src/map_beta_hmm_loso.py`; esse módulo reaproveita a causalidade e o
protocolo descritos aqui, com treino Presente ≥50 dB e prioris editáveis.
O histórico MLE/MCMC abaixo permanece preservado, sem nova busca automática.

Decisão final do usuário em 30/09/2026: manter a emissão Beta dos p-valores,
com Ausente ESP e Presente real. Histogramas são experimentos preservados.
A referência completa está na Seção 20 de `docs/CONTEXTO_GET_MATRIX.md`.

## Implementação e reprodução

`phase_transition_hmm.py` constrói trajetórias por paciente × frequência na
ordem ESP, 30, 40, 50, 60 e 70 dB. Janelas ficam dentro de cada arquivo. Só as
observações são concatenadas; delta e consecutividade continuam entre fases.
Alvos e laterais, frequências e pacientes mantêm trajetórias independentes.

Fluxo Rayleigh: EEG → FFT complexa por época → normalização de fase → PLV² →
p=exp(-M*PLV²) → logpdf de duas Betas → estado terminal Viterbi causal. Não há
labels nem histograma na emissão. Os p-valores são limitados a [1e-9;1-1e-9].

Betas são ajustadas globalmente por configuração em
`continuous_rayleigh_hmm._fit_beta`: alvos ESP para Ausente e alvos reais de
todos os níveis estimulados para Presente. Laterais não ajustam a emissão.
Essa modelagem Presente é um proxy exploratório da condição estimulada, não
confirma que toda janela possua uma resposta fisiológica.

Para plots/reprodução, leia modelos e parâmetros de
`results/phase_transition_hmm.json` e chame `_estados_hmm` com os valores salvos.
**O main desse módulo executa search**; não rodá-lo para uma mera visualização
nem chamar `_melhor_hmm`/`_melhor_bruto`. O artefato
`results/compare_histogram_phase_rayleigh.json` contém fases e a reprodução
antiga em `hmm_antigo_recalculado` / `detalhes_hmm_antigo`.

## Referência fixa

Rayleigh, janela/passo 120/60, A=[[0,99;0,01],[0,15;0,85]], pi=[0,95;0,05],
uma consecutiva. Betas salvas: Ausente a=1,184151213734531,
b=0,8862635970365302; Presente a=0,39052284005446136,
b=0,9567359610170701. A veio da seleção histórica, não da heurística de duração.

Detecção alvo depois do início de 30 dB. FP alvo ESP é separado; FP lateral
solicitado considera qualquer disparo no protocolo completo. Por paciente:
alvos detectados/8 e laterais com disparo/8. Tempos desde 30 são reportados
somente entre detectados; não detectados recebem null e uma duração censurada
separada. A unidade global é 88 alvos e 88 laterais.

Resultado antigo reproduzido: 79,55% detecção / 9,09% FP lateral, acurácia
balanceada lateral 85,23%, tempo médio/mediano 734/730 s. O FP combinado
histórico 4,55% usa 176 oportunidades e não é FP lateral. A comparação histórica
é in-sample; LOSO tem arquivos próprios e não foi reexecutado nesta tarefa.

## Alternativa bayesiana concluída

A alternativa bayesiana experimental foi implementada separadamente em
`src/bayesian_beta_hmm.py`: posterior conjunta μ/κ e emissão preditiva,
mantendo parâmetros históricos fixos e baseline MLE reproduzida.
Ver [bayesian_beta_hmm.md](bayesian_beta_hmm.md) e Seção 21 do contexto.
Não altera o método mantido nem autoriza nova busca ou adoção automática.

A comparação fixa foi executada com três prioris: todas detectaram 68/88
alvos, com zero FP alvo ESP e 8/88 FP laterais, contra 70/88 na MLE.
Média/mediana desde 30 dB: principal 745/730 s, ampla 736/730 s,
simétrica 744/730 s, MLE 734/730 s. Não houve melhora de velocidade.
Pareamento principal: 59 empates e nove atrasos entre 68 comuns, nenhuma
antecipação e duas detecções exclusivas MLE. A MLE continua sendo o método ativo.

A posterior quantifica incerteza da emissão, mas não tornou o HMM inteiro
bayesiano nem validou generalização. O experimento e a documentação estão
concluídos; qualquer avaliação LOSO da alternativa depende de nova tarefa.

A apresentação dessa comparação está pronta em
`outputs/bayesian_beta_hmm/index.html`: 46 figuras PNG/SVG/PDF, PDF consolidado,
relatório e CSVs. O prompt `docs/prompts/apresentacao_metodo_hmm_beta.md`
é histórico e não representa uma próxima tarefa automaticamente autorizada.

## Registro da apresentação anterior

Disponível em `outputs/apresentacao_metodo_beta/index.html` e
`apresentacao.pdf`. Execute `.venv/bin/python
outputs/apresentacao_metodo_beta/gerar.py` na raiz para reproduzir figuras,
textos e auditoria com os parâmetros salvos, sem busca. O README da pasta
explica fontes, exemplos, CSVs e limites. A apresentação confere EEG,
p-valores, Viterbi por prefixo e eventos; não altera o método ou `results/`.

Na árvore inspecionada após a entrega bayesiana, `apresentacao.pdf` está
presente, mas a pasta `outputs/apresentacao_metodo_beta/` não está disponível.
Para o material pronto e conferido da comparação atual, use a galeria
`outputs/bayesian_beta_hmm/` e os comandos de `bayesian_beta_hmm.md`.
