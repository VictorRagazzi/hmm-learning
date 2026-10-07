# Documentação para agentes

Este diretório é o ponto de entrada para quem for trabalhar no projeto.
**Decisão vigente mais recente:** search contínuo **in-sample**, todos os
pacientes no ajuste de B e avaliação, sem LOSO. Execute
`src/search_hmm_parameters.py --continuous-map` (padrão in-sample); veja
Seção 24 e `search_map_beta_hmm.md`. Resultados/galeria possuem sufixo
`_in_sample`. LOSO preservado com `--validation loso`, sem execução automática.
**Retomada de 05/10/2026:** leia [`map_beta_hmm_loso.md`](map_beta_hmm_loso.md)
e a Seção 22 do contexto. Nova comparação com Beta única MAP Presente,
prioris editáveis de μ/κ, corte ≥50 dB e LOSO das emissões. Entrada:
`.venv/bin/python src/map_beta_hmm_loso.py`. A referência MLE e os experimentos
anteriores abaixo permanecem históricos; o novo módulo não usa senoide.
Critérios temporais disponíveis: `--min-consecutive`, `--min-percent` e
`--decision-mode AND|OR`; porcentagem calculada causalmente desde ESP.
Busca compatível: `src/search_hmm_parameters.py --continuous-map`, agora padrão
in-sample; modo LOSO explícito com `--validation loso`. Veja
[`search_map_beta_hmm.md`](search_map_beta_hmm.md).
Antes de alterar processamento de EEG ou as matrizes do HMM, leia também,
integralmente, [`../CONTEXTO_GET_MATRIX.md`](../CONTEXTO_GET_MATRIX.md) e as
instruções de `AGENTS.md` na raiz.

## Ordem de leitura sugerida

1. [`status.md`](status.md): estado observado da árvore, artefatos e pendências.
2. [`architecture.md`](architecture.md): fluxo de dados e responsabilidades.
3. Documento do módulo que será alterado:
   - [`detectors.md`](detectors.md)
   - [`get_obs_matrix.md`](get_obs_matrix.md)
   - [`get_tran_matrix.md`](get_tran_matrix.md)
   - [`hmm_inference.md`](hmm_inference.md)
   - [`search_hmm_parameters.md`](search_hmm_parameters.md)
   - [`continuous_rayleigh_hmm.md`](continuous_rayleigh_hmm.md)
   - [`early_detection_rayleigh_hmm.md`](early_detection_rayleigh_hmm.md)
   - [`compare_detectors_early_hmm.md`](compare_detectors_early_hmm.md)
   - [`histogram_hmm.md`](histogram_hmm.md)
   - [`compare_histogram_phase_rayleigh.md`](compare_histogram_phase_rayleigh.md)
   - [`phase_transition_hmm.md`](phase_transition_hmm.md)
   - [`bayesian_beta_hmm.md`](bayesian_beta_hmm.md)
   - [`map_beta_hmm_loso.md`](map_beta_hmm_loso.md)
   - [`search_map_beta_hmm.md`](search_map_beta_hmm.md)

A referência histórica mantida após a decisão de 30/09/2026 está em `status.md`,
`phase_transition_hmm.md` e na Seção 20 do contexto: paciente × frequência,
fases concatenadas, emissão Beta dos p-valores (Ausente ESP, Presente real),
parâmetros históricos fixos e inferência causal. Histogramas foram testados e
preservados como histórico comparativo, sem serem adotados.

A alternativa bayesiana está **implementada, executada e documentada**, com
posteriores MCMC, diagnósticos, comparação fixa e 46 figuras para apresentação.
Leia [`bayesian_beta_hmm.md`](bayesian_beta_hmm.md) e a Seção 21 do contexto.
Não houve melhora de detecção, FP ou tempo nesta comparação in-sample;
nessa etapa o método mantido foi Beta por MLE, sem adoção automática da mistura.
A etapa MAP/LOSO de 05/10/2026 está descrita acima e não usa essa mistura.
A entrega pronta está na [galeria](../../outputs/bayesian_beta_hmm/index.html),
no [PDF de 46 páginas](../../outputs/bayesian_beta_hmm/apresentacao_figuras.pdf)
e no [relatório](../../outputs/bayesian_beta_hmm/README.md).
O prompt `../prompts/apresentacao_metodo_hmm_beta.md` é um registro anterior,
não uma tarefa pendente ou autorização para iniciar outro trabalho.
Os módulos antigos continuam disponíveis; não confundir resultados por arquivo
com resultados por trajetória de fases. O
`status.md` separa deliberadamente o comportamento observado das intenções e
resultados históricos documentados. Ao mudar código, configuração, saída ou
decisão metodológica, atualize a documentação afetada na mesma tarefa.
