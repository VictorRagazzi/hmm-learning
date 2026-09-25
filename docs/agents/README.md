# Documentação para agentes

Este diretório é o ponto de entrada para quem for trabalhar no projeto.
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

Estes documentos descrevem o código presente na árvore em 25/09/2026. O
`status.md` separa deliberadamente o comportamento observado das intenções e
resultados históricos documentados. Ao mudar código, configuração, saída ou
decisão metodológica, atualize a documentação afetada na mesma tarefa.
