# HMM e Viterbi

Exemplo didático de inferência MPE em um HMM discreto. O projeto usa Python 3.11+ e UV.

```bash
uv sync
uv run python src/run_inference.py --method mpe
```

O método `mpe` chama `mpe_inference` em `viterbi.py`. A execução imprime o trellis, os backpointers, o caminho escolhido e seu score. Cada execução acrescenta um registro em `src/logs/inference.jsonl`, com data, método, parâmetros do HMM, observações e resultado. Esse arquivo é gerado localmente e ignorado pelo Git.

Para executar somente o exemplo original:

```bash
uv run python src/viterbi.py
```

Para gerar as figuras da análise comparativa:

```bash
uv run python src/plot_comparison.py
```

## ASSR com compilação temporal semântica

O pipeline em `src/assr_semantic_hmm.py` usa CSM agregada nas frequências
físicas 81, 83, ..., 95 Hz. A LLM local participa somente da compilação
offline: uma chamada gera notas para todos os estados e uma chamada compila
cada linha da tabela.

```bash
uv run python src/assr_semantic_hmm.py compile
uv run python src/assr_semantic_hmm.py validate
```

O primeiro comando salva `semantic_table.json`. O segundo somente carrega essa
tabela, processa cada gravação independentemente e salva o relatório em
`results/assr_validation.json`; ele não chama a LLM. O subcomando `all` executa
as duas etapas em sequência.
