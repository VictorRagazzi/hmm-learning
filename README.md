# HMM e Viterbi

Exemplo didático de inferência MPE em um HMM discreto. O projeto usa Python 3.11+ e UV, sem dependências externas.

```bash
uv sync
uv run python run_inference.py --method mpe
```

O método `mpe` chama `mpe_inference` em `viterbi.py`. A execução imprime o trellis, os backpointers, o caminho escolhido e seu score. Cada execução acrescenta um registro em `logs/inference.jsonl`, com data, método, parâmetros do HMM, observações e resultado. Esse arquivo é gerado localmente e ignorado pelo Git.

Para executar somente o exemplo original:

```bash
uv run python viterbi.py
```
