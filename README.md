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

## Ondas de Mayer

```bash
uv run python src/mayer_waves.py
```

O script acumula os intervalos RR da primeira coluna como no MATLAB
(`T = cumsum(RR)`), classifica RR, seleciona as IMFs indicadas por `NUM_IMF`
e gera `figs/resultados_mayer.png`. Para classificar a pressão arterial média
(quarta coluna), altere `TIPO_SINAL` para `"MAP"` em `src/mayer_waves.py`.
Os valores padrão de `fp`, `range_f`, `NumIMF` e `suma` ficam no início de
`src/mayer_waves.py`. A função `classificador(sinal, t_s, fp, range_f,
num_imf, suma)` também aceita esses parâmetros diretamente.

A potência usa a normalização amostral e a grade descritas para `plomb`.
O limiar FAP é uma aproximação baseada no número estimado de frequências
independentes: a documentação do MATLAB não publica a fórmula exata de `pth`.
Além disso, `PyEMD` e `emd` do MATLAB podem produzir IMFs diferentes.
Portanto, a classificação ainda pode diferir do MATLAB; para comparar valores
exatos, use os mesmos vetores de entrada e exporte `p`, `f` e `pth` do MATLAB.

### Teste de perda de dados

```bash
uv run python src/mayer_loss_test.py
```

O teste percorre 0%, 5%, ..., 70% de perda nos 21 pares de arquivos. Ele usa
`RR` por padrão, como nas chamadas `classificador(RR_S, T_S, ...)` do trecho
MATLAB. Use `--signal MAP` para testar a pressão arterial média. As opções
`--seed`, `--fp`, `--range-f`, `--num-imf` e `--suma` controlam o sorteio e o
classificador. A perda é progressiva para cada arquivo; os tempos originais das
amostras mantidas são preservados.

O script salva `figs/perda_dados_mayer.png`, `figs/perda_dados_resumo.csv` e
`figs/perda_dados_detalhes.csv`. A comparação com 0% reproduz os argumentos
do `testcholdout` do trecho: os rótulos de 0% são usados como referência no
teste de McNemar bilateral mid-p, com `alpha=0,05`. Isso mede estabilidade das
classificações, não acurácia contra diagnósticos clínicos. A porcentagem
informada no terminal é a última antes da primeira diferença significativa.
Como a exclusão de amostras é aleatória, mudar `--seed` pode alterar os
resultados.
