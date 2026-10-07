# Beta única MAP: ponto de entrada da comparação LOSO

**Histórico fixo LOSO.** A decisão vigente posterior é search in-sample sem
retidos: `search_hmm_parameters.py --continuous-map`. Leia Seção 24 e
`search_map_beta_hmm.md`. O módulo descrito aqui permanece preservado, mas
não é o comando para a execução in-sample atual.

Nova etapa autorizada em 05/10/2026. Leia a Seção 22 de
[`../CONTEXTO_GET_MATRIX.md`](../CONTEXTO_GET_MATRIX.md) para resultados e limites.

`src/map_beta_hmm_loso.py` compara Rayleigh janelado bruto, HMM com Betas MLE e
HMM com Beta Presente por MAP. Uma única Beta por estado/dobra, global entre
frequências. Ausente permanece MLE idêntico nos dois HMMs. Não há senoide,
labels, emissão por histograma ou mistura preditiva. Os scripts anteriores
continuam com seus comportamentos; `hmm_inference.py` ainda é discreto.

Busca posterior autorizada: `search_hmm_parameters.py --continuous-map` varia
janela/passo, A, pi e regras nos treinos LOSO. Leia
[`search_map_beta_hmm.md`](search_map_beta_hmm.md). O comando sem search deste
documento continua usando valores fixos do topo; resultados da busca ficam
separados e não substituem automaticamente suas constantes.

## Configuração e comandos

No topo do script estão μ~Beta(2,4), κ~Gamma(2,1) shape/rate e corte ≥50 dB.
Altere `MU_PRIOR_ALPHA/BETA`, `KAPPA_PRIOR_SHAPE/RATE`, `PRESENT_MIN_DB`, ou use:

```bash
.venv/bin/python src/map_beta_hmm_loso.py --mu-prior 2 4 --kappa-prior 2 1 --present-min-db 50
.venv/bin/python src/map_beta_hmm_loso.py --plots-only
.venv/bin/python -m unittest discover -s tests -p test_map_beta_hmm_loso.py
```

`--plots-only` lê o JSON salvo, recria figuras e tabela sem reajustar; novas
prioris requerem execução normal. `--output` e `--output-dir` permitem
preservar variantes. Imagens podem ser apagadas e regeneradas. Não apagar
arquivos históricos por iniciativa própria.

### Critérios de detecção (extensão de 05/10/2026)

No topo: `MIN_CONSECUTIVE=1`, `MIN_PERCENT=None`,
`MODO_REGRA_DECISAO="OR"`. `None` desativa um limiar; pelo menos um deve
estar ativo. Porcentagem é uma **fração** em (0,1]: 0.10 corresponde a 10%.
Consecutivas é um inteiro ≥1. `AND` exige ambos os critérios ativos; `OR`
aceita qualquer um. Com um único critério ativo, AND/OR são equivalentes.
Mesma regra aplicada ao bruto, HMM MLE e HMM MAP.

```bash
.venv/bin/python src/map_beta_hmm_loso.py --min-consecutive 4 --min-percent 0.10 --decision-mode AND
.venv/bin/python src/map_beta_hmm_loso.py --min-consecutive none --min-percent 0.10
.venv/bin/python src/map_beta_hmm_loso.py --min-consecutive 4 --min-percent none
```

A cada janela, porcentagem = positivos / janelas observadas no **prefixo
desde ESP**. Denominador não inclui futuro, não é quantidade de épocas e não
reinicia entre intensidades; consecutivas também persistem. O primeiro prefixo
que atende à regra dispara a decisão; uma queda posterior na porcentagem não
revoga esse evento. Poucas janelas iniciais podem atingir porcentagens altas.
Detecção alvo só conta após começo de 30 dB; FP alvo ESP é avaliado separado;
FP lateral inclui qualquer fase. O relatório registra limiares, modo e
definição do denominador. `--plots-only` conserva a regra do JSON; mudar os
limiares exige executar normalmente. Padrão preserva os resultados anteriores.

## MAP e visualização

MAP maximiza a posterior nas coordenadas μ,κ, usando SciPy com gradiente
analítico, três inicializações e detecção de convergência/borda. Logit/log são
somente coordenadas numéricas: sem Jacobiano no objetivo. A emissão final é
BetaPDF no par MAP conjunto. μ_MAP/κ_MAP não precisam ser modos marginais.

As posteriores completas dos parâmetros são aproximadas por quadratura local
adaptativa para os plots, com Jacobiano na integração em log(κ), normalização,
checagem de bordas e refinamento de grade. Não são usadas como emissão. Figuras
por dobra mostram prioris, posteriores marginais/conjunta e dados com curvas
MLE/MAP. Beta da priori de μ não é a distribuição dos p-valores.

## Dados e comparação

LOSO retém uma pessoa inteira. Treino usa alvos ESP para Ausente e alvos de
50/60/70 dB para Presente em ambos os HMMs; laterais não ajustam. Teste inclui
ESP e 30–70 dB. ≥50 dB é proxy de resposta forte; não garante resposta por
janela e pode representar pior 30–40 dB. Metadados/intervalos/R²/p-valores e
manifesto dos 66 EEG reais são preservados.

Rayleigh 120/60, A=[[.99,.01],[.15,.85]], pi=[.95,.05], uma consecutiva;
bruto alpha=.005. Mesmo janelamento e regra temporal para todos. Nenhuma busca
nova. Esses valores históricos foram selecionados nos mesmos pacientes;
**LOSO das emissões, não validação independente de todos os hiperparâmetros**.

Trajetórias causais por paciente×frequência concatenam ESP→30→40→50→60→70,
sem janelas cruzando arquivos. FP lateral inclui ESP; FP alvo ESP separado.
Tempos desde 30 só entre detectados; não detectados null e duração censurada.

Resultados desta execução: bruto 58/88 alvos e 3/88 laterais, 824,5/810 s;
MLE e MAP 70/88 e 11/88, 744,6/730 s, um FP alvo ESP cada. Acurácia balanceada
lateral 81,25% bruto e 83,52% HMMs. MAP/MLE empatam em todos os 70 eventos e
tempos comuns. Não houve ganho bayesiano demonstrado, nem FP HMM ≤5%.

## Artefatos e dependências

- `results/map_beta_hmm_loso.json`: dobras, parâmetros, calibração, trajetórias,
  resultados/tempos por paciente e frequência, logpdf externa e pareamento.
- `outputs/map_beta_hmm_loso/index.html`: galeria e tabela dos três métodos.
- `comparacao.csv`, `README.md`, onze figuras em PNG/SVG/PDF na mesma pasta.
- `parametros_por_dobra.csv` e `posterior_parametros_<pessoa>.csv`: parâmetros
  e densidades dos gráficos em formato numérico.
- `tests/test_map_beta_hmm_loso.py`: sete testes numéricos/metodológicos,
  incluindo AND/OR/limiares desativados, porcentagem causal entre fases e
  equivalência do avaliador padrão com todos os eventos do utilitário antigo.

Reutiliza leitores/FFT de `continuous_rayleigh_hmm.py`, causalidade de
`phase_transition_hmm.py` e protocolo/métricas de
`compare_histogram_phase_rayleigh.py`; não executa os mains nem usa seus dados
sintéticos. SciPy/NumPy/Matplotlib existentes bastam; PyMC não é necessário.
Não modifica A/B discretas, artefatos MCMC ou resultados anteriores.

Prioris devem ser positivas/finita; se um MAP for singular/de borda ou a grade
não cobrir a posterior, o script falha explicitamente. Janelas dependentes
limitam a interpretação da posterior iid. Explorar prioris olhando o teste
é sensibilidade, não seleção independente; não executar busca automaticamente.
