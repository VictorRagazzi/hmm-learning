# Detecção de ASSR com HMM — ELT610

MVP para estudar um HMM de dois estados (`Ausente`, `Presente`) em EEG.
O detector escolhido gera p-valores; distribuições Beta descrevem as emissões
B; Viterbi acompanha a evidência ao longo das intensidades.

O ajuste usa **todos os pacientes**. Ajuste, otimização e avaliação compartilham
os mesmos dados (in-sample): os números são exploratórios.

## Executar

Python 3.11 ou superior. Para instalar as dependências com uv:

```bash
uv sync
```

Coloque os `.mat` reais em `data/`, com nomes como `AbESP.mat`, `Ab30dB.mat`,
`Ab40dB.mat`, `Ab50dB.mat`, `Ab60dB.mat`, `Ab70dB.mat`.

Execute o MVP com a configuração escolhida:

```bash
.venv/bin/python src/hmm_inference.py
```

Esse comando calcula A e B, aplica Viterbi, grava três JSON em `results/` e
as curvas de B e das prioris em `outputs/`. Não inicia a busca.

Os cálculos também podem ser executados separadamente:

```bash
.venv/bin/python src/get_tran_matrix.py
.venv/bin/python src/get_obs_matrix.py
```

## Alterar parâmetros e otimizar

Edite [src/config.py](src/config.py): canal, janela/passo, prioris, corte de
Presente, A, pi, regra de decisão e grades da busca ficam no mesmo arquivo.
`METODO="map"` usa priori; `METODO="mle"` usa máxima verossimilhança.

`DETECTOR_NAME="rayleigh"` é o padrão. Também estão disponíveis `msc`, `mmsc`,
`csm`, `hotelling` e `spectral_f`. Um detector é aplicado por execução, no
ajuste de B, Viterbi e busca. Ao mudar essa configuração, execute novamente
o MVP para recalibrar B; para otimizar o novo detector, refaça a busca.
`--usar-busca` rejeita um vencedor salvo para outro detector.

```bash
.venv/bin/python src/search_hmm_parameters.py
.venv/bin/python src/hmm_inference.py --usar-busca
```

A busca compara MLE e MAP, maximiza acurácia balanceada sob FP lateral <=5%
na própria amostra e salva o vencedor de cada método. O segundo comando usa
o vencedor do método configurado, incluindo sua janela/passo, A, pi e regra.
As constantes de `config.py` não são modificadas automaticamente.

Para conferir a busca com uma grade pequena:

```bash
.venv/bin/python src/search_hmm_parameters.py --rapida
```

A execução reduzida tem saída separada e não substitui a busca completa.

## Ler o código

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Parâmetros e grades |
| `dados.py` | Leitura, FFT, bins de ruído, janelas e concatenação |
| `detectors.py` | Estatística e p-valor dos seis detectores |
| `get_tran_matrix.py` | A calculada pela duração dos arquivos |
| `get_obs_matrix.py` | B por MLE/MAP, PDF, CDF e gráficos |
| `viterbi.py` | Recorrência, backtracking e estados causais |
| `hmm_inference.py` | Primeiro disparo, métricas e execução do MVP |
| `search_hmm_parameters.py` | Busca dos parâmetros usando todos os pacientes |

A sequência é independente por paciente e frequência:
`ESP → 30 → 40 → 50 → 60 → 70 dB`. Janelas não atravessam arquivos; a memória
do HMM e os contadores continuam entre intensidades.

B é contínua: `B_s(p) = BetaPDF(p; a_s, b_s)`. O JSON também traz uma tabela
2×5 para explicar probabilidades de intervalos; ela não alimenta Viterbi.
A heurística calculada por duração é uma referência; a inferência padrão usa
A selecionada anteriormente pela busca, declarada em `config.py`.

## Resultado de referência

Configuração escolhida: janela/passo 90/45, A=[[.8,.2],[.45,.55]], pi=[.9,.1],
três janelas consecutivas OU 60% de positivos no prefixo observado.

MLE e MAP reproduzem **66/88 alvos detectados (75%)**, **4/88 FP laterais
(4,55%)** e zero FP alvo ESP. São 66 arquivos, 11 pacientes, oito alvos e
oito laterais por paciente. Os parâmetros de B usam 128 janelas Ausente e
672 Presente; gravações curtas não geram janelas, mas os pacientes permanecem
na avaliação. Tempos entre detectados: média 784,6 s, mediana 737,5 s desde 30 dB.

Leia as fórmulas, limitações e resultados em
[docs/CONTEXTO_GET_MATRIX.md](docs/CONTEXTO_GET_MATRIX.md).

## Verificar

```bash
.venv/bin/python -m unittest discover -s tests
```

Os testes conferem Viterbi por enumeração de caminhos, causalidade, MAP,
normalização, busca em lote e reprodução com os EEG reais.
