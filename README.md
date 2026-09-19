# Detecção de ASSR com HMM semântico

Este projeto implementa, em [`src/assr_semantic_hmm.py`](src/assr_semantic_hmm.py), um pipeline para detectar resposta auditiva de estado estável (ASSR) em gravações de EEG. O programa combina processamento tradicional de sinais com uma tabela temporal compilada por uma LLM local.

A LLM participa somente da compilação offline da tabela. Depois que o arquivo `semantic_table.json` existe, toda a inferência é determinística e feita por consultas a essa tabela, sem chamadas ao modelo.

## Visão geral do fluxo

O arquivo executa o seguinte pipeline para cada gravação `.mat`:

1. carrega e valida o EEG;
2. remove o nível DC, descarta o início da coleta e rejeita épocas com artefatos;
3. agrupa as épocas válidas em janelas sobrepostas;
4. calcula uma estatística única de sincronismo de fase para toda a banda analisada e todos os canais;
5. converte a estatística contínua em um nível ordinal de evidência;
6. consulta a tabela semântica usando o histórico recente de estados e o nível observado;
7. produz a sequência de estados `rest`/`response` e um veredito final;
8. agrega os resultados por condição de estímulo e grava um relatório JSON.

```text
arquivo .mat
    → limpeza e rejeição de artefatos
    → janelas de épocas
    → CSM agregada e -log10(p)
    → nível ordinal de evidência
    → consulta à tabela temporal
    → estados por janela
    → veredito e relatório
```

## Configuração central

As constantes editáveis ficam no topo de `assr_semantic_hmm.py`:

- `ESTADOS`: estados que a tabela pode produzir. Atualmente, `rest` e `response`.
- `NIVEIS_OBSERVACAO`: categorias ordinais usadas para discretizar a evidência. Atualmente, `absent`, `low`, `medium` e `strong`.
- `ESTADO_INICIAL`: estado usado para preencher o histórico antes da primeira janela.
- `ESTADO_RESPOSTA`: identifica explicitamente qual item de `ESTADOS` conta como resposta no veredito final.
- `ORDEM_CONTEXTO`: quantidade de estados anteriores usada em cada consulta. O valor atual `1` cria `2¹ = 2` históricos possíveis.
- `FREQUENCIAS_ESPERADAS_HZ`: frequências efetivamente esperadas pelo código e pelos dados, atualmente `81, 83, ..., 95 Hz`.
- `TAMANHO_JANELA_EPOCAS` e `PASSO_JANELA_EPOCAS`: tamanho da janela e avanço entre janelas, em épocas. Com os valores `20` e `5`, janelas vizinhas se sobrepõem em 15 épocas.
- `SEGUNDOS_INICIAIS_DESCARTADOS`: quantidade de épocas iniciais descartadas. O nome reflete o formato esperado, no qual cada época tem um segundo.
- `LIMIAR_ARTEFATO_VOLTS`: amplitude absoluta máxima aceita em uma época.
- `P_VALOR_INICIO_EVIDENCIA` e `P_VALOR_EVIDENCIA_MAXIMA`: limites usados para gerar automaticamente os cortes da discretização.
- `FRACAO_MINIMA_RESPOSTA`, `JANELAS_CONSECUTIVAS_RESPOSTA` e `COMBINACAO_CRITERIOS`: regra configurável do veredito final.
- `OLLAMA_URL`, `OLLAMA_MODEL` e `OLLAMA_TIMEOUT_SEGUNDOS`: conexão com a LLM local usada na compilação.
- `PASTA_DADOS`, `ARQUIVO_TABELA` e `ARQUIVO_RESULTADOS`: caminhos padrão de entrada e saída.

Se `ESTADOS`, `NIVEIS_OBSERVACAO` ou `ORDEM_CONTEXTO` forem alterados, a tabela deve ser compilada novamente. O carregamento rejeita tabelas incompatíveis com a configuração atual.

## Entrada `.mat`

`carregar_mat` lê arquivos MATLAB v7.3 por meio do `h5py`. Cada arquivo precisa conter:

- `x`: EEG, convertido internamente para a forma `(amostras, épocas, canais)`;
- `Fs`: taxa de amostragem em hertz;
- `freqEstim`: vetor das frequências analisadas;
- `binsM`, se presente: deve ser igual a `freqEstim + 1`, conforme a indexação usada no MATLAB.

O programa exige que o número de amostras de cada época seja igual a `round(Fs)` e que `freqEstim` coincida com `FREQUENCIAS_ESPERADAS_HZ`. Portanto, cada época representa um segundo, e as frequências inteiras coincidem exatamente com bins da FFT de 1 Hz.

O nome do arquivo também identifica a condição. Nomes terminados em `ESP.mat` são classificados como controle `ESP`; nomes como `Ab40dB.mat` são classificados como `40dB`. O arquivo auxiliar `eletrodos.mat` é ignorado.

## Pré-processamento e janelas

`preparar_epocas` subtrai, de cada combinação época/canal, sua média ao longo das amostras. Em seguida, descarta as primeiras `SEGUNDOS_INICIAIS_DESCARTADOS` épocas. Uma época inteira é rejeitada se o maior valor absoluto, considerando todas as amostras e canais, ultrapassar `LIMIAR_ARTEFATO_VOLTS`.

`criar_janelas` percorre apenas as épocas que sobreviveram à rejeição. Cada janela contém `TAMANHO_JANELA_EPOCAS` e a próxima começa `PASSO_JANELA_EPOCAS` depois. Se restarem menos épocas que o tamanho configurado, nenhuma janela será criada.

## Estatística agregada da banda

`calcular_estatistica_banda` gera um único valor por janela:

1. calcula a FFT real no eixo de amostras;
2. seleciona os bins de todas as frequências-alvo;
3. descarta a magnitude e conserva apenas a fase de cada coeficiente complexo;
4. calcula a *Component Synchrony Measure* (CSM), isto é, o módulo ao quadrado da média dos fasores entre épocas;
5. tira a média da CSM entre todas as frequências e canais;
6. converte a evidência agregada em um valor-p pela distribuição qui-quadrado;
7. retorna `-log10(p)`, limitado ao intervalo de `0` a `300`.

Quanto maior o resultado, mais forte é a evidência de sincronismo de fase consistente entre as épocas. Frequências e canais não geram decisões separadas: todos contribuem para a mesma estatística da janela.

## Discretização da observação

`limiares_discretizacao` cria automaticamente `N - 1` cortes para `N` itens em `NIVEIS_OBSERVACAO`. Os cortes são igualmente espaçados na escala `-log10(p)` entre `P_VALOR_INICIO_EVIDENCIA` e `P_VALOR_EVIDENCIA_MAXIMA`.

Com a configuração atual, os quatro níveis correspondem aproximadamente a:

| Nível | Faixa de valor-p |
|---|---:|
| `absent` | `p > 0,05` |
| `low` | `0,007071 < p <= 0,05` |
| `medium` | `0,001 < p <= 0,007071` |
| `strong` | `p <= 0,001` |

`discretizar` usa esses cortes sem pressupor quatro categorias. Assim, a lista pode ter 1, 3, 6 ou outra quantidade de níveis sem que a lógica da função precise ser modificada. A ordem da lista sempre deve ir da menor para a maior evidência.

## Compilação semântica offline

`compilar_tabela_semantica` enumera todos os históricos possíveis de comprimento `ORDEM_CONTEXTO`. Para os dois estados e a ordem atual 1, são duas linhas: `rest` e `response`.

A compilação ocorre em duas rodadas:

1. para cada histórico, a LLM recebe os estados permitidos, os cortes estatísticos e o significado das observações; ela devolve uma `expert_note` específica sobre persistência, início ou interrupção da resposta e evidência ambígua;
2. para cada histórico, uma nova chamada recebe sua nota e a lista completa de níveis; ela escolhe o próximo estado para todos os níveis em uma única resposta JSON.

Com a configuração atual, isso resulta em `2 + 2 = 4` chamadas. De forma geral, o total é:

```text
2 × len(ESTADOS) ** ORDEM_CONTEXTO
```

As chamadas usam temperatura `0.1` e semente `42`. A resposta precisa ser JSON, conter exatamente as chaves esperadas e usar somente estados declarados em `ESTADOS`; qualquer violação interrompe a compilação com erro. O resultado completo, incluindo metadados, notas e tabela, é salvo em `semantic_table.json`.

Todos os prompts e os valores do esquema enviados à LLM estão em inglês.

## Inferência online

`inferir_sequencia` começa com um histórico preenchido por `ORDEM_CONTEXTO` ocorrências de `ESTADO_INICIAL`. Para cada janela, a função:

1. discretiza `-log10(p)`;
2. transforma o histórico atual em uma chave como `rest` (ordem 1) ou `rest -> rest -> response` (ordem 3);
3. obtém o próximo estado por `tabela[chave][nivel]`;
4. registra estado, nível e histórico consultado;
5. remove o estado mais antigo e acrescenta o novo ao histórico.

Nenhuma função de inferência chama `_chamar_ollama`. O modelo local pode inclusive estar desligado durante `validate`, desde que `semantic_table.json` já exista.

O veredito final considera dois critérios:

- a fração de janelas no estado definido por `ESTADO_RESPOSTA` deve ser pelo menos `FRACAO_MINIMA_RESPOSTA` (30% por padrão);
- a maior sequência consecutiva nesse estado deve ter pelo menos `JANELAS_CONSECUTIVAS_RESPOSTA` janelas (3 por padrão).

`COMBINACAO_CRITERIOS = "ou"` detecta resposta se qualquer critério for satisfeito. O valor `"e"` exige ambos.

## Instalação

Requisitos:

- Python 3.11 ou mais recente;
- [UV](https://docs.astral.sh/uv/) para instalar e executar o ambiente;
- Ollama acessível em `http://localhost:11434` somente para `compile` ou `all`;
- o modelo configurado em `OLLAMA_MODEL`, por padrão `qwen3-coder:30b`.

Na raiz do projeto, instale as dependências:

```bash
uv sync
```

Antes da compilação, certifique-se de que o Ollama esteja em execução e que o modelo exista localmente. Por exemplo:

```bash
ollama pull qwen3-coder:30b
```

Os arquivos `.mat` ficam, por padrão, no diretório `data/`.

## Como executar

### 1. Compilar a tabela

```bash
uv run python src/assr_semantic_hmm.py compile
```

Esse comando chama a LLM e salva `semantic_table.json`. Para escolher outro modelo ou caminho:

```bash
uv run python src/assr_semantic_hmm.py compile \
  --model outro-modelo \
  --table caminho/tabela.json
```

### 2. Validar os dados sem chamar a LLM

```bash
uv run python src/assr_semantic_hmm.py validate
```

O comando carrega `semantic_table.json`, processa separadamente cada `.mat` de `data/`, imprime o resultado de cada gravação e salva `results/assr_validation.json`.

Os caminhos podem ser substituídos:

```bash
uv run python src/assr_semantic_hmm.py validate \
  --data caminho/dados \
  --table caminho/tabela.json \
  --output caminho/resultado.json
```

### 3. Compilar e validar em sequência

```bash
uv run python src/assr_semantic_hmm.py all
```

`all` aceita `--model`, `--data`, `--table` e `--output`. Como recompila a tabela, esse modo precisa do Ollama em execução.

### 4. Executar os testes

```bash
uv run python -m unittest discover -s tests
```

## Saídas

`semantic_table.json` contém a versão do esquema, data de compilação, modelo utilizado, ordem do contexto, contagem de chamadas, estados, níveis, notas da LLM e a tabela de transição.

`results/assr_validation.json` contém:

- uma cópia da configuração relevante;
- resumo por intensidade/condição, com total de arquivos, detecções, taxa de detecção e mediana da fração de resposta;
- resultado detalhado de cada arquivo, incluindo qualidade das épocas, estatística de cada janela, níveis discretizados, históricos consultados, estados e veredito.

As condições com som e o controle `ESP` são processados como sequências independentes. A comparação esperada entre detecções com estímulo e ausência de detecção em `ESP` serve apenas como validação de sanidade; ela não treina nem modifica a tabela.
