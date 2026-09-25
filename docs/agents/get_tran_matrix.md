# `src/get_tran_matrix.py`

## Responsabilidade

Constrói e salva uma matriz A de dois estados a partir da duração, em janelas,
dos arquivos ESP e com estímulo. É uma heurística inicial de persistência.

## Cálculo

Com janela W e passo P, um arquivo de E épocas produz:

```text
N = max(floor((E-W)/P) + 1, 0)
```

Cada arquivo válido contribui N-1 autopassagens e uma saída implícita. Para uma
condição, `p_self = soma(N-1) / soma(N)`. A linha da matriz recebe `p_self` na
diagonal e `1-p_self` fora dela.

Arquivos `*ESP.mat` representam Ausente e `*dB.mat`, Presente. A transição de
saída não é observada; é adicionada pela regra. Portanto A não é Baum–Welch,
não usa o detector e não mede diretamente persistência fisiológica.

## Configuração e validação

O código usa janela 10 e passo 5. Valida a existência de `x` e `Fs`, três
dimensões, exatamente 16 canais e `amostras == int(Fs)`. Arquivos sem uma
janela completa são descartados.

## API e saída

- `n_janelas_de`: converte épocas em janelas.
- `carregar_n_epocas`: faz validação geométrica sem carregar todo o EEG.
- `calcular_p_self`: agrega uma condição e retém detalhe por arquivo.
- `construir_matriz_transicao`: monta o documento completo.
- `salvar_tabela_transicao`: grava `results/transition_matrix.json`.

O artefato atual usa 279 janelas Ausente e 3.225 Presente, gerando:

```text
[[0.960573, 0.039427],
 [0.017054, 0.982946]]
```

## Pontos de atenção

- O passo atual não coincide com B/inferência (5 contra 10). Não regenere A ou
  interprete durações sem resolver/documentar essa escolha.
- Arquivos longos têm mais peso que curtos.
- Fixar 16 canais e `amostras == Fs` codifica o formato atual de épocas de um
  segundo; mudanças no dataset exigem inspeção real antes de relaxar isso.
- Mudar janela ou passo altera A mesmo sem qualquer mudança fisiológica.
