# `src/get_obs_matrix.py`

## Responsabilidade

Calibra a matriz de emissão B e centraliza a leitura do EEG, a extração FFT, a
seleção do detector e a discretização reutilizadas por inferência e busca.

## Pipeline

1. Descobre todos os `data/*ESP.mat`.
2. Lê e valida `x`, `Fs`, `freqEstim` e `binsM` de cada arquivo.
3. Confirma que alvos e laterais têm a mesma ordem entre arquivos.
4. Para cada alvo, extrai um coeficiente FFT complexo por época.
5. Aplica o detector ativo em janelas completas.
6. Repete após injetar uma senoide coerente de amplitude
   `K_SINTETICO * std(época)` para simular `Presente`.
7. Discretiza pela cauda nula teórica do detector.
8. Soma contagens de todas as frequências, aplica smoothing e normaliza uma
   única vez para obter B global.
9. Mantém resultados por frequência como diagnóstico e salva um resumo JSON.

## Configuração atual

O código usa canal 0, detector padrão CSM, janela 10, passo 10, cinco labels,
alpha 0,05, `K_SINTETICO=0.01` e pseudocontagem 0,5. `ASSR_DETECTOR` tem
precedência sobre o padrão; ao rodar diretamente sem a variável, há menu.

`PI_INICIAL=[0.99,0.01]` também fica neste arquivo e é importado pela
inferência, embora não participe da estimação de B.

## API e saída

- `construir_matrizes_observacao(pasta)` devolve `global` e
  `por_frequencia`, incluindo registros auditáveis em memória.
- `construir_tabela_para_salvar` remove os registros volumosos e monta o
  schema persistido.
- `salvar_tabela_observacao` grava `results/observation_matrix.json`.

Cada registro em memória preserva participante, condição, canal, alvo,
controle, bins FFT, intervalo de épocas, `Fs`, estado de calibração, valor,
p-valor e label. Por compatibilidade, campos genéricos ainda se chamam
`valor_msc`, `thresholds_msc` e `msc_critica`.

## Invariantes

- B tem duas linhas na ordem de `ESTADOS` e cinco colunas na ordem de labels.
- A B final é única e global; diagnósticos por frequência não são matrizes a
  serem selecionadas durante inferência.
- `binsM` é metadado/controle lateral, nunca denominador da estatística.
- `B(Presente)` é sintética e não pode ser descrita como estimativa clínica.
- Alterar detector, M, canal ou agregação exige rever o nulo e regenerar B.

## Dívidas atuais

O passo 10 diverge do passo 5 de `get_tran_matrix.py` e das instruções atuais
do projeto. Além disso, o JSON descreve a distribuição nula como Beta mesmo
com CSM ativo. Consulte `status.md` antes de fazer mudanças.
