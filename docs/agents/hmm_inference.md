# `src/hmm_inference.py`

## Responsabilidade

Carrega A e B, produz uma sequência de observações por arquivo/frequência,
decodifica estados com Viterbi e aplica uma regra exploratória de detecção.
Também executa o detector bruto como baseline sob a mesma regra temporal.
Adiciona uma baseline separada em que cada detector usa todas as épocas de uma
gravação em uma única estatística e agrupa as métricas por intensidade.

## Compatibilidade antes da execução

O script seleciona o detector via `get_obs_matrix.selecionar_detector()` e
valida no JSON de B detector, janela, passo, fronteiras, estados, labels e somas
de linha. A valida somente ordem de estados e somas de linha. Assim, ele impede
usar B calibrada por outro detector, mas hoje não detecta que A foi gerada com
passo diferente.

## Unidade de processamento

São lidos os 55 `*dB.mat`. Para cada arquivo:

- cada `freqEstim` gera uma sequência independente;
- cada `binsM` gera outra sequência independente de controle;
- não se concatenam frequências, participantes ou condições;
- cada frequência conta como um experimento nas taxas agregadas.

`construir_sequencia_labels` extrai valores, p-valores, labels e intervalos por
janela usando exatamente o detector ativo em `get_obs_matrix.py`.

## Viterbi e regra de decisão

O Viterbi local opera em log-espaço, acrescentando `1e-300` antes do log. O
caminho ótimo é transformado em uma sequência booleana `estado == Presente`.
A baseline usa `p_value <= alpha`.

A regra atual detecta uma frequência quando há pelo menos uma janela positiva
OU pelo menos 5% de janelas positivas. `None` desativa um critério; com ambos
ativos, os modos válidos são `OR` e `AND`. A função trata qualquer modo diferente
de `AND` como OR, portanto a configuração deveria ser validada explicitamente
antes de aceitar novos valores.

## API e saída

- `rodar_inferencia`: carrega tabelas e processa todos os arquivos.
- `processar_arquivo`: calcula estímulos, laterais, HMM e baseline.
- agregações por intensidade de estímulo; não imprime desempenho por
  participante.
- `construir_relatorio`: serializa configuração, resumo e detalhe completo.
- `salvar_relatorio`: grava `results/inference_analysis.json`.

O detalhe persistido atual inclui labels, valores do detector, p-valores,
intervalos e caminhos, além das decisões. O nome `valores_msc` permanece mesmo
quando o detector é CSM/Rayleigh/MMSC.

## Interpretação

As frequências laterais são medidas nos próprios arquivos estimulados, o que
captura artefatos de aquisição, mas elas não são uma amostra clínica validada
de ausência. As taxas por dB e a comparação HMM versus baseline são
exploratórias e não substituem validação fora da amostra. A baseline de
gravação completa executa vários testes por gravação, então alfa por frequência
não garante controle de 5% no nível da gravação.
