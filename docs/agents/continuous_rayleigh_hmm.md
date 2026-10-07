# `src/continuous_rayleigh_hmm.py`

Após a decisão final de 30/09/2026, suas funções de ajuste Beta/leitura continuam
como base do método mantido. O protocolo ativo é por paciente × frequência com
fases concatenadas em `phase_transition_hmm.py`, documentado em
`phase_transition_hmm.md`. O experimento por arquivo descrito abaixo continua
disponível, mas não substitui esse protocolo.

## Responsabilidade

Executa um experimento isolado com Rayleigh, vários tamanhos de janela e
emissões contínuas ajustadas em dados reais. Ele não modifica as matrizes A/B
do pipeline discreto nem os JSON antigos.

## Emissão contínua

Para cada janela, o detector produz `PLV²` e o p-valor aproximado
`exp(-M * PLV²)`. O programa ajusta uma distribuição Beta aos p-valores ESP
para o estado `Ausente` e outra aos p-valores das frequências estimuladas para
`Presente`. O Viterbi usa as log-densidades dessas duas distribuições, sem
converter valores em labels.

As Betas são recalibradas separadamente para janelas de 10, 20, 30 e 60 épocas.
O passo é metade da janela. A busca varia A, distribuição inicial e regra
temporal, escolhendo a maior acurácia balanceada entre candidatos com FP
lateral global de no máximo 5%.

## Vazamento conhecido

Todas as frequências estimuladas dos 55 arquivos entram no ajuste de
`Presente` e os mesmos arquivos entram na avaliação. A seleção de parâmetros
também é in-sample. Isso é deliberado para testar viabilidade, não validação.
Não apresente os resultados como desempenho fora da amostra ou clínico.

## Saída atual

`results/continuous_rayleigh_hmm.json` contém configuração, parâmetros Beta,
contagens de calibração, melhor HMM e melhor Rayleigh bruto janelado por tamanho,
resultados por dB e baseline de gravação completa.

O melhor resultado atual usa janela/passo 60/30: 45,91% de detecção, 5,00% de
FP e 70,45% de acurácia balanceada. Rayleigh na gravação completa permanece
melhor, com 53,41%, 3,86% e 74,77%, respectivamente.
