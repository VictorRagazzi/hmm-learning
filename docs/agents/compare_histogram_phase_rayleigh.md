# `src/compare_histogram_phase_rayleigh.py`

Este é um experimento comparativo histórico. Após a decisão final de
30/09/2026, manter a emissão Beta, documentada em `phase_transition_hmm.md` e
na Seção 20 do contexto. Para reproduzir especificamente a comparação histórica:

```bash
.venv/bin/python src/compare_histogram_phase_rayleigh.py
```

Não executa search: lê as 14 configurações Rayleigh já salvas de
`phase_transition_hmm.json` e o vencedor Rayleigh discreto de
`phase_transition_discrete_hmm.json`. Mantém janela/passo, A, pi e
consecutividade HMM, alpha e consecutividade bruta. O destaque é o vencedor
histórico, definido antes do teste novo; não seleciona vencedor pelo resultado.

Calibra um histograma global para cada janela/passo, com 20 bins uniformes em
[0,1] e pseudocontagem 0,5. Ausente usa estatística Rayleigh PLV² em ESP;
Presente usa ESP + senoide com K=0,01 × std(época), coerente em fase. Nenhum
estímulo real entra no ajuste da emissão. Estatística, não p-valor, alimenta
`log_emissoes`. O método é constante por bin; não é uma densidade suave.

Há 11 pacientes, oito alvos e oito laterais, totalizando 88 trajetórias de cada
tipo. Cada trajetória concatena ESP → 30 → 40 → 50 → 60 → 70 dB. Calcula-se
janela dentro de cada arquivo; não há janela cruzando fronteira. Delta e
consecutividade não reiniciam entre fases. O Viterbi é causal: estado terminal
por prefixo, sem usar observações futuras. Rayleigh bruto é janelado, não
acumulado. Tempos usam finais das janelas somados à duração das fases passadas.
Antes de executar, verifica-se que uma época realmente dura um segundo em
todos os arquivos (amostras/Fs).

Métricas por paciente e globais:

- Detecção: alvos disparados após início de 30 dB / alvos estimulados.
- FP lateral: laterais com qualquer disparo, incluindo ESP / laterais medidas.
- FP do alvo em ESP: separado, não misturado no denominador lateral.
- Tempo médio/mediano desde início de 30 dB: somente entre detectados; nulo
  para uma trajetória não detectada.
- Duração média incluindo não detectados: separada, atribui duração máxima a
  não detectados. Não chamar essa métrica de latência média entre detectados.
- Primeiro nível dB, tempo total e tempo dentro da intensidade: registros por
  paciente/frequência. Pareamento mede exclusividades e quem disparou antes
  entre trajetórias detectadas por ambos.

O FP combinado histórico é preservado como diagnóstico para reproduzir os
artefatos antigos, mas não substitui o FP lateral solicitado. Um FP combinado
de 4,55% com zero FP ESP pode corresponder a FP lateral de 9,09%.

Saídas: `results/compare_histogram_phase_rayleigh.json` (parâmetros, histogramas,
protocolos, métricas por paciente/frequência e pareamento) e `.md` (tabelas).
Há também dois controles: `hmm_antigo_recalculado` reaplica as emissões Beta
armazenadas (ou B categórica no teste discreto); e
`histograma_presente_real_diagnostico` mantém a origem real de Presente em um
histograma da estatística, somente para diagnóstico in-sample. Esse controle
não substitui a emissão ESP/sintética ativa. Ambos mantêm A/pi/regras fixos.
As baselines brutas e o HMM antigo precisam reproduzir as taxas históricas, com assertivas
durante a execução. Testes de causalidade, relógio e FP lateral em
`tests/test_histogram_phase_rayleigh.py`; resultados na Seção 19 de
`docs/CONTEXTO_GET_MATRIX.md`.

O comparativo anterior `compare_histogram_rayleigh.py` é por arquivo, com
backtracking, e não atende a este protocolo. Mantê-lo apenas como histórico.
Os parâmetros foram previamente escolhidos no próprio conjunto; essa execução
fixa não é validação externa nem clínica. Nenhuma busca foi autorizada nesta
etapa.

Diagnóstico do destaque 120/60: antigo Beta real 79,55% de detecção, histograma
real 35,23%, histograma sintético 29,55%. Em 20 bins uniformes, 100% das 104
observações Ausente e 91,28% das 1744 Presente reais ocupam [0;0,05). O limiar
bruto alpha=0,05, PLV²≈0,02496, está dentro desse bin. Portanto o histograma
perde resolução nessa faixa; não descrever a queda como causada apenas por
troca real/sintético ou por retirar a transformação para p-valor. Não houve
mudança automática de bins para compensar o resultado.
