# `src/histogram_hmm.py`

Entrada para emissão por histograma da estatística ORD, com calibração ESP e
ESP + senoide. Consulte a Seção 18 de `docs/CONTEXTO_GET_MATRIX.md` para contrato,
fórmulas, comandos e limites. A observação é a estatística; a densidade do bin é
a emissão. Usa os mesmos bins para os dois estados, suavização 0,5 e modelo
global compartilhado por frequências. Não ajusta Beta nem usa p-valores na
inferência. A vem do artefato heurístico; inferência tem backtracking completo.

Saída independente: `results/histogram_hmm.json`. Os experimentos Beta e
discretos anteriores permanecem disponíveis para comparação. Testes numéricos
em `tests/test_histogram_hmm.py`.

O histograma não foi adotado após a comparação: manter Beta dos p-valores,
conforme `phase_transition_hmm.md` e a Seção 20 do contexto técnico.
Para reproduzir o experimento histórico de fases concatenadas com histograma, use
`src/compare_histogram_phase_rayleigh.py`, documentado em
`compare_histogram_phase_rayleigh.md`. `histogram_hmm.py` sozinho permanece
uma inferência por arquivo com backtracking; não representa a comparação
causal por paciente × frequência.
