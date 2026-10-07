# `src/compare_detectors_early_hmm.py`

## Responsabilidade

Executa o protocolo causal de detecção precoce separadamente para Rayleigh,
MSC, MMSC, Hotelling T² geral e F espectral local. Para cada detector, compara
HMM contínuo e o mesmo teste acumulado; não há fusão de detectores.

## Reuso e checkpoint

O resultado Rayleigh previamente validado pode ser reutilizado de
`results/early_detection_rayleigh_hmm.json`. Depois de cada detector, o script
grava checkpoint em `results/early_detection_detectors_hmm.json`, permitindo
retomar sem repetir combinações concluídas.

As tabelas nulas numéricas do MMSC ficam em cache durante todo o processo. Essa
mudança evita recomputação para prefixos de comprimentos diferentes e não muda
a estatística ou seus p-valores.

## Resultado atual

Nenhum HMM superou seu detector acumulado. MSC teve o melhor desempenho:
HMM 46,59% de detecção e 5,00% de FP contra MSC acumulada 53,64% e 5,00%.
Rayleigh perdeu 7,95 pontos percentuais de detecção e MMSC perdeu 6,59 pontos.
Hotelling geral perdeu 3,64 pontos e F espectral local perdeu 3,41 pontos.
Os resultados usam os mesmos participantes para ajuste e avaliação.
