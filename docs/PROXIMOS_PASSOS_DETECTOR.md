# Próximos passos do detector (trabalho futuro)

Nota de planejamento. Nenhum código foi alterado. Detector atual: Rayleigh puro
com fasores unitários, `Z = (1/N)|Σ e^{iθ}|²` (`src/legado/detectors.py`,
`_rayleigh_stat`), janelas de 10 épocas com passo 5, alimentando o HMM.

Referências do projeto antigo, em
`C:\Users\T.i\Desktop\trampo\Academico\UFV\Nias\Numero_Deteccoes_consecutiva\`
(abreviada abaixo como `<NDC>\`).

## 1. Rayleigh com postos (rank-weighted)

**O que é.** Em vez de fasores de módulo 1, cada época recebe como módulo o
posto (1..N) do seu |FFT| no bin de interesse. Em `<NDC>\protocolo_deteccao.m:12-18`:

```matlab
[~,Rank] = sort(Y(:));  [~,Rank] = sort(Rank);   % posto de |Y|
Y = Y./abs(Y).*Rank;                              % fase × posto
ord(M,N) = sqrt(mean(real(Y))^2 + mean(imag(Y))^2)/sqrt(N);
```

A variante `<NDC>\protocolo_deteccao_qsample.m:13-20` ranqueia juntos o bin e os
harmônicos (`[M 2M 3M]`) e usa `Σ(Cm²+Sm²)/N`.

**Motivação.** Usa a informação de amplitude sem depender do valor absoluto.
Épocas com artefato de amplitude muito alta recebem no máximo o maior posto e
deixam de dominar a soma. O Rayleigh puro ignora a amplitude, e o MSC e a soma
de coeficientes brutos são sensíveis a ela.

**O que falta fazer.**
- **Distribuição nula e limiar.** A estatística com pesos de posto não segue a
  `Beta`/exponencial do Rayleigh puro. Verificar se há teoria fechada (família
  dos testes de Rayleigh ponderado) ou obter a distribuição nula por simulação
  Monte Carlo, com ruído gaussiano ou ESP real embaralhado por fase. Gerar uma
  tabela por `M` (como `_mmsc_null_table`) e usar os quantis para os limiares
  por faixa de p-valor.
- **Encaixe em `detectors.py`.** Novo `Detector` com `estatistica`, `pvalue` e
  `thresholds`, seguindo `_rayleigh_*`. A estatística recebe a janela de
  coeficientes complexos (épocas × 1) e calcula os postos dentro da janela,
  não na sequência inteira. Atenção: no código antigo o posto é calculado de
  1 a N cumulativo, e aqui seria por janela de 10 épocas.
- **Recalibração de B.** Refazer `B(Ausente)` (janelas ESP) e `B(Presente)`
  (injeção sintética) com o novo detector, pois a distribuição e os labels
  mudam. A matriz A heurística depende só do tamanho e passo das janelas e
  permanece.
- **Validação.** Falso positivo global e por frequência lateral (inclusive
  83 Hz, que já mostrou desvio). Não ajustar limiares com estímulos reais e
  avaliar no mesmo conjunto. Comparar a taxa de detecção com a do Rayleigh
  puro nas mesmas janelas.

## 2. NDC com alfa corrigido (baseline para o HMM)

**O que é.** Regra de decisão sequencial do projeto antigo: o detector é
testado em épocas `MM = Mmin:Mstep:Mmax`. Declara-se detecção quando ocorrem
**NDC detecções consecutivas** acima do limiar
(`<NDC>\ETS.m:23-37`, contador `cont = det(ii)+cont*det(ii)`). Exigir várias
detecções seguidas reduz o falso positivo acumulado por testes repetidos.

**Alfa corrigido.** `<NDC>\funcao_NDC_alfaCorrigido_Mmax.m` simula `nRuns` vezes
sob H0 (ruído gaussiano) e, para cada protocolo `(Mmin, Mstep, Mmax)` gerado por
`<NDC>\parametros_protocolo.m`, busca o par `(alfa_corrigido, NDC_minimo)` que
leva o falso positivo global ao valor desejado (0,05). Os resultados estão em
`<NDC>\NDC_AlfaCorrigido_Mmax{20,50,240}_alfa_0.05_FPdesejado0.05.mat`.
Avaliação em `<NDC>\estimarNDC.m`, `<NDC>\testar_NDC_AlfaCorrigido.m` e
`<NDC>\Avaliando_NDC_FP.m`.

**Uso como baseline.** Comparar HMM contra regra de N detecções consecutivas
nas mesmas janelas e dados:
- Reaplicar a simulação H0 com o detector atual (Rayleigh) e com
  `fs`/épocas reais, sem fixar 180 épocas ou `Fs`, para obter `NDC` e limiar
  com FP global de 5%.
- Medir **FP por frequência lateral** e **taxa de detecção** em ambas as
  abordagens, com o mesmo critério de FP.
- Medir **tempo de exame** (época da primeira decisão, `time` em `ETS.m`) para
  o HMM (decisão por posterior) e para a regra NDC, e comparar na fronteira
  taxa de detecção × tempo, como na fronteira de Pareto de
  `<NDC>\sinal_eeg.m:159-170`.
- Manter calibração e avaliação em dados separados.
