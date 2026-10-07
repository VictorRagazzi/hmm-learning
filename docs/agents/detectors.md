# `src/detectors.py`

## Responsabilidade

Define uma interface imutável `Detector` e o registro de estatísticas que
transformam coeficientes FFT complexos de M épocas em evidência no intervalo
`[0,1]`. Cada detector fornece `estatistica`, `p_value` e `thresholds`.

`calcular_estatistica_janelas` é o único laço genérico de janelamento: aceita
tamanho e passo, descarta janelas incompletas e devolve valores e intervalos
semiabertos `(inicio, fim)`.

## Detectores registrados

| Chave | Estatística | Nulo/limitação declarada |
|---|---|---|
| `msc` | `|sum X|² / (M sum |X|²)` | `Beta(1, M-1)` sob hipóteses gaussianas circulares e independência |
| `rayleigh` | `PLV²` | aproximação `p=exp(-M*PLV²)`; ignora magnitude |
| `csm` | transformação do T² circular de Hotelling | formulação F algebricamente equivalente à MSC sob as mesmas hipóteses gaussianas; não combinar com MSC |
| `mmsc` | média da MSC em duas metades | convolução numérica das duas `Beta(1, floor(M/2)-1)` independentes; estatística split-half experimental |
| `hotelling` | T² sobre média de `[Re(X), Im(X)]` com covariância 2×2 completa | forma escalada `F(2, M-2)` sob normalidade multivariada; requer `M > 2` |
| `spectral_f` | `M·|média(X_alvo)|² / média(|X_ruído|²)` | `F(2, 4M)` com dois bins locais complexos, independentes e de mesma variância |

O padrão efetivo é escolhido em `get_obs_matrix.py`, atualmente `csm`, não
neste módulo.

## Extensão segura

Para adicionar detector, implemente as três funções, crie um `Detector` e
inclua-o em `DETECTOR_REGISTRY`. Garanta que os thresholds sejam finitos e
estritamente crescentes na escala de evidência. Depois regenere B, execute a
inferência com a mesma chave e atualize o contexto técnico e os documentos de
agentes.

Uma nova estatística ou nova aproximação nula é decisão metodológica. Ela deve
ser justificada e validada; apenas encaixar a função na interface não torna o
teste calibrado.

## Pontos de atenção

- `EPS` evita divisão por zero, mas pode mascarar janelas degeneradas.
- Rayleigh divide cada coeficiente por `abs(X)+EPS`; coeficientes quase nulos
  merecem inspeção se o detector for adotado.
- Em MMSC, uma época é ignorada quando M é ímpar, pois são usadas duas metades
  de tamanho `M//2`. A cauda agora vem da convolução numérica, não da Beta
  aproximada usada anteriormente.
- As tabelas nulas MMSC ficam em cache sem limite durante o processo. Isso evita
  recalcular a convolução ao avaliar muitos comprimentos acumulados; o cache
  afeta apenas desempenho, não valores ou hipóteses estatísticas.
- CSM e MSC são duas expressões da mesma estatística sob as hipóteses atuais;
  não as use juntas como observações independentes na fusão.
- Hotelling geral difere do CSM circular porque estima covariância completa das
  partes real e imaginária. Covariâncias quase singulares usam pseudoinversa.
- `spectral_f` requer entrada matricial com uma coluna alvo e duas de ruído. O
  pipeline causal escolhe os bins FFT mais próximos que não pertençam a
  `freqEstim`; ao avaliar uma lateral, isso evita colocar um alvo estimulado no
  denominador. Seu F teórico depende de independência e variância comum entre
  esses bins, hipóteses ainda não validadas nos EEG reais.
- Os nomes CSM/MMSC não são universais; preserve a fórmula explícita nos
  relatórios em vez de depender apenas da sigla.
- O menu interativo é apropriado para uso manual; automações devem definir
  `ASSR_DETECTOR`.
