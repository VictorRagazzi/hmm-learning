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
| `csm` | transformação de T²/F sobre a média complexa | usa `F(2, 2(M-1))` antes da transformação |
| `mmsc` | média da MSC em duas metades | aproximação `Beta(1, floor(M/2)-1)`, não distribuição exata da média |

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
  de tamanho `M//2`.
- Os nomes CSM/MMSC não são universais; preserve a fórmula explícita nos
  relatórios em vez de depender apenas da sigla.
- O menu interativo é apropriado para uso manual; automações devem definir
  `ASSR_DETECTOR`.
