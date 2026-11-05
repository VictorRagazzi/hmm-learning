# Instruções para agentes — Rayleigh × HMM Gamma bayesiano (detecção de ASSR)

## Referência principal

Leia [src/rayleigh_hmm/README.md](src/rayleigh_hmm/README.md) antes de mexer no
projeto: desenho, configuração, resultados oficiais e limitações. Atualize esse
README na mesma tarefa sempre que mudar resultado, configuração ou decisão
metodológica.

## Mapa do código (`src/rayleigh_hmm/`)

- **Núcleo:** `dados.py`, `emissao.py`, `hmm.py`, `experimento.py`,
  `avaliacao.py`, `graficos.py`.
- **Validação oficial:** `explorar_loso_fc.py` (LOSO, fora da amostra, braço
  `HMM-REAL`). `explorar_loso.py` é o motor do LOSO usado por ele.
- **Secundários:** `selecao_sinal.py`, `explorar_protocolo*.py`,
  `explorar_busca.py` (suas grades são importadas por outros scripts; não mover).
- **`exploracoes/`:** scripts desatualizados (sinal antigo); só consulta.
- **`src/legado/`** (+ `tests/legado/`, `docs/legado/`): fase antiga (MSC, matriz
  B discreta, HMM Beta). Não alterar a menos que pedido; instruções antigas em
  `docs/legado/AGENTS_legado.md`.

## Regras metodológicas

- `experimento.py` com calibração real é in-sample com vazamento direto. O
  número a reportar é o do LOSO (`explorar_loso_fc.py`).
- Cada participante × condição × frequência é uma sequência independente. Não
  concatenar sequências, exceto na exploração explícita de protocolo.
- Teto de falso positivo lateral = FP do Rayleigh de referência (α = 0,05, 3
  consecutivas).
- Sinal: canal FC, sem rejeição de artefatos.
- Não apresentar resultados como estimativas clínicas validadas (11
  participantes; SNR sintética não é dB acústico).

## Verificação

```bash
.venv/bin/python -m py_compile src/rayleigh_hmm/*.py src/rayleigh_hmm/exploracoes/*.py
.venv/bin/python -m unittest tests.test_rayleigh_hmm
.venv/bin/python src/rayleigh_hmm/explorar_loso_fc.py   # validação oficial, vários minutos
```

Depois de rodar, confira `git diff --stat results/rayleigh_hmm/` e o braço
`HMM-REAL` em `results/rayleigh_hmm/exploratorio_loso_fc/tabelas.md`.

## Estilo

Poucas funções claras, configuração no topo do arquivo, fluxo explícito.
Documentação em português. Preserve alterações não relacionadas na árvore.
