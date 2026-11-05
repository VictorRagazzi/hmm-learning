# hmm-learning

Detecção de ASSR em EEG: teste de Rayleigh como observação de um HMM de 2
estados (Ausente/Presente) com emissão Gamma bayesiana.

## Estrutura

**Projeto final**

- `src/rayleigh_hmm/` — código, README técnico e resultados oficiais (LOSO em
  `explorar_loso_fc.py`); `exploracoes/` guarda scripts desatualizados.
- `tests/test_rayleigh_hmm.py` — testes do projeto final.
- `results/rayleigh_hmm/` — saídas; `docs/PROXIMOS_PASSOS_DETECTOR.md` — notas.
- `AGENTS.md` — instruções para agentes.

**Legado (fase MSC / matriz B discreta / HMM Beta)**

- `src/legado/`, `tests/legado/`, `docs/legado/`.

Dados em `data/` (não versionados). Detalhes: [src/rayleigh_hmm/README.md](src/rayleigh_hmm/README.md).
