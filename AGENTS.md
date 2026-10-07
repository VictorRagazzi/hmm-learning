# Instruções para agentes — MVP ASSR / ELT610

Antes de alterar o processamento, leia `docs/CONTEXTO_GET_MATRIX.md` por completo.
Atualize esse documento quando mudar configuração, fórmulas, resultados ou formatos.

O escopo vigente é um detector configurável (Rayleigh, MSC, MMSC, CSM, Hotelling
ou F espectral local), A heurística por duração, B contínua Beta por MLE/MAP,
Viterbi e busca in-sample com todos os pacientes. Não reintroduza experimentos
históricos, LOSO, histogramas de emissão ou dados sintéticos.

- Configurações e grades ficam em `src/config.py`.
- `DETECTOR_NAME` seleciona um detector para B, inferência e busca. Recalibre B
  ao trocar o detector; respeite sua distribuição nula. MSC/CSM são equivalentes,
  MMSC é split-half experimental e F espectral precisa dos bins de ruído locais.
- Leia `Fs`, `freqEstim` e `binsM` dos EEG reais; preserve dados e metadados.
- Uma emissão global por estado, compartilhada entre pacientes/frequências.
- Concatene ESP→30→40→50→60→70 somente dentro de cada paciente/frequência.
  Calcule janelas dentro dos arquivos; mantenha delta e contadores entre fases.
- Ausente: alvos ESP. Presente: alvos reais >=50 dB (corte configurável).
  Laterais medem FP e não calibram B. Priori somente em Presente.
- B no HMM é uma densidade Beta, não a tabela ilustrativa de massas.
- Viterbi completo usa backtracking; primeiro disparo usa estados causais.
- FP lateral é separado de FP dos alvos ESP, inclusive por frequência.
- A calculada por duração e A selecionada pela busca têm interpretações
  diferentes; nenhuma é transição fisiológica validada ou Baum–Welch.
- Todos os resultados são in-sample, sem validação clínica/independente.
- Prefira funções curtas, nomes em português, fluxo explícito e comentários
  sobre as fórmulas; público com programação básica.

Verificação após alterações:

```bash
.venv/bin/python -m py_compile src/*.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python src/get_obs_matrix.py
.venv/bin/python src/get_tran_matrix.py
.venv/bin/python src/hmm_inference.py
.venv/bin/python src/search_hmm_parameters.py --rapida
```

Confirme JSON válidos, linhas de A e massas ilustrativas B somando 1, PDFs
normalizadas, contagens de arquivos/janelas e diagnósticos por frequência.
Execute a busca completa quando alterar seu objetivo, grade ou recorrência.
Não altere dados nem descarte pacientes porque algum arquivo não tem janela.
