# Implementar emissão Beta bayesiana para o HMM de ASSR

Implemente uma alternativa bayesiana ao ajuste por máxima verossimilhança das
emissões Beta do HMM atual. Preserve o método atual como baseline reproduzível;
não substitua seus resultados nem adote a alternativa automaticamente.

## Contexto e leitura

Leia AGENTS.md, docs/CONTEXTO_GET_MATRIX.md por completo (especialmente a
Seção 20), docs/agents/phase_transition_hmm.md e os módulos
src/continuous_rayleigh_hmm.py e src/phase_transition_hmm.py antes de editar.
Inspecione também os parâmetros e resultados salvos. As instruções iniciais
descrevem o pipeline histórico MSC/categórico; esta tarefa trata explicitamente
do método Rayleigh/Beta contínuo mantido na Seção 20 e de seu protocolo
concatenado, sem modificar o pipeline histórico.

Atualmente: EEG → FFT por época → Rayleigh R² → p=exp(-M*R²) → densidades
Beta por estado → Viterbi causal. Ausente usa alvos ESP; Presente usa alvos
de gravações estimuladas reais, como proxy de resposta. O ajuste atual usa
beta.fit com loc=0 e scale=1, por máxima verossimilhança. Escolher a família
Beta para os p-valores não é, por si só, uma priori bayesiana dos parâmetros.

## Modelo solicitado

Para cada estado s, modele:

```text
p_i | mu_s, kappa_s ~ Beta(mu_s*kappa_s, (1-mu_s)*kappa_s)
mu_s ~ Beta(u_s, v_s)
kappa_s ~ Gamma(shape_s, rate_s)
a_s = mu_s*kappa_s
b_s = (1-mu_s)*kappa_s
```

As prioris de mu e kappa podem ser independentes dentro de cada estado nesta
primeira implementação. Mu é a média dos p-valores; kappa é a concentração.
Gamma usa parametrização shape/rate: explicite a conversão se a biblioteca
usar scale. Não confunda a Beta dos dados com a Beta escolhida como priori de mu.

Escolha e justifique hiperparâmetros iniciais com simulações preditivas a priori.
Mantenha-os configuráveis no topo do arquivo. Não use resultados de detecção ou
FP do conjunto avaliado para escolher prioris. Compare um pequeno conjunto
predefinido de prioris plausíveis para verificar sensibilidade, sem selecionar
um vencedor pela avaliação. Não interprete uma priori uniforme de mu como
ausência de informação sobre a forma da distribuição dos p-valores.

Estime a posterior conjunta de mu e kappa por um método numérico apropriado,
preferencialmente MCMC com diagnósticos de convergência. Justifique a biblioteca,
registre versões e use sementes fixas. Não apresente somente MLE ou MAP como
se fosse inferência posterior completa. Salve amostras e resumos posteriores,
incluindo intervalos de credibilidade para mu, kappa, a e b.

## Como usar a posterior no HMM

Implemente como emissão alternativa a densidade preditiva posterior:

```text
f_s(p_novo | dados_calibracao)
  ≈ (1/S) * soma_r BetaPDF(p_novo; a_s[r], b_s[r])
```

Avalie-a em log-espaço com logsumexp(logpdf_r) - log(S), usando um conjunto
fixo de amostras posteriores durante cada execução. Não use a média dos logs
nem confunda a média das densidades com a densidade nos parâmetros médios.
Reutilize a estabilização dos p-valores nos extremos e documente seu efeito.
As emissões são densidades: podem exceder 1 e devem integrar 1; não precisam
somar 1 em uma grade de pontos.

Use essas emissões na mesma atualização Viterbi causal. Documente que usar
marginais preditivas por janela no HMM é uma aproximação: isso não integra
conjuntamente a incerteza dos parâmetros compartilhados por toda a trajetória
e não torna o HMM inteiro um modelo bayesiano conjunto. A e pi ficam fixas.

## Comparação controlada

Reutilize as observações e parâmetros históricos salvos, sem nova busca:

```text
janela/passo = 120/60
A = [[0.99, 0.01], [0.15, 0.85]]
pi = [0.95, 0.05]
min_consecutive = 1
```

Confira os valores nos artefatos. Preserve detector, canal, FFT, conversão em
p-valor, fontes de calibração, compartilhamento global entre frequências e
regra de decisão. Não use histogramas, labels ou injeção sintética nesta tarefa.
Não execute o main de phase_transition_hmm.py, pois ele realiza search.

Mantenha uma trajetória independente por participante × frequência, na ordem
ESP → 30 → 40 → 50 → 60 → 70 dB. Janelas não atravessam arquivos; as pontuações
Viterbi e consecutivas continuam entre fases. Alvos e laterais são separados;
laterais não entram no ajuste das emissões.

Faça primeiro uma comparação in-sample controlada entre MLE e emissão
preditiva bayesiana, rotulada explicitamente como exploratória. Reproduza a
baseline salva antes de comparar. A referência é 70/88 alvos detectados e
8/88 laterais positivas, com tempo médio/mediano 734/730 s entre detectados.
Não prometa melhora de desempenho nem apresente esse teste como validação
independente. Não execute nova busca de A, pi, janela, regras ou thresholds.
Uma avaliação LOSO poderá ser proposta como trabalho posterior; não misture
seus resultados com a comparação fixa solicitada.

Reporte detecção após início de 30 dB, FP alvo em ESP separadamente e FP
lateral em todo o protocolo. Denominadores: 88 alvos e 88 laterais; por
participante, oito de cada. Preserve tempos null dos não detectados e reporte
tempo de detecção somente entre detectados, além de comparação pareada entre
trajetórias detectadas pelos dois métodos. Inclua métricas por participante e
frequência; não confunda FP lateral com FP combinado histórico.

## Verificação e entrega

- Separe código e artefatos novos dos resultados históricos; use nomes claros
  como results/bayesian_beta_hmm.json e arquivos próprios de amostras.
- Verifique recuperação de parâmetros em dados simulados, convergência das
  cadeias, tamanho efetivo das amostras e divergências quando aplicável.
- Verifique normalização das densidades, estabilidade perto de 0 e 1,
  cálculo preditivo em log-espaço e causalidade da inferência por prefixos.
- Preserve e registre a proveniência dos dados e configurações, quantidades
  de arquivos/janelas por estado, intervalos de épocas e metadados reais.
- Discuta a dependência entre janelas sobrepostas e observações do mesmo
  participante: a verossimilhança fatorizada é uma aproximação que pode
  subestimar a incerteza. Não trate número de janelas como número de
  observações independentes comprovadas.
- Atualize docs/CONTEXTO_GET_MATRIX.md e a documentação dos módulos afetados,
  distinguindo o método mantido da alternativa experimental.
- Execute verificações pertinentes e as exigidas pelas instruções locais
  quando os respectivos scripts forem alterados. Preserve alterações não
  relacionadas na árvore de trabalho.

Ao finalizar, explique o modelo implementado, as prioris escolhidas e sua
justificativa, os diagnósticos posteriores, a comparação com MLE e os comandos
para reproduzir. Mostre separadamente mudanças na emissão e seus efeitos nas
decisões do HMM. Não afirme validação clínica nem substitua o método atual sem
uma decisão explícita posterior do usuário.
