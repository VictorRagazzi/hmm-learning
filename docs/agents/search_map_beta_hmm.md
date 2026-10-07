# Busca de parâmetros Beta contínua MLE/MAP

Implementação autorizada em 05/10/2026: `src/search_map_beta_hmm.py`.
O search discreto original não era compatível com emissão contínua/MAP e
não variava janelamento. Foi removida uma linha isolada `j` que causava
NameError no import/execução. O fluxo antigo permanece disponível sem a flag.

**Decisão vigente posterior:** o usuário pediu retirar LOSO para explorar
viabilidade com vazamento deliberado. O padrão atual é **in-sample**, ajustando
B, pesquisando parâmetros e avaliando os mesmos onze pacientes. Os resultados
LOSO anteriores abaixo são históricos preservados. Ver Seção 24 do contexto.

## Execução

```bash
.venv/bin/python src/search_hmm_parameters.py --continuous-map
.venv/bin/python src/search_hmm_parameters.py --continuous-map --resume
# Para executar explicitamente o protocolo LOSO histórico:
.venv/bin/python src/search_hmm_parameters.py --continuous-map --validation loso
# Entrada direta equivalente:
.venv/bin/python src/search_map_beta_hmm.py --resume
```

Prioris e corte vêm dos valores atuais de `map_beta_hmm_loso.py`, ou podem
ser fornecidos: `--mu-prior 2 4 --kappa-prior 2 1 --present-min-db 50`.
Esses parâmetros NÃO são pesquisados nesta tarefa. O bruto usa alpha fixo .005.
Os valores individuais MIN_CONSECUTIVE/MIN_PERCENT do módulo MAP não limitam
a busca; as grades estão no topo de `search_map_beta_hmm.py`.

`--output`/`--output-dir` permitem variantes independentes. `--windows 60 120`
restringe tamanhos; `--step-modes half_overlap no_overlap` define os passos.
`--quick` é diagnóstico de grade reduzida, não resultado da busca completa;
use saída separada. Checkpoint atômico por dobra; `--resume` valida grade,
prioris, corte, hashes dos dados e código e retoma somente dobras completas.
Uma dobra interrompida é recalculada. Resultados não mudam valores no módulo
MAP automaticamente nem sobrescrevem tabelas discretas.

`--validation in-sample` (padrão) não executa dobras: faz somente a busca global
em todos e avalia essa mesma amostra. Saídas automáticas próprias:
`results/search_map_beta_hmm_in_sample.json` e
`outputs/search_map_beta_hmm_in_sample/`. `--validation loso` usa as saídas
históricas sem `_in_sample`; os arquivos anteriores permanecem preservados até
execução explícita nesse modo. O checkpoint valida também o modo; não retomar
checkpoint LOSO como in-sample nem reutilizar checkpoint anterior a mudança
de código. Use outra saída ou execute sem `--resume` para um ajuste novo.

## Grade completa e seleção

- Janelas: 10,20,30,60,90,120,180 épocas.
- Passos: metade da janela ou janela inteira (14 combinações).
- Consecutivas: 1–12; percentuais: .01,.025,.05,.075,.09,.10,.20,.30,.40,.50,
  .60,.70,.80,.90,1; modos OR/AND.
- Inclui consecutivas apenas e porcentagem apenas: 387 regras.
- P(Ausente) inicial: .99,.95,.90,.85,.80,.75,.70,.65,.60,.55,.50.
- A: produto cartesiano desses mesmos 11 valores nas duas autopermanências,
  mais a A heurística salva e a matriz atual do módulo MAP, deduplicadas.
  Na árvore atual são 122 A e 1.342 combinações A/pi.
- São 7.270.956 candidatos completos por método HMM/dobra. Os dois HMMs
  fazem busca separada. Bruto pesquisa janelas/passos/regras (5.418 candidatos).

Maximiza acurácia balanceada lateral, sujeito a **FP lateral de treino ≤5%**
nos três métodos. Desempata por maior detecção, menor FP e ordem determinística
da grade. Tempo não é objetivo de desempate. Este teto é lateral no protocolo
inteiro, não o FP combinado histórico. FP alvo ESP é reportado separadamente.
O FP externo nunca exclui/reajusta uma dobra. Se não houver candidato, falha
explicitamente; não remove silenciosamente o participante.

Cada uma das 11 dobras retém uma pessoa inteira. Nos outros dez:
1. recalcula p-valores para cada janelamento;
2. ajusta Beta Ausente MLE em ESP e Beta Presente MLE/MAP no corte configurado;
3. escolhe janela/passo, A, pi e regra somente no treino;
4. aplica o vencedor ao paciente retido uma única vez.

Essa descrição de dobras vale somente para `--validation loso`. No padrão
in-sample nenhum paciente é retido, B usa todas as janelas elegíveis e a
avaliação usa todos os mesmos pacientes. Não há onze pares μ/κ: existe um
modelo global por estado/configuração/método. Arquivos mais curtos que a
janela não contribuem para o ajuste, mas ficam no protocolo de avaliação.

A emissão é global entre frequências por estado/configuração/dobra. Ausente é
compartilhado por MLE/MAP; Presente é uma Beta única (nunca mistura, labels ou
senoide). As frequências laterais não calibram B, mas entram no controle de FP
de treino, como no search original. FFT/detector/metadados reais preservados.

Após as dobras, uma busca adicional em todos os 11 pacientes salva a melhor
configuração para uso (`deployment_fit`). Não confundir essa configuração com
os onze modelos externos nem sua métrica de treino com desempenho LOSO.

## Causalidade e otimização

Protocolo ESP→30→40→50→60→70 por paciente/frequência, janela dentro do arquivo,
delta e consecutivas continuando entre fases. Porcentagem usa positivos no
prefixo desde ESP dividido pelas janelas observadas até o instante corrente.
Detecção alvo só depois do início de 30; lateral em qualquer fase.

EEG/FFT carregados uma vez; observações cacheadas por janelamento. Betas
ajustadas uma vez por treino/janelamento. Viterbi em lotes de A/pi, sem
backtracking; máscaras impedem padding de gerar eventos. Regras reaproveitam
estados. Para AND, o máximo de porcentagem só é calculado em prefixos que
também satisfazem consecutivas: não juntar máximos de instantes diferentes.
Vencedores são reaplicados pelo avaliador escalar com assertivas nos eventos
de treino. Não modifica globais para inferir com cada A/pi.

Onze testes passam nas suites MAP/search, incluindo Viterbi/prefixos escalares
versus lote, todas as regras versus cálculo escalar, sequências vazias/padding,
validade da grade e invariantes do MAP. Verificação:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test*map_beta*.py'
.venv/bin/python -m py_compile src/search_hmm_parameters.py src/search_map_beta_hmm.py src/map_beta_hmm_loso.py
```

## Artefatos e limites

- `results/search_map_beta_hmm.json`: especificação/hashes, checkpoint por
  dobra, indivíduos treino/teste, modelos Beta e parâmetros escolhidos,
  contagens de candidatos/rejeições, métricas de treino/teste, eventos externos,
  comparação por pessoa/frequência e ajuste final com todos.
- `outputs/search_map_beta_hmm/index.html`, `README.md`, `comparacao.csv`:
  tabela dos três métodos, configurações finais e por dobra.
- `tests/test_search_map_beta_hmm.py`: contrato do motor de busca.

Melhor = melhor na grade, não ótimo contínuo global ou Baum–Welch. O search
agora seleciona parâmetros temporais dentro do treino LOSO, corrigindo a
limitação de reutilizar vencedores históricos no teste fixo anterior. Ainda
é experimento com prioris/corte/desenho discutidos a partir desta base, proxy
Presente real e dependência de janelas; não validação clínica. ≤5% de treino
não garante ≤5% no teste. Comparações externas não devem escolher prioris
automaticamente. MLE/MAP com configurações próprias não isolam apenas o efeito
da priori; o experimento fixo anterior mantém essa comparação isolada.

## Busca completa concluída e resultados

**Resultados LOSO históricos.** A execução in-sample vigente está abaixo.

Onze dobras mais reajuste final em todos, com corte Presente ≥50 dB,
μ~Beta(2,4), κ~Gamma(2,1). Foram avaliados 174.567.960 candidatos/regra:
87.251.472 em cada HMM e 65.016 no bruto (inclui reajuste final).

| Método | Detecção /88 | FP lateral /88 | FP alvo ESP | BA lateral | Média/mediana desde 30 entre detectados (s) |
|---|---:|---:|---:|---:|---:|
| Bruto | 55 (62,50%) | 3 (3,41%) | 0 | 79,55% | 860,4/900 |
| HMM MLE | 61 (69,32%) | 15 (17,05%) | 0 | 76,14% | 732,9/705 |
| HMM MAP | 61 (69,32%) | 15 (17,05%) | 0 | 76,14% | 732,9/705 |

Eventos externos MLE/MAP idênticos. Os HMMs não controlaram FP externo ≤5% e
tiveram BA menor que bruto. Não prometer ganho MAP nem interpretar diferença
de médias entre conjuntos de detectados distintos como antecipação pareada.

Reajuste final MAP/MLE em todos: janela/passo 90/45, pi=[.9,.1],
A=[[.8,.2],[.45,.55]], regra 3 consecutivas OR 60% no prefixo desde ESP.
Treino: 66/88 detecção, 4/88 FP lateral, BA=85,23%; não é resultado LOSO.
Bruto final 120/60 com somente fração mínima .025, consecutivas=None.

Betas finais MAP: Ausente MLE a=1.1698079195,b=1.0134264024,n=128;
Presente MAP a=.3035875494,b=.9561090720,μ=.2410005268,κ=1.2596966214,n=672.
O JSON preserva as Betas e hiperparâmetros específicos de cada dobra.
As constantes atuais do módulo de inferência não foram substituídas.
Auditoria final reproduziu todos os eventos externos, calibrações vencedoras,
exclusões completas dos retidos e hashes/metadados dos EEG reais.

## Resultado in-sample vigente

Execução completa, mesmos prioris/corte/grade: 14.547.330 avaliações (7.270.956
por HMM e 5.418 bruto). Sem LOSO, 11 sujeitos no ajuste e avaliação, 88 alvos
e 88 laterais. Resultado reproduz exatamente o `deployment_fit` da busca
anterior, agora com eventos, métricas por pessoa/frequência e tabela próprios.

| Método | Detecção /88 | FP lateral /88 | FP alvo ESP | BA lateral | Média/mediana desde 30 entre detectados (s) |
|---|---:|---:|---:|---:|---:|
| Bruto | 58 (65,91%) | 3 (3,41%) | 0 | 81,25% | 824,5/810 |
| HMM MLE | 66 (75,00%) | 4 (4,55%) | 0 | 85,23% | 784,6/737,5 |
| HMM MAP | 66 (75,00%) | 4 (4,55%) | 0 | 85,23% | 784,6/737,5 |

MLE/MAP: mesmos eventos e tempos. Configuração HMM 90/45, A=[[.8,.2],[.45,.55]],
pi=[.9,.1], 3 consecutivas OR fração .60, Presente ≥50 dB. Bruto 120/60,
fração .025 apenas, alpha=.005. Betas MAP iguais às finais acima: Ausente
128 janelas (oito ESP suficientemente longos); Presente 672 janelas reais.
Todos os onze pacientes continuam na avaliação.

FP in-sample caiu de 17,05% LOSO para 4,55%; isso não demonstra melhora de
generalização. Emissão, escolha de parâmetros e medição compartilham os dados;
registrar como demonstração exploratória de viabilidade no conjunto ajustado.
Nenhum ganho específico de MAP sobre MLE foi demonstrado. Não fazer buscas
extras de prioris/corte/frequência automaticamente.

Doze testes passaram, incluindo que in-sample usa todos, não cria dobras,
avalia o mesmo conjunto e rejeita retomar com outro modo. Retomada completa
in-sample verificada; comparações reproduzem o ajuste global histórico.
O comando `map_beta_hmm_loso.py` continua sendo o experimento fixo LOSO antigo;
para a etapa vigente usar o search com `--continuous-map`.
