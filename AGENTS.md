# Contexto

Você vai implementar um pipeline de detecção de resposta evocada auditiva
(ASSR - Auditory Steady-State Response) em sinais de EEG, usando uma
arquitetura chamada "Semantic Temporal Compilation": um HMM binário
{repouso, resposta} cuja tabela de transição/emissão não é aprendida
estatisticamente, mas compilada offline por uma LLM que raciocina
semanticamente sobre cada combinação (estado_anterior, observação).

Isso é inspirado em trabalho anterior onde uma LLM substituiu inferência MPE
em Redes Bayesianas: em vez de usar CPTs numéricas, o valor mais plausível
de uma variável era decidido por raciocínio semântico da LLM sobre o
contexto causal, e essa decisão era pré-compilada em uma tabela de lookup.

# Etapa 1 — Investigação (fazer ANTES de qualquer implementação)

Os dados estão em arquivos .mat contendo ondas de EEG coletadas com janelas
de estimulação em diferentes intensidades (40, 60, 80 dB e ESP — estimulação
sem som, usada como controle negativo). O objetivo é detectar se existe
resposta evocada na banda de frequências 82:2:96 Hz (ou seja, 82, 84, 86,
88, 90, 92, 94, 96 Hz).

Antes de implementar, investigue os arquivos .mat e leia código MATLAB (.m)
relacionado, se existir, para responder:

1. Como o sinal está estruturado (taxa de amostragem, duração das
   gravações, quantidade de canais, formato das janelas de estimulação).
2. Se já existe algum processamento estatístico de referência para detectar
   resposta na banda 82-96 Hz (F-test, Hotelling T², magnitude espectral,
   SNR, etc.) — seja em código MATLAB existente ou em literatura de ASSR
   padrão.
3. Proponha um método concreto de calcular, para uma janela temporal do
   sinal contínuo, um valor estatístico agregado que resuma a evidência de
   resposta na banda 82-96 Hz inteira (não por frequência individual —
   queremos UM valor por janela, agregando a banda toda).
4. Relate o que encontrou antes de prosseguir para a implementação.

# Etapa 2 — Arquitetura (o que implementar)

## Estados ocultos
Dois estados: `repouso` e `resposta`. A lista de estados deve ser uma
variável declarada no topo do código (ex: `ESTADOS = ["repouso", "resposta"]`),
não hardcoded em lógica espalhada — facilita estender para mais estados no
futuro se necessário.

## Observação
A observação de cada janela temporal vem do processamento de sinal
tradicional decidido na Etapa 1 (ex: um valor de F-test ou SNR agregado na
banda 82-96 Hz). Esse valor contínuo deve ser discretizado em categorias
ordinais de evidência, por exemplo:

    NIVEIS_OBSERVACAO = ["ausente", "fraco", "forte", "muito_forte"]

Importante: a função de discretização deve ser genérica em relação ao
número de categorias em NIVEIS_OBSERVACAO — se eu mudar essa lista para ter
3 ou 6 níveis, a função de discretização (ex: dividir a faixa de
p-valores/SNR observada nos dados em N faixas, ou usar cortes definidos por
percentil) deve se adaptar automaticamente, sem precisar editar a lógica de
discretização. Não hardcode "4 categorias" na função.

## Compilação offline da tabela semântica (uso de LLM)

Para cada estado em ESTADOS, fazer UMA chamada à LLM contendo:

  - o estado anterior sendo considerado;
  - a lista completa de níveis de observação possíveis (NIVEIS_OBSERVACAO);
  - "expert notes" sobre esse estado (ver abaixo);
  - pedir que a LLM decida, para CADA nível de observação da lista, qual é
    o próximo estado mais plausível (repouso ou resposta).

Ou seja: número de chamadas à LLM = número de estados (2, nesse caso), não
uma chamada por combinação (estado, observação). Cada chamada deve cobrir
todos os níveis de observação de uma vez, para que a LLM decida com
consistência relativa entre os níveis, evitando fronteiras arbitrárias
entre categorias vizinhas.

Antes da compilação da tabela, gerar "expert notes" para cada estado — uma
única vez, também via LLM — descrevendo o conhecimento de domínio relevante
sobre transições em ASSR. Por exemplo, para o estado `resposta`: sob que
condições uma resposta seguiria em curso vs. voltaria a repouso; que peso
dar a uma observação fortemente ausente vs. uma observação ambigua/fraca;
etc. Essas notas servem para MODULAR a decisão da LLM na etapa de
compilação, não para determiná-la rigidamente — a observação atual não deve
ser ignorada só porque o estado anterior "sugeriria" outra coisa.

O resultado da compilação é uma tabela/dicionário:

    tabela[estado_anterior][nivel_observacao] = proximo_estado

## Inferência online

Depois de compilada a tabela, a inferência sobre uma sequência de janelas
do sinal é pura consulta a essa tabela — SEM nenhuma chamada à LLM:

    estado = "repouso"  # estado inicial
    for janela in sequencia_de_janelas:
        observacao_continua = calcular_estatistica_banda(janela)
        nivel = discretizar(observacao_continua, NIVEIS_OBSERVACAO)
        estado = tabela[estado][nivel]
        registrar(estado)

O veredito final da sequência (detectou ou não detectou resposta) deve ser
derivado da sequência de estados resultante — defina um critério simples e
explícito (ex: "resposta" ocorreu em pelo menos X% das janelas, ou por N
janelas consecutivas) e deixe esse critério também como parâmetro
configurável no topo do código, não fixo no meio da lógica.

## Validação

Cada intensidade de estimulação (40, 60, 80 dB, ESP) é um experimento/
sequência independente — rode o pipeline separadamente para cada uma. Como
ground truth aproximado: espera-se NÃO detecção em estimulação ESP
(controle) e espera-se detecção nos testes com estímulo real. Use isso para
uma validação de sanidade do pipeline, não como treinamento.

# Estilo de código

- Funções claras e diretas, mesmo que grandes — não fragmentar em excesso.
- Poucos arquivos.
- Sem otimizações prematuras ou padrões de "clean code" que compliquem
  legibilidade — priorize entendimento total do fluxo sobre elegância.
- Toda constante/configuração relevante (ESTADOS, NIVEIS_OBSERVACAO,
  critério de decisão final, etc.) declarada no topo do arquivo principal,
  não espalhada implicitamente pelo código.

# O que NÃO fazer

- Não chamar a LLM durante a inferência online — isso quebra a premissa
  central da arquitetura (custo online deve ser zero chamadas).
- Não compilar célula a célula (uma chamada por combinação estado×
  observação) — compilar por estado, cobrindo todos os níveis de
  observação em uma única chamada.
- Não hardcode o número de níveis de observação em nenhuma lógica de
  discretização ou de prompt de compilação.