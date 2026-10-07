# `src/early_detection_rayleigh_hmm.py`

## Responsabilidade

Compara causalmente o tempo de detecção do HMM Rayleigh de emissão contínua e
do Rayleigh acumulado, mantendo FP lateral global máximo de 5%.

## Temporalidade

O HMM usa apenas o delta terminal disponível a cada janela. Não aplica
backtracking sobre observações futuras para declarar um tempo passado. A
primeira satisfação da regra de consecutivas/percentual encerra o exame.

Rayleigh é recalculado usando o prefixo completo de épocas no final de cada
janela. Seu alpha sequencial é buscado entre 0,05 e 0,0001 porque múltiplas
consultas aumentam FP. Uma época equivale a um segundo nos arquivos atuais.

## Espaço avaliado

- janelas: 10, 20, 30, 60, 90, 120 e 180 épocas;
- passos: metade da janela e janela inteira;
- Rayleigh apenas;
- emissões Beta reais de `continuous_rayleigh_hmm.py`;
- A, distribuição inicial e regra temporal escolhidas in-sample;
- FP lateral global máximo de 5%.

## Saída e interpretação

`results/early_detection_rayleigh_hmm.json` guarda taxas, tempo entre casos
detectados, duração do exame incluindo não detectados e comparação pareada das
frequências. Configurações com janela maior que o arquivo mantêm o caso no
denominador, mas não podem detectá-lo.

O melhor global usa 60/60: HMM 43,64% de detecção e 4,77% de FP; Rayleigh
51,59% e 4,77%. Ambos tiveram mediana de 120 s, e Rayleigh teve menor duração
média do exame. Em 180/90 o HMM mostrou uma vantagem exploratória, mas com FP
mais próximo do teto. Todo o ajuste e avaliação ainda usa os mesmos
participantes, portanto não estima generalização.
