"""Exemplo didático do algoritmo de Viterbi para um HMM discreto."""

# Os índices destas listas identificam estados e observações nas matrizes.
states = ["Sol", "Chuva"]
symbols = ["caminhar", "comprar", "limpar"]

# pi[i] = P(estado inicial = i).
pi = [0.6, 0.4]

# A[i][j] = P(próximo estado = j | estado atual = i).
A = [
    [0.7, 0.3],
    [0.4, 0.6],
]

# B[i][k] = P(observação = k | estado atual = i).
B = [
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1],
]

# A sequência fixa permite comparar o resultado com contas feitas à mão.
observations = [1, 1, 1, 0 , 1, 1, 1, 1, 2, 1, 1, 1]


def mpe_inference(pi, A, B, observations):
    """
    Recebe os parâmetros do HMM e uma sequência de observações (índices de B).
    Retorna a sequência de estados mais provável (MPE) e o score.
    Esta função concentra TODO o algoritmo de Viterbi.
    No futuro, pode ser substituída por uma versão que consulta um LLM.
    """
    if not observations:
        raise ValueError("A sequência de observações não pode ser vazia.")

    number_of_states = len(pi)
    number_of_times = len(observations)

    # trellis[estado][tempo] guarda a probabilidade do melhor caminho que
    # termina naquele estado, após observar os símbolos até aquele tempo.
    # backpointers[estado][tempo] guarda o estado anterior desse melhor caminho.
    trellis = [[0.0] * number_of_times for _ in range(number_of_states)]
    backpointers = [[None] * number_of_times for _ in range(number_of_states)]

    # Inicialização: no tempo 0, só há a probabilidade inicial do estado
    # multiplicada pela probabilidade de emitir a primeira observação.
    for state in range(number_of_states):
        trellis[state][0] = pi[state] * B[state][observations[0]]

    # Recursão: para cada estado atual, experimentamos todos os estados
    # anteriores. Cada candidato multiplica a melhor probabilidade anterior
    # pela transição e pela emissão da observação atual.
    for time in range(1, number_of_times):
        observation = observations[time]
        for state in range(number_of_states):
            best_previous_state = 0
            best_probability = (
                trellis[0][time - 1] * A[0][state] * B[state][observation]
            )

            for previous_state in range(1, number_of_states):
                probability = (
                    trellis[previous_state][time - 1]
                    * A[previous_state][state]
                    * B[state][observation]
                )
                if probability > best_probability:
                    best_probability = probability
                    best_previous_state = previous_state

            trellis[state][time] = best_probability
            backpointers[state][time] = best_previous_state

    # Terminação: escolhemos o estado final com a maior probabilidade
    # acumulada. Esse valor é o score do caminho inteiro, não uma soma.
    final_state = 0
    score = trellis[0][number_of_times - 1]
    for state in range(1, number_of_states):
        if trellis[state][number_of_times - 1] > score:
            score = trellis[state][number_of_times - 1]
            final_state = state

    # Backtracking: partimos do melhor estado final e seguimos os ponteiros
    # para trás até reconstruir o caminho ótimo desde o tempo 0.
    path = [0] * number_of_times
    path[-1] = final_state
    for time in range(number_of_times - 1, 0, -1):
        path[time - 1] = backpointers[path[time]][time]

    # Exibimos o trellis completo. O asterisco marca a célula escolhida
    # no caminho ótimo em cada coluna de tempo.
    print("Trellis (probabilidade; * = caminho escolhido):")
    print(" " * 12 + "".join(f"{'t=' + str(time):>15}" for time in range(number_of_times)))
    for state in range(number_of_states):
        cells = [
            f"{trellis[state][time]:.6e}{'*' if path[time] == state else ' '}"
            for time in range(number_of_times)
        ]
        print(f"Estado {state:<5}" + "".join(f"{cell:>15}" for cell in cells))

    # Cada ponteiro indica a linha de origem da melhor transição.
    # No tempo 0 não existe estado anterior, por isso mostramos '-'.
    print("\nBackpointers (estado anterior escolhido):")
    print(" " * 12 + "".join(f"{'t=' + str(time):>15}" for time in range(number_of_times)))
    for state in range(number_of_states):
        cells = [
            f"{backpointers[state][time] if time > 0 else '-':>15}"
            for time in range(number_of_times)
        ]
        print(f"Estado {state:<5}" + "".join(cells))

    return path, score


if __name__ == "__main__":
    print("Observações:", [symbols[index] for index in observations])
    path, score = mpe_inference(pi, A, B, observations)
    print("\nEstados escolhidos (índices):", path)
    print("Sequência de estados:", [states[index] for index in path])
    print(f"Score do caminho: {score:.6e}")
