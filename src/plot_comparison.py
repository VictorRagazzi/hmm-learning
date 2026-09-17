"""Gera figuras comparativas para métodos de inferência MPE.

Este script lê os registros JSONL criados por run_inference.py e compara
cada método contra o método "mpe", que aqui representa o Viterbi clássico.
As figuras são salvas em ../figs para facilitar inspeção depois.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hmm-learning-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(__file__).resolve().parent / "logs" / "inference.jsonl"
FIGS_DIR = PROJECT_ROOT / "figs"


def load_records():
    """Carrega uma linha JSON por execução de inferência."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Nenhum log encontrado em {LOG_PATH}. Rode primeiro: "
            "uv run python src/run_inference.py --method mpe"
        )

    records = []
    with LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"O arquivo {LOG_PATH} existe, mas está vazio.")

    return records


def calculate_viterbi_trellis(pi, A, B, observations):
    """Replica o Viterbi apenas para medir margem/ambiguidade por posição."""
    number_of_states = len(pi)
    number_of_times = len(observations)
    trellis = [[0.0] * number_of_times for _ in range(number_of_states)]

    # Inicialização: probabilidade inicial vezes emissão observada em t=0.
    for state in range(number_of_states):
        trellis[state][0] = pi[state] * B[state][observations[0]]

    # Recursão: melhor caminho até cada estado atual, como no Viterbi normal.
    for time in range(1, number_of_times):
        observation = observations[time]
        for state in range(number_of_states):
            best_probability = trellis[0][time - 1] * A[0][state] * B[state][observation]
            for previous_state in range(1, number_of_states):
                probability = (
                    trellis[previous_state][time - 1]
                    * A[previous_state][state]
                    * B[state][observation]
                )
                if probability > best_probability:
                    best_probability = probability
            trellis[state][time] = best_probability

    return trellis


def save_bar_chart(path, title, labels, values, ylabel, ylim=None):
    """Desenha barras simples para métricas agregadas por método."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Método")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    records = load_records()
    FIGS_DIR.mkdir(exist_ok=True)

    # Agrupamos por sequência observada. Dentro de cada grupo, o método "mpe"
    # vira a referência de Viterbi para calcular Hamming e erros por posição.
    records_by_observations = defaultdict(list)
    for record in records:
        records_by_observations[tuple(record["observations"])].append(record)

    comparisons = []
    position_errors = defaultdict(lambda: defaultdict(int))
    position_counts = defaultdict(lambda: defaultdict(int))
    ambiguity_points = defaultdict(list)
    length_accuracy_points = defaultdict(list)
    cost_points = defaultdict(list)

    for observation_key, group in records_by_observations.items():
        baseline_candidates = [record for record in group if record["method"] == "mpe"]
        if not baseline_candidates:
            continue

        # Se houver várias execuções do mpe para a mesma sequência, usamos a
        # última do log como referência mais recente.
        baseline = baseline_candidates[-1]
        baseline_path = baseline["path"]
        model = baseline["model"]
        trellis = calculate_viterbi_trellis(
            model["pi"],
            model["A"],
            model["B"],
            list(observation_key),
        )

        # Margem pequena entre os dois melhores estados indica maior ambiguidade.
        margins_by_time = []
        for time in range(len(observation_key)):
            column = sorted((trellis[state][time] for state in range(len(trellis))), reverse=True)
            best = column[0]
            second_best = column[1] if len(column) > 1 else 0.0
            if best == 0:
                margins_by_time.append(0.0)
            else:
                margins_by_time.append((best - second_best) / best)

        number_of_states = len(model["pi"])
        sequence_length = len(observation_key)
        estimated_viterbi_ops = sequence_length * number_of_states * number_of_states

        for record in group:
            method = record["method"]
            path = record["path"]
            mismatches = [index for index, state in enumerate(path) if state != baseline_path[index]]
            accuracy = 1.0 - (len(mismatches) / len(baseline_path))

            comparisons.append(
                {
                    "method": method,
                    "accuracy": accuracy,
                    "length": sequence_length,
                    "errors": len(mismatches),
                }
            )
            length_accuracy_points[method].append((sequence_length, accuracy))

            # Tokens podem aparecer em métodos futuros. Para o Viterbi puro,
            # deixamos zero porque não houve chamada a LLM.
            usage = record.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            cost_points[method].append((estimated_viterbi_ops, total_tokens))

            for time in range(sequence_length):
                position_counts[method][time] += 1
                if time in mismatches:
                    position_errors[method][time] += 1
                    ambiguity_points[method].append((margins_by_time[time], 1))
                else:
                    ambiguity_points[method].append((margins_by_time[time], 0))

    if not comparisons:
        raise ValueError('Não encontrei registros com method="mpe" para usar como Viterbi.')

    methods = sorted({item["method"] for item in comparisons})

    # Figura 1: acurácia média, isto é, 1 - distância de Hamming normalizada.
    average_accuracy = []
    for method in methods:
        values = [item["accuracy"] for item in comparisons if item["method"] == method]
        average_accuracy.append(sum(values) / len(values))
    save_bar_chart(
        FIGS_DIR / "accuracy_by_method.png",
        "Acurácia média contra Viterbi",
        methods,
        average_accuracy,
        "Acurácia",
        ylim=(0, 1.05),
    )

    # Figura 2: taxa de erro por posição para ver início, meio e fim.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        max_position = max(position_counts[method])
        xs = list(range(max_position + 1))
        ys = [
            position_errors[method][position] / position_counts[method][position]
            for position in xs
        ]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_title("Erros por posição")
    ax.set_xlabel("Posição na sequência")
    ax.set_ylabel("Taxa de erro")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "position_errors.png", dpi=160)
    plt.close(fig)

    # Figura 3: sensibilidade à ambiguidade. Margem menor significa decisão
    # mais apertada no trellis do Viterbi.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        xs = [point[0] for point in ambiguity_points[method]]
        ys = [point[1] for point in ambiguity_points[method]]
        ax.scatter(xs, ys, alpha=0.65, label=method)
    ax.set_title("Sensibilidade à ambiguidade")
    ax.set_xlabel("Margem relativa entre melhor e segundo melhor estado")
    ax.set_ylabel("Erro contra Viterbi")
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "ambiguity_sensitivity.png", dpi=160)
    plt.close(fig)

    # Figura 4: escalabilidade com T. Com mais logs de comprimentos variados,
    # este gráfico mostra se a acurácia cai em sequências longas.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        points = sorted(length_accuracy_points[method])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_title("Escalabilidade com T")
    ax.set_xlabel("Comprimento da sequência (T)")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "scalability_T.png", dpi=160)
    plt.close(fig)

    # Figura 5: custo computacional. O eixo x estima operações do Viterbi;
    # o eixo y fica pronto para tokens quando métodos futuros registrarem usage.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        xs = [point[0] for point in cost_points[method]]
        ys = [point[1] for point in cost_points[method]]
        ax.scatter(xs, ys, alpha=0.75, label=method)
    ax.set_title("Custo computacional")
    ax.set_xlabel("Operações estimadas do Viterbi")
    ax.set_ylabel("Tokens consumidos")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "computational_cost.png", dpi=160)
    plt.close(fig)

    print("Figuras salvas em:", FIGS_DIR)
    for figure in sorted(FIGS_DIR.glob("*.png")):
        print("-", figure)


if __name__ == "__main__":
    main()
