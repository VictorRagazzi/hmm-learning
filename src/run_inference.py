"""Executa um método de inferência MPE e registra o resultado em JSONL."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from viterbi import A, B, observations, pi, states, symbols, mpe_inference


def run_inference(method="mpe"):
    """Executa o método escolhido com o exemplo fixo e registra a execução."""
    if method == "mpe":
        print("Observações:", [symbols[index] for index in observations])
        path, score = mpe_inference(pi, A, B, observations)
    else:
        raise ValueError(f"Método não disponível: {method}")

    print("\nMétodo:", method)
    print("Estados escolhidos (índices):", path)
    print("Sequência de estados:", [states[index] for index in path])
    print(f"Score do caminho: {score:.6e}")

    # Uma linha por execução permite adicionar métodos e comparar resultados
    # sem reescrever os registros anteriores. Guardamos também o HMM usado.
    log_path = Path(__file__).parents[1] / "logs" / "inference.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "model": {"states": states, "symbols": symbols, "pi": pi, "A": A, "B": B},
        "observations": observations,
        "path": path,
        "score": score,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("Log salvo em:", log_path)
    return path, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa inferência MPE no HMM de exemplo.")
    parser.add_argument("--method", default="mpe", help="Método de inferência (atual: mpe).")
    args = parser.parse_args()
    run_inference(args.method)
