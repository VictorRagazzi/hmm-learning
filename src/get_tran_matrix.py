"""Calcula A pela duração: N-1 permanências e uma saída implícita por arquivo."""

import numpy as np

import config
from dados import carregar_dados, salvar_json


def construir_matriz_transicao(dados, tamanho, passo):
    if tamanho < 10 or passo < 1:
        raise ValueError("Janela >=10 e passo >=1")
    resumo = {estado: {"arquivos": [], "janelas": 0, "autopassagens": 0}
              for estado in config.ESTADOS}
    for fases in dados.values():
        for nivel, arquivo in fases.items():
            estado = "Ausente" if nivel == "ESP" else "Presente"
            n = max(0, (arquivo["n_epocas"] - tamanho) // passo + 1)
            resumo[estado]["arquivos"].append({"arquivo": arquivo["arquivo"], "janelas": n})
            resumo[estado]["janelas"] += n
            resumo[estado]["autopassagens"] += max(0, n - 1)
    permanencias = []
    for estado in config.ESTADOS:
        if resumo[estado]["janelas"] == 0:
            raise ValueError(f"Sem janelas para calcular A({estado})")
        permanencias.append(resumo[estado]["autopassagens"] / resumo[estado]["janelas"])
    ausente, presente = permanencias
    matriz = [[ausente, 1 - ausente], [1 - presente, presente]]
    return {"estados": config.ESTADOS, "matriz": matriz, "janela": tamanho,
            "passo": passo, "resumo": resumo,
            "metodo": "Duração: N-1 autopassagens e uma saída implícita por arquivo; heurística."}


if __name__ == "__main__":
    dados = carregar_dados()
    resultado = construir_matriz_transicao(
        dados, config.WINDOW_SIZE_EPOCHS, config.WINDOW_STEP_EPOCHS)
    salvar_json(resultado, config.RESULTS_DIR / "transition_matrix.json")
    print("A heurística (linhas/colunas: Ausente, Presente):")
    print(np.asarray(resultado["matriz"]))
    print("Janelas por estado:", {s: r["janelas"] for s, r in resultado["resumo"].items()})
