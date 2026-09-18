"""Teste de perda de amostras para os 21 pares EuroBaVar.

Segue as chamadas MATLAB com RR como sinal padrão. Para reproduzir a versão
com pressão arterial média, execute com --signal MAP.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from mayer_waves import (
    FAIXA_MAYER_HZ,
    FP_PERCENT,
    NUM_IMF,
    SUMA,
    carregar_arquivo,
    classificador,
    listar_arquivos,
)


PERDAS_PERCENTUAIS = tuple(range(0, 71, 5))


def pares_de_arquivos():
    """Ordena cada par de arquivos por série e paciente."""
    por_id = {}
    for arquivo in listar_arquivos():
        par = por_id.setdefault(arquivo["id"], {})
        if arquivo["posicao"] in par:
            raise ValueError(f"posição duplicada no paciente {arquivo['id']}")
        par[arquivo["posicao"]] = arquivo
    if not por_id:
        raise ValueError("nenhum arquivo beat-to-beat encontrado em data/")
    for id_paciente, par in por_id.items():
        if set(par) != {"S", "L"}:
            raise ValueError(f"par standing/lying incompleto: {id_paciente}")
    return [(id_paciente, por_id[id_paciente]) for id_paciente in sorted(por_id)]


def pvalor_midp_vs_referencia(rotulos_perda, rotulos_referencia):
    """testcholdout(perda, referência, referência), padrão mid-p bilateral.

    A segunda predição coincide com o rótulo verdadeiro em todos os casos.
    Assim, se k classificações mudam, o mid-p de McNemar é 2**(-k).
    """
    perda = np.asarray(rotulos_perda, dtype=bool)
    referencia = np.asarray(rotulos_referencia, dtype=bool)
    if perda.shape != referencia.shape or perda.ndim != 1:
        raise ValueError("os vetores de rótulos devem ser unidimensionais e pareados")
    mudancas = int(np.count_nonzero(perda != referencia))
    return float(2.0 ** (-mudancas))


def executar_teste(tipo_sinal="RR", seed=0, fp=FP_PERCENT,
                   range_f=FAIXA_MAYER_HZ, num_imf=NUM_IMF, suma=SUMA):
    """Executa uma realização reproduzível do teste de perda de dados."""
    pares = pares_de_arquivos()
    rng = np.random.default_rng(seed)
    # Eixo: porcentagem, paciente, posição (0=em pé, 1=supino).
    rotulos = np.zeros((len(PERDAS_PERCENTUAIS), len(pares), 2), dtype=bool)
    detalhes = []

    for indice_paciente, (id_paciente, par) in enumerate(pares):
        for indice_posicao, posicao in enumerate(("S", "L")):
            arquivo = par[posicao]
            t, sinal = carregar_arquivo(arquivo["caminho"], tipo_sinal)
            # A mesma ordem aleatória é usada em todos os níveis. A perda é
            # progressiva e t mantém os instantes originais das amostras.
            ordem_remocao = rng.permutation(len(sinal))
            for indice_perda, porcentagem in enumerate(PERDAS_PERCENTUAIS):
                n_removidas = int(np.floor(len(sinal) * porcentagem / 100 + 0.5))
                manter = np.sort(ordem_remocao[n_removidas:])
                if len(manter) < 3:
                    raise ValueError(f"amostras insuficientes em {arquivo['nome']}")
                resultado = classificador(
                    sinal[manter], t[manter], fp=fp, range_f=range_f,
                    num_imf=num_imf, suma=suma,
                )
                rotulos[indice_perda, indice_paciente, indice_posicao] = resultado["detectou"]
                detalhes.append({
                    "paciente": id_paciente,
                    "posicao": "em_pe" if posicao == "S" else "supino",
                    "perda_percentual": porcentagem,
                    "amostras_originais": len(sinal),
                    "amostras_restantes": len(manter),
                    "detectou": int(resultado["detectou"]),
                    "pico": resultado["pico"],
                    "limiar_aproximado": resultado["limiar"],
                })

    resumo = []
    for indice_perda, porcentagem in enumerate(PERDAS_PERCENTUAIS):
        em_pe, supino = rotulos[indice_perda, :, 0], rotulos[indice_perda, :, 1]
        base_em_pe, base_supino = rotulos[0, :, 0], rotulos[0, :, 1]
        resumo.append({
            "perda_percentual": porcentagem,
            "em_pe": int(np.count_nonzero(em_pe)),
            "supino": int(np.count_nonzero(supino)),
            "mudancas_em_pe": int(np.count_nonzero(em_pe != base_em_pe)),
            "mudancas_supino": int(np.count_nonzero(supino != base_supino)),
            "pvalor_em_pe": pvalor_midp_vs_referencia(em_pe, base_em_pe),
            "pvalor_supino": pvalor_midp_vs_referencia(supino, base_supino),
        })
    return resumo, detalhes


def limite_antes_da_mudanca(resumo, campo_pvalor, alpha=0.05):
    """Devolve a última perda antes da primeira rejeição, e a primeira rejeição."""
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar entre 0 e 1")
    for indice, linha in enumerate(resumo):
        if linha[campo_pvalor] < alpha:
            anterior = resumo[indice - 1]["perda_percentual"] if indice else None
            return anterior, linha["perda_percentual"]
    return resumo[-1]["perda_percentual"], None


def salvar_resultados(resumo, detalhes, pasta_saida):
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    for nome, linhas in (("perda_dados_resumo.csv", resumo),
                         ("perda_dados_detalhes.csv", detalhes)):
        with (pasta_saida / nome).open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=linhas[0].keys())
            escritor.writeheader()
            escritor.writerows(linhas)

    x = np.arange(len(resumo))
    largura = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - largura / 2, [r["em_pe"] for r in resumo], largura,
           label="Posição em pé")
    ax.bar(x + largura / 2, [r["supino"] for r in resumo], largura,
           label="Posição em supino")
    ax.set_xticks(x, [str(r["perda_percentual"]) for r in resumo])
    ax.set_xlabel("Porcentagem de perda de dados (%)")
    ax.set_ylabel("Número de pacientes detectados")
    ax.set_ylim(0, max(22, max(r["em_pe"] for r in resumo) + 1,
                       max(r["supino"] for r in resumo) + 1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta_saida / "perda_dados_mayer.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", choices=("RR", "MAP"), default="RR",
                        help="RR segue as chamadas do trecho MATLAB; MAP usa a pressão média")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp", type=float, default=FP_PERCENT)
    parser.add_argument("--range-f", nargs=2, type=float, default=FAIXA_MAYER_HZ,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--num-imf", nargs="+", type=int, default=NUM_IMF)
    parser.add_argument("--suma", type=int, default=SUMA)
    parser.add_argument("--saida", type=Path, default=Path("figs"))
    args = parser.parse_args()

    resumo, detalhes = executar_teste(
        tipo_sinal=args.signal, seed=args.seed, fp=args.fp,
        range_f=args.range_f, num_imf=args.num_imf, suma=args.suma,
    )
    salvar_resultados(resumo, detalhes, args.saida)
    print("Perda (%)  Em pé  Supino  Mudanças em pé  Mudanças supino  p em pé  p supino")
    for linha in resumo:
        print(f"{linha['perda_percentual']:>9}  {linha['em_pe']:>5}  "
              f"{linha['supino']:>6}  {linha['mudancas_em_pe']:>14}  "
              f"{linha['mudancas_supino']:>15}  {linha['pvalor_em_pe']:>7.4f}  "
              f"{linha['pvalor_supino']:>8.4f}")
    for nome, campo in (("Em pé", "pvalor_em_pe"), ("Supino", "pvalor_supino")):
        anterior, primeira = limite_antes_da_mudanca(resumo, campo)
        if primeira is None:
            print(f"{nome}: nenhuma diferença significativa até {anterior}% de perda")
        else:
            print(f"{nome}: última perda antes da primeira diferença significativa: "
                  f"{anterior}%; primeira diferença: {primeira}%")
    print(f"Arquivos salvos em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
