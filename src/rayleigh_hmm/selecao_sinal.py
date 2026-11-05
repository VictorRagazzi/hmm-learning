"""Selecao barata do sinal (canal, media, PCA, rejeicao) sem HMM e sem busca.

Para cada opcao de sinal e cada (arquivo, frequencia), faz UM teste de
Rayleigh no fim, com todas as epocas validas do arquivo (alfa fixo). Metricas:
  deteccao  = alvos em 30-70 dB com p <= ALFA;
  FP lateral = laterais (todas as condicoes) com p <= ALFA;
  FP alvo ESP = alvos do ESP com p <= ALFA;
  KS ESP    = KS dos p-valores dos alvos ESP contra Uniforme(0,1) (nula);
  fracao de epocas rejeitadas e deteccao por intensidade.

Blocos: (a) cada canal isolado; (b) media/PCA de todos; (c) media/PCA dos
top-k canais de (a); (d) rejeicao de artefatos no melhor canal e na melhor
combinacao. ATENCAO, VAZAMENTO ACEITO: o ranking de canais e a escolha usam
os mesmos dados da avaliacao (in-sample); nao e estimativa de generalizacao.

Regra de escolha: maior deteccao entre as opcoes com FP lateral <= FP_MAXIMO e
KS ESP com p > KS_P_MINIMO; desempate pelo menor FP lateral. O ranking de
canais usa a mesma regra (elegiveis primeiro, depois deteccao).

Execucao (da raiz do repositorio):
    .venv/bin/python src/rayleigh_hmm/selecao_sinal.py
"""

import json
import os

import numpy as np
from scipy import stats

import dados

# ============================================================
# Configuracao
# ============================================================
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PASTA_DADOS = os.path.join(RAIZ, "data")
PASTA_RESULTADOS = os.path.join(RAIZ, "results", "rayleigh_hmm")

ALFA = 0.05
FP_MAXIMO = 0.07
KS_P_MINIMO = 0.05
TOPO_K = [2, 3, 4, 6]
FATORES_REJEICAO = [None, 1.5, 2.0, 3.0]
AVISO_RAZAO_VARIANCIA = 0.05  # media com variancia < 5% da dos canais -> aviso

NOMES = dados.nomes_eletrodos(PASTA_DADOS)
N_CANAIS = len(NOMES)


# ============================================================
# Avaliacao de uma opcao de sinal
# ============================================================
def nova_opcao(bloco, nome, combinacao, canais, rejeicao_fator=None):
    return {"bloco": bloco, "nome": nome, "combinacao": combinacao, "canais": list(canais),
            "rejeicao_fator": rejeicao_fator,
            "p": {"alvo_estimulo": [], "lateral": [], "alvo_esp": []},
            "por_condicao": {}, "rejeitada": [], "razao_variancia": []}


def processar_arquivo(bruto, condicao, opcoes):
    """Acrescenta os p-valores deste arquivo a cada opcao (bruto carregado uma vez)."""
    for op in opcoes:
        d = dados.coeficientes_do_sinal(bruto, op["combinacao"], op["canais"], op["rejeicao_fator"])
        valida = d["epoca_valida"][None]
        op["rejeitada"].append(1.0 - valida.mean())
        op["razao_variancia"].append(d["razao_variancia"])
        for tipo, chave in (("alvo", "coef_alvo"), ("lateral", "coef_lateral")):
            z, _ = dados.rayleigh_z_acumulado(d[chave][None], valida)
            p = np.exp(-z[-1])
            if tipo == "lateral":
                op["p"]["lateral"].extend(p)
            elif condicao == "ESP":
                op["p"]["alvo_esp"].extend(p)
            else:
                op["p"]["alvo_estimulo"].extend(p)
                op["por_condicao"].setdefault(condicao, []).extend(p)


def varrer(opcoes):
    """Carrega cada .mat uma vez e avalia todas as opcoes; depois fecha as metricas."""
    for caminho, _, condicao in dados.listar_arquivos(PASTA_DADOS):
        processar_arquivo(dados.carregar_x(caminho), condicao, opcoes)
    for op in opcoes:
        p = {k: np.array(v) for k, v in op.pop("p").items()}
        ks = stats.kstest(p["alvo_esp"], "uniform")
        ordem = sorted(op["por_condicao"], key=lambda c: int(c.removesuffix("dB")))
        op["deteccao"] = float(np.mean(p["alvo_estimulo"] <= ALFA))
        op["fp_lateral"] = float(np.mean(p["lateral"] <= ALFA))
        op["fp_alvo_esp"] = float(np.mean(p["alvo_esp"] <= ALFA))
        op["n"] = {k: int(v.size) for k, v in p.items()}
        op["ks_esp_estatistica"] = float(ks.statistic)
        op["ks_esp_p"] = float(ks.pvalue)
        op["fracao_rejeitada"] = float(np.mean(op.pop("rejeitada")))
        op["deteccao_por_intensidade"] = {
            c: float(np.mean(np.array(op["por_condicao"][c]) <= ALFA)) for c in ordem
        }
        del op["por_condicao"]
        razoes = np.array(op.pop("razao_variancia"))
        op["arquivos_variancia_desprezivel"] = int(np.sum(razoes < AVISO_RAZAO_VARIANCIA))
        op["razao_variancia_minima"] = float(razoes.min())
        op["elegivel"] = bool(op["fp_lateral"] <= FP_MAXIMO and op["ks_esp_p"] > KS_P_MINIMO)
    return opcoes


def chave_escolha(op):
    """Maior deteccao; desempate pelo menor FP lateral."""
    return (op["deteccao"], -op["fp_lateral"])


def melhor(opcoes):
    """Melhor opcao elegivel pela regra (None se nenhuma)."""
    elegiveis = [o for o in opcoes if o["elegivel"]]
    return max(elegiveis, key=chave_escolha) if elegiveis else None


# ============================================================
# Saidas
# ============================================================
def rotulo_canais(canais):
    return ",".join(NOMES[i] for i in canais)


def tabela_md(opcoes, com_rejeicao=False):
    cond = list(opcoes[0]["deteccao_por_intensidade"])
    cab = ["Opção", "Canais", "Detecção /440", "FP lateral /528", "FP alvo ESP /88", "KS ESP p", "Rejeitadas"]
    if com_rejeicao:
        cab.insert(2, "Fator")
    cab += [f"{c}" for c in cond] + ["Elegível"]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for o in sorted(opcoes, key=chave_escolha, reverse=True):
        col = [o["nome"], rotulo_canais(o["canais"]) if len(o["canais"]) < N_CANAIS else "todos"]
        if com_rejeicao:
            col.append("—" if o["rejeicao_fator"] is None else f"{o['rejeicao_fator']}")
        col += [f"{o['deteccao']:.1%}", f"{o['fp_lateral']:.1%}", f"{o['fp_alvo_esp']:.1%}",
                f"{o['ks_esp_p']:.3f}", f"{o['fracao_rejeitada']:.1%}"]
        col += [f"{o['deteccao_por_intensidade'][c]:.0%}" for c in cond]
        col.append("sim" if o["elegivel"] else "não")
        linhas.append("| " + " | ".join(col) + " |")
    return "\n".join(linhas)


def escrever_md(caminho, blocos, escolhida, avisos):
    e = escolhida
    partes = [
        "# Seleção do sinal — Rayleigh de teste único no fim",
        "",
        f"Gerado por `src/rayleigh_hmm/selecao_sinal.py`. Um teste de Rayleigh por (arquivo, frequência) com "
        f"todas as épocas válidas, α = {ALFA}. Detecção = alvos em 30–70 dB (n = 440); FP lateral = laterais "
        "em todas as condições (n = 528); FP alvo ESP (n = 88); KS = p-valores dos alvos ESP contra "
        "Uniforme(0,1) (checagem da nula); colunas de dB = detecção por intensidade.",
        "",
        "**Vazamento aceito:** o ranking de canais e a escolha usam os mesmos dados da avaliação "
        "(in-sample). Os números não são estimativa de generalização nem clínica.",
        "",
        f"**Regra de escolha:** maior detecção entre as opções com FP lateral ≤ {FP_MAXIMO:.0%} e "
        f"KS ESP com p > {KS_P_MINIMO}; desempate pelo menor FP lateral. O ranking de canais para o "
        "top-k usa a mesma regra (elegíveis primeiro, depois detecção).",
        "",
        "## Opção escolhida",
        "",
        f"`{e['nome']}`: combinação `{e['combinacao']}`, canais {rotulo_canais(e['canais'])}, "
        f"rejeição `{e['rejeicao_fator']}`. Detecção {e['deteccao']:.1%}, FP lateral {e['fp_lateral']:.1%}, "
        f"FP alvo ESP {e['fp_alvo_esp']:.1%}, KS ESP p = {e['ks_esp_p']:.3f}, "
        f"épocas rejeitadas {e['fracao_rejeitada']:.1%}.",
        "",
    ]
    if avisos:
        partes += ["## Avisos", ""] + [f"- {a}" for a in avisos] + [""]
    titulos = {"canais": "(a) Canais isolados", "combinacoes": "(b)/(c) Combinações",
               "rejeicao": "(d) Rejeição de artefatos"}
    for bloco, titulo in titulos.items():
        partes += [f"## {titulo}", "", tabela_md(blocos[bloco], com_rejeicao=bloco == "rejeicao"), ""]
    with open(caminho, "w") as f:
        f.write("\n".join(partes))


def main():
    # (a) canais isolados e (b) todos os canais
    todos = list(range(N_CANAIS))
    canais = [nova_opcao("canais", f"canal:{NOMES[i]}", "canal", [i]) for i in todos]
    combos = [nova_opcao("combinacoes", f"{m}:todos", m, todos) for m in ("media", "pca")]
    varrer(canais + combos)
    ranking = sorted(canais, key=lambda o: (o["elegivel"],) + chave_escolha(o), reverse=True)
    ordem = [o["canais"][0] for o in ranking]

    # (c) top-k pelo ranking de (a); (d) rejeicao no melhor canal e na melhor combinacao
    topo = [nova_opcao("combinacoes", f"{m}:top{k}", m, ordem[:k]) for k in TOPO_K for m in ("media", "pca")]
    varrer(topo)
    combos += topo
    base = {"canal": melhor(canais) or ranking[0], "combinacao": melhor(combos) or max(combos, key=chave_escolha)}
    rejeicao = [
        nova_opcao("rejeicao", f"{b['nome']}|rej{f}", b["combinacao"], b["canais"], f)
        for b in base.values() for f in FATORES_REJEICAO if f is not None
    ]
    varrer(rejeicao)
    # linhas sem rejeicao (fator None) das bases entram na tabela de rejeicao
    rejeicao += [dict(b, bloco="rejeicao") for b in base.values()]

    todas = canais + combos + rejeicao
    escolhida = melhor(todas)
    if escolhida is None:
        raise RuntimeError("Nenhuma opcao com FP lateral e KS dentro dos limites")

    avisos = list(dict.fromkeys(
        f"{o['nome']}: {o['arquivos_variancia_desprezivel']} arquivos com variância do sinal combinado "
        f"< {AVISO_RAZAO_VARIANCIA:.0%} da média dos canais (mínimo {o['razao_variancia_minima']:.3g})"
        for o in todas if o["arquivos_variancia_desprezivel"] > 0
    ))
    if not avisos:
        avisos = ["Nenhuma opção teve variância do sinal combinado desprezível "
                  f"(razão mínima observada em média de todos: {combos[0]['razao_variancia_minima']:.3g})."]
    for a in avisos:
        print("AVISO:", a)

    blocos = {"canais": canais, "combinacoes": combos, "rejeicao": rejeicao}
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    saida = {
        "configuracao": {"alfa": ALFA, "fp_maximo": FP_MAXIMO, "ks_p_minimo": KS_P_MINIMO, "topo_k": TOPO_K,
                         "fatores_rejeicao": FATORES_REJEICAO, "nomes_eletrodos": NOMES,
                         "ranking_canais": [NOMES[i] for i in ordem],
                         "vazamento": "ranking e escolha usam os mesmos dados da avaliacao (in-sample)"},
        "regra_escolha": "maior deteccao com FP lateral <= fp_maximo e KS ESP p > ks_p_minimo; desempate pelo menor FP lateral",
        "escolhida": escolhida,
        "avisos": avisos,
        **blocos,
    }
    with open(os.path.join(PASTA_RESULTADOS, "selecao_sinal.json"), "w") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    escrever_md(os.path.join(PASTA_RESULTADOS, "selecao_sinal.md"), blocos, escolhida, avisos)
    print("Escolhida:", escolhida["nome"], f"det={escolhida['deteccao']:.1%} fp={escolhida['fp_lateral']:.1%}")


if __name__ == "__main__":
    main()
