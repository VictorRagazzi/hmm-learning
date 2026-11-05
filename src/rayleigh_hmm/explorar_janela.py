"""Teste exploratorio: observacao do HMM = Rayleigh Z POR JANELA (nao acumulado).

Unica mudanca em relacao ao experimento atual: Z_w = |soma dos fasores das
epocas validas da janela w|^2 / n_w, com n_w = epocas validas da janela (<= 10).
Sinal, rejeicao, janelas, sequencias, regras, grade (a estendida de
explorar_busca.py) e FP lateral <= 0.07 ficam iguais. O baseline
bruto_referencia continua com Z acumulado. Variante: hmm_viterbi (emissao
marginal com N -> n_w). Resultados in-sample (11 participantes), nao validacao.
Saidas: results/rayleigh_hmm/exploratorio_janela/{resultados.json,tabelas.md}
Nao altera nenhum arquivo existente.
"""

import json
import os
import time

import numpy as np

import dados
import emissao
import experimento as ex
import explorar_busca as eb
import hmm

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_janela")
ATUAL_VENCEDOR = {"p1": 0.01, "a01": 0.005, "a10": 0.3,
                  "regra": {"familia": "fracao_ou_consecutivas", "fracao": 0.1, "consecutivas": 2}}


# ---------------- Z por janela ----------------
def rayleigh_z_por_janela(janelas, valida_janelas):
    """Z_w = |soma fasores validos da janela|^2 / n_w (0 se n_w = 0). Retorna z (J, F), n_w (J,)."""
    fases = janelas / (np.abs(janelas) + dados.EPS) * valida_janelas[:, :, None]
    n_w = valida_janelas.sum(axis=1)
    z = np.abs(fases.sum(axis=1)) ** 2 / np.maximum(n_w, 1)[:, None]
    return z, n_w


def sequencias_por_janela(sequencias_acum):
    """Copia das sequencias do experimento com z, n_ep trocados por Z por janela (mesma ordem)."""
    novas, esp, ordem = [], [], iter(sequencias_acum)
    for caminho, participante, condicao in dados.listar_arquivos(ex.PASTA_DADOS):
        d = dados.carregar_coeficientes(caminho, **ex.SINAL)
        if d["n_epocas"] // ex.EPOCAS_POR_JANELA < ex.MINIMO_AVALIACOES:
            continue
        valida = dados.janelar(d["epoca_valida"], ex.EPOCAS_POR_JANELA)
        if condicao == "ESP":
            esp.append((d["coef_alvo"], d["epoca_valida"]))
        for chave, freqs in (("coef_alvo", d["freq_estim"]), ("coef_lateral", d["bins_m"])):
            z, n_w = rayleigh_z_por_janela(dados.janelar(d[chave], ex.EPOCAS_POR_JANELA), valida)
            for i in range(len(freqs)):
                s = next(ordem)
                assert s["participante"] == participante and s["condicao"] == condicao
                assert abs(s["frequencia_hz"] - freqs[i]) < 1e-9 and len(s["z"]) == len(z)
                novas.append({**s, "z": z[:, i], "n_ep": n_w})
    return novas, esp


def presente_sintetico_janela(esp):
    """Z por janela das ESP + senoide (5 SNRs fixas); descarta janelas com n_w = 0."""
    zs, ns, ids, prox = [], [], [], 0
    for snr in ex.SNR_SINTETICA_DB:
        for coef, valida in esp:
            z, n_w = rayleigh_z_por_janela(
                dados.janelar(dados.injetar_senoide(coef, snr, valida), ex.EPOCAS_POR_JANELA),
                dados.janelar(valida, ex.EPOCAS_POR_JANELA))
            ok = np.repeat((n_w > 0)[:, None], z.shape[1], axis=1)
            idj = np.tile(np.arange(z.shape[1]), z.shape[0]).reshape(z.shape) + prox
            zs.append(z[ok]); ns.append(np.broadcast_to(n_w[:, None], z.shape)[ok]); ids.append(idj[ok])
            prox += z.shape[1]
    return np.concatenate(zs), np.concatenate(ns), np.concatenate(ids)


def log_emissoes_janela(s, emissoes):
    """Emissao marginal de ex.log_emissoes; janelas com n_w = 0 sao nao informativas (0, 0)."""
    le = ex.log_emissoes(s, emissoes, "marginal")
    le[s["n_ep"] == 0] = 0.0
    return le


# ---------------- fluxo ----------------
def preparar():
    seq_acum, _ = ex.carregar_sequencias()
    sequencias, esp = sequencias_por_janela(seq_acum)
    masc = ex.mascaras(sequencias)
    _, ref = ex.buscar_bruto(seq_acum, masc)  # baseline inalterado (Z acumulado)

    esp_alvo = [s for s in sequencias if s["condicao"] == "ESP" and s["tipo"] == "alvo"]
    z_esp = np.concatenate([s["z"][s["n_ep"] > 0] for s in esp_alvo])
    n_zero = int(sum((s["n_ep"] == 0).sum() for s in sequencias))
    z_s, n_s, id_s = presente_sintetico_janela(esp)
    emissoes = {"ausente": emissao.ajustar_mle(z_esp), "z_sintetico": z_s, "n_sintetico": n_s, "id_sintetico": id_s}
    emissoes["marginal"] = ex.ajustar_marginal(emissoes)

    sanidade = {
        "ausente": emissoes["ausente"], "n_z_esp": int(z_esp.size),
        "frac_z_esp_acima_ln20": float((z_esp > -np.log(0.05)).mean()),
        "janelas_n_w_zero_total": n_zero,
        "n_w_distribuicao": {int(v): int(c) for v, c in zip(*np.unique(np.concatenate([s["n_ep"] for s in esp_alvo]), return_counts=True))},
        "n_sinteticos": int(z_s.size), "n_sequencias_sinteticas": int(emissoes["marginal"]["hiper"]["n_sequencias"]),
        "hiper": emissoes["marginal"]["hiper"],
    }
    grupos = [(idx, dur, np.stack([log_emissoes_janela(sequencias[i], emissoes) for i in idx]))
              for idx, dur in ex.indices_por_comprimento(sequencias)]
    return sequencias, masc, ref, grupos, sanidade


def main():
    t0 = time.time()
    sequencias, masc, ref, grupos, sanidade = preparar()
    h = sanidade["hiper"]
    print(f"Ausente por janela: {sanidade['ausente']}; FP ESP (Z>-ln .05): {sanidade['frac_z_esp_acima_ln20']:.4f}; "
          f"janelas n_w=0: {sanidade['janelas_n_w_zero_total']}")
    print(f"Hiper: k~Gamma({h['a_k']:.3f},{h['b_k']:.3f}) rho~Gamma({h['a_rho']:.3f},{h['b_rho']:.3f}) borda={h['na_borda']}")

    viaveis, total = eb.varrer(grupos, sequencias, masc, ref)
    print(f"candidatos: {total}; viaveis: {len(viaveis)}")
    a = max(viaveis, key=eb.chave_a)
    nao_lento = [c for c in viaveis if c["dif_pareada_s"] <= 0]
    b = max(nao_lento, key=eb.chave_b) if nao_lento else None
    fronteira = eb.pareto(viaveis)
    pontos = eb.representativos(fronteira, 252 / 440)

    escolhidos = {"a_chave_ordem": a}
    if b:
        escolhidos["b_nao_mais_lento"] = b
    else:
        n_ref = int(ref["detectado"][masc["alvo_estimulo"]].sum())
        quase = [c for c in viaveis if round(c["m"]["deteccao"] * 440) >= n_ref]
        if quase:
            escolhidos["b_vazio_menor_dif_det_ge_bruto"] = min(quase, key=lambda c: (c["dif_pareada_s"], -c["m"]["deteccao"]))
    # parametros do vencedor acumulado atual, com Z por janela
    v = ATUAL_VENCEDOR
    k_v = next(i for i, r in enumerate(ex.REGRAS) if r == v["regra"])
    gd = eb.avaliar_parametros(grupos, len(sequencias), v["p1"], v["a01"], v["a10"])
    det, tempo = ex.decidir(gd, len(sequencias), v["regra"])
    n_pares, dif = eb.diferenca_pareada(masc, ref, det, tempo)
    escolhidos["ref_parametros_vencedor_acumulado"] = {
        "p1": v["p1"], "a01": v["a01"], "a10": v["a10"], "k": k_v, "m": ex.metricas(masc, det, tempo),
        "dif_pareada_s": dif, "n_pares": n_pares}

    res = {n: eb.completar(c, grupos, sequencias, masc, ref) for n, c in escolhidos.items()}
    res_front = [eb.completar(c, grupos, sequencias, masc, ref) for c in pontos]
    ref_m = ex.metricas(masc, ref["detectado"], ref["tempo_s"])
    n_ref = [int(ref["detectado"][masc[k]].sum()) for k in ("alvo_estimulo", "lateral", "alvo_esp")]

    saida = {"aviso": "exploratorio, in-sample (11 participantes); nao e validacao",
             "mudanca": "observacao do HMM = Z por janela (n_w epocas validas), nao acumulado",
             "sanidade": sanidade, "n_candidatos_avaliados": total, "n_viaveis": len(viaveis),
             "n_nao_mais_lentos": len(nao_lento), "n_pontos_fronteira_total": len(fronteira),
             "bruto_referencia": ref_m,
             "escolhidos": {n: eb.sem_privado(r) for n, r in res.items()},
             "fronteira_pareto": [eb.sem_privado(r) for r in res_front]}

    t = eb.fmt_t
    md = ["# Exploratorio: HMM (Viterbi) com Rayleigh Z por janela (in-sample, 11 participantes)\n",
          "## Sanidade\n",
          f"- Ausente (MLE, Z por janela, ESP alvo, n={sanidade['n_z_esp']}): k={sanidade['ausente']['k']:.4f}, mu={sanidade['ausente']['mu']:.4f} (teoria 1, 1)",
          f"- Fracao de Z_ESP > -ln(0.05): {100*sanidade['frac_z_esp_acima_ln20']:.2f}%",
          f"- Janelas com n_w = 0 (todas as sequencias): {sanidade['janelas_n_w_zero_total']}; distribuicao de n_w nas ESP alvo: {sanidade['n_w_distribuicao']}",
          f"- Presente sintetico: {sanidade['n_sequencias_sinteticas']} sequencias, {sanidade['n_sinteticos']} Z",
          f"- Hierarquico: k~Gamma({h['a_k']:.3f}, {h['b_k']:.3f}) media {h['k_media']:.3f} dp {h['k_desvio']:.3f}; "
          f"rho~Gamma({h['a_rho']:.3f}, {h['b_rho']:.3g}) media {h['rho_media']:.4f} dp {h['rho_desvio']:.4f}; "
          f"na_borda={h['na_borda']} (hiper no limite={h['hiper_no_limite']}, massa fora={h['massa_fora_grade']}, dispersao inicios={h['dispersao_relativa_inicios']:.2g})\n",
          f"Candidatos: {total}; viaveis (FP lat <= {ex.FP_MAXIMO}): {len(viaveis)}; com dif. pareada <= 0: {len(nao_lento)}; pontos Pareto (dedupe): {len(fronteira)}.\n",
          f"bruto_referencia (Z acumulado): det {n_ref[0]}/440, FP lat {n_ref[1]}/528, FP ESP {n_ref[2]}/88, tempo medio/mediano {t(ref_m['tempo_medio_s'])}/{t(ref_m['tempo_mediano_s'])}.\n",
          "## Vencedores e referencia\n", eb.CABECALHO]
    for n, r in res.items():
        md.append(eb.linha_md(n, r))
    md += ["\n## Fronteira de Pareto (deteccao x tempo medio)\n", eb.CABECALHO]
    md += [eb.linha_md(f"P{i+1}", r) for i, r in enumerate(res_front)]
    for n, r in res.items():
        linhas = eb.tabela_intensidade(sequencias, r["_cand"], ref)
        md.append(eb.tabela_intensidade_md(linhas, n))
        saida["escolhidos"][n]["por_intensidade"] = linhas
    saida["tempo_execucao_s"] = time.time() - t0
    md.append(f"\nTempo de execucao: {saida['tempo_execucao_s']:.0f} s.")

    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
