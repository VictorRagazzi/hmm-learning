"""Exploratorio: protocolo concatenado ESP -> 30 -> ... -> 70 dB por participante x frequencia.

Nao altera nenhum arquivo existente; reaproveita experimento/dados/emissao/hmm/explorar_busca.
- Janelas de EPOCAS_POR_JANELA epocas (passo = tamanho), sem cruzar gravacoes.
- Z da janela = Rayleigh sobre as epocas validas da gravacao ATUAL ate o fim da janela,
  limitado as ultimas M epocas (M=None: acumulado desde o inicio da gravacao). Reinicia
  a cada gravacao.
- Emissoes por M: Ausente = Gamma MLE no Z janelado do ESP (alvos); Presente = marginal
  hierarquica no Z janelado de todos os alvos com estimulo (calibracao "real").
- Sem heranca: cada intensidade e decidida pela sua propria gravacao.
- HMM: Viterbi causal sobre o protocolo INTEIRO (o estado atravessa as gravacoes como memoria),
  mas a decisao em cada passo e o estado atual: o estado final do melhor caminho ate t e
  Presente.
- Regras (REGRAS) e MINIMO_AVALIACOES valem DENTRO de cada segmento (participante x
  frequencia x intensidade).
IN-SAMPLE com vazamento direto (Presente ajustado nos mesmos alvos avaliados): exploratorio.

Execucao: .venv/bin/python src/rayleigh_hmm/explorar_protocolo.py
Saidas: results/rayleigh_hmm/exploratorio_protocolo/{resultados.json,tabelas.md}
"""

import itertools
import json
import os
import time

import numpy as np
from scipy import stats

import dados
import emissao
import experimento as ex
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI

# ---------------- configuracao ----------------
JANELAS_M = [30, 60, 120, 240, None]  # None = acumulado dentro da gravacao
DECISOES = ("estado",)
ORDEM_CONDICOES = ["ESP", "30dB", "40dB", "50dB", "60dB", "70dB"]
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_protocolo")
EPJ = ex.EPOCAS_POR_JANELA
GRANDE = 10**9
N_COND = len(ORDEM_CONDICOES)


# ---------------- dados e estrutura do protocolo ----------------
def carregar_fases():
    """Soma acumulada de fasores unitarios e de epocas validas, por (participante, cond)."""
    arquivos = {}
    for caminho, part, cond in dados.listar_arquivos(ex.PASTA_DADOS):
        d = dados.carregar_coeficientes(caminho, **ex.SINAL)
        n_jan = d["n_epocas"] // EPJ
        assert n_jan >= ex.MINIMO_AVALIACOES, f"{part}{cond} com {n_jan} janelas"
        valida = d["epoca_valida"][: n_jan * EPJ]
        cs = {}
        for tipo, chave in (("alvo", "coef_alvo"), ("lateral", "coef_lateral")):
            c = d[chave][: n_jan * EPJ]
            fases = c / (np.abs(c) + dados.EPS) * valida[:, None]
            cs[tipo] = np.concatenate([np.zeros((1, c.shape[1]), complex), np.cumsum(fases, axis=0)])
        arquivos[(part, cond)] = {
            "n_ep": int(d["n_epocas"]), "cs": cs, "cv": np.concatenate([[0], np.cumsum(valida)]), "n_jan": n_jan,
            "dur": d["duracao_epoca_s"], "freqs": {"alvo": d["freq_estim"], "lateral": d["bins_m"]},
        }
    return arquivos


def z_janelado(arq, tipo, m):
    """(n_jan, n_freq) Z e (n_jan,) N das janelas de uma gravacao, com cap de m epocas."""
    fim = (np.arange(arq["n_jan"]) + 1) * EPJ
    ini = np.zeros_like(fim) if m is None else np.maximum(fim - m, 0)
    soma = arq["cs"][tipo][fim] - arq["cs"][tipo][ini]
    n = arq["cv"][fim] - arq["cv"][ini]
    return np.abs(soma) ** 2 / np.maximum(n, 1)[:, None], n


def montar_estrutura(arquivos):
    """Layout plano: protocolos (tipo, participante, freq) -> segmentos -> janelas."""
    participantes = sorted({p for p, _ in arquivos})
    segs, protos = [], []
    inicio = 0
    for tipo in ("alvo", "lateral"):
        for p in participantes:
            for i in range(8):
                ini_proto = inicio
                for cond in ORDEM_CONDICOES:
                    a = arquivos[(p, cond)]
                    segs.append({"proto": len(protos), "tipo": tipo, "participante": p, "cond": cond, "i": i,
                                 "freq": float(a["freqs"][tipo][i]), "ini": inicio, "n": a["n_jan"],
                                 "dur_s": a["dur"] * EPJ * a["n_jan"], "dur_epoca": a["dur"]})
                    inicio += a["n_jan"]
                protos.append({"tipo": tipo, "participante": p, "i": i, "ini": ini_proto, "n": inicio - ini_proto})
    est = {"segs": segs, "protos": protos, "total": inicio}
    est["seg_ini"] = np.array([s["ini"] for s in segs])
    seg_id = np.repeat(np.arange(len(segs)), [s["n"] for s in segs])
    est["seg_id"] = seg_id
    est["pos"] = np.arange(inicio) - est["seg_ini"][seg_id]
    est["dur_epoca"] = np.array([s["dur_epoca"] for s in segs])
    tipo = np.array([s["tipo"] for s in segs])
    cond = np.array([s["cond"] for s in segs])
    est["m_alvo_estim"] = (tipo == "alvo") & (cond != "ESP")
    est["m_lateral"] = tipo == "lateral"
    est["m_alvo_esp"] = (tipo == "alvo") & (cond == "ESP")
    est["cond"] = cond
    est["tipo"] = tipo
    # indices para (n_proto, Tmax) <-> plano
    tmax = max(p["n"] for p in protos)
    est["tmax"] = tmax
    est["rows"] = np.repeat(np.arange(len(protos)), [p["n"] for p in protos])
    est["cols"] = np.concatenate([np.arange(p["n"]) for p in protos])
    return est


def z_plano(arquivos, est, m):
    z = np.empty(est["total"])
    n = np.empty(est["total"])
    cache = {}
    for s in est["segs"]:
        chave = (s["participante"], s["cond"], s["tipo"])
        if chave not in cache:
            cache[chave] = z_janelado(arquivos[(s["participante"], s["cond"])], s["tipo"], m)
        zz, nn = cache[chave]
        z[s["ini"]: s["ini"] + s["n"]] = zz[:, s["i"]]
        n[s["ini"]: s["ini"] + s["n"]] = nn
    return z, n


# ---------------- emissoes ----------------
def ajustar_emissoes_m(est, z, n):
    seg_id = est["seg_id"]
    no_esp = est["m_alvo_esp"][seg_id]
    no_est = est["m_alvo_estim"][seg_id]
    aus = emissao.ajustar_mle(z[no_esp])
    matriz = emissao.matriz_verossimilhanca_sequencias(z[no_est], n[no_est], seg_id[no_est],
                                                       ex.GRADE_HIER_K, ex.GRADE_HIER_RHO)
    hiper = emissao.ajustar_hierarquico(matriz, ex.GRADE_HIER_K, ex.GRADE_HIER_RHO, ex.INICIOS_HIER)
    hiper["n_sequencias"] = int(matriz.shape[0])
    del matriz
    pontos = ex.pontos_marginal(hiper)
    log_aus = emissao.log_pdf_gamma(z, aus["k"], aus["mu"])
    log_pres = emissao.log_pdf_preditiva(z, n, pontos)
    log_em = np.stack([log_aus, log_pres], axis=-1)
    resumo = {"ausente": aus, "presente": {k: v for k, v in hiper.items() if k != "corridas"},
              "n_obs_esp": int(no_esp.sum()), "n_obs_estimulo": int(no_est.sum())}
    return log_em, resumo


def padronizar(est, log_em):
    """(n_proto, Tmax, 2); o preenchimento no fim nao afeta o Viterbi causal."""
    saida = np.zeros((len(est["protos"]), est["tmax"], 2))
    saida[est["rows"], est["cols"]] = log_em
    return saida


def viterbi_estado(log_em, pi, A):
    """Para cada t: o estado final do melhor caminho de 1..t e Presente? (empate: Ausente)."""
    log_pi, log_A = np.log(pi), np.log(A)
    delta = log_pi[None, :] + log_em[:, 0]
    det = np.empty(log_em.shape[:2], dtype=bool)
    det[:, 0] = delta.argmax(axis=1) == 1
    for t in range(1, log_em.shape[1]):
        delta = (delta[:, :, None] + log_A[None, :, :]).max(axis=1) + log_em[:, t]
        det[:, t] = delta.argmax(axis=1) == 1
    return det


# ---------------- regras dentro dos segmentos ----------------
def indicadores_plano(est, det):
    d = det.astype(int)
    cs = np.cumsum(d)
    ini = est["seg_ini"][est["seg_id"]]
    antes = cs[ini] - d[ini]
    fracao = (cs - antes) / (est["pos"] + 1)
    idx = np.arange(det.size)
    ultimo_falso = np.maximum.accumulate(np.where(det, -1, idx))
    corrida = idx - np.maximum(ultimo_falso, ini - 1)
    return fracao, corrida


def mascaras_regras(est, det):
    fracao, corrida = indicadores_plano(est, det)
    ok = est["pos"] >= ex.MINIMO_AVALIACOES - 1
    pf = {f: (fracao >= f - 1e-9) & ok for f in ex.GRADE_FRACAO_MINIMA}
    pc = {c: (corrida >= c) & ok for c in ex.GRADE_CONSECUTIVAS}
    return pf, pc


def dispara(regra, pf, pc, ok):
    fam = regra["familia"]
    if fam == "fracao":
        return pf[regra["fracao"]]
    if fam == "consecutivas":
        return pc[regra["consecutivas"]]
    f, c = pf[regra["fracao"]], pc[regra["consecutivas"]]
    return (f & c) if fam == "fracao_e_consecutivas" else ((f | c) & ok)


def decidir_plano(est, det, regra):
    """detectado (n_seg,) e tempo em s (nan) com a regra aplicada em cada segmento."""
    pf, pc = mascaras_regras(est, det)
    return decidir_com(est, pf, pc, regra)


def decidir_com(est, pf, pc, regra):
    ok = est["pos"] >= ex.MINIMO_AVALIACOES - 1
    d = dispara(regra, pf, pc, ok)
    primeiro = np.minimum.reduceat(np.where(d, est["pos"], GRANDE), est["seg_ini"])
    detectado = primeiro < GRANDE
    tempo = np.where(detectado, (primeiro + 1) * EPJ * est["dur_epoca"], np.nan)
    return detectado, tempo


def contagens(est, det_seg, tempo):
    t = tempo[est["m_alvo_estim"] & det_seg]
    return (int(det_seg[est["m_alvo_estim"]].sum()), int(det_seg[est["m_lateral"]].sum()),
            int(det_seg[est["m_alvo_esp"]].sum()), float(t.mean()) if t.size else np.inf)


def chave(c):
    """Maior deteccao; desempate: menor FP lateral, menor tempo medio."""
    return (c[0], -c[1], -c[3])


def varrer_regras(est, det, teto, melhor, parametros):
    pf, pc = mascaras_regras(est, det)
    for regra in ex.REGRAS:
        ds, tempo = decidir_com(est, pf, pc, regra)
        c = contagens(est, ds, tempo)
        if c[1] > teto:
            continue
        if melhor.get("c") is None or chave(c) > chave(melhor["c"]):
            melhor.update({"c": c, "regra": regra, **parametros})


# ---------------- buscas ----------------
def buscar_bruto(est, z, teto):
    melhor = {}
    for alfa in ex.GRADE_ALFA:
        varrer_regras(est, np.exp(-z) <= alfa, teto, melhor, {"alfa": alfa})
    return melhor


def buscar_hmm(est, log_em_pad, teto):
    melhor = {}
    for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
        pi = np.array([1 - p1, p1])
        A = np.array([[1 - a01, a01], [a10, 1 - a10]])
        det_pad = viterbi_estado(log_em_pad, pi, A)
        varrer_regras(est, det_pad[est["rows"], est["cols"]], teto, melhor,
                      {"pi": pi.tolist(), "A": A.tolist(), "pi_presente": p1, "A_aus_pres": a01, "A_pres_aus": a10})
    return melhor


def finalizar(est, escolha, det_plano):
    """Arrays por segmento do candidato escolhido."""
    ds, tempo = decidir_plano(est, det_plano, escolha["regra"])
    return {**{k: v for k, v in escolha.items() if k != "c"}, "detectado": ds, "tempo_s": tempo}


def det_plano_hmm(est, log_em_pad, escolha):
    pi, A = np.array(escolha["pi"]), np.array(escolha["A"])
    d = viterbi_estado(log_em_pad, pi, A)
    return d[est["rows"], est["cols"]]


# ---------------- metricas ----------------
def metricas_metodo(est, c):
    ds, tempo = c["detectado"], c["tempo_s"]
    segs = est["segs"]
    por_int = {}
    for cond in ORDEM_CONDICOES[1:]:
        msk = est["m_alvo_estim"] & (est["cond"] == cond)
        por_int[cond] = int(ds[msk].sum())
    t = tempo[est["m_alvo_estim"] & ds]
    # por protocolo-alvo: primeira intensidade detectada e consistencia
    protos = {}
    for k, s in enumerate(segs):
        if s["tipo"] == "alvo":
            protos.setdefault(s["proto"], []).append(k)
    primeira = {r: 0 for r in ORDEM_CONDICOES + ["nenhuma"]}
    consistentes = com_det = 0
    tempos_prot = []
    for ks in protos.values():
        v = [bool(ds[k]) for k in ks]
        primeira[next((ORDEM_CONDICOES[j] for j, x in enumerate(v) if x), "nenhuma")] += 1
        est_v = v[1:]
        if any(est_v):
            com_det += 1
            f = est_v.index(True)
            consistentes += all(est_v[f:])
            acum = sum(segs[k]["dur_s"] for k in ks[1:1 + f])
            tempos_prot.append(acum + tempo[ks[1 + f]])
    d2 = ds.reshape(-1, N_COND)
    tipo_p = est["tipo"].reshape(-1, N_COND)[:, 0]
    return {
        "proto_alvo_det_estimulo": int(d2[tipo_p == "alvo", 1:].any(axis=1).sum()),
        "proto_lateral_det": int(d2[tipo_p == "lateral"].any(axis=1).sum()),
        "det": int(ds[est["m_alvo_estim"]].sum()), "det_por_intensidade": por_int,
        "fp_lateral": int(ds[est["m_lateral"]].sum()), "fp_esp": int(ds[est["m_alvo_esp"]].sum()),
        "tempo_medio_s": float(t.mean()) if t.size else None,
        "tempo_mediano_s": float(np.median(t)) if t.size else None,
        "primeira_intensidade": primeira, "protocolos_com_det_estimulo": com_det,
        "consistentes_acima": consistentes,
        "tempo_protocolo_medio_s": float(np.mean(tempos_prot)) if tempos_prot else None,
        "tempo_protocolo_mediano_s": float(np.median(tempos_prot)) if tempos_prot else None,
    }


def mcnemar(a, b):
    """a = referencia, b = metodo. Binomial exato nos discordantes."""
    so_a, so_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = stats.binomtest(so_a, so_a + so_b, 0.5).pvalue if so_a + so_b else 1.0
    return {"so_ref": so_a, "so_metodo": so_b, "p_valor": float(p)}


def por_protocolo(est, ds):
    """(alvos: alguma deteccao em 30-70 dB, laterais: algum FP em qualquer segmento) por protocolo."""
    d2 = ds.reshape(-1, N_COND)
    tp = est["tipo"].reshape(-1, N_COND)[:, 0]
    return d2[tp == "alvo", 1:].any(axis=1), d2[tp == "lateral"].any(axis=1)


def testes_pareados(est, ref, c):
    rd, cd = ref["detectado"], c["detectado"]
    ambos = est["m_alvo_estim"] & rd & cd
    ta, tb = ref["tempo_s"][ambos], c["tempo_s"][ambos]
    w = {"n_pares": int(ambos.sum()), "mediana_ref_s": float(np.median(ta)) if ta.size else None,
         "mediana_metodo_s": float(np.median(tb)) if tb.size else None,
         "diferencas_zero": int(np.sum(ta == tb)), "p_valor": None}
    if ta.size and np.sum(ta == tb) < ta.size:
        w["p_valor"] = float(stats.wilcoxon(ta, tb).pvalue)
    (ra, rl), (ca, cl) = por_protocolo(est, rd), por_protocolo(est, cd)
    return {"alvos": mcnemar(rd[est["m_alvo_estim"]], cd[est["m_alvo_estim"]]),
            "laterais": mcnemar(rd[est["m_lateral"]], cd[est["m_lateral"]]),
            "esp": mcnemar(rd[est["m_alvo_esp"]], cd[est["m_alvo_esp"]]),
            "alvos_proto": mcnemar(ra, ca), "laterais_proto": mcnemar(rl, cl), "tempo": w}


def diagnostico_intensidade(arquivos):
    """Por intensidade: epocas por gravacao, janelas, mediana de Z final (M=None) e % com p<=0,05."""
    saida = {}
    for cond in ORDEM_CONDICOES:
        arqs = [a for (p, c), a in sorted(arquivos.items()) if c == cond]
        n_ep = np.array([a["n_ep"] for a in arqs])
        n_jan = np.array([a["n_jan"] for a in arqs])
        zf = np.concatenate([z_janelado(a, "alvo", None)[0][-1] for a in arqs])
        saida[cond] = {"n_gravacoes": len(arqs), "epocas_min": int(n_ep.min()),
                       "epocas_mediana": float(np.median(n_ep)), "epocas_max": int(n_ep.max()),
                       "janelas_min": int(n_jan.min()), "janelas_mediana": float(np.median(n_jan)),
                       "janelas_max": int(n_jan.max()), "n_alvos": int(zf.size),
                       "z_final_mediana": float(np.median(zf)),
                       "alvos_p05_fim": int((np.exp(-zf) <= 0.05).sum()),
                       "pct_alvos_p05_fim": float(100 * (np.exp(-zf) <= 0.05).mean())}
    return saida


def escrever_csv(est, por_metodo):
    import csv
    with open(os.path.join(PASTA, "por_segmento.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "metodo", "proto_id", "tipo", "participante", "freq_hz", "intensidade", "detectado", "tempo_s"])
        for (m, nome), c in por_metodo.items():
            for k, s in enumerate(est["segs"]):
                t = c["tempo_s"][k]
                w.writerow([m, nome, s["proto"], s["tipo"], s["participante"], s["freq"], s["cond"],
                            int(c["detectado"][k]), "" if np.isnan(t) else f"{t:.1f}"])


# ---------------- principal ----------------
def sem_arrays(c):
    return {k: v for k, v in c.items() if k not in ("detectado", "tempo_s")}


def verificar_referencia(est, ref):
    """Confere o bruto_referencia plano (M=None) contra o pipeline atual por gravacao."""
    seqs, _ = ex.carregar_sequencias()
    r = ex.calcular_referencia(seqs, ex.mascaras(seqs))
    chave_seq = {(s["participante"], s["condicao"], s["tipo"], s["frequencia_hz"]): bool(d)
                 for s, d in zip(seqs, r["detectado"])}
    iguais = all(chave_seq[(s["participante"], s["cond"], s["tipo"], s["freq"])] == bool(d)
                 for s, d in zip(est["segs"], ref["detectado"]))
    return bool(iguais)


def main():
    t_ini = time.time()
    arquivos = carregar_fases()
    est = montar_estrutura(arquivos)
    print(f"segmentos {len(est['segs'])}, protocolos {len(est['protos'])}, janelas {est['total']}, Tmax {est['tmax']}")

    z_cum, _ = z_plano(arquivos, est, None)
    ref_c = finalizar(est, {"alfa": ex.REFERENCIA_BRUTO["alfa"], "regra": ex.REFERENCIA_BRUTO["regra"]},
                      np.exp(-z_cum) <= ex.REFERENCIA_BRUTO["alfa"])
    ref_c["metricas"] = metricas_metodo(est, ref_c)
    conf = verificar_referencia(est, ref_c)
    teto = ref_c["metricas"]["fp_lateral"]
    print(f"bruto_referencia: det {ref_c['metricas']['det']}/440 FP lat {teto}/528 FP ESP "
          f"{ref_c['metricas']['fp_esp']}/88 | confere com pipeline atual: {conf}")

    resultados = {}
    por_metodo = {("-", "bruto_referencia"): ref_c}
    for m in JANELAS_M:
        t0 = time.time()
        nome = "None" if m is None else str(m)
        z, n = z_plano(arquivos, est, m)
        log_em, fit = ajustar_emissoes_m(est, z, n)
        log_pad = padronizar(est, log_em)
        metodos = {}
        escolha = buscar_bruto(est, z, teto)
        metodos["bruto_otimizado"] = finalizar(est, escolha, np.exp(-z) <= escolha["alfa"])
        for dec in DECISOES:
            escolha = buscar_hmm(est, log_pad, teto)
            metodos[f"hmm_{dec}"] = finalizar(est, escolha, det_plano_hmm(est, log_pad, escolha))
        for c in metodos.values():
            c["metricas"] = metricas_metodo(est, c)
        pareados = {k: testes_pareados(est, ref_c, c) for k, c in metodos.items()}
        for k, c in metodos.items():
            por_metodo[(nome, k)] = c
        resultados[nome] = {"ajuste": fit, "metodos": {k: sem_arrays(c) for k, c in metodos.items()},
                            "pareados_vs_referencia": pareados, "tempo_execucao_s": time.time() - t0}
        print(f"M={nome}: {time.time()-t0:.0f}s | " + " | ".join(
            f"{k} {c['metricas']['det']}/440 FPlat {c['metricas']['fp_lateral']} FPesp {c['metricas']['fp_esp']}"
            for k, c in metodos.items()), flush=True)

    total = time.time() - t_ini
    saida = {
        "aviso": "exploratorio, IN-SAMPLE com vazamento direto (Presente ajustado nos mesmos alvos avaliados)",
        "configuracao": {"janelas_m": JANELAS_M, "epocas_por_janela": EPJ, "decisoes": DECISOES,
                         "grade_pi": GRADE_PI, "grade_a_aus_pres": GRADE_A01, "grade_a_pres_aus": GRADE_A10,
                         "minimo_avaliacoes": ex.MINIMO_AVALIACOES, "n_regras": len(ex.REGRAS),
                         "teto_fp_lateral_n": teto, "teto_fp_lateral_total": 528,
                         "referencia_confere_pipeline": conf},
        "diagnostico_intensidade": diagnostico_intensidade(arquivos), "bruto_referencia": sem_arrays(ref_c), "por_m": resultados, "tempo_total_s": total,
    }
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    escrever_md(saida)
    escrever_csv(est, por_metodo)
    print(f"tempo total {total:.0f}s; saidas em {PASTA}")


# ---------------- tabelas ----------------
def f_t(x):
    return "-" if x is None else f"{x:.0f}"


def pct(x, n):
    return f"{100 * x / n:.1f}%".replace(".", ",")


def linha(rotulo, m):
    di = m["det_por_intensidade"]
    return (f"| {rotulo} | {pct(m['det'], 440)} | " + " | ".join(pct(di[c], 88) for c in ORDEM_CONDICOES[1:])
            + f" | {pct(m['fp_lateral'], 528)} | {pct(m['fp_esp'], 88)} | {pct(m['proto_alvo_det_estimulo'], 88)} "
            f"| {pct(m['proto_lateral_det'], 88)} | {f_t(m['tempo_medio_s'])} / {f_t(m['tempo_mediano_s'])} |")


def escrever_md(s):
    cab = "| Método | Det | 30 | 40 | 50 | 60 | 70 | FP lat | FP ESP | Proto alvo | Proto lat | Tempo médio/mediano (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    md = ["# Exploratório: protocolo concatenado ESP→30→…→70 dB (IN-SAMPLE, vazamento direto)", "",
          "Janelas de 10 épocas dentro de cada gravação; Z com cap de M épocas; HMM com Viterbi causal sobre "
          "o protocolo inteiro; regras aplicadas por segmento. Sem herança: cada intensidade é decidida pela sua "
          "própria gravação; o estado do HMM atravessa as gravações como memória, mas a decisão é o estado atual. "
          "Percentuais: detecção geral sobre 440; cada intensidade sobre 88; FP lateral sobre 528; FP ESP e protocolos sobre 88. "
          f"Teto de FP lateral = {s['configuracao']['teto_fp_lateral_n']}/528 (bruto_referencia). Exploratório, não é validação.", "",
          f"bruto_referencia conferido contra o pipeline atual: {s['configuracao']['referencia_confere_pipeline']}", "",
          "## Diagnóstico por intensidade", "",
          "Z final = fim da gravação (M=None) nos alvos; p = exp(−Z).", "",
          "| Intensidade | Épocas/gravação (mín/med/máx) | Janelas de 10 (mín/med/máx) | Mediana Z final | Alvos com p≤0,05 no fim |",
          "|---|---:|---:|---:|---:|"]
    for cond, d in s["diagnostico_intensidade"].items():
        md.append(f"| {cond} | {d['epocas_min']}/{d['epocas_mediana']:.0f}/{d['epocas_max']} | "
                  f"{d['janelas_min']}/{d['janelas_mediana']:.0f}/{d['janelas_max']} | "
                  f"{d['z_final_mediana']:.2f}".replace(".", ",") + f" | {pct(d['alvos_p05_fim'], d['n_alvos'])} |")
    md.append("")
    ref = s["bruto_referencia"]["metricas"]
    for nome, p in s["por_m"].items():
        md += [f"## M = {nome}", "", cab, linha("bruto_referencia", ref)]
        for k, c in p["metodos"].items():
            md.append(linha(k, c["metricas"]))
        md += ["", "Configurações escolhidas:", ""]
        for k, c in p["metodos"].items():
            par = {q: c[q] for q in ("alfa", "pi", "A", "regra") if q in c}
            md.append(f"- {k}: `{json.dumps(par)}`")
        a, pr = p["ajuste"]["ausente"], p["ajuste"]["presente"]
        md += ["", f"Ausente: k={a['k']:.3f}, μ={a['mu']:.3f}, n={a['n']}. Presente: k~Gamma({pr['a_k']:.2f},{pr['b_k']:.2f}) "
               f"(média {pr['k_media']:.2f}), ρ~Gamma({pr['a_rho']:.3f},{pr['b_rho']:.2f}) (média {pr['rho_media']:.4f}), "
               f"na_borda={pr['na_borda']}, hiper_no_limite={pr['hiper_no_limite']}, disp. inícios={pr['dispersao_relativa_inicios']:.1e}.", ""]
        md += ["McNemar pareado vs bruto_referencia (só_ref / só_método, p); tempo: Wilcoxon (pares, mediana ref/método, p):", "",
               "| Método | Alvos seg | Laterais seg | ESP | Alvos protocolo | Laterais protocolo | Tempo |", "|---|---|---|---|---|---|---|"]
        for k, t in p["pareados_vs_referencia"].items():
            w = t["tempo"]
            pw = "-" if w["p_valor"] is None else f"{w['p_valor']:.3g}"
            md.append(f"| {k} | " + " | ".join(f"{t[q]['so_ref']}/{t[q]['so_metodo']}, p={t[q]['p_valor']:.3g}"
                      for q in ("alvos", "laterais", "esp", "alvos_proto", "laterais_proto"))
                      + f" | {w['n_pares']}, {f_t(w['mediana_ref_s'])}/{f_t(w['mediana_metodo_s'])}, p={pw} |")
        md += ["", "Primeira intensidade detectada por protocolo-alvo (% de 88), consistência e tempo de protocolo (desde início do 30 dB):", "",
               "| Método | " + " | ".join(ORDEM_CONDICOES + ["nenhuma"]) + " | consistentes/com det | tempo protocolo méd/med (s) |",
               "|---|" + "---:|" * 7 + "---:|---:|"]
        for k, m in [("bruto_referencia", ref)] + [(k, c["metricas"]) for k, c in p["metodos"].items()]:
            pi_ = m["primeira_intensidade"]
            md.append(f"| {k} | " + " | ".join(pct(pi_[c], 88) for c in ORDEM_CONDICOES + ["nenhuma"])
                      + f" | {m['consistentes_acima']}/{m['protocolos_com_det_estimulo']} | "
                      f"{f_t(m['tempo_protocolo_medio_s'])} / {f_t(m['tempo_protocolo_mediano_s'])} |")
        md += [f"", f"Tempo de execução: {p['tempo_execucao_s']:.0f} s.", ""]
    md.append(f"Tempo total: {s['tempo_total_s']:.0f} s.")
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
