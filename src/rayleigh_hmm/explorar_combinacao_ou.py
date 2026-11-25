"""Exploratorio: combinacao OU de Rayleigh e HMM-REAL (detecta se QUALQUER um detectar).

LOSO (fora da amostra), sinal FC, teto de FP lateral por fold = FP lateral do
bruto_referencia no treino. Nao altera nenhum outro arquivo e nao atualiza docs.

Bracos (decisoes do participante retirado, concatenadas nos 11 folds):
  referencia   Rayleigh alfa=0,05, 3 consecutivas (nada ajustado)
  HMM-REAL     busca oficial (pi, A, regra) com teto de FP do HMM sozinho
  OU-ingenuo   referencia OU HMM-REAL, cada um com sua config oficial (FP soma; so informativo)
  OU-restrito  (pi, A, regra) do HMM oficial fixos; escolhe so o lado Rayleigh (menos graus de liberdade)
  OU-conjunto  (alfa, regra) do Rayleigh e (pi, A, regra) do HMM escolhidos JUNTOS no treino,
               maximizando a deteccao do OU com FP lateral do OU <= teto (comparacao justa)
  Rayleigh-teto-ou  Rayleigh sozinho (alfa, regra) otimizado no teto, para ver se o ganho vem do OU

Tempo de exame do OU = menor tempo entre os dois disparos.
Busca exata do OU-conjunto: HMM e Rayleigh so entram se seu FP lateral sozinho ja <= teto
(necessario, pois FP do OU >= FP de cada um); HMM deduplicado por decisao; OU avaliado
em bits para todos os pares viaveis. Desempate: menor soma dos FP laterais, depois tempo.

Saidas: results/rayleigh_hmm/exploratorio_combinacao_ou/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_combinacao_ou.py
"""
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import explorar_loso as el
import explorar_loso_fc as elfc
import experimento as ex
import hmm

PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_combinacao_ou")
N_PROCESSOS = 16
METODOS = ["referencia", "Rayleigh-teto-ou", "HMM-REAL", "OU-ingenuo", "OU-conjunto", "OU-restrito"]

SEQ = None  # lido pelos filhos (fork)
COEF = None


def ou(det_a, tempo_a, det_b, tempo_b):
    """OU de duas decisoes: detecta se um detectar; tempo = menor tempo de disparo."""
    tempo = np.fmin(tempo_a, tempo_b)  # fmin ignora nan (nao detectou)
    return det_a | det_b, tempo


def contagens(masc, det):
    return tuple(int(det[masc[k]].sum()) for k in ("alvo_estimulo", "lateral", "alvo_esp"))


def tempo_medio(masc, det, tempo):
    t = tempo[masc["alvo_estimulo"] & det]
    return float(t.mean()) if t.size else np.inf


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def candidatos_rayleigh(grupos_z, n, masc):
    """Todas as (alfa, regra) com deteccao, tempo e contagens (alvos, laterais, ESP) no treino."""
    todos = []
    for alfa in ex.GRADE_ALFA:
        gd = [(idx, *ex.indicadores(np.exp(-z) <= alfa), dur) for idx, dur, z in grupos_z]
        for regra in ex.REGRAS:
            det, tempo = ex.decidir(gd, n, regra)
            todos.append({"alfa": alfa, "regra": regra, "det": det, "tempo": tempo, "c": contagens(masc, det)})
    return todos


class BuscaOu:
    """Busca exata do melhor OU entre um conjunto fixo de candidatos Rayleigh e candidatos HMM
    passados um a um. Ordem: mais alvos, menos laterais do OU, menor soma dos laterais de cada
    um (contra detector carona), menor tempo medio do OU (calculado so nos empates)."""

    def __init__(self, rays, masc, teto_lat):
        self.rays, self.masc, self.teto_lat = rays, masc, teto_lat
        self.ra = np.stack([np.packbits(r["det"][masc["alvo_estimulo"]]) for r in rays])
        self.rl = np.stack([np.packbits(r["det"][masc["lateral"]]) for r in rays])
        self.rlat = np.array([r["c"][1] for r in rays])
        self.melhor, self.empates = None, []

    def testar(self, det, tempo, info):
        m = self.masc
        ha, hl = np.packbits(det[m["alvo_estimulo"]]), np.packbits(det[m["lateral"]])
        lat = POP[self.rl | hl].sum(axis=1)
        ok = lat <= self.teto_lat
        if not ok.any():
            return
        alvos = POP[self.ra | ha].sum(axis=1)
        soma = self.rlat + int(det[m["lateral"]].sum())
        score = np.where(ok, alvos * 10**8 - lat * 10**4 - soma, -1)
        s = score.max()
        if self.melhor is not None and s < self.melhor:
            return
        if self.melhor is None or s > self.melhor:
            self.melhor, self.empates = s, []
        for j in np.flatnonzero(score == s):
            self.empates.append((j, det.copy(), tempo.copy(), info))

    def resultado(self):
        def tempo_ou(e):
            j, det, tempo, _ = e
            return tempo_medio(self.masc, det | self.rays[j]["det"], np.fmin(tempo, self.rays[j]["tempo"]))
        j, det, tempo, info = min(self.empates, key=tempo_ou)
        return {"alfa": self.rays[j]["alfa"], "regra_r": self.rays[j]["regra"], **info}


def rodar_fold(fold):
    t0 = time.time()
    participantes = sorted({s["participante"] for s in SEQ})
    h = participantes[fold]
    treino = set(participantes) - {h}
    i_tr = [i for i, s in enumerate(SEQ) if s["participante"] in treino]
    i_h = [i for i, s in enumerate(SEQ) if s["participante"] == h]
    seq_tr, seq_h = [SEQ[i] for i in i_tr], [SEQ[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    n_tr, n_h = len(seq_tr), len(seq_h)
    teto = elfc.teto_referencia(seq_tr, masc_tr)
    n_lat_tr = int(masc_tr["lateral"].sum())
    teto_lat = int(round(teto * n_lat_tr))  # contagem inteira de laterais permitida

    emi = el.ajustar_braco("REAL", seq_tr, COEF, treino, None)
    grupos_tr = el.grupos_log_em(seq_tr, emi)
    grupos_h = el.grupos_log_em(seq_h, emi)
    z_tr = [(idx, dur, np.stack([seq_tr[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_tr)]
    z_h = [(idx, dur, np.stack([seq_h[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_h)]

    todos_r = candidatos_rayleigh(z_tr, n_tr, masc_tr)
    viaveis_r = [a for a in todos_r if a["c"][1] <= teto_lat]  # exato: OU >= cada um
    # Rayleigh sozinho no teto (separa o efeito do OU do efeito de afrouxar alfa)
    melhor_r = max(viaveis_r, key=lambda a: (a["c"][0], -a["c"][1], -tempo_medio(masc_tr, a["det"], a["tempo"])))

    busca = BuscaOu(viaveis_r, masc_tr, teto_lat)
    vistos, melhores_h = set(), {}
    for p1, a01, a10 in itertools.product(el.GRADE_PI_BUSCA, el.GRADE_A01_BUSCA, el.GRADE_A10_BUSCA):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        gd = el.detectores_hmm(grupos_tr, pi, A)
        for regra in ex.REGRAS:
            det, tempo = ex.decidir(gd, n_tr, regra)
            cand = {"pi": pi.tolist(), "A": A.tolist(), "regra": regra, "metricas": ex.metricas(masc_tr, det, tempo)}
            if contagens(masc_tr, det)[1] > teto_lat:
                continue
            atual = melhores_h.get(regra["familia"])  # HMM sozinho, mesma logica de ex.avaliar_regras
            if atual is None or ex.chave_ordem(cand) > ex.chave_ordem(atual):
                melhores_h[regra["familia"]] = {**cand, "det": det, "tempo": tempo}
            chave = det.tobytes()  # dedup por decisao (empates de tempo entre duplicatas: fica o primeiro)
            if chave not in vistos:
                vistos.add(chave)
                busca.testar(det, tempo, {"pi": pi.tolist(), "A": A.tolist(), "regra_hmm": regra})
    melhor_hmm = ex.melhor_geral(melhores_h)
    melhor_ou = busca.resultado()
    # OU-restrito: (pi, A, regra) do HMM oficial fixos; busca so o lado Rayleigh
    restrito = BuscaOu(viaveis_r, masc_tr, teto_lat)
    restrito.testar(melhor_hmm["det"], melhor_hmm["tempo"],
                    {"pi": melhor_hmm["pi"], "A": melhor_hmm["A"], "regra_hmm": melhor_hmm["regra"]})
    melhor_rest = restrito.resultado()

    def aplicar_hmm(c):
        gd = el.detectores_hmm(grupos_h, np.array(c["pi"]), np.array(c["A"]))
        return ex.decidir(gd, n_h, c["regra_hmm"] if "regra_hmm" in c else c["regra"])

    def aplicar_rayleigh(alfa, regra):
        gd = [(idx, *ex.indicadores(np.exp(-z) <= alfa), dur) for idx, dur, z in z_h]
        return ex.decidir(gd, n_h, regra)

    def aplicar_ou(c):
        return ou(*aplicar_hmm(c), *aplicar_rayleigh(c["alfa"], c["regra_r"]))

    det_ref, tempo_ref = aplicar_rayleigh(ex.REFERENCIA_BRUTO["alfa"], ex.REFERENCIA_BRUTO["regra"])
    det_hmm, tempo_hmm = aplicar_hmm(melhor_hmm)
    decisoes = {
        "referencia": (det_ref, tempo_ref),
        "HMM-REAL": (det_hmm, tempo_hmm),
        "Rayleigh-teto-ou": aplicar_rayleigh(melhor_r["alfa"], melhor_r["regra"]),
        "OU-ingenuo": ou(det_ref, tempo_ref, det_hmm, tempo_hmm),
        "OU-conjunto": aplicar_ou(melhor_ou),
        "OU-restrito": aplicar_ou(melhor_rest),
    }
    return {"fold": fold, "participante": h, "indices_h": i_h, "teto": teto, "teto_lat_contagem": teto_lat,
            "decisoes": decisoes, "n_rayleigh_viaveis": len(viaveis_r), "n_hmm_unicos": len(vistos),
            "segundos": time.time() - t0,
            "escolha_ou": melhor_ou, "escolha_ou_restrito": melhor_rest,
            "escolha_hmm": {k: melhor_hmm[k] for k in ("pi", "A", "regra")},
            "escolha_rayleigh": {"alfa": melhor_r["alfa"], "regra": melhor_r["regra"]}}


def tabela(seqs, masc, c_met):
    linhas = ["| Metodo | Deteccao /440 | FP lateral /528 | FP ESP /88 | Tempo medio / mediano (s) |", "|---|---|---|---|---|"]
    for nome in METODOS:
        c = c_met[nome]
        m = ex.metricas(masc, c["detectado"], c["tempo_s"])
        n = contagens(masc, c["detectado"])
        linhas.append(f"| {nome} | {n[0]} ({100*m['deteccao']:.1f}%) | {n[1]} ({100*m['fp_lateral']:.1f}%) | "
                      f"{n[2]} ({100*m['fp_alvo_esp']:.1f}%) | {el.fmt_t(m['tempo_medio_s'])} / {el.fmt_t(m['tempo_mediano_s'])} |")
    return linhas


def pareados(seqs, masc, c_met, pares):
    out, linhas = {}, ["| A vs B | McNemar alvos (so_A-so_B, p) | laterais | ESP | Wilcoxon tempo (n, A mais rapido/mais lento/igual, dif media A-B, p) |",
                       "|---|---|---|---|---|"]
    for a, b in pares:
        sig = ex.significancia(seqs, masc, c_met[a], c_met[b])  # so_bruto = so A, so_hmm = so B
        ambos = masc["alvo_estimulo"] & c_met[a]["detectado"] & c_met[b]["detectado"]
        d = c_met[a]["tempo_s"][ambos] - c_met[b]["tempo_s"][ambos]
        w = {"n_pares": int(ambos.sum()), "a_mais_rapido": int((d < 0).sum()), "a_mais_lento": int((d > 0).sum()),
             "iguais": int((d == 0).sum()), "dif_media_s": float(d.mean()), "p_valor": sig["tempo"]["p_valor"]}
        out[f"{a} vs {b}"] = {"mcnemar": {k: sig[k] for k in ("alvos", "laterais", "esp")}, "tempo": w}
        mc = [f"{sig[k]['so_bruto']}-{sig[k]['so_hmm']}, p={sig[k]['p_valor']:.2g}" for k in ("alvos", "laterais", "esp")]
        p = "-" if w["p_valor"] is None else f"{w['p_valor']:.2g}"
        linhas.append(f"| {a} vs {b} | {mc[0]} | {mc[1]} | {mc[2]} | n={w['n_pares']}, {w['a_mais_rapido']}/"
                      f"{w['a_mais_lento']}/{w['iguais']}, {w['dif_media_s']:+.1f}, p={p} |")
    return out, linhas


def por_intensidade(seqs, c_met):
    linhas = ["| Intensidade | " + " | ".join(f"{m} det / FP lat" for m in METODOS) + " |", "|---|" + "---|" * len(METODOS)]
    for cond in el.INTENSIDADES:
        cel = []
        for m in METODOS:
            r = ex.resumo(seqs, c_met[m], lambda s, cond=cond: s["condicao"] == cond)
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} / {r['fp_lateral_detectados']}/{r['fp_lateral_total']}")
        linhas.append(f"| {cond} | " + " | ".join(cel) + " |")
    return linhas


def main():
    global SEQ, COEF
    t0 = time.time()
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    SEQ, _ = ex.carregar_sequencias()
    COEF = el.carregar_coef()
    participantes = sorted({s["participante"] for s in SEQ})
    n_seq = len(SEQ)
    masc = ex.mascaras(SEQ)
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        res = []
        for r in pool.imap_unordered(rodar_fold, range(len(participantes))):
            res.append(r)
            print(f"[{len(res)}/{len(participantes)}] {r['participante']}: {r['segundos']:.0f} s, "
                  f"Rayleigh viaveis {r['n_rayleigh_viaveis']}, HMM unicos {r['n_hmm_unicos']}", flush=True)
    res.sort(key=lambda r: r["fold"])

    c_met = {}
    for m in METODOS:
        det, tempo = np.zeros(n_seq, bool), np.full(n_seq, np.nan)
        for r in res:
            det[r["indices_h"]], tempo[r["indices_h"]] = r["decisoes"][m]
        c_met[m] = {"detectado": det, "tempo_s": tempo}

    geral = tabela(SEQ, masc, c_met)
    pares = [("OU-conjunto", "referencia"), ("OU-conjunto", "HMM-REAL"), ("OU-conjunto", "Rayleigh-teto-ou"), ("OU-restrito", "referencia"), ("Rayleigh-teto-ou", "referencia"),
             ("OU-ingenuo", "referencia"), ("HMM-REAL", "referencia")]
    sig, sig_linhas = pareados(SEQ, masc, c_met, pares)
    inten = por_intensidade(SEQ, c_met)
    folds = ["| Held-out | Teto (lat. treino) | Escolha OU: alfa, regra Rayleigh | OU: pi(Pres), A(A>P), A(P>A), regra HMM |", "|---|---|---|---|"]
    for r in res:
        e = r["escolha_ou"]
        folds.append(f"| {r['participante']} | {r['teto']:.4f} | {e['alfa']}, {el.txt_regra(e['regra_r'])} | "
                     f"{e['pi'][1]:g}, {e['A'][0][1]:g}, {e['A'][1][0]:g}, {el.txt_regra(e['regra_hmm'])} |")
    md = ["# Exploratorio: combinacao OU Rayleigh + HMM-REAL (LOSO, FC, teto = FP lateral da referencia no treino)\n",
          "Fora da amostra. OU-ingenuo soma FP (informativo); OU-conjunto escolhe as duas configs juntas com teto unico.\n",
          "## Geral\n", *geral, "\n## Testes pareados\n", *sig_linhas,
          "\n## Por intensidade (det n/88 / FP lat n/88)\n", *inten, "\n## Escolhas por fold (OU-conjunto)\n", *folds,
          f"\nTempo de execucao: {time.time()-t0:.0f} s."]
    saida = {"aviso": "exploratorio, LOSO, combinacao OU", "participantes": participantes,
             "pooled": {m: ex.metricas(masc, c["detectado"], c["tempo_s"]) for m, c in c_met.items()},
             "contagens": {m: contagens(masc, c["detectado"]) for m, c in c_met.items()},
             "pareados": sig,
             "folds": [{k: v for k, v in r.items() if k not in ("indices_h", "decisoes")} for r in res]}
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
