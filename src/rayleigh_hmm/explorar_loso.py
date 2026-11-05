"""Exploratorio: validacao leave-one-subject-out (LOSO), 11 folds.

Nao altera nenhum outro arquivo; so reaproveita funcoes de experimento.py,
emissao.py, dados.py e hmm.py. Para cada participante retirado (h), TUDO que e
ajustado ou escolhido usa so os outros 10 (T): emissao Ausente, emissao
Presente, SNRs sinteticas, (pi, A, regra) do HMM e (alfa, regra) do bruto
otimizado. Depois a configuracao escolhida e aplicada as sequencias de h.

Verdade-terreno (decisao do usuario, igual a avaliacao atual): todo
participante ouviu todos os estimulos, entao TODOS os alvos 30-70 dB sao
"resposta presente"; laterais e ESP sao "ausente".

Dois bracos para a emissao Presente:
  REAL       -> ajuste hierarquico no Z acumulado dos alvos com estimulo de T;
  SINTETICO  -> ESP de T + senoide com 5 SNRs medidas nos estimulos de T.

Saidas: results/rayleigh_hmm/exploratorio_loso/{resultados.json,tabelas.md}
Execucao: .venv/bin/python src/rayleigh_hmm/explorar_loso.py
"""

import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

import dados
import emissao
import experimento as ex
import hmm
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI  # grade estendida 12 x 12 x 9

# ---------------- configuracao ----------------
PASTA = os.path.join(ex.PASTA_RESULTADOS, "exploratorio_loso")
N_PROCESSOS = 16
GRADE_PI_BUSCA, GRADE_A01_BUSCA, GRADE_A10_BUSCA = GRADE_PI, GRADE_A01, GRADE_A10
INTENSIDADES = ["30dB", "40dB", "50dB", "60dB", "70dB"]
SNR_DOC_DB = (-23.5, -18.5, -15.0, -13.5, -11.5)  # tabela do PROXIMOS_PASSOS.md (C)
# Numeros in-sample atuais (hmm_viterbi e bruto_referencia), apenas para comparacao.
IN_SAMPLE = {
    "HMM in-sample": {"det": 252, "fp_lat": 36, "fp_esp": 5, "tempo": (132.3, 85.0)},
    "bruto_ref in-sample": {"det": 252, "fp_lat": 35, "fp_esp": 4, "tempo": (134.7, 100.0)},
}
def teto_fp(seq_tr, masc_tr):
    """Teto de FP lateral usado na busca de cada fold (padrao: teto fixo). Sobrescrevivel."""
    return ex.FP_MAXIMO


METODOS = ["HMM-REAL", "HMM-SINTETICO", "bruto_otimizado_LOSO", "bruto_referencia"]

# Estado global lido pelos processos filhos (fork)
SEQ = None
COEF = None


# ---------------- dados ----------------
def carregar_coef():
    """Coeficientes dos alvos de cada arquivo: ESP (para injecao) e estimulos (para SNR)."""
    saida = []
    for caminho, participante, condicao in dados.listar_arquivos(ex.PASTA_DADOS):
        d = dados.carregar_coeficientes(caminho, **ex.SINAL)
        saida.append({"participante": participante, "condicao": condicao, "freqs": d["freq_estim"],
                      "coef": d["coef_alvo"], "valida": d["epoca_valida"]})
    return saida


def snr_por_intensidade(coef, treino):
    """SNR (dB) por intensidade a partir de S/P medio nas sequencias-alvo de `treino`.

    Estimador do PROXIMOS_PASSOS.md (C): S = |media X|^2 - media|X|^2 / n,
    P = media|X|^2 (epocas validas); media de S/P entre sequencias; depois
    SNR = (S/P) / (1 - S/P) em dB.
    """
    saida = []
    for inten in INTENSIDADES:
        razoes = []
        for c in coef:
            if c["condicao"] != inten or c["participante"] not in treino:
                continue
            x = c["coef"][c["valida"]]
            n = x.shape[0]
            p = np.mean(np.abs(x) ** 2, axis=0)
            s = np.abs(x.mean(axis=0)) ** 2 - p / n
            razoes.extend((s / p).tolist())
        r = float(np.mean(razoes))
        saida.append(10.0 * np.log10(r / (1.0 - r)))
    return saida


# ---------------- emissoes por braco ----------------
def ajustar_braco(braco, seq_treino, coef, treino, snrs):
    """Emissoes (Ausente + Presente marginal) usando so o treino."""
    z_esp = np.concatenate([s["z"] for s in seq_treino if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    emi = {"ausente": emissao.ajustar_mle(z_esp)}
    if braco == "REAL":
        alvos = [s for s in seq_treino if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
        emi["z_sintetico"] = np.concatenate([s["z"] for s in alvos])
        emi["n_sintetico"] = np.concatenate([s["n_ep"] for s in alvos])
        emi["id_sintetico"] = np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])
    else:
        esp = [(c["coef"], c["valida"]) for c in coef if c["condicao"] == "ESP" and c["participante"] in treino]
        antigo = ex.SNR_SINTETICA_DB
        ex.SNR_SINTETICA_DB = tuple(snrs)  # so em memoria; restaurado logo depois
        try:
            emi["z_sintetico"], emi["n_sintetico"], emi["id_sintetico"] = ex.gerar_presente_sintetico(esp)
        finally:
            ex.SNR_SINTETICA_DB = antigo
    emi["marginal"] = ex.ajustar_marginal(emi)
    return emi


def grupos_log_em(seqs, emi):
    return [(idx, dur, np.stack([ex.log_emissoes(seqs[i], emi, "marginal") for i in idx]))
            for idx, dur in ex.indices_por_comprimento(seqs)]


def detectores_hmm(grupos, pi, A):
    return [(idx, *ex.indicadores(hmm.viterbi_alguma_presente(le, pi, A)), dur) for idx, dur, le in grupos]


# ---------------- um (fold, braco) ----------------
def rodar_fold_braco(args):
    """Ajusta emissoes em T, busca (pi, A, regra) em T e aplica em h."""
    fold, braco = args
    t0 = time.time()
    participantes = sorted({s["participante"] for s in SEQ})
    h = participantes[fold]
    treino = set(participantes) - {h}
    i_tr = [i for i, s in enumerate(SEQ) if s["participante"] in treino]
    i_h = [i for i, s in enumerate(SEQ) if s["participante"] == h]
    seq_tr, seq_h = [SEQ[i] for i in i_tr], [SEQ[i] for i in i_h]
    masc_tr = ex.mascaras(seq_tr)
    snrs = snr_por_intensidade(COEF, treino) if braco == "SINTETICO" else None
    emi = ajustar_braco(braco, seq_tr, COEF, treino, snrs)

    grupos_tr = grupos_log_em(seq_tr, emi)
    melhores = {}
    for p1, a01, a10 in itertools.product(GRADE_PI_BUSCA, GRADE_A01_BUSCA, GRADE_A10_BUSCA):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        ex.avaliar_regras(seq_tr, masc_tr, detectores_hmm(grupos_tr, pi, A),
                          {"pi": pi.tolist(), "A": A.tolist()}, melhores, teto_fp(seq_tr, masc_tr))
    melhor = ex.melhor_geral(melhores)

    if melhor is None:  # nenhum candidato viavel: nao detecta nada (sinalizado no resultado)
        det_h, tempo_h = np.zeros(len(seq_h), bool), np.full(len(seq_h), np.nan)
        escolha = None
    else:
        pi, A = np.array(melhor["pi"]), np.array(melhor["A"])
        gd_h = detectores_hmm(grupos_log_em(seq_h, emi), pi, A)
        det_h, tempo_h = ex.decidir(gd_h, len(seq_h), melhor["regra"])
        escolha = {"pi": melhor["pi"], "A": melhor["A"], "regra": melhor["regra"],
                   "metricas_treino": melhor["metricas"]}
    masc_h = ex.mascaras(seq_h)
    hip = emi["marginal"]["hiper"]
    return {
        "fold": fold, "braco": braco, "participante": h, "indices_h": i_h,
        "detectado_h": det_h, "tempo_h": tempo_h,
        "metricas_h": ex.metricas(masc_h, det_h, tempo_h),
        "fp_lateral_h": float(det_h[masc_h["lateral"]].mean()),
        "escolha": escolha, "viavel": melhor is not None,
        "ausente": emi["ausente"], "snrs_db": snrs,
        "hiper": {k: hip[k] for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media", "na_borda",
                                      "hiper_no_limite", "massa_fora_grade", "dispersao_relativa_inicios")},
        "convergiu": all(c["sucesso"] for c in hip["corridas"]), "n_sequencias_ajuste": hip["n_sequencias"],
        "segundos": time.time() - t0,
    }


# ---------------- bruto ----------------
def rodar_bruto_fold(fold):
    participantes = sorted({s["participante"] for s in SEQ})
    h = participantes[fold]
    i_tr = [i for i, s in enumerate(SEQ) if s["participante"] != h]
    i_h = [i for i, s in enumerate(SEQ) if s["participante"] == h]
    seq_tr, seq_h = [SEQ[i] for i in i_tr], [SEQ[i] for i in i_h]
    melhores, _ = ex.buscar_bruto(seq_tr, ex.mascaras(seq_tr), teto_fp(seq_tr, ex.mascaras(seq_tr)))
    melhor = ex.melhor_geral(melhores)
    grupos = [(idx, dur, np.stack([seq_h[i]["z"] for i in idx])) for idx, dur in ex.indices_por_comprimento(seq_h)]
    gd = [(idx, *ex.indicadores(np.exp(-z) <= melhor["alfa"]), dur) for idx, dur, z in grupos]
    det, tempo = ex.decidir(gd, len(seq_h), melhor["regra"])
    masc_h = ex.mascaras(seq_h)
    return {"fold": fold, "participante": h, "indices_h": i_h, "detectado_h": det, "tempo_h": tempo,
            "alfa": melhor["alfa"], "regra": melhor["regra"], "metricas_treino": melhor["metricas"],
            "fp_lateral_h": float(det[masc_h["lateral"]].mean())}


def tarefa(args):
    return rodar_bruto_fold(args[1]) if args[0] == "bruto" else rodar_fold_braco(args[1])


# ---------------- agregacao ----------------
def juntar(resultados, n_seq):
    det, tempo = np.zeros(n_seq, bool), np.full(n_seq, np.nan)
    for r in resultados:
        det[r["indices_h"]], tempo[r["indices_h"]] = r["detectado_h"], r["tempo_h"]
    return {"detectado": det, "tempo_s": tempo}


def fmt_t(x):
    return "-" if x is None else f"{x:.1f}"


def txt_regra(r):
    return ", ".join(f"{k}={v}" for k, v in r.items() if k != "familia") + f" ({r['familia']})"


def tabela_geral(seqs, masc, c_met):
    linhas = ["| Metodo | Deteccao | FP lateral | FP ESP | Tempo medio/mediano (s) |", "|---|---|---|---|---|"]
    for nome, c in c_met.items():
        m = ex.metricas(masc, c["detectado"], c["tempo_s"])
        n = [int(c["detectado"][masc[k]].sum()) for k in ("alvo_estimulo", "lateral", "alvo_esp")]
        linhas.append(f"| {nome} | {n[0]}/440 ({100*m['deteccao']:.1f}%) | {n[1]}/528 ({100*m['fp_lateral']:.1f}%) | "
                      f"{n[2]}/88 ({100*m['fp_alvo_esp']:.1f}%) | {fmt_t(m['tempo_medio_s'])} / {fmt_t(m['tempo_mediano_s'])} |")
    for nome, v in IN_SAMPLE.items():
        linhas.append(f"| {nome} (referencia) | {v['det']}/440 ({100*v['det']/440:.1f}%) | {v['fp_lat']}/528 | "
                      f"{v['fp_esp']}/88 | {v['tempo'][0]} / {v['tempo'][1]} |")
    return linhas


def pareados(seqs, masc, c_met):
    pares = [(a, "bruto_referencia") for a in METODOS[:3]] + [("HMM-REAL", "HMM-SINTETICO")]
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


def tabelas_cortes(seqs, c_met):
    """Por intensidade e por participante (held-out)."""
    inten = ["| Intensidade | " + " | ".join(f"{m} det / FP lat / tempo" for m in METODOS) + " |",
             "|---|" + "---|" * len(METODOS)]
    dados_i = {}
    for cond in INTENSIDADES:
        cel = []
        for m in METODOS:
            r = ex.resumo(seqs, c_met[m], lambda s, cond=cond: s["condicao"] == cond)
            dados_i[f"{cond}/{m}"] = r
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} / {r['fp_lateral_detectados']}/"
                       f"{r['fp_lateral_total']} / {fmt_t(r['tempo_medio_s'])}")
        inten.append(f"| {cond} | " + " | ".join(cel) + " |")
    part = ["| Participante | " + " | ".join(f"{m} det/FP lat/FP ESP" for m in METODOS) + " |", "|---|" + "---|" * len(METODOS)]
    dados_p = {}
    for p in sorted({s["participante"] for s in seqs}):
        cel = []
        for m in METODOS:
            r = ex.resumo(seqs, c_met[m], lambda s, p=p: s["participante"] == p)
            dados_p[f"{p}/{m}"] = r
            cel.append(f"{r['deteccao_detectados']}/{r['deteccao_total']} / {r['fp_lateral_detectados']}/"
                       f"{r['fp_lateral_total']} / {r['fp_alvo_esp_detectados']}/{r['fp_alvo_esp_total']}")
        part.append(f"| {p} | " + " | ".join(cel) + " |")
    return inten, part, dados_i, dados_p


def tabelas_folds(res_arm, res_bruto):
    linhas = ["| Fold (h) | Braco | pi(Pres) | A(Aus->Pres) | A(Pres->Aus) | Regra | FP lat treino | FP lat held-out | "
              "Ausente k, mu | a_k, b_k | a_rho, b_rho | k medio | rho medio | borda/limite | convergiu | SNRs usadas (dB) |",
              "|" + "---|" * 16]
    for r in sorted(res_arm, key=lambda r: (r["fold"], r["braco"])):
        e, hp, au = r["escolha"], r["hiper"], r["ausente"]
        if e is None:
            esc = ["-"] * 4
            fpt = "-"
        else:
            esc = [f"{e['pi'][1]:g}", f"{e['A'][0][1]:g}", f"{e['A'][1][0]:g}", txt_regra(e["regra"])]
            fpt = f"{e['metricas_treino']['fp_lateral']:.3f}"
        snr = "-" if r["snrs_db"] is None else ", ".join(f"{x:.1f}" for x in r["snrs_db"])
        linhas.append(f"| {r['participante']} | {r['braco']} | {esc[0]} | {esc[1]} | {esc[2]} | {esc[3]} | {fpt} | "
                      f"{r['fp_lateral_h']:.3f} | {au['k']:.3f}, {au['mu']:.3f} | {hp['a_k']:.3g}, {hp['b_k']:.3g} | "
                      f"{hp['a_rho']:.3g}, {hp['b_rho']:.3g} | {hp['k_media']:.3g} | {hp['rho_media']:.3g} | "
                      f"{hp['na_borda']}/{hp['hiper_no_limite']} | {r['convergiu']} | {snr} |")
    bru = ["| Fold (h) | alfa | Regra | FP lat treino | FP lat held-out |", "|---|---|---|---|---|"]
    for r in sorted(res_bruto, key=lambda r: r["fold"]):
        bru.append(f"| {r['participante']} | {r['alfa']} | {txt_regra(r['regra'])} | "
                   f"{r['metricas_treino']['fp_lateral']:.3f} | {r['fp_lateral_h']:.3f} |")
    return linhas, bru


def sem_arrays(r):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
            if k not in ("indices_h", "detectado_h", "tempo_h")}


# ---------------- principal ----------------
def main():
    global SEQ, COEF
    t0 = time.time()
    SEQ, _ = ex.carregar_sequencias()
    COEF = carregar_coef()
    participantes = sorted({s["participante"] for s in SEQ})
    n_seq = len(SEQ)
    masc = ex.mascaras(SEQ)
    print(f"{n_seq} sequencias; {len(participantes)} participantes: {participantes}")

    # Checagem: estimador de SNR com os 11 participantes reproduz a tabela do documento?
    snr_todos = snr_por_intensidade(COEF, set(participantes))
    print("SNR (11 participantes):", [round(x, 2) for x in snr_todos], "doc:", SNR_DOC_DB)

    tarefas = [("bruto", f) for f in range(len(participantes))] + \
              [("hmm", (f, b)) for f in range(len(participantes)) for b in ("REAL", "SINTETICO")]
    # a primeira tarefa HMM mede o tempo; todas rodam em paralelo (fork herda SEQ/COEF)
    with mp.get_context("fork").Pool(N_PROCESSOS) as pool:
        saidas = []
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saidas.append(r)
            if "braco" in r:
                print(f"[{k}/{len(tarefas)}] {r['participante']} {r['braco']}: {r['segundos']:.0f} s "
                      f"(decorrido {time.time()-t0:.0f} s)", flush=True)
    res_bruto = [r for r in saidas if "braco" not in r]
    res_arm = [r for r in saidas if "braco" in r]

    # decisoes held-out concatenadas
    c_met = {
        "HMM-REAL": juntar([r for r in res_arm if r["braco"] == "REAL"], n_seq),
        "HMM-SINTETICO": juntar([r for r in res_arm if r["braco"] == "SINTETICO"], n_seq),
        "bruto_otimizado_LOSO": juntar(res_bruto, n_seq),
    }
    # bruto_referencia: nada ajustado, identico em todos os folds -> calculado uma vez em todas as sequencias
    _, ref = ex.buscar_bruto(SEQ, masc)
    c_met["bruto_referencia"] = {"detectado": ref["detectado"], "tempo_s": ref["tempo_s"]}

    geral = tabela_geral(SEQ, masc, c_met)
    sig_dados, sig_linhas = pareados(SEQ, masc, c_met)
    inten, part, dados_i, dados_p = tabelas_cortes(SEQ, c_met)
    folds_md, bruto_md = tabelas_folds(res_arm, res_bruto)
    excedeu = {b: sorted(r["participante"] for r in res_arm if r["braco"] == b and r["fp_lateral_h"] > ex.FP_MAXIMO)
               for b in ("REAL", "SINTETICO")}
    excedeu["bruto_otimizado_LOSO"] = sorted(r["participante"] for r in res_bruto if r["fp_lateral_h"] > ex.FP_MAXIMO)
    tempo = time.time() - t0

    md = ["# Exploratorio LOSO (leave-one-subject-out), N = 11 participantes\n",
          "Resultados FORA DA AMOSTRA por participante: tudo que e ajustado/escolhido usa so os outros 10; "
          "as decisoes sao do participante retirado, concatenadas nos 11 folds. Verdade-terreno: todos os alvos "
          "com estimulo = presente. bruto_referencia (alfa 0,05, 3 consecutivas) nao ajusta nada.\n",
          f"Grade HMM estendida: pi {len(GRADE_PI_BUSCA)} x A(Aus->Pres) {len(GRADE_A01_BUSCA)} x "
          f"A(Pres->Aus) {len(GRADE_A10_BUSCA)} x {len(ex.REGRAS)} regras; FP lateral de treino <= {ex.FP_MAXIMO}.\n",
          "## Geral (pooled, fora da amostra)\n", *geral,
          "\n## Testes pareados (A vs B; McNemar so_A-so_B; Wilcoxon nos alvos detectados por ambos)\n", *sig_linhas,
          "\n## Por intensidade (det n/88 / FP lat n/88 / tempo medio s)\n", *inten,
          "\n## Por participante held-out (det n/40 / FP lat n/48 / FP ESP n/8)\n", *part,
          "\n## Por fold: HMM\n", *folds_md,
          "\n## Por fold: bruto otimizado\n", *bruto_md,
          "\n## FP lateral held-out acima de 7% (folds)\n",
          *[f"- {k}: {len(v)} de {len(participantes)} ({', '.join(v) or '-'})" for k, v in excedeu.items()],
          f"\nSNR com os 11 participantes: {', '.join(f'{x:.1f}' for x in snr_todos)} dB "
          f"(documento: {', '.join(str(x) for x in SNR_DOC_DB)}).",
          f"\nTempo de execucao: {tempo:.0f} s."]

    saida = {"aviso": "exploratorio, LOSO (fora da amostra por participante), N=11", "participantes": participantes,
             "grades": {"pi": GRADE_PI_BUSCA, "A_aus_pres": GRADE_A01_BUSCA, "A_pres_aus": GRADE_A10_BUSCA},
             "snr_11_participantes_db": snr_todos, "snr_documento_db": list(SNR_DOC_DB),
             "folds_hmm": [sem_arrays(r) for r in sorted(res_arm, key=lambda r: (r["fold"], r["braco"]))],
             "folds_bruto": [sem_arrays(r) for r in sorted(res_bruto, key=lambda r: r["fold"])],
             "pooled": {m: ex.metricas(masc, c["detectado"], c["tempo_s"]) for m, c in c_met.items()},
             "pareados": sig_dados, "por_intensidade": dados_i, "por_participante": dados_p,
             "folds_fp_acima_limite": excedeu, "in_sample_referencia": IN_SAMPLE, "tempo_execucao_s": tempo}
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
