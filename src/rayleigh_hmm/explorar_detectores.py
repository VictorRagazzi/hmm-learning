"""Exploratorio: outros detectores de ASSR como observacao do HMM (LOSO, dois protocolos).

Cada detector vira Z = -ln p (p-valor do teste no acumulado; ver detectores.py). Emissoes
(Ausente = Gamma MLE no ESP; Presente = marginal hierarquica, media 1 + (N-1) rho), HMM,
grades e busca ficam EXATAMENTE como no experimento oficial; so o Z muda.
Teto de FP lateral por fold = FP lateral do Rayleigh de referencia (alfa 0,05, 3 consecutivas)
no treino, calculado SEMPRE com o Z do Rayleigh (nao com o do detector).
Protocolos:
  normal      -> 1 sequencia por arquivo x tipo x frequencia (como explorar_loso_fc.py);
  concatenado -> ESP -> 30 -> ... -> 70 dB, melhor M por fold (como explorar_protocolo_loso_fp.py).
Metodos: referencia do detector (alfa 0,05, 3 consecutivas, sem ajuste), otimizado (alfa e regra
buscados no treino) e HMM.

Emissao Presente (EMISSAO_PRESENTE / --emissao): "mlii" (hiperparametros por maxima verossimilhanca
marginal, padrao) ou "mcmc" (Metropolis-dentro-de-Gibbs completo em k_s, rho_s e hiperparametros, a variante
GIBBS de monte_carlo/loso_mc.py; registra R-hat e ESS por ajuste).

Execucao (de qualquer pasta):
  .venv/bin/python src/rayleigh_hmm/explorar_detectores.py [rayleigh,msc,...] [--emissao mlii|mcmc]
      [--pasta NOME] [--tetos 0.06,0.05]
Saidas: results/rayleigh_hmm/exploratorio_detectores/{tabelas.md,resultados.json,por_detector/*.json}
(com --pasta, em results/rayleigh_hmm/NOME)
Um detector rodado antes fica em por_detector/ e e reaproveitado nas tabelas.
"""

import argparse
import os

# ---------------- configuracao ----------------
EMISSAO_PRESENTE = "mlii"  # "mlii" ou "mcmc"
TETOS_CONCAT = ["atual", "0.05"]
PASTA_NOME = None          # None: exploratorio_detectores (mlii) ou exploratorio_detectores_mcmc (mcmc)

_ap = argparse.ArgumentParser()
_ap.add_argument("detectores", nargs="?", default="", help="lista separada por virgula (padrao: todos)")
_ap.add_argument("--emissao", choices=["mlii", "mcmc"], default=EMISSAO_PRESENTE)
_ap.add_argument("--pasta", default=PASTA_NOME)
_ap.add_argument("--tetos", default=None, help="tetos do concatenado, ex.: 0.06 ou atual,0.05")
_ap.add_argument("--processos", type=int, default=None, help="processos por Pool (menos = menos RAM)")
_args = _ap.parse_args()
EMISSAO_PRESENTE = _args.emissao
if _args.tetos:
    TETOS_CONCAT = [t for t in _args.tetos.split(",") if t]
if EMISSAO_PRESENTE == "mcmc":
    # um fio de BLAS por processo: o paralelismo e por ajuste (como monte_carlo/loso_mc.py)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import glob  # noqa: E402
import itertools
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import dados
import detectores
import emissao
import experimento as ex
import explorar_loso as el
import explorar_protocolo as ep
import explorar_protocolo_loso as pl
from explorar_busca import GRADE_A01, GRADE_A10, GRADE_PI

if EMISSAO_PRESENTE == "mcmc":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "monte_carlo"))
    import mc_comum as mc  # noqa: E402
    import mcmc_completo as mcc  # noqa: E402

_argv = sys.argv
sys.argv = _argv[:1]  # explorar_protocolo_loso_fp le sys.argv ao ser importado
import explorar_protocolo_loso_fp as plfp  # noqa: E402

sys.argv = _argv

DETECTORES = [d for d in _args.detectores.split(",") if d] or list(detectores.DETECTORES)
PROTOCOLOS = ["normal", "concatenado"]
# mlii: a matriz do ajuste hierarquico ocupa centenas de MB; mcmc: o concatenado nao monta a matriz
N_PROCESSOS = {"normal": 16, "concatenado": 10 if EMISSAO_PRESENTE == "mlii" else 14}
if _args.processos:
    N_PROCESSOS = {k: _args.processos for k in N_PROCESSOS}
PASTA = os.path.join(ex.PASTA_RESULTADOS, _args.pasta or ("exploratorio_detectores" if EMISSAO_PRESENTE == "mlii"
                                                          else "exploratorio_detectores_mcmc"))
JANELAS_M = pl.JANELAS_M
EPJ = ex.EPOCAS_POR_JANELA
INTENSIDADES = ["30dB", "40dB", "50dB", "60dB", "70dB"]
ROTULO_TETO = {t: ("teto atual" if t == "atual" else f"teto {float(t) * 100:g}%") for t in TETOS_CONCAT}
CABECALHO = ("Exploratório, LOSO, 11 participantes; teto de FP lateral = FP lateral do Rayleigh de referência "
             "(α = 0,05, 3 consecutivas) no treino; canal FC, sem rejeição de épocas. Só o Z muda; emissões, "
             "HMM, grades e busca são os do experimento oficial."
             + (" Emissão Presente por MCMC completo (Metropolis-dentro-de-Gibbs em k_s, ρ_s e hiperparâmetros)."
                if EMISSAO_PRESENTE == "mcmc" else ""))

SOMAS = {}  # (participante, condicao) -> somas; herdado pelos filhos (fork)
SEQ_RAY = SEQ_DET = TETOS_NORMAL = None
DET_ATUAL = ""
HIPER_REGISTRADO = []
DIAG = {}  # diagnostico do ultimo ajuste MCMC do processo (emissao "mcmc")
ROTULO_AJUSTE = ""


# ---------------- dados ----------------
def carregar_somas_todas():
    return {(p, c): detectores.carregar_somas(cam) for cam, p, c in dados.listar_arquivos(ex.PASTA_DADOS)}


def sequencias_do_detector(seq_ray, det):
    """Copia as sequencias do Rayleigh trocando so o "z" (n_ep e metadados iguais)."""
    z_por_arquivo, contador, saida = {}, {}, []
    for s in seq_ray:
        chave = (s["participante"], s["condicao"], s["tipo"])
        if chave not in z_por_arquivo:
            fim = (np.arange(len(s["z"])) + 1) * EPJ
            z_por_arquivo[chave] = detectores.z_intervalo(SOMAS[chave[:2]], det, s["tipo"], np.zeros_like(fim), fim)[0]
        i = contador[chave] = contador.get(chave, -1) + 1  # frequencia i-esima, na ordem de carregar_sequencias
        saida.append({**s, "z": z_por_arquivo[chave][:, i]})
    return saida


def z_plano_detector(det, est, m):
    """Layout plano do protocolo concatenado (como ep.z_plano), com o Z do detector."""
    z, n, cache = np.empty(est["total"]), np.empty(est["total"]), {}
    for s in est["segs"]:
        chave = (s["participante"], s["cond"], s["tipo"])
        if chave not in cache:
            som = SOMAS[chave[:2]]
            fim = (np.arange(s["n"]) + 1) * EPJ
            ini = np.zeros_like(fim) if m is None else np.maximum(fim - m, 0)
            cache[chave] = detectores.z_intervalo(som, det, s["tipo"], ini, fim)
        zz, nn = cache[chave]
        z[s["ini"]: s["ini"] + s["n"]] = zz[:, s["i"]]
        n[s["ini"]: s["ini"] + s["n"]] = nn
    return z, n


# ---------------- metricas comuns aos dois protocolos ----------------
def resumir(det, tempo, tipo, cond):
    """Contagens e tempos de um metodo; tipo/cond: arrays por sequencia (ou segmento)."""
    alvo = (tipo == "alvo") & (cond != "ESP")
    lat, esp = tipo == "lateral", (tipo == "alvo") & (cond == "ESP")
    t = tempo[alvo & det]
    return {"det": int(det[alvo].sum()), "n_alvo": int(alvo.sum()), "fp_lat": int(det[lat].sum()), "n_lat": int(lat.sum()),
            "fp_esp": int(det[esp].sum()), "n_esp": int(esp.sum()),
            "tempo_medio_s": float(t.mean()) if t.size else None, "tempo_mediano_s": float(np.median(t)) if t.size else None,
            "por_intensidade": {c: int(det[alvo & (cond == c)].sum()) for c in INTENSIDADES},
            "n_por_intensidade": int((alvo & (cond == INTENSIDADES[0])).sum())}


def medias_hiper(lista):
    chaves = ("a_k", "b_k", "a_rho", "b_rho", "rho_media")
    return {**{k: float(np.mean([h[k] for h in lista])) for k in chaves},
            "folds_na_borda": int(sum(bool(h["na_borda"]) or bool(h["hiper_no_limite"]) for h in lista)),
            "n_folds": len(lista)}


def registrar_hiper(h):
    return {k: h[k] for k in ("a_k", "b_k", "a_rho", "b_rho", "k_media", "rho_media", "na_borda", "hiper_no_limite")}


# ---------------- emissao Presente por MCMC completo ----------------
def pontos_e_diagnostico_mcmc(emissoes):
    """MCMC completo como o ramo GIBBS de loso_mc.py: pontos (k, rho), diagnostico e medias posteriores."""
    res = mcc.ajustar_completo(emissoes, rotulo=ROTULO_AJUSTE)
    pontos = mc.pontos_mcmc(res, burn_in=mcc.BURN_IN, thin=mcc.THIN)
    d = mc.diagnosticos(res, mcc.BURN_IN, mcc.THIN)
    mp_ = d["media_posteriori"]
    hiper = {"a_k": mp_["a_k"], "b_k": mp_["b_k"], "a_rho": mp_["a_rho"], "b_rho": mp_["b_rho"],
             "k_media": float(pontos["k"].mean()), "rho_media": float(pontos["rho"].mean()),
             "na_borda": False, "hiper_no_limite": False}
    diag = {"rhat_max": float(d["rhat_max"]), "ess_min": float(d["ess_min_thin"]), "segundos": float(res["segundos"]),
            "rotulo": ROTULO_AJUSTE}
    return pontos, hiper, diag


EX_AJUSTAR_MARGINAL = ex.ajustar_marginal


def ajustar_marginal_mcmc(emissoes):
    """ML-II como hoje; so substitui os pontos pelos do MCMC completo (mesma semente)."""
    global DIAG
    marg = EX_AJUSTAR_MARGINAL(emissoes)
    marg["pontos"], marg["hiper_mcmc"], DIAG = pontos_e_diagnostico_mcmc(emissoes)
    return marg


def ajustar_emissoes_treino_mcmc(z, n, no_treino):
    """Versao de pl.ajustar_emissoes_treino com Presente por MCMC completo (sequencia = segmento)."""
    global DIAG
    seg_id = pl.EST["seg_id"]
    no_esp = pl.EST["m_alvo_esp"][seg_id] & no_treino
    no_est = pl.EST["m_alvo_estim"][seg_id] & no_treino
    aus = emissao.ajustar_mle(z[no_esp])
    emissoes = {"z_sintetico": z[no_est], "n_sintetico": n[no_est], "id_sintetico": seg_id[no_est]}
    pontos, hiper, DIAG = pontos_e_diagnostico_mcmc(emissoes)
    HIPER_REGISTRADO.append(hiper)
    log_em = np.stack([emissao.log_pdf_gamma(z, aus["k"], aus["mu"]), emissao.log_pdf_preditiva(z, n, pontos)], axis=-1)
    aviso = {"na_borda": False, "hiper_no_limite": False, "dispersao_inicios": 0.0,
             "a_k": hiper["a_k"], "b_k": hiper["b_k"], "ausente_k": float(aus["k"]), "ausente_mu": float(aus["mu"])}
    return log_em, aviso


def resumo_diagnosticos(lista):
    """Pior R-hat, menor ESS e numero de ajustes com R-hat > 1,01."""
    return {"n_ajustes": len(lista), "rhat_max": max(d["rhat_max"] for d in lista),
            "ess_min": min(d["ess_min"] for d in lista),
            "n_rhat_acima_1_01": int(sum(d["rhat_max"] > 1.01 for d in lista)), "ajustes": lista}


# ---------------- protocolo normal ----------------
def ajustar_emissao(seq_tr):
    """Ausente no ESP de treino e Presente marginal em todos os alvos com estimulo (calibracao real)."""
    z_esp = np.concatenate([s["z"] for s in seq_tr if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    emi = {"ausente": emissao.ajustar_mle(z_esp)}
    alvos = [s for s in seq_tr if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    emi["z_sintetico"] = np.concatenate([s["z"] for s in alvos])
    emi["n_sintetico"] = np.concatenate([s["n_ep"] for s in alvos])
    emi["id_sintetico"] = np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])
    emi["marginal"] = ex.ajustar_marginal(emi)
    return emi


def tarefa_normal(args):
    """Um fold do protocolo normal: "hmm" ou "otimizado" (bruto com alfa e regra buscados)."""
    global ROTULO_AJUSTE
    modo, fold = args
    t0 = time.time()
    h = sorted({s["participante"] for s in SEQ_DET})[fold]
    i_h = [i for i, s in enumerate(SEQ_DET) if s["participante"] == h]
    seq_tr = [s for s in SEQ_DET if s["participante"] != h]
    seq_h = [SEQ_DET[i] for i in i_h]
    masc_tr, teto = ex.mascaras(seq_tr), TETOS_NORMAL[fold]
    out = {"modo": modo, "fold": fold, "participante": h, "indices_h": i_h, "teto": teto}
    if modo == "hmm":
        ROTULO_AJUSTE = f"{DET_ATUAL} normal fold {fold} {h}"
        emi = ajustar_emissao(seq_tr)
        grupos_tr = el.grupos_log_em(seq_tr, emi)
        melhores = {}
        for p1, a01, a10 in itertools.product(GRADE_PI, GRADE_A01, GRADE_A10):
            pi, A = np.array([1.0 - p1, p1]), np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
            ex.avaliar_regras(seq_tr, masc_tr, el.detectores_hmm(grupos_tr, pi, A),
                              {"pi": pi.tolist(), "A": A.tolist()}, melhores, teto)
        out["hiper"] = registrar_hiper(emi["marginal"].get("hiper_mcmc", emi["marginal"]["hiper"]))
        if EMISSAO_PRESENTE == "mcmc":
            out["diag"] = {**DIAG, "participante": h}
        melhor = ex.melhor_geral(melhores)
        if melhor is not None:
            pi, A = np.array(melhor["pi"]), np.array(melhor["A"])
            gd = el.detectores_hmm(el.grupos_log_em(seq_h, emi), pi, A)
    else:
        melhores, _ = ex.buscar_bruto(seq_tr, masc_tr, teto)
        melhor = ex.melhor_geral(melhores)
        if melhor is not None:
            gd = [(idx, *ex.indicadores(np.exp(-np.stack([seq_h[i]["z"] for i in idx])) <= melhor["alfa"]), dur)
                  for idx, dur in ex.indices_por_comprimento(seq_h)]
    out["viavel"] = melhor is not None
    if melhor is None:
        out["detectado_h"], out["tempo_h"] = np.zeros(len(seq_h), bool), np.full(len(seq_h), np.nan)
    else:
        out["detectado_h"], out["tempo_h"] = ex.decidir(gd, len(seq_h), melhor["regra"])
        out["escolha"] = {k: melhor[k] for k in ("alfa", "pi", "A", "regra") if k in melhor}
    out["segundos"] = time.time() - t0
    return out


def rodar_normal(det, seq_ray):
    global SEQ_DET, DET_ATUAL
    SEQ_DET = sequencias_do_detector(seq_ray, det)
    DET_ATUAL = det
    n_seq, masc = len(SEQ_DET), ex.mascaras(SEQ_DET)
    tipo = np.array([s["tipo"] for s in SEQ_DET])
    cond = np.array([s["condicao"] for s in SEQ_DET])
    tarefas = [(m, f) for m in ("hmm", "otimizado") for f in range(len(TETOS_NORMAL))]
    t0, saidas = time.time(), []
    with mp.get_context("fork").Pool(N_PROCESSOS["normal"]) as pool:
        for k, r in enumerate(pool.imap_unordered(tarefa_normal, tarefas), 1):
            saidas.append(r)
            print(f"  [{det} normal {k}/{len(tarefas)}] {r['modo']} {r['participante']}: {r['segundos']:.0f}s "
                  f"(decorrido {time.time() - t0:.0f}s)", flush=True)
    ref = ex.calcular_referencia(SEQ_DET, masc)
    c = {"referencia": ref, **{m: el.juntar([r for r in saidas if r["modo"] == m], n_seq) for m in ("hmm", "otimizado")}}
    hip = [r["hiper"] for r in saidas if r["modo"] == "hmm"]
    diag = {"diagnosticos": resumo_diagnosticos([r["diag"] for r in saidas if "diag" in r])} if EMISSAO_PRESENTE == "mcmc" else {}
    return {
        **diag,
        "metodos": {m: resumir(v["detectado"], v["tempo_s"], tipo, cond) for m, v in c.items()},
        "hiper": {"normal": medias_hiper(hip)},
        "avisos": {"folds_inviaveis": [f"{r['participante']} ({r['modo']})" for r in saidas if not r["viavel"]],
                   "ajuste_na_borda": [r["participante"] for r in saidas if r["modo"] == "hmm"
                                       and (r["hiper"]["na_borda"] or r["hiper"]["hiper_no_limite"])]},
        "folds": [{"modo": r["modo"], "participante": r["participante"], "teto": r["teto"], **r.get("escolha", {})}
                  for r in sorted(saidas, key=lambda r: (r["modo"], r["fold"]))],
    }


# ---------------- protocolo concatenado ----------------
def ajustar_hierarquico_registrando(*args, **kw):
    h = HIPER_ORIGINAL(*args, **kw)
    HIPER_REGISTRADO.append(registrar_hiper(h))
    return h


HIPER_ORIGINAL = emissao.ajustar_hierarquico


def tarefa_concatenada(args):
    """plfp.tarefa (busca com teto atual e 5%) mais os hiperparametros do ajuste hierarquico."""
    global ROTULO_AJUSTE
    HIPER_REGISTRADO.clear()
    ROTULO_AJUSTE = f"{DET_ATUAL} concatenado fold {pl.PARTICIPANTES[args[0]]} M={args[1]}"
    r = plfp.tarefa(args)
    r["hiper"] = dict(HIPER_REGISTRADO[-1])
    if EMISSAO_PRESENTE == "mcmc":
        r["diag"] = {**DIAG, "participante": pl.PARTICIPANTES[args[0]], "m": str(args[1])}
    return r


def agregar_concatenado(saidas, tk, nome):
    """Melhor M por fold (criterio no treino) e juntada fora da amostra, como no script original."""
    n_seg = pl.N_SEG
    det, tempo = np.zeros(n_seg, bool), np.full(n_seg, np.nan)
    folds = []
    for f, h in enumerate(pl.PARTICIPANTES):
        no_h = pl.PART_SEG == h
        melhor, m_esc = None, None
        for m in JANELAS_M:
            r = next(x for x in saidas if x["fold"] == f and x["m"] == m)[nome][tk]
            if r.get("c") is not None and (melhor is None or ep.chave(r["c"]) > ep.chave(melhor["c"])):
                melhor, m_esc = r, m
        if melhor is None:
            folds.append({"participante": h, "viavel": False})
            continue
        det[no_h], tempo[no_h] = melhor["detectado"][no_h], melhor["tempo_s"][no_h]
        folds.append({"participante": h, "viavel": True, "m": str(m_esc), "regra": melhor["regra"],
                      **{k: melhor[k] for k in ("alfa", "pi", "A") if k in melhor}})
    return det, tempo, folds


def preparar_concatenado():
    """Estrutura do protocolo e tetos (Rayleigh de referencia) compartilhados por todos os detectores."""
    pl.ARQ = ep.carregar_fases()
    pl.EST = est = ep.montar_estrutura(pl.ARQ)
    pl.PARTICIPANTES = sorted({p for p, _ in pl.ARQ})
    pl.PART_SEG = np.array([s["participante"] for s in est["segs"]])
    z_cum, _ = ep.z_plano(pl.ARQ, est, None)
    ref = ep.finalizar(est, {"alfa": ex.REFERENCIA_BRUTO["alfa"], "regra": ex.REFERENCIA_BRUTO["regra"]},
                       np.exp(-z_cum) <= ex.REFERENCIA_BRUTO["alfa"])
    pl.TETOS = [int(ref["detectado"][est["m_lateral"] & (pl.PART_SEG != h)].sum()) for h in pl.PARTICIPANTES]
    return z_cum


def salvar_por_segmento(det, est, por_seg):
    """CSV por segmento (detectado, tempo, tipo, condicao, participante, frequencia) de cada metodo."""
    import csv
    with open(os.path.join(PASTA, f"por_segmento_{det}_concatenado.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "participante", "tipo", "condicao", "freq_hz", "detectado", "tempo_s"])
        for nome, c in por_seg.items():
            for k, sg in enumerate(est["segs"]):
                t = c["tempo_s"][k]
                w.writerow([nome, sg["participante"], sg["tipo"], sg["cond"], sg["freq"], int(c["detectado"][k]),
                            "" if np.isnan(t) else f"{t:.1f}"])


def numeros_tabela_i(est, hmm, ray, rotulo_teto):
    """Numeros da Tabela I: McNemar exato (A = Rayleigh otimizado, B = HMM), Wilcoxon e 1a avaliacao em 70 dB."""
    tipo, cond = est["tipo"], est["cond"]
    t = ep.testes_pareados(est, ray, hmm)  # ref = Rayleigh otimizado, metodo = HMM
    primeira = ex.MINIMO_AVALIACOES * EPJ * est["dur_epoca"]
    d70 = hmm["detectado"] & (tipo == "alvo") & (cond == "70dB")
    na_primeira = d70 & (hmm["tempo_s"] <= primeira + 1e-9)
    return {"teto": rotulo_teto, "mcnemar_so_rayleigh_so_hmm": {"alvos": t["alvos"], "laterais": t["laterais"]},
            "wilcoxon_tempo": t["tempo"],
            "hmm_70dB": {"deteccoes": int(d70.sum()), "na_primeira_avaliacao": int(na_primeira.sum()),
                         "primeira_avaliacao_s": float(np.unique(primeira[d70]).min()) if d70.any() else None}}


def rodar_concatenado(det):
    global DET_ATUAL
    DET_ATUAL = det
    est = pl.EST
    pl.ZN = {m: z_plano_detector(det, est, m) for m in JANELAS_M}
    pl.ZN["teto"] = pl.TETOS
    z_cum = pl.ZN[None][0]
    ref = ep.finalizar(est, {"alfa": ex.REFERENCIA_BRUTO["alfa"], "regra": ex.REFERENCIA_BRUTO["regra"]},
                       np.exp(-z_cum) <= ex.REFERENCIA_BRUTO["alfa"])
    plfp.TETOS_FP = [float(t) for t in TETOS_CONCAT if t != "atual"]
    emissao.ajustar_hierarquico = ajustar_hierarquico_registrando
    tarefas = [(f, m) for m in JANELAS_M for f in range(len(pl.PARTICIPANTES))]
    t0, saidas = time.time(), []
    try:
        with mp.get_context("fork").Pool(N_PROCESSOS["concatenado"]) as pool:
            for k, r in enumerate(pool.imap_unordered(tarefa_concatenada, tarefas), 1):
                saidas.append(r)
                print(f"  [{det} concatenado {k}/{len(tarefas)}] fold {pl.PARTICIPANTES[r['fold']]} M={r['m']}: "
                      f"{r['segundos']:.0f}s (decorrido {time.time() - t0:.0f}s)", flush=True)
    finally:
        emissao.ajustar_hierarquico = HIPER_ORIGINAL
    tipo, cond = est["tipo"], est["cond"]
    metodos, folds = {"referencia": resumir(ref["detectado"], ref["tempo_s"], tipo, cond)}, {}
    por_seg = {"referencia": ref}
    for tk in TETOS_CONCAT:
        for nome, rotulo in (("rayleigh_loso", "otimizado"), ("hmm_loso", "hmm")):
            d, t, fs = agregar_concatenado(saidas, tk, nome)
            metodos[f"{rotulo} [{ROTULO_TETO[tk]}]"] = resumir(d, t, tipo, cond)
            folds[f"{rotulo} [{ROTULO_TETO[tk]}]"] = fs
            por_seg[f"{rotulo} [{ROTULO_TETO[tk]}]"] = {"detectado": d, "tempo_s": t}
    extra = {}
    if EMISSAO_PRESENTE == "mcmc":
        salvar_por_segmento(det, est, por_seg)
        tks = [t for t in TETOS_CONCAT if t != "atual"]
        if tks:
            extra["tabela_I"] = numeros_tabela_i(est, por_seg[f"hmm [{ROTULO_TETO[tks[0]]}]"],
                                                 por_seg[f"otimizado [{ROTULO_TETO[tks[0]]}]"], ROTULO_TETO[tks[0]])
    por_m = {str(m): [r["hiper"] for r in saidas if r["m"] == m] for m in JANELAS_M}
    avisos_aj = {str(m): [a for a in (r["aviso"] for r in saidas if r["m"] == m)] for m in JANELAS_M}
    diag = {"diagnosticos": resumo_diagnosticos([r["diag"] for r in saidas])} if EMISSAO_PRESENTE == "mcmc" else {}
    return {
        **diag, **extra,
        "metodos": metodos,
        "hiper": {f"M={m}": medias_hiper(v) for m, v in por_m.items()},
        "avisos": {"folds_inviaveis": sorted({f"{x['participante']} ({k})" for k, v in folds.items() for x in v if not x["viavel"]}),
                   "ajuste_na_borda_por_m": {m: int(sum(a["na_borda"] or a["hiper_no_limite"] for a in v)) for m, v in avisos_aj.items()},
                   "dispersao_inicios_acima_1e-3_por_m": {m: int(sum(a["dispersao_inicios"] > 1e-3 for a in v)) for m, v in avisos_aj.items()}},
        "folds": folds,
    }


# ---------------- calibracao sob H0 ----------------
def calibracao_h0(seq):
    """Z medio e fracao de p <= 0,05 em laterais (todas as condicoes) e alvos ESP: 1a janela, final e todas."""
    grupos = {"laterais": [s for s in seq if s["tipo"] == "lateral"],
              "alvos ESP": [s for s in seq if s["tipo"] == "alvo" and s["condicao"] == "ESP"]}
    saida = {}
    for nome, ss in grupos.items():
        posicoes = {"primeira": np.array([s["z"][0] for s in ss]), "final": np.array([s["z"][-1] for s in ss]),
                    "todas": np.concatenate([s["z"] for s in ss])}
        saida[nome] = {p: {"z_medio": float(z.mean()), "frac_p_005": float((np.exp(-z) <= 0.05).mean()), "n": int(z.size)}
                       for p, z in posicoes.items()}
    return saida


# ---------------- tabelas ----------------
def pct(x, n):
    return f"{100 * x / n:.1f}%".replace(".", ",")


def seg(x):
    return "-" if x is None else f"{x:.0f}"


def num(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def escrever_md(res, total):
    ordem = [d for d in detectores.DETECTORES if d in res]
    md = [f"# Detectores de ASSR como observação do HMM", "", CABECALHO, "",
          "Taxas em %: detecção sobre 440 alvos de estímulo (30 a 70 dB), FP lateral sobre 528 laterais, FP ESP sobre 88 alvos ESP. "
          "Tempo médio/mediano em s, só entre os alvos detectados. Normal: uma sequência por arquivo × tipo × frequência. "
          "Concatenado: ESP → 30 → … → 70 dB, M escolhido por fold.", "",
          "## Resultado principal", "",
          "| Protocolo | Detector | Método | Detecção | FP lateral | FP ESP | Tempo médio/mediano (s) |", "|---|---|---|---|---|---|---|"]
    for prot in PROTOCOLOS:
        for d in ordem:
            if prot not in res[d]:
                continue
            for nome, m in res[d][prot]["metodos"].items():
                md.append(f"| {prot} | {d} | {nome} | {pct(m['det'], m['n_alvo'])} | {pct(m['fp_lat'], m['n_lat'])} | "
                          f"{pct(m['fp_esp'], m['n_esp'])} | {seg(m['tempo_medio_s'])} / {seg(m['tempo_mediano_s'])} |")
    md += ["", "## Detecção por intensidade (% de 88 alvos por condição)", "",
           "| Protocolo | Detector | Método | " + " | ".join(INTENSIDADES) + " |", "|---|---|---|" + "---|" * len(INTENSIDADES)]
    for prot in PROTOCOLOS:
        for d in ordem:
            if prot not in res[d]:
                continue
            for nome, m in res[d][prot]["metodos"].items():
                md.append(f"| {prot} | {d} | {nome} | " + " | ".join(pct(m["por_intensidade"][c], m["n_por_intensidade"]) for c in INTENSIDADES) + " |")
    md += ["", "## Calibração sob H0 (protocolo normal; esperado: Z médio ≈ 1 e ≈ 5,0% com p ≤ 0,05)", "",
           "| Detector | Grupo | Janela | Z médio | p ≤ 0,05 | n |", "|---|---|---|---:|---:|---:|"]
    for d in ordem:
        for grupo, v in res[d]["calibracao"].items():
            for pos, x in v.items():
                md.append(f"| {d} | {grupo} | {pos} | {num(x['z_medio'])} | {num(100 * x['frac_p_005'], 1)}% | {x['n']} |")
    md += ["", "## Hiperparâmetros médios do Presente (média dos folds)", "",
           "| Protocolo | Detector | Ajuste | a_k | b_k | a_rho | b_rho | rho médio | folds na borda |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for prot in PROTOCOLOS:
        for d in ordem:
            for chave, h in res[d].get(prot, {}).get("hiper", {}).items():
                md.append(f"| {prot} | {d} | {chave} | {num(h['a_k'], 3)} | {num(h['b_k'], 3)} | {num(h['a_rho'], 3)} | "
                          f"{num(h['b_rho'], 1)} | {num(h['rho_media'], 4)} | {h['folds_na_borda']}/{h['n_folds']} |")
    ti = res.get("rayleigh", {}).get("concatenado", {}).get("tabela_I")
    if ti:
        me = res["rayleigh"]["concatenado"]["metodos"]
        h, o = me[f"hmm [{ti['teto']}]"], me[f"otimizado [{ti['teto']}]"]
        md += ["", f"## Rayleigh concatenado, MCMC — números para a Tabela I ({ti['teto']})", "",
               "| Método | Detecção | FP lateral | FP ESP | Tempo médio/mediano (s) |", "|---|---|---|---|---|"]
        for nome, m in (("HMM", h), ("Rayleigh otimizado", o)):
            md.append(f"| {nome} | {pct(m['det'], m['n_alvo'])} | {pct(m['fp_lat'], m['n_lat'])} | {pct(m['fp_esp'], m['n_esp'])} | "
                      f"{seg(m['tempo_medio_s'])} / {seg(m['tempo_mediano_s'])} |")
        md += ["", "Detecção por intensidade (% de 88 alvos):", "", "| Método | " + " | ".join(INTENSIDADES) + " |",
               "|---|" + "---|" * len(INTENSIDADES)]
        for nome, m in (("HMM", h), ("Rayleigh otimizado", o)):
            md.append(f"| {nome} | " + " | ".join(pct(m["por_intensidade"][c], m["n_por_intensidade"]) for c in INTENSIDADES) + " |")
        a, l, w = ti["mcnemar_so_rayleigh_so_hmm"]["alvos"], ti["mcnemar_so_rayleigh_so_hmm"]["laterais"], ti["wilcoxon_tempo"]
        pv = lambda p: "-" if p is None else f"{p:.4g}".replace(".", ",")
        md += ["", "McNemar exato, A = Rayleigh otimizado, B = HMM (só A / só B / p):", "",
               f"- Alvos: {a['so_ref']} / {a['so_metodo']} / p = {pv(a['p_valor'])}",
               f"- Laterais: {l['so_ref']} / {l['so_metodo']} / p = {pv(l['p_valor'])}",
               f"- Wilcoxon pareado do tempo nos alvos detectados por ambos: n = {w['n_pares']}, mediana Rayleigh "
               f"{num(w['mediana_ref_s'], 1)} s, mediana HMM {num(w['mediana_metodo_s'], 1)} s, p = {pv(w['p_valor'])}", "",
               f"Em 70 dB, detecções do HMM na primeira avaliação permitida ({num(ti['hmm_70dB']['primeira_avaliacao_s'], 0)} s): "
               f"{ti['hmm_70dB']['na_primeira_avaliacao']} de {ti['hmm_70dB']['deteccoes']}.",
               "", "Decisões e tempos por segmento: por_segmento_rayleigh_concatenado.csv."]
    if EMISSAO_PRESENTE == "mcmc":
        md += ["", "## Convergência do MCMC (R-hat máximo e ESS mínimo entre a_k, b_k, a_rho, b_rho; 4 cadeias)", "",
               "| Detector | Protocolo | Ajustes | Pior R-hat | Menor ESS | Ajustes com R-hat > 1,01 |", "|---|---|---:|---:|---:|---:|"]
        for d in ordem:
            for prot in PROTOCOLOS:
                g = res[d].get(prot, {}).get("diagnosticos")
                if g:
                    md.append(f"| {d} | {prot} | {g['n_ajustes']} | {num(g['rhat_max'], 3)} | {num(g['ess_min'], 0)} | "
                              f"{g['n_rhat_acima_1_01']} |")
    md += ["", "## Avisos", ""]
    for d in ordem:
        for prot in PROTOCOLOS:
            if prot in res[d]:
                a = res[d][prot]["avisos"]
                md.append(f"- {d}, {prot}: " + "; ".join(f"{k}: {v if v else 'nenhum'}" for k, v in a.items()))
    md += ["", f"Tempo desta execução: {total:.0f} s."]
    with open(os.path.join(PASTA, "tabelas.md"), "w") as f:
        f.write("\n".join(md) + "\n")


# ---------------- principal ----------------
def carregar_resultados_anteriores():
    res = {}
    for caminho in glob.glob(os.path.join(PASTA, "por_detector", "*.json")):
        res[os.path.basename(caminho)[:-5]] = json.load(open(caminho))
    return res


def main():
    global SOMAS, TETOS_NORMAL
    assert ex.SINAL == {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}
    t0 = time.time()
    if EMISSAO_PRESENTE == "mcmc":  # antes dos Pools (o fork herda)
        ex.ajustar_marginal = ajustar_marginal_mcmc
        pl.ajustar_emissoes_treino = ajustar_emissoes_treino_mcmc
    os.makedirs(os.path.join(PASTA, "por_detector"), exist_ok=True)
    print("carregando somas acumuladas dos arquivos...", flush=True)
    SOMAS = carregar_somas_todas()
    seq_ray, _ = ex.carregar_sequencias()
    participantes = sorted({s["participante"] for s in seq_ray})
    TETOS_NORMAL = []
    for h in participantes:
        seq_tr = [s for s in seq_ray if s["participante"] != h]
        TETOS_NORMAL.append(ex.calcular_referencia(seq_tr, ex.mascaras(seq_tr))["metricas"]["fp_lateral"])
    if "concatenado" in PROTOCOLOS:
        z_cum_ray = preparar_concatenado()
    res = carregar_resultados_anteriores()
    for det in DETECTORES:
        t_det = time.time()
        print(f"=== {det} ===", flush=True)
        r = {}
        seq_det = sequencias_do_detector(seq_ray, det)
        if det == "rayleigh":
            dif = max(float(np.max(np.abs(a["z"] - b["z"]))) for a, b in zip(seq_det, seq_ray))
            print(f"  rayleigh por somas vs dados.rayleigh_z_acumulado (normal): diferenca maxima {dif:.2e}", flush=True)
            dif_c = float(np.max(np.abs(z_plano_detector(det, pl.EST, None)[0] - z_cum_ray))) if "concatenado" in PROTOCOLOS else None
            print(f"  idem concatenado (M=None): diferenca maxima {dif_c}", flush=True)
        r["calibracao"] = calibracao_h0(seq_det)
        if "normal" in PROTOCOLOS:
            r["normal"] = rodar_normal(det, seq_ray)
        if "concatenado" in PROTOCOLOS:
            r["concatenado"] = rodar_concatenado(det)
        r["segundos"] = time.time() - t_det
        res[det] = r
        with open(os.path.join(PASTA, "por_detector", f"{det}.json"), "w") as f:
            json.dump(r, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        for prot in PROTOCOLOS:
            for nome, m in r.get(prot, {}).get("metodos", {}).items():
                print(f"  {prot:12s} {nome:28s} det {pct(m['det'], m['n_alvo'])} FP lat {pct(m['fp_lat'], m['n_lat'])} "
                      f"FP ESP {pct(m['fp_esp'], m['n_esp'])} tempo {seg(m['tempo_medio_s'])}/{seg(m['tempo_mediano_s'])}", flush=True)
    total = time.time() - t0
    with open(os.path.join(PASTA, "resultados.json"), "w") as f:
        json.dump({"aviso": CABECALHO, "detectores": res, "tempo_ultima_execucao_s": total}, f, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    escrever_md(res, total)
    print(f"tempo total {total:.0f}s; saidas em {PASTA}", flush=True)


if __name__ == "__main__":
    main()
