"""Diagnostico exploratorio: a fase da resposta e estavel ao longo da gravacao?

O Rayleigh acumulado supoe fase constante. Aqui, para cada sequencia
(arquivo x frequencia alvo), usamos fasores unitarios u_k = exp(i*theta_k) das
epocas VALIDAS, mantendo o indice ORIGINAL da epoca como tempo (epocas
rejeitadas deixam lacunas). Analises: fasor medio por bloco, indice de
estabilidade S, deriva de fase (frequencia residual), assinatura de relogio
(delta_f proporcional a f), primeira x segunda metade e impacto no Z final.
Controle: sequencias-alvo dos arquivos ESP. Somente descritivo; nao altera
detector nem HMM.

Execucao (da raiz do repositorio):
    .venv/bin/python src/rayleigh_hmm/diagnostico_fase.py
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import dados

# ============================================================
# Configuracao
# ============================================================
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PASTA_DADOS = os.path.join(RAIZ, "data")
PASTA_SAIDA = os.path.join(RAIZ, "results", "rayleigh_hmm", "diagnostico_fase")
SINAL = {"combinacao": "media", "canais": [0, 13, 15, 1], "rejeicao_fator": 2.0}

BLOCOS = (30, 60)          # tamanhos de bloco em epocas originais (30 = principal)
MIN_EPOCAS_BLOCO = 10      # bloco com menos epocas validas e descartado
MIN_BLOCOS = 3             # sequencia precisa de >= 3 blocos
P_FORTE = 1e-3             # sequencia "forte": Rayleigh final p < P_FORTE
P_FORTE_FROUXO = 1e-2      # sensibilidade
N_SIM = 2000               # simulacoes dos nulos de S
DF_MAX = 0.02             # deriva maxima testada, Hz (ciclos/s); limitada a 1/(2B) para evitar alias entre blocos
PASSO_DF = 2e-5
N_FIGURA_FASE = 12
SEMENTE = 0
ALFA = 0.05


# ============================================================
# Carga
# ============================================================
def carregar_alvos():
    """Uma entrada por (arquivo, frequencia alvo): fasores unitarios, mascara e metadados."""
    seqs = []
    for caminho, part, cond in dados.listar_arquivos(PASTA_DADOS):
        d = dados.carregar_coeficientes(caminho, **SINAL)
        c = d["coef_alvo"]
        u = c / (np.abs(c) + dados.EPS)
        for i, f in enumerate(d["freq_estim"]):
            seqs.append({"arquivo": os.path.basename(caminho), "participante": part, "condicao": cond,
                         "esp": cond == "ESP", "freq": float(f), "fs": d["fs"],
                         "dur": d["duracao_epoca_s"], "n_epocas": d["n_epocas"],
                         "u": u[:, i], "valida": d["epoca_valida"]})
    return seqs


# ============================================================
# Nucleo: blocos, S, deriva
# ============================================================
def blocos(seq, b):
    """Soma de fasores por bloco de b epocas originais. Retorna dict ou None."""
    idx = np.flatnonzero(seq["valida"])
    lab = idx // b
    ids = np.unique(lab)
    soma, n, t, rotulo = [], [], [], []
    for k in ids:
        sel = idx[lab == k]
        if len(sel) < MIN_EPOCAS_BLOCO:
            continue
        soma.append(seq["u"][sel].sum()); n.append(len(sel))
        t.append((sel.mean() + 0.5) * seq["dur"]); rotulo.append(k)
    if len(n) < MIN_BLOCOS:
        return None
    soma, n = np.array(soma), np.array(n)
    m = soma / n
    return {"soma": soma, "n": n, "m": m, "phi": np.angle(m), "mod": np.abs(m),
            "p_bloco": np.exp(-n * np.abs(m) ** 2), "t": np.array(t), "idx": idx, "lab": lab,
            "rotulos": np.array(rotulo)}


def indice_S(soma):
    """S = |sum n_b m_b| / sum n_b |m_b| (soma pode ter eixo de simulacoes na frente)."""
    return np.abs(soma.sum(-1)) / np.abs(soma).sum(-1)


def nulos_S(seq, bl, rng):
    """S sob (a) rotacao aleatoria das fases de bloco e (b) embaralhamento das epocas validas."""
    s_rot = indice_S(np.abs(bl["soma"]) * np.exp(1j * rng.uniform(0, 2 * np.pi, (N_SIM, len(bl["n"])))))
    idx, lab = bl["idx"], bl["lab"]
    cols = np.searchsorted(bl["rotulos"], lab)
    ok = np.isin(lab, bl["rotulos"])
    ind = np.zeros((ok.sum(), len(bl["n"])))
    ind[np.arange(ok.sum()), cols[ok]] = 1
    uv = seq["u"][idx][ok]
    perm = np.argsort(rng.random((N_SIM, ok.sum())), axis=1)
    s_emb = indice_S(uv[perm] @ ind)
    return s_rot, s_emb


def grade_df(b):
    lim = min(DF_MAX, 0.5 / b)
    return np.arange(-lim, lim + 1e-9, PASSO_DF)


def ajustar_deriva(bl, b):
    """Inclinacao (Hz) que maximiza |sum w_b exp(i(phi_b - 2 pi s t_b))|, w = n|m|^2."""
    grade = grade_df(b)
    w = bl["n"] * bl["mod"] ** 2
    fase = np.exp(1j * (bl["phi"][None, :] - 2 * np.pi * grade[:, None] * bl["t"][None, :]))
    return float(grade[np.argmax(np.abs(fase @ w))])


def z_final(seq, df=0.0):
    """Z = |soma u|^2 / N no fim, com a deriva df removida (tempo = indice original)."""
    idx = np.flatnonzero(seq["valida"])
    t = (idx + 0.5) * seq["dur"]
    v = seq["u"][idx] * np.exp(-2j * np.pi * df * t)
    return float(np.abs(v.sum()) ** 2 / len(idx)), len(idx)


def analisar(seq, b, rng):
    """Todas as estatisticas de uma sequencia para o tamanho de bloco b (None se curta)."""
    bl = blocos(seq, b)
    z0, n = z_final(seq)
    r = {"arquivo": seq["arquivo"], "participante": seq["participante"], "condicao": seq["condicao"],
         "freq": seq["freq"], "fs": seq["fs"], "n_validas": n, "z_final": z0, "p_final": float(np.exp(-z0))}
    if bl is None:
        return None, r
    s = float(indice_S(bl["soma"]))
    s_rot, s_emb = nulos_S(seq, bl, rng)
    df = ajustar_deriva(bl, b)
    w = bl["n"] * bl["mod"] ** 2
    corr = bl["soma"] * np.exp(-2j * np.pi * df * bl["t"])
    s_apos = float(indice_S(corr))
    z1, _ = z_final(seq, df)
    # primeira x segunda metade (metade pelo indice original)
    meio = seq["n_epocas"] // 2
    v = np.flatnonzero(seq["valida"])
    a, c = seq["u"][v[v < meio]], seq["u"][v[v >= meio]]
    dif = float(np.angle(c.mean() * np.conj(a.mean()))) if len(a) and len(c) else np.nan
    r.update({"n_blocos": len(bl["n"]), "S": s, "S_nulo_rot_med": float(np.median(s_rot)),
              "S_nulo_rot_q95": float(np.quantile(s_rot, 0.95)),
              "p_estavel": float((1 + (s_rot >= s).sum()) / (N_SIM + 1)),  # S alto = fase comum
              "S_nulo_emb_med": float(np.median(s_emb)), "S_nulo_emb_q05": float(np.quantile(s_emb, 0.05)),
              "p_deriva": float((1 + (s_emb <= s).sum()) / (N_SIM + 1)),  # S baixo = deriva
              "df_hz": df, "S_apos": s_apos, "z_apos": z1, "razao_z": z1 / max(z0, 1e-12),
              "dif_metades": dif, "phi_blocos": bl["phi"].tolist(), "mod_blocos": bl["mod"].tolist(),
              "t_blocos": bl["t"].tolist(), "n_blocos_ep": bl["n"].tolist()})
    return bl, r


# ============================================================
# Resumos
# ============================================================
def med(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if len(x) else float("nan")


def iqr(x):
    return [float(np.nanquantile(x, 0.25)), float(np.nanquantile(x, 0.75))] if len(x) else [np.nan, np.nan]


def media_circular(ang):
    ang = np.asarray(ang)[np.isfinite(ang)]
    if len(ang) == 0:
        return {"n": 0}
    R = np.abs(np.exp(1j * ang).mean())
    mu = float(np.angle(np.exp(1j * ang).mean()))
    n = len(ang)
    return {"n": n, "media_rad": mu, "R": float(R), "dp_circular_rad": float(np.sqrt(-2 * np.log(max(R, 1e-12)))),
            "p_rayleigh": float(np.exp(-n * R ** 2)),
            "p_V_zero": float(stats.norm.sf(R * np.sqrt(2 * n) * np.cos(mu)))}  # V-test, direcao 0


def resumo_grupo(rs, limiar_p, rng):
    """Resumo de um conjunto de resultados ja filtrados pela regra de 'forte'."""
    if not rs:
        return {"n": 0}
    S = [r["S"] for r in rs]
    return {"n": len(rs), "S_med": med(S), "S_iqr": iqr(S),
            "S_nulo_emb_med": med([r["S_nulo_emb_med"] for r in rs]),
            "pct_deriva_sig": 100 * float(np.mean([r["p_deriva"] < ALFA for r in rs])),
            "pct_estavel_sig": 100 * float(np.mean([r["p_estavel"] < ALFA for r in rs])),
            "df_abs_med": med(np.abs([r["df_hz"] for r in rs])), "df_med": med([r["df_hz"] for r in rs]),
            "df_iqr": iqr([r["df_hz"] for r in rs]),
            "S_apos_sobre_antes_med": med([r["S_apos"] / r["S"] for r in rs]),
            "razao_z_med": med([r["razao_z"] for r in rs]),
            "z_final_med": med([r["z_final"] for r in rs]),
            "metades": media_circular([r["dif_metades"] for r in rs])}


def checar_relogio(rs):
    """Por gravacao com >= 3 frequencias fortes: consistencia de df/f."""
    por = {}
    for r in rs:
        por.setdefault(r["arquivo"], []).append(r)
    saida = []
    for arq, lst in por.items():
        if len(lst) < 3:
            continue
        f = np.array([r["freq"] for r in lst]); df = np.array([r["df_hz"] for r in lst])
        eps = df / f
        beta = float((f * df).sum() / (f * f).sum())  # df = beta*f por origem
        res = df - beta * f
        r2 = 1 - (res ** 2).sum() / max(((df - df.mean()) ** 2).sum(), 1e-30)
        saida.append({"arquivo": arq, "participante": lst[0]["participante"], "fs": lst[0]["fs"],
                      "n_freq": len(lst), "eps_med": float(np.median(eps)), "eps_mad": float(np.median(np.abs(eps - np.median(eps)))),
                      "eps_dp": float(eps.std()), "beta": beta, "r2_prop_f": float(r2),
                      "df_hz": df.tolist(), "freqs": f.tolist()})
    return saida


# ============================================================
# Figuras e tabelas
# ============================================================
def fig_fase(pasta, fortes_bl):
    fig, eixos = plt.subplots(3, 4, figsize=(15, 9), sharey=True)
    cores = {"30dB": "tab:blue", "40dB": "tab:green", "50dB": "tab:orange", "60dB": "tab:red", "70dB": "tab:purple"}
    for ax, (r, bl) in zip(eixos.ravel(), fortes_bl):
        ax.scatter(bl["t"], bl["phi"], s=20 + 120 * bl["mod"], c=cores.get(r["condicao"], "gray"))
        ts = np.linspace(bl["t"].min(), bl["t"].max(), 50)
        ax.plot(ts, np.angle(np.exp(1j * (np.angle(bl["soma"].sum()) + 2 * np.pi * r["df_hz"] * (ts - bl["t"].mean())))),
                ".", ms=2, color="k")
        ax.set_title(f"{r['arquivo'][:-4]} {r['freq']:.0f}Hz  S={r['S']:.2f} df={r['df_hz']*1e3:.1f}mHz", fontsize=8)
        ax.set_ylim(-np.pi, np.pi); ax.set_xlabel("tempo (s)")
    for ax in eixos[:, 0]:
        ax.set_ylabel("fase do bloco (rad)")
    fig.suptitle("Fase do fasor medio por bloco (tamanho ~ comprimento), 12 sequencias mais fortes; pontos pretos: deriva ajustada")
    fig.tight_layout(); fig.savefig(os.path.join(pasta, "fase_vs_tempo.png"), dpi=110); plt.close(fig)


def fig_hist_df(pasta, forte, esp):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-0.02, 0.02, 41)
    ax.hist([r["df_hz"] for r in esp], bins, alpha=0.5, density=True, label=f"ESP (controle, n={len(esp)})")
    ax.hist([r["df_hz"] for r in forte], bins, alpha=0.5, density=True, label=f"estimulo forte (n={len(forte)})")
    ax.set_xlabel("deriva ajustada df (Hz = ciclos/s)"); ax.set_ylabel("densidade"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(pasta, "hist_deriva.png"), dpi=110); plt.close(fig)


def fig_S(pasta, forte, esp):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8))
    for rs, nome, c in ((esp, "ESP", "gray"), (forte, "forte", "tab:red")):
        a.scatter([r["S_nulo_emb_med"] for r in rs], [r["S"] for r in rs], s=14, alpha=0.6, c=c, label=nome)
        b.scatter([r["S_nulo_rot_q95"] for r in rs], [r["S"] for r in rs], s=14, alpha=0.6, c=c, label=nome)
    for ax, tx in ((a, "mediana de S sob nulo de embaralhamento (fase constante)"),
                   (b, "quantil 95% de S sob nulo de rotacao aleatoria")):
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel(tx); ax.set_ylabel("S observado"); ax.legend()
    a.set_title("abaixo da diagonal = mais deriva que o esperado"); b.set_title("acima da diagonal = fase comum entre blocos")
    fig.tight_layout(); fig.savefig(os.path.join(pasta, "S_observado_vs_nulo.png"), dpi=110); plt.close(fig)


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}g}"


def tabela_md(titulo, grupos):
    l = [f"### {titulo}", "", "| grupo | n | S med | S nulo-emb med | % deriva sig | % estavel sig | med abs(df) mHz | med S apos/antes | med Z apos/antes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nome, g in grupos:
        if g["n"] == 0:
            l.append(f"| {nome} | 0 | | | | | | | |"); continue
        l.append(f"| {nome} | {g['n']} | {fmt(g['S_med'])} | {fmt(g['S_nulo_emb_med'])} | {g['pct_deriva_sig']:.0f} | "
                 f"{g['pct_estavel_sig']:.0f} | {1e3*g['df_abs_med']:.2f} | {fmt(g['S_apos_sobre_antes_med'])} | {fmt(g['razao_z_med'])} |")
    return "\n".join(l) + "\n"


# ============================================================
# Fluxo principal
# ============================================================
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEMENTE)
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    seqs = carregar_alvos()
    saida, md = {"config": {"sinal": SINAL, "blocos": BLOCOS, "min_epocas_bloco": MIN_EPOCAS_BLOCO,
                            "n_sim": N_SIM, "grade_df_hz": {f"B{b}": [float(-min(DF_MAX, 0.5 / b)), float(min(DF_MAX, 0.5 / b)), PASSO_DF] for b in BLOCOS},
                            "p_forte": P_FORTE, "semente": SEMENTE}}, []
    stim = [s for s in seqs if not s["esp"]]
    saida["n_sequencias"] = {"estimulo": len(stim), "esp": len(seqs) - len(stim)}
    for b in BLOCOS:
        res_s, res_e, bls = [], [], {}
        for s in seqs:
            bl, r = analisar(s, b, rng)
            if bl is not None:
                (res_e if s["esp"] else res_s).append(r); bls[id(r)] = bl
        todos_s = [r for r in res_s]
        forte = [r for r in res_s if r["p_final"] < P_FORTE]
        frouxo = [r for r in res_s if r["p_final"] < P_FORTE_FROUXO]
        inten = ["30dB", "40dB", "50dB", "60dB", "70dB"]
        n_forte_int = {c: int(sum(r["p_final"] < P_FORTE and r["condicao"] == c for r in res_s)) for c in inten}
        grupos = [(f"forte p<{P_FORTE:g}", resumo_grupo(forte, P_FORTE, rng)),
                  (f"frouxo p<{P_FORTE_FROUXO:g}", resumo_grupo(frouxo, 0, rng)),
                  ("todas estimulo", resumo_grupo(todos_s, 0, rng)),
                  ("ESP (todas)", resumo_grupo(res_e, 0, rng)),
                  (f"ESP p<{P_FORTE_FROUXO:g} (ESP 'fortes')", resumo_grupo([r for r in res_e if r["p_final"] < P_FORTE_FROUXO], 0, rng))]
        por_int = [(c, resumo_grupo([r for r in forte if r["condicao"] == c], 0, rng)) for c in inten]
        relogio = checar_relogio(forte)
        chave = f"B{b}"
        saida[chave] = {"n_analisadas_estimulo": len(res_s), "n_analisadas_esp": len(res_e),
                        "n_fortes_por_intensidade": n_forte_int, "grupos": dict(grupos), "fortes_por_intensidade": dict(por_int),
                        "relogio": relogio,
                        "sequencias": {"estimulo": res_s, "esp": res_e}}
        md.append(f"## Blocos de {b} epocas\n\nSequencias com >= {MIN_BLOCOS} blocos: estimulo {len(res_s)}, ESP {len(res_e)}. "
                  f"Fortes por intensidade (p<{P_FORTE:g}): {n_forte_int}\n")
        md.append(tabela_md("Resumo por grupo", grupos))
        md.append(tabela_md("Fortes por intensidade", por_int))
        md.append("### Primeira x segunda metade (diferenca de fase 2a - 1a, fortes / ESP)\n")
        for nome, g in grupos[:1] + grupos[3:4]:
            m = g.get("metades", {})
            if m.get("n"):
                md.append(f"- {nome}: n={m['n']} media={m['media_rad']:.2f} rad R={m['R']:.2f} dp={m['dp_circular_rad']:.2f} "
                          f"p_Rayleigh={m['p_rayleigh']:.3g} p_V(0)={m['p_V_zero']:.3g}")
        md.append("\n### Assinatura de relogio (fortes; gravacoes com >= 3 frequencias fortes)\n")
        md.append("| arquivo | Fs | n freq | eps=df/f mediana | MAD | dp | R2 (df ~ f) |\n|---|---|---|---|---|---|---|")
        for c in relogio:
            md.append(f"| {c['arquivo']} | {c['fs']:.0f} | {c['n_freq']} | {c['eps_med']:.2e} | {c['eps_mad']:.2e} | {c['eps_dp']:.2e} | {c['r2_prop_f']:.2f} |")
        if relogio:
            e = np.array([c["eps_med"] for c in relogio])
            md.append(f"\nMediana de |eps| = {np.median(np.abs(e)):.2e}; mediana do dp intra-gravacao de eps = {np.median([c['eps_dp'] for c in relogio]):.2e}; "
                      f"mediana R2 = {np.median([c['r2_prop_f'] for c in relogio]):.2f}.")
            por_p = {}
            for c in relogio:
                por_p.setdefault(c["participante"], []).append(c["eps_med"])
            md.append("Mediana de eps por gravacao, agrupada por participante: " +
                      "; ".join(f"{p}: " + ", ".join(f"{x:.1e}" for x in v) for p, v in por_p.items()))
        # ganho esperado ao acaso: ajuste de deriva em todas as sequencias de estimulo e ESP
        saida[chave]["ganho_todas"] = {"estimulo_razao_z_med": med([r["razao_z"] for r in res_s]),
                                       "esp_razao_z_med": med([r["razao_z"] for r in res_e])}
        md.append(f"\nGanho de Z por correcao em TODAS as sequencias: estimulo mediana {fmt(med([r['razao_z'] for r in res_s]))}, "
                  f"ESP (acaso) {fmt(med([r['razao_z'] for r in res_e]))}\n")
        if b == BLOCOS[0]:
            fortes_ord = sorted(forte, key=lambda r: -r["z_final"])[:N_FIGURA_FASE]
            fig_fase(PASTA_SAIDA, [(r, bls[id(r)]) for r in fortes_ord])
            fig_hist_df(PASTA_SAIDA, forte, res_e)
            fig_S(PASTA_SAIDA, forte, res_e)
    saida["tempo_execucao_s"] = time.time() - t0
    md.append(f"\nTempo de execucao: {saida['tempo_execucao_s']:.1f} s\n")
    with open(os.path.join(PASTA_SAIDA, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=1, default=float)
    with open(os.path.join(PASTA_SAIDA, "tabelas.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
