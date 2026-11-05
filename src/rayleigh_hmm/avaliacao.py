"""Avaliacao complementar do exame sequencial (sem refazer a busca).

Reutiliza as configuracoes escolhidas por experimento.py (resultados.json) e as
mesmas emissoes (a injecao sintetica e deterministica). Tres blocos:
  A. Curva deteccao x FP lateral (tipo ROC): varre UM controle continuo
     (alfa no bruto, LIMIAR_POSTERIORI no HMM) mantendo o resto fixo, e uma
     referencia com teste unico no fim da gravacao ("ORD classico").
  B. Deteccao acumulada x tempo de exame, com o total de alvos no denominador.
  C. Truncamento: como se cada gravacao durasse no maximo T s, so com as
     sequencias cuja gravacao tem pelo menos T s.
Antes de tudo, confere que as metricas das gravacoes completas reproduzem as
de resultados.json.

Execucao (da raiz do repositorio):
    .venv/bin/python src/rayleigh_hmm/avaliacao.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import experimento as exp
import graficos
import hmm
from graficos import COR_1, COR_2, COR_3, COR_HIST, TINTA

# ============================================================
# Configuracao
# ============================================================
PASTA_RESULTADOS = exp.PASTA_RESULTADOS
PASTA_FIGURAS = exp.PASTA_FIGURAS

N_PONTOS_CURVA = 40
GRADE_ALFA_CURVA = np.logspace(-4, np.log10(0.5), N_PONTOS_CURVA)  # bruto e referencia
GRADE_LIMIAR_CURVA = np.linspace(0.02, 0.995, N_PONTOS_CURVA)  # HMM
FP_ALVOS = [0.05, 0.07, 0.10]  # FP lateral em que a deteccao e interpolada
FP_AREA_MAX = 0.10  # area parcial sob a curva em FP in [0, FP_AREA_MAX]

TEMPO_MAXIMO_S = 500  # a gravacao mais longa (30 dB) tem 500 s
TEMPOS_TABELA_S = [30, 60, 120, 180, 300, 480]
INTENSIDADES = ["30dB", "40dB", "50dB", "60dB", "70dB"]
DURACOES_TRUNCAMENTO_S = [60, 120, 180, 300]
CONDICOES = ["ESP"] + INTENSIDADES
METODOS_B_C = ["bruto", "hmm_preditiva"]
ROTULOS = {
    "bruto": "Rayleigh otimizado",
    "hmm_preditiva": "HMM (preditiva)",
    "hmm_map": "HMM (MAP)",
    "referencia": "Rayleigh, teste único no fim",
}
TOLERANCIA_REPRODUCAO = 1e-9


# ============================================================
# 1) Exame sequencial sobre series de evidencia
# ============================================================
def preparar(scores, duracoes, passos_max=None):
    """Agrupa sequencias pelo numero efetivo de avaliacoes (truncado em passos_max).

    scores: lista de series 1D (p-valor no bruto, P(Presente) no HMM).
    Retorna [(indices, duracao_epoca_s, matriz (n, k))]. Grupos com menos de
    MINIMO_AVALIACOES avaliacoes nunca disparam e ficam de fora.
    """
    grupos = {}
    for i, s in enumerate(scores):
        k = len(s) if passos_max is None else min(len(s), int(passos_max[i]))
        if k >= exp.MINIMO_AVALIACOES:
            grupos.setdefault(k, []).append(i)
    return [
        (np.array(idx), np.array([duracoes[i] for i in idx]), np.stack([scores[i][:k] for i in idx]))
        for k, idx in grupos.items()
    ]


def exame(grupos, n_seq, detecta, controle, regra):
    """detectado (n_seq,) e tempo de exame em s (nan se nao detectou)."""
    grupos_det = [(idx, *exp.indicadores(detecta(m, controle)), dur) for idx, dur, m in grupos]
    return exp.decidir(grupos_det, n_seq, regra)


def bruto_detecta(p, alfa):
    return p <= alfa


def hmm_detecta(post, limiar):
    return post >= limiar


def teste_unico(p_final, duracao_gravacao_s, alfa):
    """Referencia: um unico Rayleigh com todas as epocas, sem regra sequencial."""
    detectado = p_final <= alfa
    return detectado, np.where(detectado, duracao_gravacao_s, np.nan)


# ============================================================
# 2) Curva deteccao x FP e area parcial
# ============================================================
def envelope(fp, det):
    """Pontos (fp, det) ordenados, com (0, 0) e deteccao maxima por FP, nao decrescente."""
    melhor = {0.0: 0.0}
    for x, y in zip(fp, det):
        melhor[x] = max(melhor.get(x, 0.0), y)
    xs = np.array(sorted(melhor))
    return xs, np.maximum.accumulate(np.array([melhor[x] for x in xs]))


def interpolar(fp, det, alvo):
    """Deteccao em FP = alvo; acima do maior FP da curva, repete o ultimo valor."""
    xs, ys = envelope(fp, det)
    return float(np.interp(alvo, xs, ys))


def area_parcial(fp, det, fp_max=FP_AREA_MAX):
    """Area sob a curva em [0, fp_max] (regra do trapezio), normalizada por fp_max."""
    xs, ys = envelope(fp, det)
    grade = np.unique(np.concatenate([xs[xs < fp_max], [0.0, fp_max]]))
    return float(np.trapezoid(np.interp(grade, xs, ys), grade) / fp_max)


def varrer(grupos, n_seq, detecta, grade, regra, masc):
    pontos = []
    for c in grade:
        detectado, tempo = exame(grupos, n_seq, detecta, float(c), regra)
        pontos.append({"controle": float(c), **exp.metricas(masc, detectado, tempo)})
    return pontos


def resumo_curva(pontos):
    fp = [p["fp_lateral"] for p in pontos]
    det = [p["deteccao"] for p in pontos]
    return {
        "pontos": pontos,
        "deteccao_em_fp": {f"{f:.2f}": interpolar(fp, det, f) for f in FP_ALVOS},
        "area_parcial": area_parcial(fp, det),
    }


# ============================================================
# 3) Deteccao acumulada x tempo
# ============================================================
def curva_acumulada(tempo_s, t):
    """Fracao de sequencias detectadas ate cada t; nao detectadas (nan) contam no denominador."""
    tempo_s = np.asarray(tempo_s, dtype=float)
    if tempo_s.size == 0:
        return np.full(len(t), np.nan)
    ordenado = np.sort(tempo_s[~np.isnan(tempo_s)])
    return np.searchsorted(ordenado, t, side="right") / tempo_s.size


# ============================================================
# 4) Tabelas
# ============================================================
def ordem(condicao):
    return exp.ordem_condicao(condicao)


def tabela_a(curvas):
    cab = "| Método | " + " | ".join(f"Detecção a FP {f:.0%}" for f in FP_ALVOS) + " | Área parcial (FP ≤ 10%) |"
    linhas = [cab, "|---|" + "---:|" * (len(FP_ALVOS) + 1)]
    for nome, c in curvas.items():
        cel = " | ".join(f"{c['deteccao_em_fp'][f'{f:.2f}']:.1%}" for f in FP_ALVOS)
        linhas.append(f"| {ROTULOS[nome]} | {cel} | {c['area_parcial']:.3f} |")
    return "\n".join(linhas)


def tabela_b(b):
    cab = "| Painel | Método | " + " | ".join(f"{t} s" for t in TEMPOS_TABELA_S) + " |"
    linhas = [cab, "|---|---|" + "---:|" * len(TEMPOS_TABELA_S)]
    for painel in INTENSIDADES + ["total", "fp_lateral"]:
        for nome in METODOS_B_C:
            tab = b["tabela"][painel][nome]
            rotulo = {"total": "Total (alvos)", "fp_lateral": "FP lateral"}.get(painel, painel)
            linhas.append(f"| {rotulo} | {ROTULOS[nome]} | "
                          + " | ".join(f"{tab[str(t)]:.1%}" for t in TEMPOS_TABELA_S) + " |")
    return "\n".join(linhas)


def tabela_c(linhas_t):
    linhas = ["| Condição | Método | Arquivos | Detecção | FP lateral | FP alvo ESP | Tempo médio / mediano (s) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for l in linhas_t:
        linhas.append(f"| {l['condicao']} | {ROTULOS[l['metodo']]} | {l['arquivos']} | {exp.fmt_taxa(l, 'deteccao')} | "
                      f"{exp.fmt_taxa(l, 'fp_lateral')} | "
                      f"{exp.fmt_taxa(l, 'fp_alvo_esp') if l['condicao'] == 'ESP' else '—'} | {exp.fmt_tempo(l)} |")
    return "\n".join(linhas)


def escrever_markdown(caminho, saida):
    a, b, c = saida["A"], saida["B"], saida["C"]
    partes = [
        "# Avaliação complementar — Rayleigh × HMM",
        "",
        "Gerado por `src/rayleigh_hmm/avaliacao.py`. Usa as configurações escolhidas por "
        "`experimento.py` (sem refazer a busca) e as mesmas emissões. Validação in-sample.",
        "",
        "Reprodução das gravações completas: "
        + ("conferida (detecção, FP lateral, FP alvo ESP e tempo médio idênticos a `resultados.json`)."
           if saida["reproducao"]["ok"] else "FALHOU."),
        "",
        "## A. Detecção × FP lateral",
        "",
        "Cada curva varre um único controle (α no bruto e na referência; limiar da posteriori no HMM), "
        "com o resto da configuração escolhida fixo. A curva parte de (0, 0); o FP é o lateral global "
        "(528 sequências) e a detecção, a dos 440 alvos de estímulo. Acima do maior FP alcançado, "
        "a detecção interpolada repete o último valor. A referência é um teste único com todas as "
        "épocas completas da gravação (sem regra sequencial); o tempo de exame é a gravação inteira.",
        "",
        tabela_a(a["curvas"]),
        "",
        "## B. Detecção acumulada × tempo de exame",
        "",
        "Fração dos alvos (denominador = todos os alvos do painel, detectados ou não) já detectados "
        "até t segundos, com as configurações escolhidas. FP lateral: laterais de todas as condições. "
        "Durações máximas de gravação: "
        + ", ".join(f"{k} = {v:.0f} s" for k, v in b["duracao_maxima_s"].items()) + ".",
        "",
        tabela_b(b),
        "",
        "## C. Truncamento das gravações",
        "",
        "Cada gravação é truncada em T s (primeiras ⌊T / 10⌋ avaliações), com as mesmas emissões e "
        "parâmetros. Só entram sequências com gravação de pelo menos T s, para que todas as condições "
        "tenham a mesma duração máxima. A coluna Arquivos conta arquivos (participante × condição).",
        "",
    ]
    for t, bloco in c.items():
        partes += [f"### T = {t} s", "",
                   "Arquivos por condição: " + ", ".join(f"{k} = {v}" for k, v in bloco["arquivos_por_condicao"].items())
                   + f". Excluídos (gravação < {t} s): {bloco['n_arquivos_excluidos']} de {bloco['n_arquivos_total']}"
                   + (" (" + ", ".join(f"{k} = {v}" for k, v in bloco["excluidos_por_condicao"].items()) + ")"
                      if bloco["excluidos_por_condicao"] else "") + ".",
                   "", tabela_c(bloco["linhas"]), ""]
    with open(caminho, "w") as f:
        f.write("\n".join(partes))


# ============================================================
# 5) Figuras
# ============================================================
def formatar_pct(ax, eixo="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)


def figura_curva_fp(pasta, curvas, escolhidos, fp_maximo):
    estilo = {"referencia": (COR_HIST, "-", "s"), "bruto": (COR_1, "-", "o"),
              "hmm_preditiva": (COR_2, "-", "o"), "hmm_map": (COR_3, "--", "o")}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axvline(fp_maximo, color=TINTA, lw=1, ls=":", label=f"teto de FP lateral = {fp_maximo:.1%}")
    for nome in ("referencia", "bruto", "hmm_preditiva", "hmm_map"):
        cor, ls, _ = estilo[nome]
        xs, ys = envelope([p["fp_lateral"] for p in curvas[nome]["pontos"]],
                          [p["deteccao"] for p in curvas[nome]["pontos"]])
        ax.plot(xs, ys, color=cor, ls=ls, lw=2, label=ROTULOS[nome])
        if nome in escolhidos:
            m = escolhidos[nome]
            ax.plot(m["fp_lateral"], m["deteccao"], "o", color=cor, ms=9, mec=TINTA, mew=1.2, zorder=5)
    ax.plot([], [], "o", color="white", mec=TINTA, ms=9, label="ponto escolhido pela busca")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1)
    formatar_pct(ax, "x")
    formatar_pct(ax, "y")
    ax.set_xlabel("falso positivo lateral")
    ax.set_ylabel("taxa de detecção (alvos)")
    ax.set_title("Detecção × falso positivo lateral", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "curva_deteccao_fp.png"), dpi=150)
    plt.close(fig)


def figura_curva_tempo(pasta, b):
    t = np.array(b["t_s"])
    paineis = INTENSIDADES + ["total", "fp_lateral"]
    titulos = {**{k: k for k in INTENSIDADES}, "total": "Total (alvos de estímulo)", "fp_lateral": "FP lateral acumulado"}
    cores = {"bruto": COR_1, "hmm_preditiva": COR_2}
    fig, eixos = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    eixos = eixos.ravel()
    for ax, painel in zip(eixos, paineis):
        for nome in METODOS_B_C:
            ax.plot(t, b["curvas"][painel][nome], color=cores[nome], lw=2, label=ROTULOS[nome])
        if painel in b["duracao_maxima_s"]:
            dur = b["duracao_maxima_s"][painel]
            ax.axvline(dur, color=TINTA, ls="--", lw=1)
            ax.text(dur - 6, 0.98, f"gravação máx. {dur:.0f} s", rotation=90, ha="right", va="top",
                    fontsize=8, color=TINTA)
        ax.set_title(titulos[painel], loc="left", fontsize=10)
        ax.set_xlim(0, TEMPO_MAXIMO_S)
        ax.set_ylim(0, 1 if painel != "fp_lateral" else None)
        formatar_pct(ax)
        ax.set_xlabel("tempo de exame (s)")
        if painel in ("30dB", "total"):
            ax.set_ylabel("fração acumulada")
    ax_leg = eixos[-1]
    ax_leg.axis("off")
    h, l = eixos[0].get_legend_handles_labels()
    ax_leg.legend(h, l, frameon=False, loc="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "curva_deteccao_tempo.png"), dpi=150)
    plt.close(fig)


def figura_truncamento(pasta, c, completa):
    cores_t = {60: "#9ec5ee", 120: COR_1, 180: COR_3, 300: COR_2}
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(INTENSIDADES))
    for ax, nome in zip(eixos, METODOS_B_C):
        ax.plot(x, [completa[nome][k] for k in INTENSIDADES], color=COR_HIST, lw=2, ls="--", marker="s",
                label="gravação completa")
        for t, bloco in c.items():
            ys = []
            for k in INTENSIDADES:
                l = next(l for l in bloco["linhas"] if l["metodo"] == nome and l["condicao"] == k)
                ys.append(l["deteccao"] if l["deteccao_total"] else np.nan)
            ax.plot(x, ys, color=cores_t[int(t)], lw=2, marker="o", label=f"T = {t} s")
        ax.set_xticks(x, INTENSIDADES)
        ax.set_ylim(0, 1)
        formatar_pct(ax)
        ax.set_xlabel("intensidade")
        ax.set_title(ROTULOS[nome], loc="left")
        ax.grid(axis="x", visible=False)
    eixos[0].set_ylabel("taxa de detecção (alvos)")
    h, l = eixos[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=len(l), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(pasta, "truncamento.png"), dpi=150)
    plt.close(fig)


# ============================================================
# 6) Fluxo principal
# ============================================================
def conferir_reproducao(escolhidos_json, resultados):
    """Compara metricas das gravacoes completas com resultados.json."""
    diffs, comparacoes = [], {}
    for nome, (detectado, tempo, masc) in resultados.items():
        novo = exp.metricas(masc, detectado, tempo)
        ref = escolhidos_json[nome]["metricas"]
        comparacoes[nome] = {"novo": novo, "referencia": ref}
        for chave in ("deteccao", "fp_lateral", "fp_alvo_esp", "tempo_medio_s"):
            if abs(novo[chave] - ref[chave]) > TOLERANCIA_REPRODUCAO:
                diffs.append(f"{nome}.{chave}: novo={novo[chave]} resultados.json={ref[chave]}")
    return not diffs, diffs, comparacoes


def main():
    with open(os.path.join(PASTA_RESULTADOS, "resultados.json")) as f:
        resultados_json = json.load(f)
    metodos = resultados_json["metodos"]
    fp_maximo = resultados_json["configuracao"]["fp_maximo"]

    sequencias, esp = exp.carregar_sequencias()
    emissoes = exp.ajustar_emissoes(sequencias, esp)
    masc = exp.mascaras(sequencias)
    n_seq = len(sequencias)
    duracoes = [s["duracao_epoca_s"] for s in sequencias]
    grav = np.array([s["duracao_gravacao_s"] for s in sequencias])
    condicoes_seq = np.array([s["condicao"] for s in sequencias])

    # Series de evidencia por sequencia: p-valor (bruto) e posteriori filtrada (HMM)
    scores = {"bruto": [np.exp(-s["z"]) for s in sequencias]}
    for nome, variante in (("hmm_preditiva", "preditiva"), ("hmm_map", "map")):
        pi, A = np.array(metodos[nome]["pi"]), np.array(metodos[nome]["A"])
        post = [None] * n_seq
        for idx, _ in exp.indices_por_comprimento(sequencias):
            log_em = np.stack([exp.log_emissoes(sequencias[i], emissoes, variante) for i in idx])
            for i, p in zip(idx, hmm.posteriori_filtrada(log_em, pi, A)):
                post[i] = p
        scores[nome] = post
    detecta = {"bruto": bruto_detecta, "hmm_preditiva": hmm_detecta, "hmm_map": hmm_detecta}
    controle_escolhido = {"bruto": metodos["bruto"]["alfa"], "hmm_preditiva": exp.LIMIAR_POSTERIORI,
                          "hmm_map": exp.LIMIAR_POSTERIORI}
    regra = {nome: metodos[nome]["regra"] for nome in detecta}

    # ---- checagem de reproducao (gravacoes completas) ----
    grupos_completos = {nome: preparar(scores[nome], duracoes) for nome in detecta}
    completos = {}
    for nome in detecta:
        det, tempo = exame(grupos_completos[nome], n_seq, detecta[nome], controle_escolhido[nome], regra[nome])
        completos[nome] = (det, tempo, masc)
    ok, diffs, comparacoes = conferir_reproducao(metodos, completos)
    reproducao = {"ok": ok, "diferencas": diffs, "comparacao": comparacoes}
    if not ok:
        raise SystemExit("Reproducao falhou:\n  " + "\n  ".join(diffs))
    print("Reproducao das metricas de resultados.json: OK")

    # ---- Parte A ----
    escolhidos_metricas = {nome: comparacoes[nome]["novo"] for nome in detecta}
    curvas = {}
    p_final = np.array([s_[-1] for s_ in scores["bruto"]])
    pontos = []
    for alfa in GRADE_ALFA_CURVA:
        det, tempo = teste_unico(p_final, grav, alfa)
        pontos.append({"controle": float(alfa), **exp.metricas(masc, det, tempo)})
    curvas["referencia"] = resumo_curva(pontos)
    grade_alfa = np.unique(np.append(GRADE_ALFA_CURVA, controle_escolhido["bruto"]))
    grade_lim = np.unique(np.append(GRADE_LIMIAR_CURVA, exp.LIMIAR_POSTERIORI))
    for nome, grade in (("bruto", grade_alfa), ("hmm_preditiva", grade_lim), ("hmm_map", grade_lim)):
        curvas[nome] = resumo_curva(varrer(grupos_completos[nome], n_seq, detecta[nome], grade, regra[nome], masc))
    ordem_curvas = {k: curvas[k] for k in ("referencia", "bruto", "hmm_preditiva", "hmm_map")}
    figura_curva_fp(PASTA_FIGURAS, curvas, escolhidos_metricas, fp_maximo)

    # ---- Parte B ----
    t = np.arange(0, TEMPO_MAXIMO_S + 1)
    alvo_est = masc["alvo_estimulo"]
    sel = {c: alvo_est & (condicoes_seq == c) for c in INTENSIDADES}
    sel["total"] = alvo_est
    sel["fp_lateral"] = masc["lateral"]
    b = {"t_s": t.tolist(), "curvas": {p: {} for p in sel}, "tabela": {p: {} for p in sel},
         "duracao_maxima_s": {c: float(grav[sel[c]].max()) for c in INTENSIDADES},
         "denominador": {p: int(m.sum()) for p, m in sel.items()}}
    for nome in METODOS_B_C:
        tempo = completos[nome][1]
        for painel, m in sel.items():
            curva = curva_acumulada(tempo[m], t)
            b["curvas"][painel][nome] = curva.tolist()
            b["tabela"][painel][nome] = {str(tt): float(curva[tt]) for tt in TEMPOS_TABELA_S}
    figura_curva_tempo(PASTA_FIGURAS, b)

    # ---- Parte C ----
    arquivos = [(s["participante"], s["condicao"]) for s in sequencias]
    c_saida = {}
    for T in DURACOES_TRUNCAMENTO_S:
        incl = np.where(grav >= T)[0]
        k_max = np.array([int(T // (exp.EPOCAS_POR_JANELA * d) + 1e-9) for d in duracoes])
        todos = sorted(set(arquivos))
        usados = sorted({arquivos[i] for i in incl})
        excluidos = [a for a in todos if a not in set(usados)]
        bloco = {
            "arquivos_por_condicao": {c: sum(1 for a in usados if a[1] == c) for c in CONDICOES},
            "excluidos_por_condicao": {c: sum(1 for a in excluidos if a[1] == c) for c in CONDICOES
                                       if any(a[1] == c for a in excluidos)},
            "n_arquivos_total": len(todos), "n_arquivos_excluidos": len(excluidos),
            "avaliacoes_por_sequencia": int(T // exp.EPOCAS_POR_JANELA), "linhas": [],
        }
        sub = [sequencias[i] for i in incl]
        for nome in METODOS_B_C:
            sub_scores = [scores[nome][i] for i in incl]
            grupos = preparar(sub_scores, [duracoes[i] for i in incl], k_max[incl])
            det, tempo = exame(grupos, len(sub), detecta[nome], controle_escolhido[nome], regra[nome])
            c_ = {"detectado": det, "tempo_s": tempo}
            for cond in CONDICOES:
                r = exp.resumo(sub, c_, lambda s, cond=cond: s["condicao"] == cond)
                bloco["linhas"].append({"metodo": nome, "condicao": cond,
                                        "arquivos": bloco["arquivos_por_condicao"][cond], **r})
        bloco["linhas"].sort(key=lambda l: (ordem(l["condicao"]), METODOS_B_C.index(l["metodo"])))
        c_saida[str(T)] = bloco
    completa = {}
    for nome in METODOS_B_C:
        c_ = {"detectado": completos[nome][0], "tempo_s": completos[nome][1]}
        completa[nome] = {k: exp.resumo(sequencias, c_, lambda s, k=k: s["condicao"] == k)["deteccao"]
                          for k in INTENSIDADES}
    figura_truncamento(PASTA_FIGURAS, c_saida, completa)

    saida = {
        "configuracao": {
            "semente": exp.SEMENTE, "sinal": exp.SINAL, "epocas_por_janela": exp.EPOCAS_POR_JANELA,
            "minimo_avaliacoes": exp.MINIMO_AVALIACOES, "limiar_posteriori_escolhido": exp.LIMIAR_POSTERIORI,
            "n_pontos_curva": N_PONTOS_CURVA, "grade_alfa_curva": [float(x) for x in grade_alfa],
            "grade_limiar_curva": [float(x) for x in grade_lim], "fp_alvos": FP_ALVOS,
            "fp_area_max": FP_AREA_MAX, "tempos_tabela_s": TEMPOS_TABELA_S,
            "duracoes_truncamento_s": DURACOES_TRUNCAMENTO_S,
            "metodos_escolhidos": metodos,
            "validacao": "in-sample (parametros escolhidos nos mesmos pacientes)",
        },
        "reproducao": reproducao,
        "A": {"curvas": ordem_curvas, "escolhidos": escolhidos_metricas},
        "B": b,
        "C": c_saida,
        "C_gravacao_completa": completa,
    }
    with open(os.path.join(PASTA_RESULTADOS, "avaliacao.json"), "w") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    escrever_markdown(os.path.join(PASTA_RESULTADOS, "avaliacao.md"), saida)
    print(tabela_a(ordem_curvas))
    print(f"\nResultados em {PASTA_RESULTADOS}/avaliacao.json e avaliacao.md\nFiguras em {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
