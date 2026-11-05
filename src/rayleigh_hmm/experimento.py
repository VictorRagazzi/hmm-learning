"""Experimento enxuto: teste de Rayleigh bruto x HMM com emissao Gamma bayesiana.

Fluxo:
  1. Rayleigh ACUMULADO: a cada janela de EPOCAS_POR_JANELA epocas (sem
     sobreposicao), Z = N * PLV^2 com todas as N epocas desde o inicio;
     p = exp(-Z).
  2. Emissao Ausente: Gamma por MLE nos Z do ESP (alvos).
  3. Emissao Presente: Gamma com media 1 + (N - 1) * rho; priori em (k, rho),
     verossimilhanca nos Z de calibracao (CALIBRACAO_PRESENTE: "real" = alvos das
     gravacoes com estimulo; "sintetica" = ESP somado a uma senoide); emissao =
     preditiva posterior. Variante marginal: pdf fixa p(z | N), integral de
     Gamma(z | k, mu(N, rho)) nas Gammas de (k, rho), cujos hiperparametros vem
     da verossimilhanca marginal por sequencia de calibracao.
  4. HMM de 2 estados por arquivo (sem concatenar condicoes), observando o
     mesmo Z acumulado. Posteriori FILTRADA (causal).
  5. Exame sequencial: a cada avaliacao, a avaliacao detecta se p <= alfa
     (bruto) ou P(Presente | obs 1..t) >= LIMIAR_POSTERIORI (HMM). O exame
     para quando a regra de decisao dispara; o tempo de exame e o numero de
     epocas acumuladas nesse momento.
  6. Busca em grade de alfa (bruto), pi/A (HMM) e regra, maximizando a
     deteccao sujeita a FP lateral <= teto (FP do bruto_referencia). Ajuste e avaliacao usam os
     mesmos pacientes (vazamento aceito, experimento exploratorio).

ATENCAO (CALIBRACAO_PRESENTE = "real"): a avaliacao deste script e IN-SAMPLE COM
VAZAMENTO DIRETO, pois a emissao Presente e ajustada nos proprios alvos com
estimulo que depois sao avaliados. NAO reportar estes numeros como desempenho.
A estimativa a reportar e a do LOSO (explorar_loso_fc.py, braco HMM-REAL).

Execucao (da raiz do repositorio):
    .venv/bin/python src/rayleigh_hmm/experimento.py
"""

import csv
import itertools
import json
import os

import numpy as np
from scipy import stats

import dados
import emissao
import graficos
import hmm

# ============================================================
# Configuracao
# ============================================================
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PASTA_DADOS = os.path.join(RAIZ, "data")
PASTA_RESULTADOS = os.path.join(RAIZ, "results", "rayleigh_hmm")
PASTA_FIGURAS = os.path.join(RAIZ, "outputs", "rayleigh_hmm")

# Sinal analisado: "canal" (um canal), "media" ou "pca" (varios canais,
# combinados no tempo antes da FFT). rejeicao_fator: None ou fator da mediana
# do desvio padrao por epoca acima do qual a epoca e rejeitada.
# Escolha: so o canal FC (indice 0), sem rejeicao de epocas. FC e o melhor canal
# isolado em selecao_sinal.md (deteccao 53,4%, in-sample). Decisao do usuario em
# 04/11/2026: abandonar a media de 4 canais (FC,Fz,Cz,F4) e a rejeicao.
SINAL = {"combinacao": "canal", "canais": [0], "rejeicao_fator": None}  # FC
AVISO_RAZAO_VARIANCIA = 0.05  # media com variancia < 5% da dos canais -> aviso
NOMES_ELETRODOS = dados.nomes_eletrodos(PASTA_DADOS)
EPOCAS_POR_JANELA = 10  # passo da avaliacao; SoESP tem 30 epocas -> 3 avaliacoes
# Restricao da busca: FP lateral global maximo dos metodos buscados (bruto
# otimizado e HMM). Decisao do usuario em 04/11/2026: o teto e o FP lateral do
# bruto_referencia (Rayleigh fixo), para escolher o HMM na mesma faixa de
# deteccao da referencia sem mais alarmes falsos que ela. Com
# FP_IGUAL_REFERENCIA = False, vale o teto fixo FP_MAXIMO_FIXO.
FP_IGUAL_REFERENCIA = True
FP_MAXIMO_FIXO = 0.07
FP_MAXIMO = FP_MAXIMO_FIXO  # alias usado pelos scripts exploratorios (teto fixo de 7%)
SEMENTE = 0  # so registrada em resultados.json: o experimento nao sorteia mais nada

# Regras de parada do exame (bruto e HMM), avaliadas a cada passo com os dados
# ja gravados. Nenhuma regra dispara antes de MINIMO_AVALIACOES avaliacoes.
#   fracao       -> fracao das avaliacoes ate agora com deteccao >= f
#   consecutivas -> as ultimas c avaliacoes detectaram
#   fracao_e_consecutivas / fracao_ou_consecutivas -> combinacoes
MINIMO_AVALIACOES = 3
GRADE_FRACAO_MINIMA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GRADE_CONSECUTIVAS = [1, 2, 3, 4, 5, 6, 8, 10]

# Calibracao da emissao Presente (decisao do usuario em 04/11/2026):
#   "real"      -> Z e N acumulados de TODAS as sequencias-alvo das gravacoes com
#                  estimulo (30-70 dB x 8 frequencias x 11 participantes = 440);
#                  todo participante ouviu todos os estimulos, logo todas sao
#                  "resposta presente". A avaliacao deste script passa a ser
#                  IN-SAMPLE COM VAZAMENTO DIRETO (Presente ajustado nos mesmos
#                  alvos avaliados); a estimativa a reportar e a do LOSO
#                  (explorar_loso_fc.py, braco HMM-REAL).
#   "sintetica" -> comportamento anterior (ESP + senoide, SNR_SINTETICA_DB).
CALIBRACAO_PRESENTE = "real"

# Presente sintetico (so com CALIBRACAO_PRESENTE = "sintetica"): nada e sorteado. Cada sequencia ESP (arquivo x frequencia)
# recebe uma senoide de fase 0 para cada SNR abaixo, constante na gravacao
# (11 ESP x 8 frequencias x 5 SNRs = 440 sequencias). Os valores sao a S/P media
# medida no bin-alvo dos proprios arquivos com estimulo (PROXIMOS_PASSOS.md, C),
# um por intensidade: 30, 40, 50, 60 e 70 dB. Nao e dB acustico. Medir nos
# mesmos arquivos da avaliacao e vazamento leve (5 medias, nao limiares).
# Valores medidos em 04/11/2026 com o sinal atual (canal FC, sem rejeicao).
SNR_SINTETICA_DB = (-25.0, -20.5, -17.5, -14.0, -12.0)

# Priori ("chute") para Presente: k ~ Gamma(forma, taxa), rho ~ Gamma(forma, taxa).
# rho = PLV^2 verdadeiro da resposta; media a priori 0,01 (~ SNR de -20 dB).
PRIORI_PRESENTE = {"k_forma": 2.0, "k_taxa": 1.0, "rho_forma": 2.0, "rho_taxa": 200.0}
GRADE_K = np.logspace(-1.0, 1.5, 250)
GRADE_RHO = np.logspace(-5.0, -0.5, 250)

# Presente marginal: grade fixa da integral do ajuste, inicios da otimizacao
# (a_k, b_k, a_rho, b_rho) e nos de quadratura da emissao p(z | N).
GRADE_HIER_K = np.geomspace(0.01, 300.0, 170)
GRADE_HIER_RHO = np.geomspace(1e-10, 1.0, 800)
INICIOS_HIER = [(9.0, 6.0, 1.0, 40.0), (4.0, 3.0, 2.0, 100.0), (20.0, 2.0, 0.7, 30.0)]
J_MARGINAL_K = 10
J_MARGINAL_RHO = 24

# Grade de busca do HMM.
GRADE_PI_PRESENTE = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
GRADE_A_AUS_PRES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
GRADE_A_PRES_AUS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
LIMIAR_POSTERIORI = 0.5  # avaliacao "Presente" se P(Presente | obs) >= limiar

# Grade do detector bruto (p = exp(-Z) <= alfa).
GRADE_ALFA = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
REFERENCIA_BRUTO = {"alfa": 0.05, "regra": {"familia": "consecutivas", "consecutivas": 3}}

METODOS_TABELA = {
    "bruto_referencia": "Rayleigh α=0,05, 3 consecutivas",
    "bruto": "Rayleigh otimizado",
    "hmm_preditiva": "HMM (preditiva)",
    "hmm_map": "HMM (MAP)",
    "hmm_marginal": "HMM (pdf marginal)",
    "hmm_viterbi": "HMM (Viterbi)",
}


# ============================================================
# 1) Dados e estatistica de Rayleigh acumulada
# ============================================================
def carregar_sequencias():
    """Uma sequencia por (arquivo, tipo, frequencia), com metadados."""
    sequencias, esp = [], []
    for caminho, participante, condicao in dados.listar_arquivos(PASTA_DADOS):
        d = dados.carregar_coeficientes(caminho, **SINAL)
        if d["razao_variancia"] < AVISO_RAZAO_VARIANCIA:
            print(f"AVISO: {os.path.basename(caminho)}: sinal combinado com variancia "
                  f"{d['razao_variancia']:.3g} da media dos canais (referencia media comum?)")
        n_janelas = d["n_epocas"] // EPOCAS_POR_JANELA
        if n_janelas < MINIMO_AVALIACOES:
            print(f"AVISO: {os.path.basename(caminho)} tem {n_janelas} janelas; ignorado")
            continue
        if condicao == "ESP":
            esp.append((d["coef_alvo"], d["epoca_valida"]))
        valida = dados.janelar(d["epoca_valida"], EPOCAS_POR_JANELA)

        for tipo, chave, freqs in (
            ("alvo", "coef_alvo", d["freq_estim"]),
            ("lateral", "coef_lateral", d["bins_m"]),
        ):
            z, n_ep = dados.rayleigh_z_acumulado(dados.janelar(d[chave], EPOCAS_POR_JANELA), valida)
            for i, freq in enumerate(freqs):
                sequencias.append(
                    {
                        "participante": participante,
                        "condicao": condicao,
                        "tipo": tipo,
                        "frequencia_hz": float(freq),
                        "fs": d["fs"],
                        "n_epocas": d["n_epocas"],
                        "n_janelas": n_janelas,
                        "duracao_epoca_s": d["duracao_epoca_s"],
                        "duracao_gravacao_s": n_janelas * EPOCAS_POR_JANELA * d["duracao_epoca_s"],
                        "z": z[:, i],
                        "n_ep": n_ep,  # epocas VALIDAS acumuladas (media da emissao Presente)
                        "fracao_rejeitada": float(1.0 - d["epoca_valida"][: n_janelas * EPOCAS_POR_JANELA].mean()),
                    }
                )
    return sequencias, esp


def gerar_presente_sintetico(esp):
    """Z acumulado das sequencias ESP apos somar senoide (cada ESP recebe todas as SNRs fixas).

    esp: [(coef_alvo, epoca_valida)] do mesmo sinal combinado e da mesma mascara.
    Retorna z, n e o id da sequencia (arquivo x frequencia x SNR) de cada z. Deterministica.
    """
    zs, ns, ids = [], [], []
    proximo_id = 0
    for snr in SNR_SINTETICA_DB:
        for coef, valida in esp:
            injetado = dados.injetar_senoide(coef, snr, valida)
            z, n_ep = dados.rayleigh_z_acumulado(
                dados.janelar(injetado, EPOCAS_POR_JANELA), dados.janelar(valida, EPOCAS_POR_JANELA)
            )
            zs.append(z.ravel())
            ns.append(np.repeat(n_ep, z.shape[1]))
            ids.append(np.tile(np.arange(z.shape[1]), z.shape[0]) + proximo_id)
            proximo_id += z.shape[1]
    return np.concatenate(zs), np.concatenate(ns), np.concatenate(ids)


def gerar_presente_real(sequencias):
    """Z acumulado e N das sequencias-alvo com estimulo (calibracao "real").

    Retorna z, n e o id da sequencia (arquivo x frequencia) de cada z, no mesmo
    formato de gerar_presente_sintetico. Mesma logica do braco REAL do LOSO.
    """
    alvos = [s for s in sequencias if s["tipo"] == "alvo" and s["condicao"] != "ESP"]
    z = np.concatenate([s["z"] for s in alvos])
    n = np.concatenate([s["n_ep"] for s in alvos])
    ids = np.concatenate([np.full(len(s["z"]), i) for i, s in enumerate(alvos)])
    return z, n, ids


# ============================================================
# 2) Emissoes
# ============================================================
def ajustar_emissoes(sequencias, esp):
    z_esp = np.concatenate(
        [s["z"] for s in sequencias if s["condicao"] == "ESP" and s["tipo"] == "alvo"]
    )
    # KS so no ultimo passo: os passos acumulados de uma sequencia sao correlacionados
    z_esp_final = np.array([s["z"][-1] for s in sequencias if s["condicao"] == "ESP" and s["tipo"] == "alvo"])
    ks = stats.kstest(z_esp_final, "expon")

    # As chaves *_sintetico guardam o conjunto de calibracao Presente, seja qual for a origem.
    if CALIBRACAO_PRESENTE == "real":
        z_sint, n_sint, id_sint = gerar_presente_real(sequencias)
    elif CALIBRACAO_PRESENTE == "sintetica":
        z_sint, n_sint, id_sint = gerar_presente_sintetico(esp)
    else:
        raise ValueError(f"CALIBRACAO_PRESENTE invalida: {CALIBRACAO_PRESENTE!r}")
    posteriori = emissao.posteriori_em_grade(z_sint, n_sint, PRIORI_PRESENTE, GRADE_K, GRADE_RHO)
    if posteriori["na_borda"]:
        print("AVISO: MAP da posteriori na borda da grade; amplie GRADE_K/GRADE_RHO")

    return {
        "z_esp": z_esp,
        "calibracao": CALIBRACAO_PRESENTE,
        "z_sintetico": z_sint,
        "n_sintetico": n_sint,
        "id_sintetico": id_sint,
        "ausente": emissao.ajustar_mle(z_esp),
        "ks_esp_vs_exp1": {"estatistica": float(ks.statistic), "p_valor": float(ks.pvalue), "n": int(z_esp_final.size)},
        "posteriori": posteriori,
    }


def pontos_marginal(hiper, j_k=J_MARGINAL_K, j_rho=J_MARGINAL_RHO):
    """Nos de pesos iguais no formato de emissao.log_pdf_preditiva."""
    k, rho = emissao.nos_quadratura(hiper, j_k, j_rho)
    return {"k": k, "rho": rho, "peso": np.full(k.size, 1.0 / k.size)}


def ajustar_marginal(emissoes):
    """Hiperparametros de (k, rho) por sequencia e nos da emissao; nao usa o rng."""
    matriz = emissao.matriz_verossimilhanca_sequencias(
        emissoes["z_sintetico"], emissoes["n_sintetico"], emissoes["id_sintetico"], GRADE_HIER_K, GRADE_HIER_RHO)
    hiper = emissao.ajustar_hierarquico(matriz, GRADE_HIER_K, GRADE_HIER_RHO, INICIOS_HIER)
    hiper["n_sequencias"] = int(matriz.shape[0])
    if hiper["na_borda"]:
        print(f"AVISO: marginal na borda: massa fora {hiper['massa_fora_grade']}, "
              f"hiper no limite={hiper['hiper_no_limite']}; amplie GRADE_HIER_*")
    return {"hiper": hiper, "pontos": pontos_marginal(hiper)}


def log_emissoes(s, emissoes, variante):
    """(T, 2) com log p(z | Ausente) e log p(z | N, Presente)."""
    aus = emissao.log_pdf_gamma(s["z"], emissoes["ausente"]["k"], emissoes["ausente"]["mu"])
    if variante == "preditiva":
        pres = emissao.log_pdf_preditiva(s["z"], s["n_ep"], emissoes["posteriori"]["pontos_preditiva"])
    elif variante == "marginal":
        pres = emissao.log_pdf_preditiva(s["z"], s["n_ep"], emissoes["marginal"]["pontos"])
    else:  # "map"
        mapa = emissoes["posteriori"]["map"]
        pres = emissao.log_pdf_gamma(s["z"], mapa["k"], emissao.media_presente(s["n_ep"], mapa["rho"]))
    return np.stack([aus, pres], axis=-1)


# ============================================================
# 3) Exame sequencial: regras de parada e metricas
# ============================================================
def mascaras(sequencias):
    return {
        "alvo_estimulo": np.array([s["tipo"] == "alvo" and s["condicao"] != "ESP" for s in sequencias]),
        "lateral": np.array([s["tipo"] == "lateral" for s in sequencias]),
        "alvo_esp": np.array([s["tipo"] == "alvo" and s["condicao"] == "ESP" for s in sequencias]),
    }


def metricas(masc, detectado, tempo_s):
    detectado = np.asarray(detectado, dtype=bool)
    tempos = tempo_s[masc["alvo_estimulo"] & detectado]
    return {
        "deteccao": float(detectado[masc["alvo_estimulo"]].mean()),
        "fp_lateral": float(detectado[masc["lateral"]].mean()),
        "fp_alvo_esp": float(detectado[masc["alvo_esp"]].mean()),
        "tempo_medio_s": float(tempos.mean()) if tempos.size else None,
        "tempo_mediano_s": float(np.median(tempos)) if tempos.size else None,
    }


def regras_de_decisao():
    regras = [{"familia": "fracao", "fracao": f} for f in GRADE_FRACAO_MINIMA]
    regras += [{"familia": "consecutivas", "consecutivas": c} for c in GRADE_CONSECUTIVAS]
    for modo in ("e", "ou"):
        regras += [
            {"familia": f"fracao_{modo}_consecutivas", "fracao": f, "consecutivas": c}
            for f, c in itertools.product(GRADE_FRACAO_MINIMA, GRADE_CONSECUTIVAS)
        ]
    return regras


REGRAS = regras_de_decisao()


def indicadores(det):
    """Para cada passo t: fracao de deteccoes em 1..t e corrida de deteccoes terminando em t."""
    t = np.arange(1, det.shape[1] + 1)
    fracao = np.cumsum(det, axis=1) / t[None, :]
    corrida = np.zeros(det.shape, dtype=int)
    atual = np.zeros(det.shape[0], dtype=int)
    for j in range(det.shape[1]):
        atual = (atual + 1) * det[:, j]
        corrida[:, j] = atual
    return fracao, corrida


def primeiro_disparo(fracao, corrida, regra):
    """Indice do primeiro passo em que a regra dispara (-1 se nunca)."""
    familia = regra["familia"]
    por_fracao = fracao >= regra.get("fracao", 0.0) - 1e-9
    por_consecutivas = corrida >= regra.get("consecutivas", 0)
    if familia == "fracao":
        dispara = por_fracao
    elif familia == "consecutivas":
        dispara = por_consecutivas
    elif familia == "fracao_e_consecutivas":
        dispara = por_fracao & por_consecutivas
    else:
        dispara = por_fracao | por_consecutivas
    dispara[:, : MINIMO_AVALIACOES - 1] = False
    return np.where(dispara.any(axis=1), dispara.argmax(axis=1), -1)


def decidir(grupos_det, n_seq, regra):
    """detectado (n_seq,) e tempo de exame em s (nan se nao detectou).

    grupos_det: [(indices, fracao, corrida, duracao_epoca_s)] por comprimento.
    No passo p (base 0) o exame acumulou (p + 1) * EPOCAS_POR_JANELA epocas.
    """
    detectado = np.zeros(n_seq, dtype=bool)
    tempo = np.full(n_seq, np.nan)
    for idx, fracao, corrida, duracao in grupos_det:
        passo = primeiro_disparo(fracao, corrida, regra)
        ok = passo >= 0
        detectado[idx] = ok
        tempo[idx[ok]] = (passo[ok] + 1) * EPOCAS_POR_JANELA * duracao[ok]
    return detectado, tempo


def chave_ordem(c):
    """Maior deteccao; desempate pelo menor FP lateral e depois menor tempo medio."""
    m = c["metricas"]
    return (m["deteccao"], -m["fp_lateral"], -(m["tempo_medio_s"] or np.inf))


def avaliar_regras(sequencias, masc, grupos_det, parametros, melhores, fp_maximo=FP_MAXIMO):
    """Atualiza o melhor candidato viavel (FP lateral <= fp_maximo) de cada familia."""
    for regra in REGRAS:
        detectado, tempo = decidir(grupos_det, len(sequencias), regra)
        candidato = {**parametros, "regra": regra, "metricas": metricas(masc, detectado, tempo)}
        if candidato["metricas"]["fp_lateral"] > fp_maximo + 1e-12:  # folga p/ comparar fracoes n/N
            continue
        atual = melhores.get(regra["familia"])
        if atual is None or chave_ordem(candidato) > chave_ordem(atual):
            melhores[regra["familia"]] = {**candidato, "detectado": detectado, "tempo_s": tempo}


def melhor_geral(melhores):
    return max(melhores.values(), key=chave_ordem) if melhores else None


def indices_por_comprimento(sequencias):
    """[(indices, duracao_epoca_s)] agrupando sequencias de mesmo comprimento."""
    grupos = {}
    for i, s in enumerate(sequencias):
        grupos.setdefault(len(s["z"]), []).append(i)
    return [
        (np.array(idx), np.array([sequencias[i]["duracao_epoca_s"] for i in idx]))
        for idx in grupos.values()
    ]


def calcular_referencia(sequencias, masc):
    """Rayleigh fixo (REFERENCIA_BRUTO): sem busca, so a regra e o alfa dados."""
    grupos_det = [
        (idx, *indicadores(np.exp(-np.stack([sequencias[i]["z"] for i in idx])) <= REFERENCIA_BRUTO["alfa"]), dur)
        for idx, dur in indices_por_comprimento(sequencias)
    ]
    detectado, tempo = decidir(grupos_det, len(sequencias), REFERENCIA_BRUTO["regra"])
    return {**REFERENCIA_BRUTO, "metricas": metricas(masc, detectado, tempo),
            "detectado": detectado, "tempo_s": tempo}


def buscar_bruto(sequencias, masc, fp_maximo=FP_MAXIMO):
    grupos = [
        (idx, dur, np.stack([sequencias[i]["z"] for i in idx]))
        for idx, dur in indices_por_comprimento(sequencias)
    ]
    melhores = {}
    for alfa in GRADE_ALFA:
        grupos_det = [(idx, *indicadores(np.exp(-z) <= alfa), dur) for idx, dur, z in grupos]
        avaliar_regras(sequencias, masc, grupos_det, {"alfa": alfa}, melhores, fp_maximo)
    return melhores, calcular_referencia(sequencias, masc)


def detector_filtro(log_em, pi, A):
    """Avaliacao detectada se P(Presente | obs 1..t) >= LIMIAR_POSTERIORI."""
    return hmm.posteriori_filtrada(log_em, pi, A) >= LIMIAR_POSTERIORI


def buscar_hmm(sequencias, masc, emissoes, variante, detector=detector_filtro, fp_maximo=FP_MAXIMO):
    grupos = [
        (idx, dur, np.stack([log_emissoes(sequencias[i], emissoes, variante) for i in idx]))
        for idx, dur in indices_por_comprimento(sequencias)
    ]
    melhores = {}
    for p1, a01, a10 in itertools.product(GRADE_PI_PRESENTE, GRADE_A_AUS_PRES, GRADE_A_PRES_AUS):
        pi = np.array([1.0 - p1, p1])
        A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
        grupos_det = [
            (idx, *indicadores(detector(log_em, pi, A)), dur)
            for idx, dur, log_em in grupos
        ]
        avaliar_regras(sequencias, masc, grupos_det, {"pi": pi.tolist(), "A": A.tolist()}, melhores, fp_maximo)
    return melhores, melhor_geral(melhores)


# ============================================================
# 4) Tabelas de resultado
# ============================================================
def ordem_condicao(condicao):
    return -1 if condicao == "ESP" else int(condicao.removesuffix("dB"))


def resumo(sequencias, c, filtro):
    """Deteccao, FP lateral, FP alvo ESP e tempo de exame nas sequencias do filtro."""
    alvo, lateral, alvo_esp, tempos = [], [], [], []
    for s, d, t in zip(sequencias, c["detectado"], c["tempo_s"]):
        if not filtro(s):
            continue
        if s["tipo"] == "lateral":
            lateral.append(d)
        elif s["condicao"] == "ESP":
            alvo_esp.append(d)
        else:
            alvo.append(d)
            if d:
                tempos.append(t)

    def taxa(v):
        return (int(np.sum(v)), len(v), float(np.mean(v)) if v else None)

    det, fpl, fpe = taxa(alvo), taxa(lateral), taxa(alvo_esp)
    return {
        "deteccao_detectados": det[0], "deteccao_total": det[1], "deteccao": det[2],
        "fp_lateral_detectados": fpl[0], "fp_lateral_total": fpl[1], "fp_lateral": fpl[2],
        "fp_alvo_esp_detectados": fpe[0], "fp_alvo_esp_total": fpe[1], "fp_alvo_esp": fpe[2],
        "tempo_medio_s": float(np.mean(tempos)) if tempos else None,
        "tempo_mediano_s": float(np.median(tempos)) if tempos else None,
        "tempo_min_s": float(np.min(tempos)) if tempos else None,
        "tempo_max_s": float(np.max(tempos)) if tempos else None,
    }


def tabelas_resumo(sequencias, escolhidos):
    participantes = sorted({s["participante"] for s in sequencias})
    condicoes = sorted({s["condicao"] for s in sequencias}, key=ordem_condicao)
    por_paciente, por_intensidade, geral = [], [], []
    for nome in METODOS_TABELA:
        c = escolhidos[nome]
        for p in participantes:
            por_paciente.append({"metodo": nome, "participante": p,
                                 **resumo(sequencias, c, lambda s, p=p: s["participante"] == p)})
        for cond in condicoes:
            por_intensidade.append({"metodo": nome, "condicao": cond,
                                    **resumo(sequencias, c, lambda s, cond=cond: s["condicao"] == cond)})
        geral.append({"metodo": nome, **resumo(sequencias, c, lambda s: True)})
    return por_paciente, por_intensidade, geral


def escrever_csv(caminho, linhas):
    with open(caminho, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def fmt_taxa(linha, chave):
    if linha[f"{chave}_total"] == 0:
        return "—"
    return f"{linha[f'{chave}_detectados']}/{linha[f'{chave}_total']} ({linha[chave]:.1%})"


def fmt_tempo(linha):
    if linha["tempo_medio_s"] is None:
        return "—"
    return f"{linha['tempo_medio_s']:.0f} / {linha['tempo_mediano_s']:.0f}"


def tabela_markdown(linhas, chave_grupo, titulo_grupo):
    cab = f"| {titulo_grupo} | Detecção | FP lateral | FP alvo ESP | Tempo médio / mediano (s) |"
    saida = [cab, "|---|---:|---:|---:|---:|"]
    for l in linhas:
        grupo = l[chave_grupo] if chave_grupo else METODOS_TABELA[l["metodo"]]
        saida.append(f"| {grupo} | {fmt_taxa(l, 'deteccao')} | {fmt_taxa(l, 'fp_lateral')} | "
                     f"{fmt_taxa(l, 'fp_alvo_esp')} | {fmt_tempo(l)} |")
    return "\n".join(saida)


def tabela_significancia(comp):
    """Linhas markdown dos testes de significancia (bruto x HMM)."""
    saida = []
    for parceiro, blocos in comp["parceiros"].items():
        saida += [f"### Contra {METODOS_TABELA[parceiro]}", "",
                  "| HMM | Alvos (só bruto / só HMM, p) | Laterais (só bruto / só HMM, p) | ESP (só bruto / só HMM, p) | Tempo: n pares, mediana bruto / HMM (s), diferenças nulas, p |",
                  "|---|---|---|---|---|"]
        for v, b in blocos.items():
            t = b["tempo"]
            p_t = "—" if t["p_valor"] is None else f"{t['p_valor']:.4f}"
            saida.append(
                f"| {METODOS_TABELA[v]} | " + " | ".join(
                    f"{b[k]['so_bruto']} / {b[k]['so_hmm']}, p={b[k]['p_valor']:.4f}" for k in ("alvos", "laterais", "esp"))
                + f" | {t['n_pares']}, {t['mediana_bruto_s']:.0f} / {t['mediana_hmm_s']:.0f}, {t['diferencas_zero']}, p={p_t} |")
        saida.append("")
    return "\n".join(saida)


AVISO_VAZAMENTO = (
    "**ATENÇÃO: com calibração real do Presente, esta avaliação é IN-SAMPLE COM VAZAMENTO DIRETO "
    "(a emissão Presente é ajustada nos próprios alvos avaliados). Não reportar estes números como "
    "desempenho; a estimativa a reportar é a do LOSO (`explorar_loso_fc.py`, braço HMM-REAL).**"
    if CALIBRACAO_PRESENTE == "real" else
    "Calibração sintética do Presente (ESP + senoide com SNR medida nos arquivos de estímulo: vazamento leve).")


def escrever_markdown(caminho, por_paciente, por_intensidade, geral, escolhidos, comp):
    partes = [
        "# Tabelas do exame — Rayleigh × HMM",
        "",
        "Gerado por `src/rayleigh_hmm/experimento.py`. Validação in-sample: ajuste e "
        "avaliação nos mesmos pacientes. " + AVISO_VAZAMENTO + " Detecção = alvos (81–95 Hz) em 30–70 dB; "
        "FP lateral = laterais (82–96 Hz) em todas as condições; FP alvo ESP = alvos "
        "na gravação sem estímulo. Tempo de exame = épocas acumuladas até a regra "
        "disparar (1 época = 1 s), só entre os alvos detectados.",
        "",
        "## Configurações escolhidas",
        "",
    ]
    for nome, rotulo in METODOS_TABELA.items():
        c = escolhidos[nome]
        params = {k: v for k, v in c.items() if k in ("alfa", "pi", "A", "regra")}
        partes.append(f"- **{rotulo}**: `{json.dumps(params, ensure_ascii=False)}`")
    partes += ["", "## Geral", "", tabela_markdown(geral, None, "Método"), "", "## Por intensidade", ""]
    for nome, rotulo in METODOS_TABELA.items():
        linhas = [l for l in por_intensidade if l["metodo"] == nome]
        partes += [f"### {rotulo}", "", tabela_markdown(linhas, "condicao", "Condição"), ""]
    partes += ["## Por paciente", ""]
    for nome, rotulo in METODOS_TABELA.items():
        linhas = [l for l in por_paciente if l["metodo"] == nome]
        partes += [f"### {rotulo}", "", tabela_markdown(linhas, "participante", "Participante"), ""]
    nota = (" O Rayleigh otimizado decide igual à referência, então os testes são os mesmos."
            if comp["bruto_identico_a_referencia"] else "")
    partes += ["## Significância: Rayleigh × HMM", "",
               "McNemar exato por alvo de estímulo, lateral e alvo ESP (pares discordantes); "
               "Wilcoxon pareado (padrão do scipy) no tempo de exame, só nos alvos detectados "
               "pelos dois métodos. Sem correção para múltiplas comparações." + nota, "",
               tabela_significancia(comp), ""]
    with open(caminho, "w") as f:
        f.write("\n".join(partes))


def tabela_paciente_intensidade(sequencias, escolhidos):
    """Detectados/8 por participante x condicao (usada no mapa de calor)."""
    linhas = []
    chaves = sorted({(s["participante"], s["condicao"]) for s in sequencias},
                    key=lambda c: (c[0], ordem_condicao(c[1])))
    for participante, condicao in chaves:
        linha = {"participante": participante, "condicao": condicao}
        for nome, c in escolhidos.items():
            r = resumo(sequencias, c, lambda s: s["participante"] == participante and s["condicao"] == condicao)
            linha[f"{nome}_alvo_detectados"] = r["deteccao_detectados"] if condicao != "ESP" else r["fp_alvo_esp_detectados"]
            linha[f"{nome}_alvo_total"] = r["deteccao_total"] if condicao != "ESP" else r["fp_alvo_esp_total"]
            linha[f"{nome}_lateral_detectados"] = r["fp_lateral_detectados"]
            linha[f"{nome}_lateral_total"] = r["fp_lateral_total"]
            linha[f"{nome}_tempo_medio_s"] = r["tempo_medio_s"]
        linhas.append(linha)
    return linhas


def mcnemar(sequencias, det_a, det_b, selecao=None):
    """McNemar exato (binomial) nas sequencias de `selecao` (padrao: alvos de estimulo)."""
    if selecao is None:
        selecao = [s["tipo"] == "alvo" and s["condicao"] != "ESP" for s in sequencias]
    a = np.asarray(det_a, bool)[selecao]
    b = np.asarray(det_b, bool)[selecao]
    so_a, so_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = stats.binomtest(so_a, so_a + so_b, 0.5).pvalue if so_a + so_b else 1.0
    return {"so_bruto": so_a, "so_hmm": so_b, "p_valor": float(p)}


def wilcoxon_tempo(masc, a, b):
    """Wilcoxon pareado (padrao do scipy) no tempo de exame, so nos alvos detectados pelos dois."""
    ambos = masc["alvo_estimulo"] & a["detectado"] & b["detectado"]
    ta, tb = a["tempo_s"][ambos], b["tempo_s"][ambos]
    zeros = int(np.sum(ta == tb))
    r = {"n_pares": int(ambos.sum()), "mediana_bruto_s": float(np.median(ta)), "mediana_hmm_s": float(np.median(tb)),
         "diferencas_zero": zeros, "estatistica": None, "p_valor": None}
    if zeros < ambos.sum():
        w = stats.wilcoxon(ta, tb)
        r["estatistica"], r["p_valor"] = float(w.statistic), float(w.pvalue)
    return r


def significancia(sequencias, masc, bruto, hmm_c):
    """McNemar (alvos, laterais, ESP) e Wilcoxon do tempo: detector bruto x variante HMM."""
    return {
        "alvos": mcnemar(sequencias, bruto["detectado"], hmm_c["detectado"], masc["alvo_estimulo"]),
        "laterais": mcnemar(sequencias, bruto["detectado"], hmm_c["detectado"], masc["lateral"]),
        "esp": mcnemar(sequencias, bruto["detectado"], hmm_c["detectado"], masc["alvo_esp"]),
        "tempo": wilcoxon_tempo(masc, bruto, hmm_c),
    }


VARIANTES_HMM = ("hmm_preditiva", "hmm_map", "hmm_marginal", "hmm_viterbi")


def comparacao_significancia(sequencias, masc, escolhidos):
    """Contra bruto_referencia e contra bruto; se os dois decidem igual, so a referencia."""
    ref, otimo = escolhidos["bruto_referencia"], escolhidos["bruto"]
    identicos = bool(np.array_equal(ref["detectado"], otimo["detectado"])
                     and np.array_equal(ref["tempo_s"], otimo["tempo_s"], equal_nan=True))
    parceiros = ["bruto_referencia"] if identicos else ["bruto_referencia", "bruto"]
    return {
        "bruto_identico_a_referencia": identicos,
        "parceiros": {
            p: {v: significancia(sequencias, masc, escolhidos[p], escolhidos[v]) for v in VARIANTES_HMM}
            for p in parceiros
        },
    }


def sem_arrays(c):
    return {k: v for k, v in c.items() if k not in ("detectado", "tempo_s")}


def salvar(sequencias, masc, emissoes, escolhidos, familias, fp_maximo):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    por_paciente, por_intensidade, geral = tabelas_resumo(sequencias, escolhidos)
    escrever_csv(os.path.join(PASTA_RESULTADOS, "tabela_por_paciente.csv"), por_paciente)
    escrever_csv(os.path.join(PASTA_RESULTADOS, "tabela_por_intensidade.csv"), por_intensidade)
    escrever_csv(os.path.join(PASTA_RESULTADOS, "tabela_geral.csv"), geral)
    comparacao = comparacao_significancia(sequencias, masc, escolhidos)
    escrever_markdown(os.path.join(PASTA_RESULTADOS, "tabelas.md"), por_paciente, por_intensidade, geral, escolhidos, comparacao)

    linhas_mapa = tabela_paciente_intensidade(sequencias, escolhidos)
    escrever_csv(os.path.join(PASTA_RESULTADOS, "por_paciente_intensidade.csv"), linhas_mapa)

    campos = ["participante", "condicao", "tipo", "frequencia_hz", "fs", "n_epocas", "n_janelas",
              "duracao_gravacao_s", "fracao_rejeitada", "p_final"]
    seq_linhas = []
    for i, s in enumerate(sequencias):
        linha = {c: s[c] for c in campos[:-1]}
        linha["p_final"] = float(np.exp(-s["z"][-1]))
        for nome, c in escolhidos.items():
            linha[f"detectado_{nome}"] = bool(c["detectado"][i])
            linha[f"tempo_s_{nome}"] = None if np.isnan(c["tempo_s"][i]) else float(c["tempo_s"][i])
        seq_linhas.append(linha)
    escrever_csv(os.path.join(PASTA_RESULTADOS, "por_sequencia.csv"), seq_linhas)

    post = emissoes["posteriori"]
    saida = {
        "configuracao": {
            "sinal": {**SINAL, "nomes_eletrodos": [NOMES_ELETRODOS[i] for i in SINAL["canais"]]},
            "fracao_rejeitada_media": float(np.mean([s["fracao_rejeitada"] for s in sequencias])),
            "epocas_por_janela": EPOCAS_POR_JANELA,
            "estatistica": "Rayleigh acumulado desde a primeira epoca",
            "exame": "sequencial: para na primeira avaliacao em que a regra dispara",
            "hmm_posteriori": "filtrada (causal)",
            "hmm_viterbi": "emissao marginal; avaliacao t detectada se o caminho de Viterbi de 1..t tem algum Presente",
            "minimo_avaliacoes": MINIMO_AVALIACOES,
            "grade_fracao_minima": GRADE_FRACAO_MINIMA,
            "grade_consecutivas": GRADE_CONSECUTIVAS,
            "fp_maximo": fp_maximo,
            "fp_maximo_regra": ("FP lateral do bruto_referencia (<=, comparado com folga de 1e-12)"
                                if FP_IGUAL_REFERENCIA else "teto fixo FP_MAXIMO_FIXO"),
            "fp_maximo_laterais": int(round(fp_maximo * masc["lateral"].sum())),
            "fp_maximo_n_laterais": int(masc["lateral"].sum()),
            "fp_maximo_fixo_alternativo": FP_MAXIMO_FIXO,
            "calibracao_presente": CALIBRACAO_PRESENTE,
            "calibracao_presente_descricao": (
                "real: Z e N acumulados de todas as sequencias-alvo com estimulo (30-70 dB); "
                "avaliacao in-sample com vazamento direto, estimativa a reportar = LOSO (explorar_loso_fc.py, HMM-REAL)"
                if CALIBRACAO_PRESENTE == "real" else
                "sintetica: ESP + senoide com 5 SNRs fixas (uma por intensidade), fase 0, sem sorteio; medidas nos arquivos de estimulo (vazamento leve)"),
            "snr_sintetica_db": SNR_SINTETICA_DB,  # usada so com calibracao "sintetica"
            "priori_presente": PRIORI_PRESENTE,
            "limiar_posteriori": LIMIAR_POSTERIORI,
            "semente": SEMENTE,
            "validacao": ("in-sample com vazamento direto (Presente ajustado nos alvos avaliados)"
                          if CALIBRACAO_PRESENTE == "real" else "in-sample (ajuste e avaliacao nos mesmos pacientes)"),
        },
        "dados": {
            "arquivos": len({(s["participante"], s["condicao"]) for s in sequencias}),
            "participantes": sorted({s["participante"] for s in sequencias}),
            "observacoes_esp_alvo": int(emissoes["z_esp"].size),
            "calibracao_presente": CALIBRACAO_PRESENTE,
            "sequencias_calibracao_presente": int(emissoes["marginal"]["hiper"]["n_sequencias"]),
            "pares_zn_calibracao_presente": int(emissoes["z_sintetico"].size),
            "observacoes_sinteticas": int(emissoes["z_sintetico"].size) if CALIBRACAO_PRESENTE == "sintetica" else 0,
        },
        "emissao": {
            "ausente_mle": emissoes["ausente"],
            "ks_esp_vs_exp1": emissoes["ks_esp_vs_exp1"],
            "presente_map": post["map"],
            "presente_media_posteriori": post["media"],
            "pontos_preditiva": int(post["pontos_preditiva"]["peso"].size),
            "presente_marginal": {**emissoes["marginal"]["hiper"], "j_k": J_MARGINAL_K, "j_rho": J_MARGINAL_RHO,
                                  "j": J_MARGINAL_K * J_MARGINAL_RHO},
        },
        "metodos": {nome: sem_arrays(c) for nome, c in escolhidos.items()},
        "melhor_por_familia": {
            metodo: {familia: sem_arrays(c) for familia, c in melhores.items()}
            for metodo, melhores in familias.items()
        },
        "tabela_geral": geral,
        "tabela_por_intensidade": por_intensidade,
        "tabela_por_paciente": por_paciente,
        "comparacao_mcnemar": {
            nome: mcnemar(sequencias, escolhidos["bruto"]["detectado"], escolhidos[nome]["detectado"])
            for nome in VARIANTES_HMM
        },
        "comparacao_significancia": comparacao,
    }
    with open(os.path.join(PASTA_RESULTADOS, "resultados.json"), "w") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    return saida, linhas_mapa


def imprimir(saida):
    e = saida["emissao"]
    print(f"\nArquivos: {saida['dados']['arquivos']} | observacoes ESP alvo: {saida['dados']['observacoes_esp_alvo']}"
          f" | calibracao Presente: {saida['dados']['calibracao_presente']}, "
          f"{saida['dados']['sequencias_calibracao_presente']} sequencias, "
          f"{saida['dados']['pares_zn_calibracao_presente']} pares (Z,N)")
    print(f"Ausente (MLE ESP): k={e['ausente_mle']['k']:.3f} mu={e['ausente_mle']['mu']:.3f}"
          f" | KS vs Exp(1) p={e['ks_esp_vs_exp1']['p_valor']:.3f}")
    print(f"Presente MAP: k={e['presente_map']['k']:.3f} rho={e['presente_map']['rho']:.5f}")
    h = e["presente_marginal"]
    print(f"Presente marginal: k~Gamma({h['a_k']:.3f},{h['b_k']:.3f}) media {h['k_media']:.3f} dp {h['k_desvio']:.3f}"
          f" | rho~Gamma({h['a_rho']:.3f},{h['b_rho']:.3f}) media {h['rho_media']:.4f} dp {h['rho_desvio']:.4f}"
          f" | J={h['j']} na_borda={h['na_borda']} dispersao inicios={h['dispersao_relativa_inicios']:.2e}")
    print("\nConfiguracoes escolhidas:")
    for nome, m in saida["metodos"].items():
        params = {k: v for k, v in m.items() if k in ("alfa", "pi", "A", "regra")}
        print(f"  {nome:<18}{params}")
    print("\nMelhor por familia (deteccao / FP lateral / tempo medio):")
    for metodo, melhores in saida["melhor_por_familia"].items():
        print(f"  {metodo}")
        for familia, c in melhores.items():
            m = c["metricas"]
            regra = {k: v for k, v in c["regra"].items() if k != "familia"}
            print(f"    {familia:<24}{m['deteccao']:>8.2%}{m['fp_lateral']:>8.2%}{m['tempo_medio_s']:>8.0f}s  {regra}")
    print("\nGeral:")
    print(tabela_markdown(saida["tabela_geral"], None, "Metodo"))
    print("\nPor intensidade:")
    for nome in ("bruto", "hmm_preditiva"):
        print(f"\n{METODOS_TABELA[nome]}")
        print(tabela_markdown([l for l in saida["tabela_por_intensidade"] if l["metodo"] == nome], "condicao", "Condicao"))
    print("\nMcNemar (alvos de estimulo, bruto otimizado x HMM):")
    for nome, r in saida["comparacao_mcnemar"].items():
        print(f"  {nome}: so bruto={r['so_bruto']} so HMM={r['so_hmm']} p={r['p_valor']:.4f}")
    comp = saida["comparacao_significancia"]
    print("\nSignificancia (McNemar alvos/laterais/ESP e Wilcoxon do tempo; bruto x HMM)"
          + (" - bruto identico a bruto_referencia, calculado uma vez:" if comp["bruto_identico_a_referencia"] else ":"))
    for parceiro, blocos in comp["parceiros"].items():
        print(f"  contra {parceiro}")
        for v, b in blocos.items():
            t = b["tempo"]
            p_t = "-" if t["p_valor"] is None else f"{t['p_valor']:.4f}"
            mc = " | ".join(f"{k} {b[k]['so_bruto']}x{b[k]['so_hmm']} p={b[k]['p_valor']:.4f}" for k in ("alvos", "laterais", "esp"))
            print(f"    {v:<14}{mc} | tempo n={t['n_pares']} med {t['mediana_bruto_s']:.0f}/{t['mediana_hmm_s']:.0f}s "
                  f"zeros={t['diferencas_zero']} W={t['estatistica']} p={p_t}")


def main():
    sequencias, esp = carregar_sequencias()
    emissoes = ajustar_emissoes(sequencias, esp)

    masc = mascaras(sequencias)
    familias = {}
    referencia = calcular_referencia(sequencias, masc)
    fp_maximo = referencia["metricas"]["fp_lateral"] if FP_IGUAL_REFERENCIA else FP_MAXIMO_FIXO
    print(f"Teto de FP lateral da busca: {fp_maximo:.4%}")
    familias["bruto"], _ = buscar_bruto(sequencias, masc, fp_maximo)
    escolhidos = {"bruto_referencia": referencia, "bruto": melhor_geral(familias["bruto"])}
    emissoes["marginal"] = ajustar_marginal(emissoes)
    for variante in ("preditiva", "map", "marginal"):
        nome = f"hmm_{variante}"
        familias[nome], escolhidos[nome] = buscar_hmm(sequencias, masc, emissoes, variante, fp_maximo=fp_maximo)
    familias["hmm_viterbi"], escolhidos["hmm_viterbi"] = buscar_hmm(
        sequencias, masc, emissoes, "marginal", hmm.viterbi_alguma_presente, fp_maximo)
    faltando = [nome for nome, c in escolhidos.items() if c is None]
    if faltando:
        raise RuntimeError(f"Nenhuma configuracao com FP lateral <= {fp_maximo}: {faltando}")

    saida, linhas = salvar(sequencias, masc, emissoes, escolhidos, familias, fp_maximo)
    graficos.gerar(PASTA_FIGURAS, emissoes, saida, linhas)
    imprimir(saida)
    print(f"\nResultados em {PASTA_RESULTADOS}\nFiguras em {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
