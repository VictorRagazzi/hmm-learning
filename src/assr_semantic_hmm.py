"""Pipeline ASSR por CSM e compilação temporal semântica.

A LLM é usada somente pelo comando ``compile``. A inferência carrega uma
tabela JSON pronta e faz apenas consultas determinísticas.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.error
import urllib.request
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import chi2


# Configuração central
ESTADOS = ["rest", "response"]
NIVEIS_OBSERVACAO = ["absent", "low", "medium", "strong"]
ESTADO_INICIAL = "rest"
ESTADO_RESPOSTA = "response"
ORDEM_CONTEXTO = 3

FREQUENCIAS_ESPERADAS_HZ = list(range(81, 96, 2))
TAMANHO_JANELA_EPOCAS = 20
PASSO_JANELA_EPOCAS = 5
SEGUNDOS_INICIAIS_DESCARTADOS = 2
LIMIAR_ARTEFATO_VOLTS = 0.1 / 200

# Os N-1 cortes são interpolados em escala -log10(p). Alterar a quantidade
# de níveis não exige alterar a função de discretização.
P_VALOR_INICIO_EVIDENCIA = 0.05
P_VALOR_EVIDENCIA_MAXIMA = 0.001

FRACAO_MINIMA_RESPOSTA = 0.30
JANELAS_CONSECUTIVAS_RESPOSTA = 3
COMBINACAO_CRITERIOS = "ou"  # "ou" ou "e"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-coder:30b"
OLLAMA_TIMEOUT_SEGUNDOS = 300

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ_PROJETO / "data"
ARQUIVO_TABELA = RAIZ_PROJETO / "semantic_table.json"
ARQUIVO_RESULTADOS = RAIZ_PROJETO / "results" / "assr_validation.json"


def carregar_mat(caminho: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Carrega MATLAB v7.3 e devolve sinal como (amostras, épocas, canais)."""
    with h5py.File(caminho, "r") as arquivo:
        ausentes = {"x", "Fs", "freqEstim"} - set(arquivo.keys())
        if ausentes:
            raise ValueError(f"{caminho.name}: variáveis ausentes: {sorted(ausentes)}")
        sinal = np.transpose(arquivo["x"][()], (2, 1, 0)).astype(float)
        fs = float(np.asarray(arquivo["Fs"]).squeeze())
        frequencias = np.asarray(arquivo["freqEstim"]).squeeze().astype(float)
        if "binsM" in arquivo:
            bins_matlab = np.asarray(arquivo["binsM"]).squeeze().astype(float)
            if not np.allclose(bins_matlab, frequencias + 1):
                raise ValueError(f"{caminho.name}: binsM não corresponde a freqEstim + 1")

    if sinal.ndim != 3 or sinal.shape[0] != round(fs):
        raise ValueError(f"{caminho.name}: forma {sinal.shape} incompatível com Fs={fs:g}")
    if not np.allclose(frequencias, FREQUENCIAS_ESPERADAS_HZ):
        raise ValueError(f"{caminho.name}: frequências inesperadas: {frequencias.tolist()}")
    return sinal, fs, frequencias


def preparar_epocas(sinal: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Remove DC, início da coleta e épocas com artefato de amplitude."""
    sinal = sinal - sinal.mean(axis=0, keepdims=True)
    sinal = sinal[:, SEGUNDOS_INICIAIS_DESCARTADOS:, :]
    picos = np.max(np.abs(sinal), axis=(0, 2))
    validas = picos <= LIMIAR_ARTEFATO_VOLTS
    resumo = {
        "epocas_apos_descarte_inicial": int(sinal.shape[1]),
        "epocas_rejeitadas": int(np.count_nonzero(~validas)),
        "epocas_validas": int(np.count_nonzero(validas)),
    }
    return sinal[:, validas, :], resumo


def criar_janelas(sinal: np.ndarray) -> list[np.ndarray]:
    """Cria janelas sobre o eixo de épocas sem assumir sua duração total."""
    return [
        sinal[:, inicio : inicio + TAMANHO_JANELA_EPOCAS, :]
        for inicio in range(
            0, sinal.shape[1] - TAMANHO_JANELA_EPOCAS + 1, PASSO_JANELA_EPOCAS
        )
    ]


def calcular_estatistica_banda(
    janela: np.ndarray, fs: float, frequencias_hz: np.ndarray
) -> float:
    """Retorna -log10(p) da CSM aditiva em toda a banda e todos os canais."""
    if janela.ndim != 3:
        raise ValueError("A janela deve ter forma (amostras, épocas, canais).")
    amostras, epocas, canais = janela.shape
    if epocas < 2:
        raise ValueError("CSM requer pelo menos duas épocas.")
    grade = np.fft.rfftfreq(amostras, d=1.0 / fs)
    indices = np.array([int(np.argmin(abs(grade - f))) for f in frequencias_hz])
    if not np.allclose(grade[indices], frequencias_hz, atol=1e-9):
        raise ValueError("As frequências-alvo não coincidem com bins da FFT.")

    espectro = np.fft.rfft(janela, axis=0)[indices, :, :]
    fasores = np.exp(1j * np.angle(espectro))
    csm = np.abs(fasores.mean(axis=1)) ** 2
    componentes = len(frequencias_hz) * canais
    evidencia_normalizada = epocas * float(csm.mean())
    log_p = chi2.logsf(2 * componentes * evidencia_normalizada, 2 * componentes)
    if not np.isfinite(log_p):
        return 300.0
    return max(0.0, min(300.0, -float(log_p) / math.log(10)))


def limiares_discretizacao(numero_niveis: int) -> np.ndarray:
    """Gera automaticamente N-1 cortes de evidência para N categorias."""
    if numero_niveis < 1:
        raise ValueError("Deve existir ao menos um nível de observação.")
    if numero_niveis == 1:
        return np.array([], dtype=float)
    minimo = -math.log10(P_VALOR_INICIO_EVIDENCIA)
    maximo = -math.log10(P_VALOR_EVIDENCIA_MAXIMA)
    return np.linspace(minimo, maximo, numero_niveis - 1)


def discretizar(
    estatistica: float, niveis: list[str] | tuple[str, ...] = NIVEIS_OBSERVACAO
) -> str:
    """Converte evidência contínua em categoria ordinal genérica."""
    if not np.isfinite(estatistica):
        raise ValueError("A estatística deve ser finita.")
    if len(set(niveis)) != len(niveis) or not niveis:
        raise ValueError("Os níveis devem ser não vazios e sem repetições.")
    indice = bisect_right(limiares_discretizacao(len(niveis)), estatistica)
    return niveis[indice]


def _descricao_estatistica_niveis() -> str:
    """Descreve para a LLM os cortes produzidos pela configuração atual."""
    cortes = limiares_discretizacao(len(NIVEIS_OBSERVACAO))
    limites_p = [10 ** (-corte) for corte in cortes]
    descricoes = []
    for indice, nivel in enumerate(NIVEIS_OBSERVACAO):
        if not limites_p:
            intervalo = "all values"
        elif indice == 0:
            intervalo = f"p > {limites_p[0]:.4g}"
        elif indice == len(NIVEIS_OBSERVACAO) - 1:
            intervalo = f"p <= {limites_p[-1]:.4g}"
        else:
            intervalo = f"{limites_p[indice]:.4g} < p <= {limites_p[indice - 1]:.4g}"
        descricoes.append(f"{nivel}: {intervalo}")
    return "; ".join(descricoes)


def _chamar_ollama(prompt: str, modelo: str) -> dict[str, Any]:
    """Faz uma chamada offline ao Ollama e exige resposta JSON."""
    corpo = {
        "model": modelo,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in ASSR, EEG, and temporal inference. "
                    "Reply only with the requested JSON object, without Markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "seed": 42},
    }
    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(corpo, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=OLLAMA_TIMEOUT_SEGUNDOS) as resposta:
            envelope = json.loads(resposta.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as erro:
        raise RuntimeError(f"Falha ao chamar Ollama em {OLLAMA_URL}: {erro}") from erro
    conteudo = envelope.get("message", {}).get("content", "")
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError as erro:
        raise ValueError(f"A LLM não retornou JSON válido: {conteudo!r}") from erro


def _validar_chaves_exatas(objeto: dict[str, Any], esperadas: list[str], nome: str) -> None:
    if set(objeto) != set(esperadas):
        raise ValueError(f"{nome}: chaves {sorted(objeto)}; esperadas {sorted(esperadas)}")


def gerar_historicos() -> list[tuple[str, ...]]:
    """Enumera as len(ESTADOS) ** ORDEM_CONTEXTO linhas da tabela."""
    if ORDEM_CONTEXTO < 1:
        raise ValueError("ORDEM_CONTEXTO deve ser pelo menos 1.")
    return list(product(ESTADOS, repeat=ORDEM_CONTEXTO))


def chave_historico(historico: tuple[str, ...] | list[str]) -> str:
    """Representa um contexto de forma legível e serializável em JSON."""
    if len(historico) != ORDEM_CONTEXTO or any(e not in ESTADOS for e in historico):
        raise ValueError(f"Histórico inválido: {historico}")
    return " -> ".join(historico)


def compilar_tabela_semantica(modelo: str = OLLAMA_MODEL) -> dict[str, Any]:
    """Faz uma chamada de notas e uma de compilação para cada histórico."""
    historicos = gerar_historicos()
    notas: dict[str, str] = {}

    # Primeira rodada: uma expert note exclusiva por linha de contexto.
    for historico in historicos:
        chave = chave_historico(historico)
        prompt_notas = f"""
Generate SPECIFIC expert notes for one context row of a temporal ASSR detection
model. History, from oldest to most recent: {chave}.
Context order: {ORDEM_CONTEXTO}. Allowed states:
{json.dumps(ESTADOS, ensure_ascii=False)}. The observation is derived from phase
synchrony aggregated across 8 frequencies and 16 channels. Statistical cutoffs:
{_descricao_estatistica_niveis()}.

Capture the particular temporal meaning of this sequence: persistence, recent
response onset or interruption, how to handle ambiguous evidence, and when the
current observation should override the tendency suggested by the history. Do
not rigidly determine the table, and do not produce a generic note reusable for
another context. Write the note in English. Return exactly:
{{"expert_note": "..."}}.
""".strip()
        resposta = _chamar_ollama(prompt_notas, modelo)
        if set(resposta) != {"expert_note"}:
            raise ValueError(f"Expert note fora do esquema para {chave}.")
        nota = resposta["expert_note"]
        if not isinstance(nota, str) or not nota.strip():
            raise ValueError(f"Expert note vazia para {chave}.")
        notas[chave] = nota.strip()
        print(f"Expert note gerada para {chave}: {notas[chave]}")

    # Segunda rodada: uma chamada por linha, cobrindo todos os níveis.
    tabela: dict[str, dict[str, str]] = {}
    for historico in historicos:
        chave = chave_historico(historico)
        exemplo = {nivel: "one allowed state" for nivel in NIVEIS_OBSERVACAO}
        prompt = f"""
Compile one row of a deterministic temporal table for ASSR detection.
History, from oldest to most recent: {chave}.
Context order: {ORDEM_CONTEXTO}.
Allowed states: {json.dumps(ESTADOS, ensure_ascii=False)}.
Complete observation levels, in increasing evidence order:
{json.dumps(NIVEIS_OBSERVACAO, ensure_ascii=False)}.
Statistical cutoffs: {_descricao_estatistica_niveis()}.
Expert notes EXCLUSIVE to this history: {notas[chave]}

Decide the next state for EACH level in a single coherent analysis. The note
modulates the decision, but it must not cause the current observation to be
ignored. Stronger evidence must never favor the response state less than weaker
evidence. Use the complete history to resolve ambiguous observations.
Return exactly: {{"transicoes": {json.dumps(exemplo, ensure_ascii=False)}}}.
""".strip()
        resposta = _chamar_ollama(prompt, modelo)
        if set(resposta) != {"transicoes"} or not isinstance(resposta["transicoes"], dict):
            raise ValueError(f"Resposta de compilação inválida para {chave}.")
        transicoes = resposta["transicoes"]
        _validar_chaves_exatas(transicoes, NIVEIS_OBSERVACAO, chave)
        if any(destino not in ESTADOS for destino in transicoes.values()):
            raise ValueError(f"{chave}: estado desconhecido retornado pela LLM.")
        tabela[chave] = transicoes
        print(f"Tabela compilada para {chave}: {transicoes}")

    return {
        "schema_version": 2,
        "compilado_em_utc": datetime.now(timezone.utc).isoformat(),
        "modelo_llm": modelo,
        "ordem_contexto": ORDEM_CONTEXTO,
        "numero_historicos": len(historicos),
        "chamadas_expert_notes": len(historicos),
        "chamadas_compilacao": len(historicos),
        "numero_chamadas_llm": 2 * len(historicos),
        "estados": ESTADOS,
        "niveis_observacao": NIVEIS_OBSERVACAO,
        "notas_especialista": notas,
        "tabela": tabela,
    }


def salvar_tabela(compilacao: dict[str, Any], caminho: Path = ARQUIVO_TABELA) -> None:
    caminho.write_text(json.dumps(compilacao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def carregar_tabela(caminho: Path = ARQUIVO_TABELA) -> dict[str, dict[str, str]]:
    if not caminho.exists():
        raise FileNotFoundError(f"Tabela ausente em {caminho}; execute primeiro 'compile'.")
    compilacao = json.loads(caminho.read_text(encoding="utf-8"))
    if compilacao.get("estados") != ESTADOS:
        raise ValueError("Estados da tabela diferem da configuração atual.")
    if compilacao.get("niveis_observacao") != NIVEIS_OBSERVACAO:
        raise ValueError("Níveis da tabela diferem da configuração atual.")
    if compilacao.get("ordem_contexto") != ORDEM_CONTEXTO:
        raise ValueError("Ordem de contexto da tabela difere da configuração atual.")
    tabela = compilacao.get("tabela", {})
    chaves = [chave_historico(h) for h in gerar_historicos()]
    _validar_chaves_exatas(tabela, chaves, "tabela")
    for chave in chaves:
        _validar_chaves_exatas(tabela[chave], NIVEIS_OBSERVACAO, chave)
        if any(destino not in ESTADOS for destino in tabela[chave].values()):
            raise ValueError(f"Destino inválido em {chave}.")
    return tabela


def inferir_sequencia(
    estatisticas: list[float], tabela: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Inferência online por lookup puro; nunca chama a LLM."""
    if ESTADO_RESPOSTA not in ESTADOS:
        raise ValueError("ESTADO_RESPOSTA deve pertencer a ESTADOS.")
    historico = [ESTADO_INICIAL] * ORDEM_CONTEXTO
    niveis, estados, historicos_consultados = [], [], []
    for estatistica in estatisticas:
        nivel = discretizar(estatistica, NIVEIS_OBSERVACAO)
        chave = chave_historico(historico)
        estado = tabela[chave][nivel]
        niveis.append(nivel)
        estados.append(estado)
        historicos_consultados.append(chave)
        historico = historico[1:] + [estado]

    fracao = sum(e == ESTADO_RESPOSTA for e in estados) / len(estados) if estados else 0.0
    maior_sequencia = atual = 0
    for observado in estados:
        atual = atual + 1 if observado == ESTADO_RESPOSTA else 0
        maior_sequencia = max(maior_sequencia, atual)
    criterios = (fracao >= FRACAO_MINIMA_RESPOSTA, maior_sequencia >= JANELAS_CONSECUTIVAS_RESPOSTA)
    if COMBINACAO_CRITERIOS == "ou":
        detectou = any(criterios)
    elif COMBINACAO_CRITERIOS == "e":
        detectou = all(criterios)
    else:
        raise ValueError("COMBINACAO_CRITERIOS deve ser 'ou' ou 'e'.")
    return {
        "niveis": niveis,
        "estados": estados,
        "historicos_consultados": historicos_consultados,
        "fracao_resposta": fracao,
        "maior_sequencia_resposta": maior_sequencia,
        "detectou": detectou,
    }


def _condicao_do_nome(nome: str) -> str:
    if nome.endswith("ESP.mat"):
        return "ESP"
    correspondencia = re.search(r"(\d+)dB\.mat$", nome)
    if not correspondencia:
        raise ValueError(f"Condição não reconhecida: {nome}")
    return f"{correspondencia.group(1)}dB"


def processar_arquivo(caminho: Path, tabela: dict[str, dict[str, str]]) -> dict[str, Any]:
    sinal, fs, frequencias = carregar_mat(caminho)
    sinal, qualidade = preparar_epocas(sinal)
    janelas = criar_janelas(sinal)
    estatisticas = [calcular_estatistica_banda(j, fs, frequencias) for j in janelas]
    inferencia = inferir_sequencia(estatisticas, tabela)
    return {
        "arquivo": caminho.name,
        "condicao": _condicao_do_nome(caminho.name),
        "fs_hz": fs,
        "frequencias_hz": frequencias.tolist(),
        **qualidade,
        "numero_janelas": len(janelas),
        "estatisticas_log10_p": estatisticas,
        **inferencia,
    }


def validar_dados(
    pasta_dados: Path = PASTA_DADOS,
    caminho_tabela: Path = ARQUIVO_TABELA,
    caminho_saida: Path = ARQUIVO_RESULTADOS,
) -> dict[str, Any]:
    """Processa cada gravação independentemente e resume por intensidade."""
    tabela = carregar_tabela(caminho_tabela)
    arquivos = [p for p in sorted(pasta_dados.glob("*.mat")) if p.name != "eletrodos.mat"]
    if not arquivos:
        raise FileNotFoundError(f"Nenhum EEG encontrado em {pasta_dados}")
    resultados = []
    for indice, caminho in enumerate(arquivos, 1):
        resultado = processar_arquivo(caminho, tabela)
        resultados.append(resultado)
        veredito = "RESPOSTA" if resultado["detectou"] else "repouso"
        print(f"[{indice:02d}/{len(arquivos)}] {caminho.name:<12} {veredito:<8} janelas={resultado['numero_janelas']:<3} fração={resultado['fracao_resposta']:.2f}")

    por_condicao: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for resultado in resultados:
        por_condicao[resultado["condicao"]].append(resultado)
    resumo = {}
    for condicao in sorted(por_condicao, key=lambda x: (x == "ESP", x)):
        grupo = por_condicao[condicao]
        deteccoes = sum(r["detectou"] for r in grupo)
        resumo[condicao] = {
            "arquivos": len(grupo),
            "deteccoes": deteccoes,
            "taxa_deteccao": deteccoes / len(grupo),
            "mediana_fracao_resposta": float(np.median([r["fracao_resposta"] for r in grupo])),
        }
    relatorio = {
        "schema_version": 1,
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "configuracao": {
            "estados": ESTADOS,
            "niveis_observacao": NIVEIS_OBSERVACAO,
            "estado_inicial": ESTADO_INICIAL,
            "estado_resposta": ESTADO_RESPOSTA,
            "ordem_contexto": ORDEM_CONTEXTO,
            "frequencias_hz": FREQUENCIAS_ESPERADAS_HZ,
            "tamanho_janela_epocas": TAMANHO_JANELA_EPOCAS,
            "passo_janela_epocas": PASSO_JANELA_EPOCAS,
            "segundos_iniciais_descartados": SEGUNDOS_INICIAIS_DESCARTADOS,
            "limiar_artefato_volts": LIMIAR_ARTEFATO_VOLTS,
            "p_valor_inicio_evidencia": P_VALOR_INICIO_EVIDENCIA,
            "p_valor_evidencia_maxima": P_VALOR_EVIDENCIA_MAXIMA,
            "fracao_minima_resposta": FRACAO_MINIMA_RESPOSTA,
            "janelas_consecutivas_resposta": JANELAS_CONSECUTIVAS_RESPOSTA,
            "combinacao_criterios": COMBINACAO_CRITERIOS,
        },
        "resumo_por_condicao": resumo,
        "resultados_por_arquivo": resultados,
    }
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return relatorio


def _imprimir_resumo(relatorio: dict[str, Any]) -> None:
    print("\nResumo por condição")
    for condicao, resumo in relatorio["resumo_por_condicao"].items():
        print(f"{condicao:>4}: {resumo['deteccoes']}/{resumo['arquivos']} detectados ({resumo['taxa_deteccao']:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    compilar = sub.add_parser("compile", help="Compila e salva a tabela via LLM.")
    compilar.add_argument("--model", default=OLLAMA_MODEL)
    compilar.add_argument("--table", type=Path, default=ARQUIVO_TABELA)
    validar = sub.add_parser("validate", help="Executa inferência sem chamar LLM.")
    validar.add_argument("--data", type=Path, default=PASTA_DADOS)
    validar.add_argument("--table", type=Path, default=ARQUIVO_TABELA)
    validar.add_argument("--output", type=Path, default=ARQUIVO_RESULTADOS)
    tudo = sub.add_parser("all", help="Compila offline e depois valida.")
    tudo.add_argument("--model", default=OLLAMA_MODEL)
    tudo.add_argument("--data", type=Path, default=PASTA_DADOS)
    tudo.add_argument("--table", type=Path, default=ARQUIVO_TABELA)
    tudo.add_argument("--output", type=Path, default=ARQUIVO_RESULTADOS)
    args = parser.parse_args()
    if args.comando in {"compile", "all"}:
        compilacao = compilar_tabela_semantica(args.model)
        salvar_tabela(compilacao, args.table)
        print(f"Tabela salva em {args.table}")
        print(json.dumps(compilacao["tabela"], ensure_ascii=False, indent=2))
    if args.comando in {"validate", "all"}:
        relatorio = validar_dados(args.data, args.table, args.output)
        _imprimir_resumo(relatorio)
        print(f"Resultados salvos em {args.output}")


if __name__ == "__main__":
    main()
