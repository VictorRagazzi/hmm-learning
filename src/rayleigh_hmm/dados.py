"""Leitura dos .mat, coeficientes FFT por epoca e estatistica de Rayleigh.

Cada arquivo tem x com formato (canais, epocas, amostras). As epocas tem
1 s (amostras == Fs), portanto a resolucao espectral e 1 Hz e cada
frequencia de freqEstim/binsM cai exatamente em um bin da FFT.
"""

import glob
import os
import re

import h5py
import numpy as np

EPS = 1e-12
PADRAO_ARQUIVO = re.compile(r"^(?P<participante>[A-Za-z]+)(?P<condicao>ESP|\d+dB)\.mat$")


def listar_arquivos(pasta_dados):
    """Retorna [(caminho, participante, condicao)] ordenado, so ESP e NNdB."""
    arquivos = []
    for caminho in sorted(glob.glob(os.path.join(pasta_dados, "*.mat"))):
        casamento = PADRAO_ARQUIVO.match(os.path.basename(caminho))
        if casamento:
            arquivos.append(
                (caminho, casamento["participante"], casamento["condicao"])
            )
    return arquivos


def nomes_eletrodos(pasta_dados):
    """Nomes dos 16 canais, lidos de eletrodos.mat (uint16, shape (2, 16))."""
    with h5py.File(os.path.join(pasta_dados, "eletrodos.mat"), "r") as arquivo:
        cod = np.asarray(arquivo["eletrodos"])
    return [chr(int(cod[0, i])) + chr(int(cod[1, i])) for i in range(cod.shape[1])]


def carregar_x(caminho):
    """Le x (canais, epocas, amostras) inteiro, mais Fs, freqEstim e binsM."""
    with h5py.File(caminho, "r") as arquivo:
        return {
            "x": np.asarray(arquivo["x"], dtype=float),
            "fs": float(np.asarray(arquivo["Fs"]).ravel()[0]),
            "freq_estim": np.asarray(arquivo["freqEstim"], dtype=float).ravel(),
            "bins_m": np.asarray(arquivo["binsM"], dtype=float).ravel(),
        }


def combinar_canais(x, combinacao, canais):
    """Monta UM sinal por epoca (epocas, amostras) a partir de x[canais].

    "canal": o unico canal listado; "media": media no tempo dos canais;
    "pca": primeira componente principal (nao supervisionada), calculada com
    todas as epocas concatenadas e cada canal centrado.
    """
    sel = x[list(canais)]
    if combinacao == "canal":
        if len(canais) != 1:
            raise ValueError("combinacao 'canal' exige exatamente um canal")
        return sel[0]
    if combinacao == "media":
        return sel.mean(axis=0)
    if combinacao == "pca":
        centrado = sel - sel.mean(axis=(1, 2), keepdims=True)
        plano = centrado.reshape(len(canais), -1)
        _, autovetores = np.linalg.eigh(plano @ plano.T)
        peso = autovetores[:, -1]  # maior autovalor; o sinal do vetor e irrelevante
        return np.tensordot(peso, centrado, axes=1)
    raise ValueError(f"combinacao desconhecida: {combinacao}")


def razao_variancia(sinal, x, canais):
    """Variancia do sinal combinado / media das variancias dos canais usados."""
    return float(sinal.var() / np.mean(x[list(canais)].var(axis=(1, 2))))


def epocas_validas(sinal, rejeicao_fator):
    """Mascara (epocas,): False se std da epoca > fator * mediana dos std."""
    if rejeicao_fator is None:
        return np.ones(sinal.shape[0], dtype=bool)
    desvios = sinal.std(axis=1)
    return desvios <= rejeicao_fator * np.median(desvios)


def coeficientes_do_sinal(bruto, combinacao, canais, rejeicao_fator=None):
    """Combina canais, rejeita epocas e devolve os coeficientes FFT por epoca.

    bruto: saida de carregar_x. Retorna dict com fs, n_epocas,
    duracao_epoca_s, freq_estim, bins_m, coef_alvo (em freqEstim),
    coef_lateral (em binsM), epoca_valida (bool, n_epocas) e razao_variancia.
    """
    fs = bruto["fs"]
    sinal = combinar_canais(bruto["x"], combinacao, canais)
    n_amostras = sinal.shape[1]
    freqs_fft = np.fft.rfftfreq(n_amostras, d=1.0 / fs)
    espectro = np.fft.rfft(sinal, axis=1)

    def indices(frequencias):
        idx = np.array([int(np.argmin(np.abs(freqs_fft - f))) for f in frequencias])
        erro = np.max(np.abs(freqs_fft[idx] - frequencias))
        if erro > 1e-6:
            raise ValueError(f"frequencia fora da grade da FFT (erro {erro} Hz)")
        return idx

    return {
        "fs": fs,
        "n_epocas": sinal.shape[0],
        "duracao_epoca_s": n_amostras / fs,
        "freq_estim": bruto["freq_estim"],
        "bins_m": bruto["bins_m"],
        "coef_alvo": espectro[:, indices(bruto["freq_estim"])],
        "coef_lateral": espectro[:, indices(bruto["bins_m"])],
        "epoca_valida": epocas_validas(sinal, rejeicao_fator),
        "razao_variancia": razao_variancia(sinal, bruto["x"], canais),
    }


def carregar_coeficientes(caminho, combinacao="canal", canais=(0,), rejeicao_fator=None):
    """Le um arquivo e devolve coeficientes FFT do sinal combinado (ver acima)."""
    return coeficientes_do_sinal(carregar_x(caminho), combinacao, canais, rejeicao_fator)


def janelar(coeficientes, m):
    """Divide (epocas, ...) em janelas SEM sobreposicao de m epocas.

    Retorna (n_janelas, m, ...). Serve tambem para a mascara (epocas,). O
    exame comeca na primeira epoca; as epocas finais que nao completam uma
    janela sao descartadas.
    """
    n_janelas = coeficientes.shape[0] // m
    return coeficientes[: n_janelas * m].reshape((n_janelas, m) + coeficientes.shape[1:])


def rayleigh_z(janelas):
    """Estatistica de Rayleigh Z = M * PLV^2 para cada (janela, freq).

    Sob H0 (fases uniformes) Z ~ Exp(1) aproximadamente; p-valor = exp(-Z).
    """
    m = janelas.shape[1]
    fases = janelas / (np.abs(janelas) + EPS)
    plv = np.abs(fases.sum(axis=1)) / m
    return m * plv**2


def rayleigh_z_acumulado(janelas, valida_janelas=None):
    """Rayleigh acumulado: no passo k usa TODAS as epocas validas das janelas 0..k.

    janelas: (n_janelas, m, n_freqs); valida_janelas: (n_janelas, m) bool ou
    None (todas validas). Soma so os fasores unitarios das epocas validas;
    N_k = epocas validas acumuladas e Z_k = |soma|^2 / N_k (0 se N_k = 0).
    Retorna z (n_janelas, n_freqs) e N_k (n_janelas,). Sob H0, Z ~ Exp(1)
    em todo passo; com resposta, E[Z] = 1 + (N - 1) * rho, com rho = PLV^2
    verdadeiro da resposta.
    """
    n_janelas, m, _ = janelas.shape
    if valida_janelas is None:
        valida_janelas = np.ones((n_janelas, m), dtype=bool)
    fases = janelas / (np.abs(janelas) + EPS) * valida_janelas[:, :, None]
    soma = np.cumsum(fases.sum(axis=1), axis=0)
    n_validas = np.cumsum(valida_janelas.sum(axis=1))
    z = np.abs(soma) ** 2 / np.maximum(n_validas, 1)[:, None]
    return z, n_validas


def injetar_senoide(coeficientes, snr_db, valida=None):
    """Soma uma senoide sintetica aos coeficientes de cada frequencia.

    Para epocas de 1 s e frequencia inteira, somar A*cos(2*pi*f*t + phi) no
    tempo equivale a somar (A*N/2)*exp(j*phi) ao coeficiente FFT do bin f.
    Por isso a injecao e feita direto no coeficiente.

    A amplitude e definida pela SNR no bin: |S|^2 / mean(|X|^2), em que o
    denominador e a potencia media (epocas validas) do proprio ESP naquela frequencia. A fase
    phi e fixa em 0 e constante entre epocas (resposta em fase com o estimulo);
    a fase nao altera a distribuicao de Z. Nada e sorteado: a mesma entrada
    gera sempre a mesma saida. snr_db e escalar ou broadcastavel para
    (epocas, n_freqs).
    """
    # ruido estimado so nas epocas validas (valida: bool (epocas,) ou None)
    base = coeficientes if valida is None else coeficientes[valida]
    potencia_ruido = np.mean(np.abs(base) ** 2, axis=0)
    snr = 10.0 ** (np.asarray(snr_db) / 10.0)
    amplitude = np.sqrt(snr * potencia_ruido[None, :])
    return coeficientes + amplitude
