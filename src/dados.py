"""EEG → FFT → detector → observações independentes por paciente/intensidade/frequência."""

import json
import re

import h5py
import numpy as np

import config
from detectors import calcular_janelas


def salvar_json(resultado, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(resultado, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")


def carregar_dados(pasta=config.DATA_DIR):
    """Lê somente o canal escolhido e guarda FFTs; não mantém todo o EEG."""
    pacientes = {}
    for caminho in sorted(pasta.glob("*.mat")):
        nome = re.fullmatch(r"(.+?)(ESP|30dB|40dB|50dB|60dB|70dB)\.mat", caminho.name)
        if nome is None:
            continue
        paciente, condicao = nome.groups()
        nivel = "ESP" if condicao == "ESP" else int(condicao[:-2])
        with h5py.File(caminho, "r") as arquivo:
            if not {"x", "Fs", "freqEstim", "binsM"} <= set(arquivo):
                raise ValueError(f"{caminho.name}: metadados obrigatórios ausentes")
            forma = arquivo["x"].shape
            fs = float(np.asarray(arquivo["Fs"]).ravel()[0])
            alvos = np.asarray(arquivo["freqEstim"]).ravel()
            laterais = np.asarray(arquivo["binsM"]).ravel()
            if (len(forma) != 3 or forma[0] != 16 or forma[1] < 1
                    or not 0 <= config.CHANNEL_INDEX < forma[0]
                    or not np.isfinite(fs) or fs <= 0
                    or not np.isclose(forma[2] / fs, 1)):
                raise ValueError(f"{caminho.name}: esperado EEG (16, épocas, Fs), épocas de 1s")
            frequencias = np.concatenate([alvos, laterais])
            if (len(alvos) == 0 or len(alvos) != len(laterais)
                    or not np.all(np.isfinite(frequencias))
                    or np.any(frequencias <= 0) or np.any(frequencias >= fs / 2)
                    or len(np.unique(frequencias)) != len(frequencias)):
                raise ValueError(f"{caminho.name}: frequências inválidas")
            eeg = np.asarray(arquivo["x"][config.CHANNEL_INDEX], dtype=float)
        if not np.all(np.isfinite(eeg)):
            raise ValueError(f"{caminho.name}: EEG contém valores não finitos")
        fft = np.fft.rfft(eeg, axis=1)
        eixo_hz = np.fft.rfftfreq(forma[2], d=1 / fs)
        bins = [int(np.argmin(abs(eixo_hz - f))) for f in frequencias]
        # F espectral: bins mais próximos, excluindo DC, alvo e todos os estímulos.
        excluidos = set(bins[:len(alvos)])
        ruido_bins = []
        for alvo in bins:
            candidatos = [i for i in range(1, len(eixo_hz)) if i not in excluidos and i != alvo]
            candidatos.sort(key=lambda i: (abs(eixo_hz[i] - eixo_hz[alvo]), i))
            if len(candidatos) < 2:
                raise ValueError(f"{caminho.name}: bins insuficientes para F espectral")
            ruido_bins.append(candidatos[:2])
        pacientes.setdefault(paciente, {})[nivel] = {
            "arquivo": caminho.name, "fs": fs, "n_epocas": forma[1],
            "alvos": alvos.tolist(), "laterais": laterais.tolist(),
            "coeficientes": fft[:, bins].T,
            "coeficientes_locais": np.asarray([fft[:, [alvo, *ruido]]
                                                for alvo, ruido in zip(bins, ruido_bins)]),
            "ruido_hz": [[float(eixo_hz[i]) for i in indices] for indices in ruido_bins],
        }
    if not pacientes:
        raise ValueError(f"Nenhum EEG encontrado em {pasta}")
    for paciente, fases in pacientes.items():
        if set(fases) != set(config.NIVEIS):
            raise ValueError(f"{paciente}: faltam fases ESP/30/40/50/60/70")
        for fase in fases.values():
            if (fase["alvos"] != fases["ESP"]["alvos"]
                    or fase["laterais"] != fases["ESP"]["laterais"]):
                raise ValueError(f"{paciente}: frequências diferentes entre fases")
    return pacientes


def preparar_protocolos(dados, tamanho, passo, detector=None):
    """Uma sequência por arquivo/frequência, com relógio local."""
    detector = config.DETECTOR_NAME if detector is None else detector
    protocolos = []
    for paciente, arquivos in sorted(dados.items()):
        n_freq = len(arquivos["ESP"]["alvos"])
        for nivel in config.NIVEIS:
            arquivo = arquivos[nivel]
            for grupo in ("alvo", "lateral"):
                for indice in range(n_freq):
                    observacoes = []
                    coluna = indice if grupo == "alvo" else n_freq + indice
                    chave = "coeficientes_locais" if detector == "spectral_f" else "coeficientes"
                    valores, p_values, intervalos = calcular_janelas(
                        arquivo[chave][coluna], tamanho, passo, detector)
                    frequencia = arquivo["alvos" if grupo == "alvo" else "laterais"][indice]
                    for valor, p, (inicio, fim) in zip(valores, p_values, intervalos):
                        observacoes.append({
                            "arquivo": arquivo["arquivo"], "nivel": nivel,
                            "canal": config.CHANNEL_INDEX, "fs": arquivo["fs"],
                            "intervalo_epocas": [inicio, fim], "fim_s": fim,
                            "detector": detector, "estatistica": valor, "p_value": p,
                            "ruido_hz": arquivo["ruido_hz"][coluna] if detector == "spectral_f" else [],
                        })
                    protocolos.append({
                        "paciente": paciente, "nivel": nivel, "arquivo": arquivo["arquivo"],
                        "detector": detector, "grupo": grupo, "frequencia_indice": indice,
                        "frequencia": frequencia, "alvo_hz": arquivo["alvos"][indice],
                        "controle_hz": arquivo["laterais"][indice],
                        "fases": [{"nivel": nivel, "inicio_s": 0, "n_epocas": arquivo["n_epocas"]}],
                        "duracao_s": arquivo["n_epocas"], "observacoes": observacoes,
                    })
    return protocolos
