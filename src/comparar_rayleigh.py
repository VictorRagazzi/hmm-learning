"""Compara vencedores HMM MLE/MAP e Rayleigh bruto, por intensidade independente."""

import csv
import json

import numpy as np

import config
from dados import carregar_dados, preparar_protocolos, salvar_json
from get_obs_matrix import construir_matrizes_observacao
from hmm_inference import avaliar, primeiro_disparo, resumir
from search_hmm_parameters import contar_regras, preparar_lote


def avaliar_bruto(protocolos, alpha, regra):
    """Positivo bruto é p <= alpha; decisão temporal usa a mesma regra causal."""
    resultados = []
    for protocolo in protocolos:
        obs = protocolo['observacoes']
        estados = [o['p_value'] <= alpha for o in obs]
        disparo = primeiro_disparo(estados, obs, regra)
        deteccao = disparo if protocolo['nivel'] != 'ESP' else None
        resultados.append({
            'paciente': protocolo['paciente'], 'grupo': protocolo['grupo'],
            'nivel': protocolo['nivel'], 'arquivo': protocolo['arquivo'],
            'frequencia': protocolo['frequencia'], 'alvo_hz': protocolo['alvo_hz'],
            'controle_hz': protocolo['controle_hz'], 'n_janelas': len(obs),
            'detectou': deteccao is not None, 'disparo_total': disparo is not None,
            'fp_esp': protocolo['nivel'] == 'ESP' and disparo is not None,
            'primeiro_nivel_db': deteccao['nivel'] if deteccao else None,
            'tempo_deteccao_s': deteccao['fim_s'] if deteccao else None,
            'duracao_avaliada_s': deteccao['fim_s'] if deteccao else protocolo['duracao_s'],
        })
    return resultados


def buscar_bruto(dados, grade):
    melhor, avaliacoes = None, 0
    alphas = np.asarray(config.RAW_ALPHA_VALUES)
    if np.any(~np.isfinite(alphas)) or np.any((alphas <= 0) | (alphas >= 1)):
        raise ValueError('Limiar bruto deve estar em (0,1)')
    for tamanho, passo in grade['janelamentos']:
        protocolos = preparar_protocolos(dados, tamanho, passo, 'rayleigh')
        selecionados = [p for p in protocolos if p['grupo'] == 'lateral' or p['nivel'] != 'ESP']
        p, validos, escopo, alvos = preparar_lote(selecionados)
        estados = p[None] <= alphas[:, None, None]
        contagens = contar_regras(estados, validos, escopo, alvos, grade['regras'])
        for i, alpha in enumerate(alphas):
            for j, regra in enumerate(grade['regras']):
                avaliacoes += 1
                det, fp = map(int, contagens[i, j])
                if fp / (~alvos).sum() > config.FP_MAXIMO:
                    continue
                ba = float((det / alvos.sum() + 1 - fp / (~alvos).sum()) / 2)
                score = [ba, det, -fp]
                if melhor is None or tuple(score) > tuple(melhor['score']):
                    melhor = {'janela': tamanho, 'passo': passo, 'alpha': float(alpha),
                              'regra': regra, 'score': score}
    if melhor is None:
        raise RuntimeError('Nenhum candidato bruto sob o teto global de FP')
    protocolos = preparar_protocolos(dados, melhor['janela'], melhor['passo'], 'rayleigh')
    eventos = avaliar_bruto(protocolos, melhor['alpha'], melhor['regra'])
    metricas = resumir(eventos)
    if [metricas['global']['detectados'], -metricas['global']['fp_lateral']] != melhor['score'][1:]:
        raise RuntimeError('Busca bruta divergiu da decisão individual')
    return {**melhor, 'metricas': metricas, 'eventos': eventos, 'avaliacoes': avaliacoes}


def formatar(x):
    return '—' if x is None else f'{x:.2f}'.replace('.', ',')


def linha(metodo, nivel, m):
    det = '—' if nivel == 'ESP' else f"{m['detectados']}/{m['n_alvos']} ({formatar(100*m['taxa_deteccao'])}%)"
    fp = f"{m['fp_lateral']}/{m['n_laterais']} ({formatar(100*m['taxa_fp_lateral'])}%)"
    return f"| {metodo} | {nivel} | {det} | {formatar(m['tempo_medio_detectados_s'])} | {fp} |"


def executar():
    busca = json.loads((config.RESULTS_DIR / 'search_hmm_parameters.json').read_text())
    cfg = busca['configuracao']
    if cfg['detector'] != 'rayleigh' or cfg['protocolo'] != 'intensidades_independentes' or busca['rapida']:
        raise ValueError('Requer busca completa Rayleigh no protocolo independente')
    if cfg['canal'] != config.CHANNEL_INDEX or cfg['p_eps'] != config.P_EPS:
        raise ValueError('Canal/clipping diferentes da busca salva')
    dados = carregar_dados()
    metodos = {}
    for metodo in ('map', 'mle'):
        escolhido = busca['melhores'][metodo]
        protocolos = preparar_protocolos(dados, escolhido['janela'], escolhido['passo'], 'rayleigh')
        b = construir_matrizes_observacao(protocolos, cfg['presente_min_db'], cfg['priori'])
        eventos = avaliar(protocolos, b[metodo], escolhido['A'], escolhido['pi'], escolhido['regra'])
        metricas = resumir(eventos)
        if metricas != escolhido['metricas']:
            raise RuntimeError('Dados/configuração atuais não reproduzem a busca HMM')
        metodos[metodo] = {k: escolhido[k] for k in ('janela', 'passo', 'A', 'pi', 'regra')}
        metodos[metodo].update(metricas=metricas, eventos=eventos)
    metodos['bruto'] = buscar_bruto(dados, busca['grade'])
    resultado = {'detector': 'rayleigh', 'protocolo': cfg['protocolo'], 'avaliacao': 'in-sample',
                 'configuracao_hmm': cfg, 'fp_maximo': config.FP_MAXIMO,
                 'grade_bruto': {'alphas': config.RAW_ALPHA_VALUES,
                                 'janelamentos': busca['grade']['janelamentos'], 'regras': busca['grade']['regras']},
                 'metodos': metodos, 'n_arquivos': sum(len(p) for p in dados.values()), 'n_pacientes': len(dados)}
    salvar_json(resultado, config.RESULTS_DIR / 'comparacao_rayleigh.json')
    linhas = ['# Rayleigh: HMM MAP, HMM MLE e detector bruto', '',
              'Melhor BA sob FP lateral global ≤5%. Cada método foi otimizado separadamente nos mesmos 11 pacientes (in-sample).',
              'HMM: busca completa salva, reproduzida com os EEG atuais. Bruto: busca completa de janela/passo, alpha e regra temporal.',
              'As intensidades são independentes. Tempo médio somente entre detectados, desde o início do arquivo.', '',
              '| Método | Intensidade | Detecções | Tempo médio (s) | FP lateral |',
              '|---|---|---:|---:|---:|']
    for metodo, resultado in metodos.items():
        linhas.append(linha(metodo.upper(), 'Global', resultado['metricas']['global']))
    for nivel in config.NIVEIS:
        for metodo, resultado in metodos.items():
            linhas.append(linha(metodo.upper(), str(nivel), resultado['metricas']['por_intensidade'][str(nivel)]))
    linhas += ['', '## Parâmetros e FP alvo ESP', '']
    for metodo, resultado in metodos.items():
        parametros = {k:v for k,v in resultado.items() if k not in ('metricas','eventos')}
        m = resultado['metricas']['global']
        linhas += [f"- {metodo.upper()}: `{json.dumps(parametros, ensure_ascii=False)}`. BA={formatar(100*m['acuracia_balanceada'])}%; FP alvo ESP={m['fp_alvo_esp']}/{m['n_alvos_esp']} ({formatar(100*m['taxa_fp_alvo_esp'])}%)."]
    linhas += ['', f"Sequências sem janela: bruto={metodos['bruto']['metricas']['global']['n_sem_janelas']}; HMM MAP={metodos['map']['metricas']['global']['n_sem_janelas']}; HMM MLE={metodos['mle']['metricas']['global']['n_sem_janelas']}. Diferenças incluem janelamento e cobertura das gravações.", '', 'O teto de FP é global, não por intensidade. Médias usam conjuntos diferentes de detectados: diferença de médias não é antecipação pareada.', '',
               '## Por paciente e intensidade', '', '| Método | Paciente | Intensidade | Detecções | Tempo médio (s) | FP lateral |', '|---|---|---|---:|---:|---:|']
    campos = ['metodo','paciente','intensidade',*metodos['map']['metricas']['global'].keys()]
    with (config.RESULTS_DIR / 'comparacao_rayleigh_por_paciente_intensidade.csv').open('w', newline='') as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for paciente in sorted(dados):
            for nivel in config.NIVEIS:
                for metodo, resultado in metodos.items():
                    m = resultado['metricas']['por_paciente_intensidade'][paciente][str(nivel)]
                    escritor.writerow({'metodo': metodo, 'paciente': paciente, 'intensidade': nivel, **m})
                    linhas.append(linha(metodo.upper(), str(nivel), m).replace(f'| {metodo.upper()} |', f'| {metodo.upper()} | {paciente} |', 1))
    (config.RESULTS_DIR / 'comparacao_rayleigh.md').write_text('\n'.join(linhas)+'\n')
    for metodo, resultado in metodos.items():
        print(metodo, resultado['metricas']['global'])
    return metodos


if __name__ == '__main__':
    executar()
