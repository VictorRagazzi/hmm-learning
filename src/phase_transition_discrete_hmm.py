"""HMM categorico causal para o protocolo concatenado de intensidades."""

import argparse
import json
import os
import tempfile
from itertools import product

import numpy as np

import continuous_rayleigh_hmm as crh
import detectors
import early_detection_rayleigh_hmm as early
import get_obs_matrix as gom
import phase_transition_hmm as phase

OUTPUT_FILE = "./results/phase_transition_discrete_hmm.json"
LABELS = tuple(gom.NIVEIS_OBSERVACAO)
SMOOTHING = 0.5


def _label(p):
    for i, boundary in enumerate(gom.P_VALUE_BOUNDARIES):
        if p > boundary:
            return LABELS[i]
    return LABELS[-1]


def _categorizar(protocolos, labels_only=False):
    saida=[]
    for protocolo in protocolos:
        novo={k:v for k,v in protocolo.items() if k!="fases"}
        fases=[]
        for fase in protocolo["fases"]:
            item={k:v for k,v in fase.items()}
            item["labels"]=[_label(float(p)) for p in fase["p_values"]]
            fases.append(item)
        novo["fases"]=fases
        saida.append(novo)
    return saida


def _montar_protocolos(esp,stim,nome,janela,passo):
    detector=detectors.obter_detector(nome)
    index=phase._indexar_arquivos(esp,stim)
    protocolos=[]; contagens=np.zeros((2,len(LABELS)),dtype=int)
    for participante in sorted(index):
        arquivos=index[participante]
        fs={}
        for nivel in phase.LEVELS:
            fs[nivel]=float(arquivos[nivel]["fs"]) if "fs" in arquivos[nivel] else float(gom.carregar_mat(arquivos[nivel]["arquivo"])["fs"])
        for grupo in ("estimulo","lateral"):
            for fi in range(len(arquivos[30][grupo])):
                fases=[];offset=0
                for fase_i,nivel in enumerate(phase.LEVELS):
                    arq=arquivos[nivel]
                    if nivel=="ESP":
                        seq="sequencias" if grupo=="estimulo" else "lateral"
                        freq,coef=arq[seq][fi]
                        freq_lat=arq["lateral"][fi][0] if grupo=="estimulo" else None
                    else:
                        freq,coef=arq[grupo][fi]
                        freq_lat=arq["lateral"][fi][0] if grupo=="estimulo" else None
                    stats,intervals=detectors.calcular_estatistica_janelas(coef,detector,janela,passo)
                    pvals=np.asarray([detector.p_value(float(v),janela) for v in stats])
                    labels=[_label(float(p)) for p in pvals]
                    obs=[]
                    for (ini,end),value,pval,label in zip(intervals,stats,pvals,labels,strict=True):
                        obs.append({"participante":participante,"frequencia_hz":float(freq),
                          "frequencia_lateral_hz":float(freq_lat) if freq_lat is not None else None,
                          "condicao":nivel,"intervalo_epocas_local":[int(ini),int(end)],
                          "intervalo_epocas_acumulado":[int(offset+ini),int(offset+end)],
                          "tempo_local_s":int(end),"tempo_acumulado_s":int(offset+end),"fs_hz":float(fs[nivel]),
                          "estatistica":float(value),"p_value":float(pval),"label":label})
                    if grupo=="estimulo":
                        state=0 if fase_i==0 else 1
                        contagens[state]+=np.bincount([LABELS.index(l) for l in labels],minlength=5)
                    fases.append({"nivel":nivel,"frequencia":float(freq),"frequencia_lateral":float(freq_lat) if freq_lat is not None else None,
                      "p_values":pvals,"labels":labels,"finais_locais":np.asarray([b for _,b in intervals],dtype=int),
                      "inicio_global":offset,"n_epocas":len(coef),"fs":float(fs[nivel]),"observacoes":obs})
                    offset+=len(coef)
                protocolos.append({"participante":participante,"grupo":grupo,"frequencia_indice":fi,
                  "fases":fases,"n_epocas_total":offset})
    b=contagens.astype(float)+SMOOTHING;b/=b.sum(axis=1,keepdims=True)
    return protocolos,contagens.tolist(),b.tolist()


def _matriz_b(protocolos):
    counts=np.zeros((2,len(LABELS)),dtype=int)
    for p in protocolos:
        if p["grupo"]!="estimulo": continue
        for i,fase in enumerate(p["fases"]):
            state=0 if i==0 else 1
            for lab in fase["labels"]: counts[state,LABELS.index(lab)]+=1
    b=counts.astype(float)+SMOOTHING
    b/=b.sum(axis=1,keepdims=True)
    return counts.tolist(),b.tolist()


def _concat_labels(p):
    obs=[]; ends=[]; phase_ids=[]; offset=0
    for i,fase in enumerate(p["fases"]):
        obs.extend(fase["labels"])
        ends.extend(offset+np.asarray(fase["finais_locais"],dtype=int))
        phase_ids.extend([i]*len(fase["labels"]))
        offset+=fase["n_epocas"]
    return np.asarray(obs,dtype=object),np.asarray(ends),np.asarray(phase_ids)


def _concat_p(p):
    values=[]; ends=[]; phase_ids=[]; offset=0
    for i,fase in enumerate(p["fases"]):
        values.extend(fase["p_values"])
        ends.extend(offset+np.asarray(fase["finais_locais"],dtype=int))
        phase_ids.extend([i]*len(fase["p_values"]))
        offset+=fase["n_epocas"]
    return np.asarray(values),np.asarray(ends),np.asarray(phase_ids)


def _online_states(labels,a,pi,b):
    if not len(labels): return np.asarray([],dtype=bool)
    ids=np.asarray([LABELS.index(l) for l in labels],dtype=int)
    loga=np.log(np.asarray(a,dtype=float)+1e-300)
    logb=np.log(np.asarray(b,dtype=float)+1e-300)
    delta=np.log(np.asarray(pi,dtype=float)+1e-300)+logb[:,ids[0]]
    out=[int(np.argmax(delta))]
    for obs in ids[1:]:
        delta=np.max(delta[:,None]+loga,axis=0)+logb[:,obs]
        out.append(int(np.argmax(delta)))
    return np.asarray(out,dtype=bool)


def _online_states_vectorized(labels,a,pi,b):
    ids=np.asarray([LABELS.index(l) for l in labels],dtype=int)
    loga=np.log(np.asarray(a,dtype=float)+1e-300)
    logb=np.log(np.asarray(b,dtype=float)+1e-300)
    emissions=logb[:,ids].T
    delta=np.log(np.asarray(pi,dtype=float)+1e-300)+emissions[0]
    states=np.empty(len(ids),dtype=bool);states[0]=int(np.argmax(delta))==1
    for t in range(1,len(ids)):
        delta=np.max(delta[:,None]+loga,axis=0)+emissions[t]
        states[t]=int(np.argmax(delta))==1
    return states


def _detalhes(protocolos,modo,candidato,b):
    saida=[]
    for p in protocolos:
        labels,ends,phases=_concat_labels(p)
        if modo=="hmm": positives=_online_states_vectorized(labels,candidato[1],candidato[2],b)
        else:
            vals,_,_=_concat_p(p); positives=vals<=candidato[1]
        n=candidato[3] if modo=="hmm" else candidato[2]
        fp=phase._primeiro_disparo(positives[phases==0],ends[phases==0],phases[phases==0],n) is not None
        trigger=phase._primeiro_disparo(positives,ends,phases,n,fase_min=1)
        found=trigger is not None; start=p["fases"][1]["inicio_global"]; total=p["n_epocas_total"]
        level=phase.LEVELS[trigger[1]] if found else None
        cond_start=p["fases"][trigger[1]]["inicio_global"] if found else None
        observations=[o for f in p["fases"] for o in f.get("observacoes",[])]
        saida.append({"participante":p["participante"],"grupo":p["grupo"],
          "frequencia_indice":p["frequencia_indice"],"frequencia":p["fases"][0]["frequencia"],
          "frequencia_lateral":p["fases"][0].get("frequencia_lateral"),"fp_esp":bool(fp),"detectou":bool(found),
          "primeiro_nivel_db":level,"tempo_total_epocas":int(trigger[0] if found else total),
          "tempo_desde_inicio_estimulo_epocas":int(trigger[0]-start if found else total-start),
          "tempo_na_condicao_epocas":int(trigger[0]-cond_start) if found else None,
          "tempo_total_protocolo_epocas":int(total),"observacoes":observations})
    return saida


def _chave_resumo(d):
    alvos=[x for x in d if x["grupo"]=="estimulo"]
    lateral=[x for x in d if x["grupo"]=="lateral"]
    fpesp=sum(x["fp_esp"] for x in alvos); fplat=sum(x["detectou"] for x in lateral)
    fp=(fpesp+fplat)/(len(alvos)+len(lateral)); det=sum(x["detectou"] for x in alvos)/len(alvos)
    tempo=float(np.mean([x["tempo_total_epocas"] for x in alvos]))
    resumo={"n_participante_frequencia":len(alvos),"detectados_ate_70db":sum(x["detectou"] for x in alvos),
      "taxa_deteccao":det,"fp_esp":fpesp,"taxa_fp_esp":fpesp/len(alvos),"fp_lateral":fplat,
      "taxa_fp_lateral":fplat/len(lateral),"taxa_fp_combinada":fp,
      "acuracia_balanceada_com_fp_combinado":(det+1-fp)/2,
      "tempo_total_medio_epocas_inclui_nao_detectados":tempo,
      "tempo_desde_estimulo_medio_epocas_inclui_nao_detectados":float(np.mean([x["tempo_desde_inicio_estimulo_epocas"] for x in alvos]))}
    return (resumo["acuracia_balanceada_com_fp_combinado"],det,-fp,-tempo),resumo


def _resumo_detalhado(det):
    targets=[x for x in det if x["grupo"]=="estimulo"]
    lateral=[x for x in det if x["grupo"]=="lateral"]
    detected=[x for x in targets if x["detectou"]]
    esp=sum(x["fp_esp"] for x in targets);lat=sum(x["detectou"] for x in lateral)
    cond=[x["tempo_na_condicao_epocas"] for x in detected if x["tempo_na_condicao_epocas"] is not None]
    since=[x["tempo_desde_inicio_estimulo_epocas"] for x in detected]
    total=[x["tempo_total_epocas"] for x in detected]
    summary={"n_alvos":len(targets),"n_laterais":len(lateral),"detectados":len(detected),
      "taxa_deteccao":len(detected)/len(targets),"fp_esp":esp,"taxa_fp_esp":esp/len(targets),
      "fp_lateral":lat,"taxa_fp_lateral":lat/len(lateral),"fp_combinado":esp+lat,
      "oportunidades_negativas":len(targets)+len(lateral),"taxa_fp_combinada":(esp+lat)/(len(targets)+len(lateral)),
      "tempo_total_medio_inclui_nao_detectados":float(np.mean([x["tempo_total_epocas"] for x in targets])),
      "tempo_desde_estimulo_medio_inclui_nao_detectados":float(np.mean([x["tempo_desde_inicio_estimulo_epocas"] for x in targets])),
      "tempo_total_medio_entre_detectados":float(np.mean(total)) if total else None,
      "tempo_total_mediano_entre_detectados":float(np.median(total)) if total else None,
      "tempo_desde_estimulo_medio_entre_detectados":float(np.mean(since)) if since else None,
      "tempo_desde_estimulo_mediano_entre_detectados":float(np.median(since)) if since else None,
      "tempo_na_condicao_medio_entre_detectados":float(np.mean(cond)) if cond else None,
      "tempo_na_condicao_mediano_entre_detectados":float(np.median(cond)) if cond else None,
      "por_participante":{},"por_frequencia":{}}
    for field,dest in (("participante","por_participante"),("frequencia_indice","por_frequencia")):
        for val in sorted({x[field] for x in targets}):
            group=[x for x in targets if x[field]==val]
            controls=[x for x in lateral if x[field]==val]
            false=sum(x["fp_esp"] for x in group)+sum(x["detectou"] for x in controls)
            key=str(group[0]["frequencia"] if field=="frequencia_indice" else val)
            dest[key]={"n":len(group),"detectados":sum(x["detectou"] for x in group),
              "taxa_deteccao":sum(x["detectou"] for x in group)/len(group),
              "fp_esp":sum(x["fp_esp"] for x in group),"fp_lateral":sum(x["detectou"] for x in controls),
              "taxa_fp_combinada":false/(len(group)+len(controls))}
    return summary


def _selecionar(protocolos,b):
    preparados=[_concat_labels(p) for p in protocolos]
    valores=[_concat_p(p) for p in protocolos]
    hmm=[]; raw=[]
    for pa,pp,pia in product(phase.P_SELF_VALUES,phase.P_SELF_VALUES,phase.PI_ABSENT_VALUES):
        a=[[pa,1-pa],[1-pp,pp]]
        states=[_online_states_vectorized(x[0],a,[pia,1-pia],b) for x in preparados]
        evaluadores=[(p,ends,phases,pos) for p,(_,ends,phases),pos in zip(protocolos,preparados,states,strict=True)]
        for n in phase.MIN_CONSECUTIVE_VALUES:
            detections=[_metricas_uma(*item,n) for item in evaluadores]
            key,summary=_agregar_metricas(protocolos,detections)
            if summary["taxa_fp_combinada"]<=phase.MAX_FALSE_POSITIVE:
                hmm.append((key,(None,a,[pia,1-pia],n),summary))
    for alpha,n in product(early.RAYLEIGH_ALPHA_VALUES,phase.MIN_CONSECUTIVE_VALUES):
        detections=[]
        for p,(vals,ends,phases) in zip(protocolos,valores,strict=True):
            detections.append(_metricas_uma(p,ends,phases,vals<=alpha,n))
        key,summary=_agregar_metricas(protocolos,detections)
        if summary["taxa_fp_combinada"]<=phase.MAX_FALSE_POSITIVE:
            raw.append((key,(None,alpha,n),summary))
    return max(hmm,key=lambda x:x[0]),max(raw,key=lambda x:x[0])


def _primeiro_run(binaria, fases, fase_min, n):
    x=np.asarray(binaria,dtype=np.int8)
    if len(x)<n: return None
    sums=np.convolve(x,np.ones(n,dtype=np.int8),mode="valid")
    candidates=np.flatnonzero(sums==n)+n-1
    candidates=candidates[np.asarray(fases)[candidates]>=fase_min]
    return int(candidates[0]) if len(candidates) else None


def _metricas_uma(p,ends,fases,positives,n):
    esp_idx=_primeiro_run(positives[fases==0],fases[fases==0],0,n)
    trigger=_primeiro_run(positives,fases,1,n)
    found=trigger is not None; start=p["fases"][1]["inicio_global"]; total=p["n_epocas_total"]
    t=int(ends[trigger]) if found else total
    return (esp_idx is not None,found,t,max(0,t-start),p["grupo"])


def _agregar_metricas(protocolos,metrics):
    targets=[m for m in metrics if m[4]=="estimulo"]; lateral=[m for m in metrics if m[4]=="lateral"]
    fpesp=sum(m[0] for m in targets); fplat=sum(m[1] for m in lateral); detections=sum(m[1] for m in targets)
    fp=(fpesp+fplat)/(len(targets)+len(lateral)); det=detections/len(targets); total=float(np.mean([m[2] for m in targets]))
    detected=[m for m in targets if m[1]]
    summary={"n_participante_frequencia":len(targets),"detectados_ate_70db":detections,"taxa_deteccao":det,
      "fp_esp":fpesp,"taxa_fp_esp":fpesp/len(targets),"fp_lateral":fplat,"taxa_fp_lateral":fplat/len(lateral),
      "taxa_fp_combinada":fp,"acuracia_balanceada_com_fp_combinado":(det+1-fp)/2,
      "tempo_total_medio_epocas_inclui_nao_detectados":total,
      "tempo_desde_estimulo_medio_epocas_inclui_nao_detectados":float(np.mean([m[3] for m in targets])),
      "tempo_total_medio_entre_detectados":float(np.mean([m[2] for m in detected])) if detected else None,
      "tempo_total_mediano_entre_detectados":float(np.median([m[2] for m in detected])) if detected else None,
      "tempo_desde_estimulo_medio_entre_detectados":float(np.mean([m[3] for m in detected])) if detected else None,
      "tempo_desde_estimulo_mediano_entre_detectados":float(np.median([m[3] for m in detected])) if detected else None}
    return (summary["acuracia_balanceada_com_fp_combinado"],det,-fp,-total),summary


def executar_detector(pasta_dados,nome,window_sizes=None,step_modes=None,compact=False,resume=None):
    esp,stim=crh._carregar_coeficientes(pasta_dados,nome)
    phase._adicionar_laterais_esp(esp,nome)
    configs={}
    if resume and os.path.exists(resume):
        with open(resume,encoding="utf-8") as f:
            old=json.load(f).get("por_detector",{}).get(nome,{})
        configs.update(old.get("por_configuracao",{}))
    for w,mode in product(window_sizes or early.WINDOW_SIZES,step_modes or early.STEP_MODES):
        step=early._step(w,mode); key=f"window_{w}_step_{step}"
        print(f"{nome}: preparando {key}",flush=True)
        if key in configs: continue
        protocols,counts,b=_montar_protocolos(esp,stim,nome,w,step)
        print(f"{nome}: selecionando {key}",flush=True)
        h,r=_selecionar(protocols,b)
        print(f"{nome}: concluido {key}",flush=True)
        configs[key]={"window_size_epochs":w,"window_step_epochs":step,"B_contagens":counts,"B":b,
          "hmm":{"matriz_a":h[1][1],"pi_inicial":h[1][2],"min_consecutive":h[1][3],**h[2]},
          "detector_bruto_janelado":{"alpha":r[1][1],"min_consecutive":r[1][2],**r[2]}}
    best_h=max(((k,v["hmm"]) for k,v in configs.items()),key=lambda x:phase._chave(x[1]))
    best_r=max(((k,v["detector_bruto_janelado"]) for k,v in configs.items()),key=lambda x:phase._chave(x[1]))
    if compact:
        return {"melhor_hmm_configuracao":best_h[0],"melhor_hmm":best_h[1],
          "melhor_bruto_configuracao":best_r[0],"melhor_bruto_janelado":best_r[1],
          "por_configuracao":configs}
    for key,mode,field in (() if compact else ((best_h[0],"hmm","hmm"),(best_r[0],"bruto","detector_bruto_janelado"))):
        w=configs[key]["window_size_epochs"]; step=configs[key]["window_step_epochs"]
        ps,_,_=_montar_protocolos(esp,stim,nome,w,step)
        c=configs[key][field]
        cand=(None,c["matriz_a"],c["pi_inicial"],c["min_consecutive"]) if mode=="hmm" else (None,c["alpha"],c["min_consecutive"])
        c["detalhes"]=_detalhes(ps,mode,cand,configs[key]["B"])
        c["metricas_recalculadas"]=_chave_resumo(c["detalhes"])[1]
    hdetails=best_h[1]["detalhes"];rdetails=best_r[1]["detalhes"]
    hmap={(x["participante"],x["frequencia_indice"],x["grupo"]):x for x in hdetails}
    rmap={(x["participante"],x["frequencia_indice"],x["grupo"]):x for x in rdetails}
    paired=[]
    for key in sorted(set(hmap)&set(rmap)):
        a=hmap[key];r=rmap[key]
        paired.append({"participante":key[0],"frequencia_indice":key[1],"grupo":key[2],
          "hmm_detectou":a["detectou"],"bruto_detectou":r["detectou"],
          "hmm_tempo_desde_30":a["tempo_desde_inicio_estimulo_epocas"],
          "bruto_tempo_desde_30":r["tempo_desde_inicio_estimulo_epocas"]})
    observation_map={f"{x['participante']}|{x['frequencia_indice']}|{x['grupo']}":x.pop("observacoes") for x in hdetails}
    for x in rdetails: x.pop("observacoes",None)
    best_h[1]["detalhes"]=hdetails;best_r[1]["detalhes"]=rdetails
    best_h[1]["comparacao_pareada_com_bruto"]=paired
    best_r[1]["comparacao_pareada_com_hmm"]=paired
    return {"melhor_hmm_configuracao":best_h[0],"melhor_hmm":best_h[1],"melhor_bruto_configuracao":best_r[0],"melhor_bruto_janelado":best_r[1],"observacoes_por_trajetoria":observation_map,"por_configuracao":configs}


def executar(pasta_dados=crh.DATA_DIR,nomes=phase.DETECTOR_NAMES,window_sizes=None,step_modes=None,compact=False,resume=None):
    return {"nome":"hmm_transicao_fases_discreto_in_sample","status":"complete",
      "metodo":"Teste in-sample: B, A, pi, janela, passo e regra ajustados e avaliados nos mesmos participantes. Emissão Presente usa gravações estimuladas reais.",
      "configuracao":{"detectores":list(nomes),"labels":list(LABELS),"p_value_boundaries":gom.P_VALUE_BOUNDARIES,"pseudocontagem":SMOOTHING,"baseline":"detector bruto janelado causal","fp_maximo_combinado":phase.MAX_FALSE_POSITIVE},
      "por_detector":{n:executar_detector(pasta_dados,n,window_sizes,step_modes,compact,resume) for n in nomes}}


def salvar(obj,caminho):
    os.makedirs(os.path.dirname(caminho) or ".",exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=os.path.dirname(caminho) or ".",delete=False,suffix=".tmp") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); tmp=f.name
    os.replace(tmp,caminho)


def consolidar_parciais(caminhos,pasta_dados=crh.DATA_DIR,detalhes=True):
    partes=[json.load(open(p,encoding="utf-8")) for p in caminhos]
    configs={}; nomes=[]
    for parte in partes:
        for nome,item in parte["por_detector"].items():
            nomes.append(nome); configs[nome]=item["por_configuracao"]
    resultado={"nome":"hmm_transicao_fases_discreto_in_sample","status":"complete",
      "metodo":"B categórica ajustada e avaliada nos mesmos participantes; resultado exploratório in-sample. Baseline bruta janelada causal.",
      "configuracao":{"detectores":nomes,"labels":list(LABELS),"p_value_boundaries":gom.P_VALUE_BOUNDARIES,
        "pseudocontagem":SMOOTHING,"emissao_presente":"gravações estimuladas reais","baseline":"janelada causal","fp_maximo_combinado":phase.MAX_FALSE_POSITIVE},"por_detector":{}}
    for nome in nomes:
        conf=configs[nome]
        kh=max(((k,v["hmm"]) for k,v in conf.items()),key=lambda x:phase._chave(x[1]))
        kr=max(((k,v["detector_bruto_janelado"]) for k,v in conf.items()),key=lambda x:phase._chave(x[1]))
        if not detalhes:
            resultado["por_detector"][nome]={"melhor_hmm_configuracao":kh[0],"melhor_hmm":kh[1],
              "melhor_bruto_configuracao":kr[0],"melhor_bruto_janelado":kr[1],"por_configuracao":conf}
            continue
        esp,stim=crh._carregar_coeficientes(pasta_dados,nome);phase._adicionar_laterais_esp(esp,nome)
        out={}
        for key,mode,field in ((kh[0],"hmm","hmm"),(kr[0],"bruto","detector_bruto_janelado")):
            w=conf[key]["window_size_epochs"];step=conf[key]["window_step_epochs"]
            ps, _=phase._montar_protocolos(esp,stim,nome,w,step);ps=_categorizar(ps)
            c=conf[key][field]
            cand=(None,c["matriz_a"],c["pi_inicial"],c["min_consecutive"]) if mode=="hmm" else (None,c["alpha"],c["min_consecutive"])
            c["detalhes"]=_detalhes(ps,mode,cand,conf[key]["B"])
            c["metricas_recalculadas"]=_chave_resumo(c["detalhes"])[1]
            out[field]=c
        hm=out["hmm"];raw=out["detector_bruto_janelado"]
        rmap={(x["participante"],x["frequencia_indice"],x["grupo"]):x for x in raw["detalhes"]}
        paired=[];observations={}
        for x in hm["detalhes"]:
            k=(x["participante"],x["frequencia_indice"],x["grupo"]);r=rmap[k]
            observations[f"{k[0]}|{k[1]}|{k[2]}"]=x.pop("observacoes")
            paired.append({"participante":k[0],"frequencia_indice":k[1],"grupo":k[2],"hmm_detectou":x["detectou"],"bruto_detectou":r["detectou"],"hmm_tempo_desde_30":x["tempo_desde_inicio_estimulo_epocas"],"bruto_tempo_desde_30":r["tempo_desde_inicio_estimulo_epocas"]})
        for x in raw["detalhes"]:x.pop("observacoes",None)
        hm["comparacao_pareada_com_bruto"]=paired;raw["comparacao_pareada_com_hmm"]=paired
        resultado["por_detector"][nome]={"melhor_hmm_configuracao":kh[0],"melhor_hmm":hm,
          "melhor_bruto_configuracao":kr[0],"melhor_bruto_janelado":raw,
          "observacoes_por_trajetoria":observations,"por_configuracao":conf}
    return resultado


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",default=crh.DATA_DIR);p.add_argument("--output",default=OUTPUT_FILE);p.add_argument("--detector",choices=phase.DETECTOR_NAMES);p.add_argument("--window",type=int,action="append");p.add_argument("--step-mode",choices=early.STEP_MODES,action="append");p.add_argument("--compact",action="store_true");p.add_argument("--resume");p.add_argument("--merge-inputs",nargs="+");p.add_argument("--no-details",action="store_true")
    a=p.parse_args()
    if a.merge_inputs: result=consolidar_parciais(a.merge_inputs,a.data_dir,not a.no_details)
    else:
        ns=(a.detector,) if a.detector else phase.DETECTOR_NAMES
        result=executar(a.data_dir,ns,a.window,a.step_mode,a.compact,a.resume)
    salvar(result,a.output)
