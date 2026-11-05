"""Validacao leave-one-subject-out completa para o HMM discreto."""

import argparse
import json
import os
import tempfile
import pickle

import continuous_rayleigh_hmm as crh
import early_detection_rayleigh_hmm as early
import phase_transition_hmm as phase
import phase_transition_discrete_hmm as discrete

OUTPUT_FILE="./results/phase_transition_discrete_hmm_loso.json"


def executar_detector(pasta,nome,window_sizes=None,step_modes=None):
    esp,stim=crh._carregar_coeficientes(pasta,nome)
    phase._adicionar_laterais_esp(esp,nome)
    participants=sorted(phase._indexar_arquivos(esp,stim))
    cache_path=f"results/.discrete_loso_cache_{nome}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path,"rb") as f: cache=pickle.load(f)
    else:
        cache={}
    for w in window_sizes or early.WINDOW_SIZES:
        for mode in step_modes or early.STEP_MODES:
            step=early._step(w,mode);key=f"window_{w}_step_{step}"
            if key in cache: continue
            protocols,counts,b=discrete._montar_protocolos(esp,stim,nome,w,step)
            cache[key]={"protocols":protocols,"counts":counts,"B":b,"window":w,"step":step}
            with open(cache_path,"wb") as f: pickle.dump(cache,f,protocol=pickle.HIGHEST_PROTOCOL)
    folds={};all_h=[];all_raw=[]
    progress_path=f"results/.discrete_loso_{nome}_progress.json"
    if os.path.exists(progress_path):
        with open(progress_path,encoding="utf-8") as f: folds=json.load(f)
        for fold in folds.values():
            all_h.extend(fold["hmm"]["detalhes_teste"]);all_raw.extend(fold["detector_bruto_janelado"]["detalhes_teste"])
    for ix,test in enumerate(participants,1):
        if test in folds: continue
        print(f"{nome}: dobra {ix}/11 ({test})",flush=True)
        best_h=best_raw=None
        candidate_path=f"results/.discrete_loso_{nome}_{test}_candidates.json"
        candidates=json.load(open(candidate_path,encoding="utf-8")) if os.path.exists(candidate_path) else {"hmm":[],"raw":[]}
        seen=set(candidates["configs_done"] if "configs_done" in candidates else [])
        for key,item in cache.items():
            if key in seen: continue
            train=[p for p in item["protocols"] if p["participante"]!=test]
            counts,b=discrete._matriz_b(train)
            hmm,raw=discrete._selecionar(train,b)
            candidates["hmm"].append([hmm[0],key,hmm[1],hmm[2],counts,b])
            candidates["raw"].append([raw[0],key,raw[1],raw[2]])
            candidates.setdefault("configs_done",[]).append(key)
            with open(candidate_path,"w",encoding="utf-8") as f:json.dump(candidates,f,ensure_ascii=False)
            print(f"{nome} {test}: selecionado {key}",flush=True)
        best_h=max(candidates["hmm"],key=lambda x:x[0]);best_raw=max(candidates["raw"],key=lambda x:x[0])
        _,kh,ch,train_h,counts_h,bh=best_h
        _,kr,cr,train_raw=best_raw
        test_h=[p for p in cache[kh]["protocols"] if p["participante"]==test]
        test_raw=[p for p in cache[kr]["protocols"] if p["participante"]==test]
        dh=discrete._detalhes(test_h,"hmm",ch,bh)
        dr=discrete._detalhes(test_raw,"bruto",cr,bh)
        all_h.extend(dh);all_raw.extend(dr)
        treino_h=[p for p in cache[kh]["protocols"] if p["participante"]!=test]
        treino_r=[p for p in cache[kr]["protocols"] if p["participante"]!=test]
        folds[test]={"participantes_treino":[p for p in participants if p!=test],
          "hmm":{"configuracao":kh,"window_size_epochs":cache[kh]["window"],"window_step_epochs":cache[kh]["step"],
            "B":bh,"B_contagens":counts_h,"matriz_a":ch[1],"pi_inicial":ch[2],"min_consecutive":ch[3],
            "metricas_treino":train_h,"metricas_teste":phase._resumir(dh),"detalhes_teste":dh},
          "detector_bruto_janelado":{"configuracao":kr,"window_size_epochs":cache[kr]["window"],"window_step_epochs":cache[kr]["step"],
            "alpha":cr[1],"min_consecutive":cr[2],"metricas_treino":train_raw,
                 "metricas_teste":phase._resumir(dr),"detalhes_teste":dr}}
        with open(progress_path,"w",encoding="utf-8") as f: json.dump(folds,f,ensure_ascii=False)
    return {"participantes":participants,"hmm":{**phase._resumir(all_h),"detalhes":all_h},
      "detector_bruto_janelado":{**phase._resumir(all_raw),"detalhes":all_raw},"dobras":folds,
      "resumo_hmm":discrete._resumo_detalhado(all_h),
      "resumo_detector_bruto_janelado":discrete._resumo_detalhado(all_raw),
      "comparacao_pareada":_comparacao_pareada(all_h,all_raw)}


def _comparacao_pareada(hmm,raw):
    rmap={(x["participante"],x["frequencia_indice"],x["grupo"]):x for x in raw}
    paired=[]
    for h in hmm:
        k=(h["participante"],h["frequencia_indice"],h["grupo"]);r=rmap[k]
        paired.append({"participante":k[0],"frequencia_indice":k[1],"grupo":k[2],
          "hmm_detectou":h["detectou"],"bruto_detectou":r["detectou"],
          "hmm_tempo_desde_30":h["tempo_desde_inicio_estimulo_epocas"],
          "bruto_tempo_desde_30":r["tempo_desde_inicio_estimulo_epocas"],
          "hmm_primeiro_nivel":h["primeiro_nivel_db"],"bruto_primeiro_nivel":r["primeiro_nivel_db"]})
    targets=[x for x in paired if x["grupo"]=="estimulo"]
    both=[x for x in targets if x["hmm_detectou"] and x["bruto_detectou"]]
    return {"n_pareado":len(targets),"hmm_exclusivo":sum(x["hmm_detectou"] and not x["bruto_detectou"] for x in targets),
      "bruto_exclusivo":sum(x["bruto_detectou"] and not x["hmm_detectou"] for x in targets),
      "ambos_detectaram":len(both),"hmm_anterior":sum(x["hmm_tempo_desde_30"]<x["bruto_tempo_desde_30"] for x in both),
      "empate":sum(x["hmm_tempo_desde_30"]==x["bruto_tempo_desde_30"] for x in both),
      "bruto_anterior":sum(x["bruto_tempo_desde_30"]<x["hmm_tempo_desde_30"] for x in both),
      "detalhes":paired}


def executar(pasta=crh.DATA_DIR,nomes=phase.DETECTOR_NAMES,window_sizes=None,step_modes=None):
    return {"nome":"hmm_transicao_fases_discreto_loso","status":"complete",
      "metodo":"11 dobras. B, janela, passo, A, pi, alpha e consecutividade escolhidos apenas nos dez participantes de treino. FP externo reportado sem reajuste.",
      "configuracao":{"detectores":list(nomes),"labels":list(discrete.LABELS),"p_value_boundaries":discrete.gom.P_VALUE_BOUNDARIES,
        "pseudocontagem":discrete.SMOOTHING,"emissao_presente":"gravacoes estimuladas reais; treino da dobra",
            "baseline":"detector bruto janelado causal","fp_maximo_treino":phase.MAX_FALSE_POSITIVE},
      "por_detector":{n:executar_detector(pasta,n,window_sizes,step_modes) for n in nomes}}


def salvar(obj,caminho):
    os.makedirs(os.path.dirname(caminho) or ".",exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=os.path.dirname(caminho) or ".",delete=False,suffix=".tmp") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);tmp=f.name
    os.replace(tmp,caminho)


def consolidar(caminhos):
    itens=[json.load(open(p,encoding="utf-8")) for p in caminhos]
    base=itens[0]; merged={}
    for item in itens: merged.update(item["por_detector"])
    base["configuracao"]["detectores"]=list(merged);base["por_detector"]=merged
    for nome,result in merged.items():
        folds=result["dobras"]
        test_subjects=list(folds)
        if len(test_subjects)!=11 or len(set(test_subjects))!=11:
            raise ValueError(f"{nome}: participante externo duplicado/ausente")
        for subject,fold in folds.items():
            if subject in fold["participantes_treino"] or len(fold["participantes_treino"])!=10:
                raise ValueError(f"{nome}/{subject}: vazamento LOSO na dobra")
    return base


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",default=crh.DATA_DIR);p.add_argument("--output",default=OUTPUT_FILE);p.add_argument("--detector",choices=phase.DETECTOR_NAMES);p.add_argument("--window",type=int,action="append");p.add_argument("--step-mode",choices=early.STEP_MODES,action="append");p.add_argument("--merge-inputs",nargs="+")
    a=p.parse_args();
    if a.merge_inputs:obj=consolidar(a.merge_inputs)
    else:
        names=(a.detector,) if a.detector else phase.DETECTOR_NAMES;obj=executar(a.data_dir,names,a.window,a.step_mode)
    salvar(obj,a.output)
