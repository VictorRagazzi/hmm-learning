"""Figuras e galeria didática do experimento Beta bayesiano fixo."""
import html
import json
from pathlib import Path
import numpy as np
from bayesian_beta_hmm import ASSETS, FIGURES, STATES, predictive_logpdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import beta, gamma
import arviz as az

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 160})
COLORS = {"Ausente": "#2874a6", "Presente": "#d35400"}


def generate(result):
    prior_sets = result["prioris"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    gallery = []
    pdf = PdfPages(FIGURES / "apresentacao_figuras.pdf")
    def save(fig, name, caption):
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        gallery.append((name, caption))
        plt.close(fig)
    data = {s: np.asarray([v for p in result["protocolos"] if p["grupo"] == "estimulo"
                           for f in p["fases"] if (f["nivel"] == "ESP") == (s == "Ausente")
                           for v in f["p_values"]]) for s in STATES}
    models = result["configuracao_historica"]["modelos_emissao_beta"]
    x = np.linspace(.0005, .9995, 800)
    xlog = np.geomspace(1e-9, .999, 800)
    fig,ax=plt.subplots(figsize=(14,5))
    ax.set_xlim(0,14);ax.set_ylim(0,5);ax.axis("off")
    boxes=[(1,3.5,"μ ~ Beta(u,v)\nMédia dos p-valores"),(5,3.5,"κ ~ Gamma(shape,rate)\nConcentração"),
           (3,2,"a=μκ; b=(1−μ)κ\nParâmetros da Beta dos dados"),(3,.5,"p | μ,κ ~ Beta(a,b)\nLikelihood das janelas"),
           (10,3.5,"Posterior conjunta de μ,κ\nMCMC: quatro cadeias NUTS"),(10,2,"Média das PDFs posteriores\nlogsumexp(logpdf) − log(S)"),
           (10,.5,"Viterbi causal com A e pi fixas\nUma trajetória por pessoa × frequência")]
    for xx,yy,label in boxes:
        ax.text(xx,yy,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",facecolor="#eaf2f8",edgecolor="#2874a6"),fontsize=10)
    for start,end in (((1,3.1),(2.4,2.45)),((5,3.1),(3.6,2.45)),((3,1.55),(3,.95)),((5,.5),(8.5,3.5)),((10,3.05),(10,2.45)),((10,1.55),(10,.95))):
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#2874a6"))
    save(fig,"modelo_hierarquico","Duas Betas distintas: priori de μ e distribuição dos dados. Posterior completa dos parâmetros, seguida de emissão marginal preditiva por janela. O HMM mantém A e pi fixas e não integra conjuntamente a trajetória.")
    for name, priors in prior_sets.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for row, s in enumerate(STATES):
            u, v, shape, rate = priors[s]
            axes[row, 0].plot(x, beta.pdf(x, u, v), color=COLORS[s])
            axes[row, 0].set(title=f"{s}: μ ~ Beta({u}, {v})", xlabel="μ: média de p", ylabel="Densidade")
            k = np.linspace(.001, gamma.ppf(.999, shape, scale=1/rate), 600)
            axes[row, 1].plot(k, gamma.pdf(k, shape, scale=1/rate), color=COLORS[s])
            axes[row, 1].set(title=f"κ ~ Gamma(shape={shape}, rate={rate})", xlabel="κ: concentração")
            prior = np.load(ASSETS / f"prior_{name}_{s}.npz")
            axes[row, 2].plot(x, np.exp(predictive_logpdf(x, prior)), color=COLORS[s], label="Mistura a priori")
            for i in range(20):
                axes[row, 2].plot(x, beta.pdf(x, prior["a"][i], prior["b"][i]), color=COLORS[s], alpha=.12)
            axes[row, 2].set(title="Distribuições de p antes dos dados", xlabel="p", ylabel="Densidade", ylim=(0, 8))
            axes[row, 2].legend()
        save(fig, f"prior_{name}", f"Priori {name}. μ e κ são independentes a priori; as curvas claras são Betas condicionais sorteadas. O teto vertical 8 facilita a leitura e corta singularidades de borda; não representa truncamento do modelo.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, s in enumerate(STATES):
        p = data[s]
        axes[i, 0].hist(p, bins=30, density=True, color=COLORS[s], alpha=.4, label=f"Dados: n={len(p)} janelas")
        axes[i, 0].plot(x, beta.pdf(x, models[s]["a"], models[s]["b"]), color="black", label="MLE histórico")
        axes[i, 0].set(title=s, xlabel="p", ylabel="Densidade", ylim=(0, 8)); axes[i, 0].legend()
        sorted_p = np.sort(p)
        axes[i, 1].plot(sorted_p, np.arange(1, len(p)+1)/len(p), color=COLORS[s], label="CDF empírica")
        axes[i, 1].plot(x, beta.cdf(x, models[s]["a"], models[s]["b"]), color="black", label="CDF MLE")
        axes[i, 1].set(xlabel="p", ylabel="Probabilidade acumulada"); axes[i, 1].legend()
    save(fig, "dados_mle", "Distribuições dos dados e ajuste MLE salvo. Histograma é somente visualização: não entra no HMM. Presente é um proxy das condições estimuladas reais; 1744 janelas não equivalem a 1744 observações independentes.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    levels = ["ESP",30,40,50,60,70]
    for ax, group in zip(axes,("estimulo","lateral")):
        distributions = [[v for p in result["protocolos"] if p["grupo"]==group
                          for f in p["fases"] if f["nivel"]==level for v in f["p_values"]] for level in levels]
        ax.boxplot([np.maximum(p,1e-9) for p in distributions], tick_labels=[str(l) for l in levels],showfliers=True)
        ax.set(yscale="log",xlabel="Fase",ylabel="p (log, estabilizado)",title=group)
        for i,p in enumerate(distributions): ax.text(i+1,1.1,f"n={len(p)}",ha="center",fontsize=8)
    save(fig,"dados_por_fase", "Dados por fase: alvos e controles separados. Laterais nunca ajustam emissões. n conta janelas agregadas de oito frequências e onze participantes, com sobreposição.")
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    participants=list(result["baseline_mle"]["metricas"]["por_paciente"])
    for ax,s in zip(axes,STATES):
        distributions=[[v for p in result["protocolos"] if p["grupo"]=="estimulo" and p["participante"]==person
                        for f in p["fases"] if (f["nivel"]=="ESP")==(s=="Ausente") for v in f["p_values"]] for person in participants]
        positions=[i+1 for i,p in enumerate(distributions) if p]
        ax.boxplot([p for p in distributions if p],positions=positions,showfliers=False)
        ax.set(xticks=np.arange(1,len(participants)+1),xticklabels=participants,ylabel="p",title=f"Calibração {s} por participante")
    save(fig,"dados_por_participante","Heterogeneidade dos p-valores de calibração. Participantes ESP com menos de 120 épocas não contribuem ao ajuste Ausente, mas continuam no denominador da avaliação. Caixas omitem outliers visuais.")
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    rng=np.random.default_rng(result["seed"]+20000)
    for row,s in enumerate(STATES):
        samples=np.load(ASSETS / f"predictive_principal_{s}.npz")
        idx=rng.choice(len(samples["a"]),size=1000,replace=False)
        # Um parâmetro por réplica inteira: preserva incerteza compartilhada
        # dentro da réplica, ao contrário das marginais usadas no HMM.
        replicas=rng.beta(samples["a"][idx,None],samples["b"][idx,None],size=(1000,len(data[s])))
        for col,(values,observed,label) in enumerate(((replicas.mean(axis=1),data[s].mean(),"Média p"),
                                                       ((replicas<.05).mean(axis=1),(data[s]<.05).mean(),"Fração p < 0,05"))):
            axes[row,col].hist(values,bins=40,density=True,color=COLORS[s],alpha=.5)
            axes[row,col].axvline(observed,color="black",ls="--",label="Observado")
            axes[row,col].set(title=f"PPC {s}",xlabel=label,ylabel="Densidade"); axes[row,col].legend()
    save(fig,"checagem_preditiva_posterior","Checagem preditiva posterior principal: 1000 réplicas iid com o mesmo número de janelas de calibração. Em cada réplica um único μ,κ é compartilhado. Verifica adequação descritiva sob o modelo, não corrige a dependência real nem constitui validação externa.")
    # Contexto visual com EEG autêntico no mesmo canal/configuração.
    import h5py
    import get_obs_matrix as gom
    protocol=next(p for p in result["protocolos"] if p["participante"]=="Ab" and p["grupo"]=="estimulo" and p["frequencia_indice"]==0)
    f=protocol["fases"][1]
    with h5py.File(f["arquivo"],"r") as mat:
        signal=np.asarray(mat["x"][f["canal"],:120,:],dtype=float)
    spectrum=np.fft.rfft(signal,axis=1); hz=np.fft.rfftfreq(signal.shape[1],d=1/f["fs"])
    idx,bin_hz=gom.localizar_bin_fft(signal.shape[1],f["fs"],f["frequencia"])
    coeff=spectrum[:,idx];unit=coeff/(np.abs(coeff)+1e-12)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    axes[0,0].plot(np.arange(signal.shape[1])/f["fs"],signal[0],lw=.7)
    axes[0,0].set(title="Ab 30 dB: primeira época, canal 0",xlabel="Tempo na época (s)",ylabel="EEG (unidade armazenada)")
    axes[0,1].plot(hz,np.mean(np.abs(spectrum),axis=0));axes[0,1].axvline(bin_hz,color="#d35400",ls="--")
    axes[0,1].set(xlim=(70,105),title="FFT média das 120 épocas",xlabel="Frequência (Hz)",ylabel="Magnitude FFT")
    angle=np.linspace(0,2*np.pi,300)
    axes[1,0].plot(np.cos(angle),np.sin(angle),color="gray",lw=.8)
    axes[1,0].scatter(unit.real,unit.imag,s=16,alpha=.5)
    mean=unit.mean();axes[1,0].arrow(0,0,mean.real,mean.imag,width=.006,color="#d35400",length_includes_head=True)
    axes[1,0].set(aspect="equal",title=f"Fases em {bin_hz:g} Hz: R²={abs(mean)**2:.5f}",xlabel="Re(u)",ylabel="Im(u)")
    axes[1,1].plot(f["finais_locais"],f["p_values"],"o-");axes[1,1].set(yscale="log",title="p=exp(−120 R²) nas janelas do arquivo",xlabel="Fim local da janela (s)",ylabel="p")
    save(fig,"eeg_fft_rayleigh","Contexto com dados reais: EEG, FFT, fases e conversão em p. A seta é a média vetorial das fases. Nenhum processamento é modificado pela alternativa bayesiana; unidades físicas do EEG não foram presumidas.")
    for name in prior_sets:
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        for row, s in enumerate(STATES):
            idata = az.from_netcdf(ASSETS / f"posterior_{name}_{s}.nc")
            prior = np.load(ASSETS / f"prior_{name}_{s}.npz")
            for col, param in enumerate(("mu", "kappa", "a", "b")):
                values = idata.posterior[param].values.ravel()
                hi = max(np.quantile(prior[param], .98), np.quantile(values, .999))
                bins = np.linspace(0, hi, 65)
                axes[row, col].hist(prior[param], bins=bins, density=True, alpha=.25, label="A priori")
                axes[row, col].hist(values, bins=bins, density=True, alpha=.55, color=COLORS[s], label="Posterior")
                axes[row, col].axvspan(*np.quantile(values, [.025,.975]), color=COLORS[s], alpha=.1)
                axes[row, col].set(title=f"{s}: {param}", xlabel=param, ylabel="Densidade")
                axes[row, col].legend()
        save(fig, f"posterior_{name}", f"Atualização {name}: prioris versus posteriores marginais. Faixa sombreada: intervalo equal-tail 95%. Limite horizontal cobre 98% da priori. a e b são derivados de μ e κ, não prioris independentes.")
        for s in STATES:
            idata = az.from_netcdf(ASSETS / f"posterior_{name}_{s}.nc")
            axes = az.plot_trace(idata, var_names=["mu", "kappa", "a", "b"], compact=False, figsize=(14, 10))
            save(axes.ravel()[0].figure, f"trace_{name}_{s}", f"{name}/{s}: densidades por cadeia e trace plots após aquecimento. Os números R-hat, ESS, divergências e BFMI estão no relatório; inspeção visual sozinha não prova convergência.")
            axes = az.plot_rank(idata, var_names=["mu", "kappa"], figsize=(11, 5))
            save(np.asarray(axes).ravel()[0].figure, f"rank_{name}_{s}", f"{name}/{s}: ranks por cadeia. Cadeias bem misturadas devem ocupar regiões semelhantes da distribuição.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, s in zip(axes, STATES):
            idata = az.from_netcdf(ASSETS / f"posterior_{name}_{s}.nc")
            ax.hexbin(idata.posterior.mu.values.ravel(), idata.posterior.kappa.values.ravel(), gridsize=45, cmap="Blues", mincnt=1)
            ax.set(title=f"Posterior conjunta: {s}", xlabel="μ", ylabel="κ")
        save(fig, f"joint_{name}", "Posterior conjunta μ × κ. Independência a priori não implica independência posterior.")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, s in enumerate(STATES):
        axes[row, 0].hist(data[s], bins=30, density=True, color=COLORS[s], alpha=.2)
        for col, grid in enumerate((x, xlog, x)):
            mle = beta.pdf(grid, models[s]["a"], models[s]["b"])
            if col < 2:
                axes[row, col].plot(grid, mle, color="black", ls="--", label="MLE histórico")
            for name in prior_sets:
                samples = np.load(ASSETS / f"predictive_{name}_{s}.npz")
                density = np.exp(predictive_logpdf(grid, samples))
                if col < 2:
                    axes[row, col].plot(grid, density, label=f"Preditiva {name}")
                else:
                    axes[row, col].plot(grid, density/mle, label=name)
                if name == "principal" and col == 0:
                    curves = beta.pdf(grid[:, None], samples["a"][::8], samples["b"][::8])
                    axes[row, col].fill_between(grid, *np.quantile(curves, [.025,.975], axis=1), color=COLORS[s], alpha=.16)
            axes[row, col].set(title=f"{s}: " + ("ajuste" if col == 0 else "extremo p→0" if col == 1 else "preditiva / MLE"), xlabel="p", ylabel="Densidade" if col < 2 else "Razão")
            if col == 0: axes[row, col].set_ylim(0,8)
            if col == 1: axes[row, col].set_xscale("log"); axes[row, col].set_yscale("log")
            axes[row, col].legend(fontsize=9)
    save(fig, "emissoes_sensibilidade", "Mudança na emissão com todos os parâmetros temporais fixos. Faixa: incerteza 95% das PDFs condicionais, distinta da curva preditiva (média das PDFs). Gráfico log-log mostra densidades altas perto de zero. As três prioris são reportadas sem selecionar vencedora.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in prior_sets:
        samples = {s: np.load(ASSETS / f"predictive_{name}_{s}.npz") for s in STATES}
        evidence = predictive_logpdf(xlog, samples["Presente"]) - predictive_logpdf(xlog, samples["Ausente"])
        axes[0].plot(xlog, evidence, label=name)
        cdf = np.mean(beta.cdf(x[:,None], samples["Presente"]["a"], samples["Presente"]["b"]), axis=1)
        axes[1].plot(x, cdf, label=name)
    mle_evidence = beta.logpdf(xlog, **{k: models["Presente"][k] for k in ("a","b")}) - beta.logpdf(xlog, **{k: models["Ausente"][k] for k in ("a","b")})
    axes[0].plot(xlog, mle_evidence, "k--", label="MLE")
    axes[0].set(xscale="log", xlabel="p", ylabel="log f_P(p) − log f_A(p)", title="Evidência de emissão"); axes[0].axhline(0,color="gray",lw=1)
    axes[1].set(xlabel="p", ylabel="CDF", title="Preditiva Presente acumulada")
    for ax in axes: ax.legend()
    save(fig, "evidencia_emissao", "Log-razão das densidades é a evidência local; não é probabilidade posterior de estado. A decisão depende também de A, pi e delta acumulado.")
    methods = {"MLE": result["baseline_mle"], **result["alternativas"]}
    labels = list(methods)
    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    for ax, key, title in zip(axes, ("taxa_deteccao", "taxa_fp_alvo_esp", "taxa_fp_lateral"), ("Detecção após 30 dB / 88", "FP alvo ESP / 88", "FP lateral protocolo todo / 88")):
        values = [methods[n]["metricas"]["global"][key]*100 for n in labels]
        ax.bar(labels, values, color=["#666", "#2874a6", "#d35400", "#148f77"])
        ax.set(title=title, ylabel="%", ylim=(0,100))
        for i, value in enumerate(values): ax.text(i,value+1,f"{value:.2f}%",ha="center")
    save(fig, "decisoes_globais", "Efeitos nas decisões: comparação fixa exploratória in-sample. FP alvo ESP e lateral têm denominadores separados. Não há seleção pelo desempenho nem nova busca.")
    for grouping, title in (("por_paciente", "Participante"), ("por_frequencia", "Índice de frequência (81 a 95 Hz)")):
        keys = list(result["baseline_mle"]["metricas"][grouping])
        fig, axes = plt.subplots(2,1,figsize=(13,8))
        for ax, metric in zip(axes,("taxa_deteccao","taxa_fp_lateral")):
            for i,n in enumerate(labels):
                vals = [methods[n]["metricas"][grouping][key][metric]*100 for key in keys]
                ax.bar(np.arange(len(keys))+(i-1.5)*.2,vals,width=.2,label=n)
            ax.set(xticks=np.arange(len(keys)),xticklabels=keys,ylabel="%",ylim=(0,110),title=metric)
            ax.legend(ncol=4)
        axes[-1].set_xlabel(title)
        save(fig, grouping, "Detecção e FP lateral por estrato. Cada participante tem oito alvos e oito laterais; cada frequência tem onze participantes. Não são testes independentes de validação.")
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    times = [[d["tempo_deteccao_desde_30_s"] for d in methods[n]["detalhes"] if d["grupo"] == "estimulo" and d["detectou"]] for n in labels]
    axes[0].boxplot(times,tick_labels=labels,showmeans=True)
    axes[0].set(ylabel="Tempo desde 30 dB (s)",title="Somente detectados")
    for name in prior_sets:
        pairs = result["alternativas"][name]["pareamento_com_mle"]["pares"]
        pairs = [p for p in pairs if p["delta_tempo_s"] is not None]
        axes[1].scatter([p["mle_tempo_s"] for p in pairs],[p["bayes_tempo_s"] for p in pairs],alpha=.5,label=name)
    axes[1].plot([0,1800],[0,1800],"k--")
    axes[1].set(xlabel="MLE (s)",ylabel="Bayes (s)",title="Pareado: detectados por ambos"); axes[1].legend()
    save(fig, "tempos_pareados", "Tempos somente entre detectados. Abaixo da diagonal: Bayes detecta antes. Trajetórias exclusivas e não detectadas estão separadas no JSON; tempos não detectados permanecem null.")
    # Uma página por participante: todas as frequências e todos os controles.
    for participant in result["baseline_mle"]["metricas"]["por_paciente"]:
        fig, axes = plt.subplots(2,1,figsize=(15,9))
        for ax, group in zip(axes,("estimulo","lateral")):
            for i, name in enumerate(labels):
                paths = [p for p in methods[name]["trajetorias"] if p["participante"]==participant and p["grupo"]==group]
                for p in paths:
                    row = p["frequencia_indice"]*5+i
                    ends = np.asarray(p["finais"])
                    ax.scatter(ends, np.full(len(ends),row), c=np.where(p["estados"], "#d35400", "#bfc9ca"),s=14,marker="s")
            protocol = next(p for p in result["protocolos"] if p["participante"]==participant)
            for f in protocol["fases"]:
                ax.axvline(f["inicio_global"],color="gray",lw=.6)
                ax.text(f["inicio_global"]+8,40,str(f["nivel"]),fontsize=9)
            ticks = [idx*5+i for idx in range(8) for i in range(4)]
            ax.set(yticks=ticks,yticklabels=[f"{81+2*idx if group=='estimulo' else 82+2*idx} Hz {n}" for idx in range(8) for n in labels],xlabel="Tempo total desde ESP (s)",title=f"{participant}: {group}",ylim=(-1,42))
            ax.tick_params(axis="y",labelsize=7)
        save(fig,f"trajetorias_{participant}","Todos os percursos causais do participante. Laranja: estado terminal Presente; cinza: Ausente. Lacunas indicam ausência de janela completa. Os estados continuam entre arquivos, mas nenhuma janela cruza a fronteira.")
    # Exemplo detalhado de atualização causal com delta em escala relativa.
    fig, axes = plt.subplots(3,1,figsize=(14,10),sharex=True)
    for name in ("MLE","principal"):
        trace = next(t for t in methods[name]["trajetorias"] if t["participante"]=="Ab" and t["grupo"]=="estimulo" and t["frequencia_indice"]==0)
        ends=np.asarray(trace["finais"]); logs=np.asarray(trace["log_emissoes"]); scores=np.asarray(trace["delta"])
        axes[0].plot(ends,np.maximum(trace["p_values"],1e-9),"o-",label=name)
        axes[1].plot(ends,logs[:,1]-logs[:,0],"o-",label=name)
        axes[2].step(ends,scores[:,1]-scores[:,0],where="post",label=name)
    axes[0].set(yscale="log",ylabel="p estabilizado",title="Exemplo Ab alvo 81 Hz")
    axes[1].set(ylabel="log f_P − log f_A")
    axes[2].set(ylabel="delta_P − delta_A",xlabel="Tempo total (s)")
    for ax in axes: ax.legend(); ax.axhline(0,color="gray",lw=.6)
    save(fig,"exemplo_causal", "Mesmos p-valores, duas emissões. Delta_P − delta_A > 0 indica estado terminal Presente. Delta não é posterior normalizada; o cálculo usa apenas o prefixo observado.")
    fig, axes = plt.subplots(2,4,figsize=(16,8))
    for row,(key,recovery) in enumerate(result["recuperacao_simulada"].items()):
        idata=az.from_netcdf(ASSETS / f"recovery_{key}.nc")
        for col,param in enumerate(("mu","kappa","a","b")):
            axes[row,col].hist(idata.posterior[param].values.ravel(),bins=50,density=True,color=COLORS[STATES[row]],alpha=.6)
            axes[row,col].axvline(recovery["truth"][param],color="black",ls="--",label="Verdade simulada")
            axes[row,col].set(title=f"Simulação {key}: {param}",xlabel=param,ylabel="Densidade"); axes[row,col].legend(fontsize=8)
    save(fig,"recuperacao_simulada","Recuperação em dois conjuntos iid simulados (n=1500). Linhas marcam os valores verdadeiros; cobertura dos intervalos 95% e erros padronizados estão no JSON. Demonstra funcionamento numérico; não testa o pressuposto de independência do EEG real.")
    pdf.close()
    records = []
    for name,item in methods.items():
        records.append({"metodo":name,**item["metricas"]["global"]})
    import csv
    with open(FIGURES / "metricas_globais.csv","w") as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    for grouping in ("por_paciente","por_frequencia"):
        rows=[{"metodo":name,"estrato":key,**g} for name,item in methods.items() for key,g in item["metricas"][grouping].items()]
        with open(FIGURES / f"metricas_{grouping}.csv","w") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    rows=[{"priori":name,"estado":s,"parametro":param,**values}
          for name,item in result["alternativas"].items() for s,posterior in item["posterior"].items()
          for param,values in posterior.items()]
    with open(FIGURES / "posteriores.csv","w") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    report=["# Relatório: emissão Beta bayesiana para ASSR", "",
            "Comparação exploratória in-sample. MLE continua sendo o método mantido. Nenhuma busca ou seleção de priori por desempenho.", "",
            "Modelo: p | μ,κ ~ Beta(μκ,(1−μ)κ), μ ~ Beta(u,v), κ ~ Gamma(shape,rate). Emissão preditiva é a média das PDFs posteriores, calculada com logsumexp. A/pi ficam fixas.", "",
            "Principal: μ Ausente Beta(8,8), μ Presente Beta(2,4), κ de ambos Gamma(2,1). Média nula ideal 0,5, alternativa com média prévia 1/3 e concentração ampla. Outras duas prioris verificam sensibilidade sem escolher vencedora.", "",
            "| Método | Detecção /88 | FP alvo ESP /88 | FP lateral /88 | Média/mediana detectados (s) |",
            "|---|---:|---:|---:|---:|"]
    for n,item in methods.items():
        g=item["metricas"]["global"]
        report.append(f"| {n} | {g['detectados']} | {g['fp_alvo_esp']} | {g['fp_lateral']} | {g['tempo_desde_30_medio_detectados_s']:.1f}/{g['tempo_desde_30_mediano_detectados_s']:.1f} |")
    report += ["", "## Diagnósticos e incerteza posterior", "",
               "| Priori | Estado | R-hat máximo | ESS bulk mínimo | ESS tail mínimo | Divergências |",
               "|---|---|---:|---:|---:|---:|"]
    for n,item in result["alternativas"].items():
        for s,d in item["diagnostics"].items():
            report.append(f"| {n} | {s} | {d['max_rhat']:.4f} | {d['min_ess_bulk']:.0f} | {d['min_ess_tail']:.0f} | {d['divergences']} |")
    report += ["", "| Priori | Estado | Parâmetro | Média | Intervalo equal-tail 95% |",
               "|---|---|---|---:|---|"]
    for n,item in result["alternativas"].items():
        for s,posterior in item["posterior"].items():
            for param,v in posterior.items():
                lo,hi=v['equal_tail_95']
                report.append(f"| {n} | {s} | {param} | {v['mean']:.4f} | [{lo:.4f}; {hi:.4f}] |")
    report += ["", "## Comparação pareada somente entre detectados pelos dois métodos", "",
               "| Priori | Comuns | Bayes antes | MLE antes | Empates | Só Bayes | Só MLE | Diferença média Bayes−MLE (s) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n,item in result["alternativas"].items():
        p=item['pareamento_com_mle']
        report.append(f"| {n} | {p['ambos']} | {p['bayes_antes']} | {p['mle_antes']} | {p['empates']} | {p['somente_bayes']} | {p['somente_mle']} | {p['diferenca_media_s_pareada']:.2f} |")
    report += ["", "## Leitura e limites", "",
               "As curvas preditivas mudam mesmo quando decisões discretas coincidem. Os CSVs e o JSON preservam os resultados completos por participante e frequência, incluindo não detectados com tempo null. O histograma aparece somente como gráfico dos dados.", "",
               "Sobreposição e dependência por participante podem subestimar incerteza sob likelihood fatorizada. Marginais preditivas por janela não integram conjuntamente parâmetros compartilhados na trajetória. Presente real é proxy e os mesmos dados são usados no ajuste e na avaliação. Não há validação clínica, comparação LOSO nova ou adoção automática.", "",
               "## Reprodução", "", "```bash", ".venv/bin/python src/bayesian_beta_hmm.py", ".venv/bin/python src/bayesian_beta_hmm.py --reuse-samples", ".venv/bin/python src/bayesian_beta_hmm.py --plots-only", ".venv/bin/python -m unittest discover -s tests -p test_bayesian_beta_hmm.py", "```", "",
               "Documentação completa: [docs/agents/bayesian_beta_hmm.md](../../docs/agents/bayesian_beta_hmm.md). Galeria: [index.html](index.html).", "", "## Figuras", ""]
    report.extend(f"- [{name}]({name}.png): {caption}" for name,caption in gallery)
    (FIGURES / "README.md").write_text("\n".join(report)+"\n")
    body = """<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Emissão Beta bayesiana para ASSR</title><style>body{font:18px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#17324d}img{width:100%}section{margin:50px 0;border-top:1px solid #ddd}a{color:#2874a6}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ddd}</style><h1>Emissão Beta bayesiana para o HMM de ASSR</h1><p>Comparação exploratória in-sample com baseline MLE reproduzida. Detector, dados, A, pi, janela/passo e regra fixos. A alternativa não foi adotada. Presente é proxy de gravações estimuladas reais.</p><p>Modelo: p | μ,κ ~ Beta(μκ,(1−μ)κ); μ ~ Beta(u,v); κ ~ Gamma(shape,rate). Emissão = média das densidades posteriores, calculada em log-espaço. Marginais por janela são uma aproximação; não há integração conjunta de parâmetros compartilhados na trajetória.</p><p><a href='apresentacao_figuras.pdf'>PDF com todas as figuras</a> · <a href='metricas_globais.csv'>Métricas CSV</a></p><table><tr><th>Método</th><th>Alvos / 88</th><th>FP ESP / 88</th><th>FP lateral / 88</th><th>Média / mediana s</th></tr>"""
    for n,item in methods.items():
        g=item["metricas"]["global"]
        body+=f"<tr><td>{n}</td><td>{g['detectados']}</td><td>{g['fp_alvo_esp']}</td><td>{g['fp_lateral']}</td><td>{g['tempo_desde_30_medio_detectados_s']:.1f} / {g['tempo_desde_30_mediano_detectados_s']:.1f}</td></tr>"
    body+="</table><p>Incerteza posterior condicional à verossimilhança fatorizada: sobreposição e dependência por participante podem produzir intervalos estreitos demais. Nenhuma figura demonstra validação clínica ou externa.</p>"
    for name,caption in gallery:
        body+=f"<section><h2>{html.escape(name.replace('_',' '))}</h2><img loading='lazy' src='{name}.png'><p>{html.escape(caption)}</p><p><a href='{name}.svg'>SVG</a> · <a href='{name}.pdf'>PDF</a></p></section>"
    (FIGURES / "index.html").write_text(body+"</html>")
    (FIGURES / "figuras.json").write_text(json.dumps(gallery,ensure_ascii=False,indent=2))
    print(f"{len(gallery)} figuras PNG/SVG/PDF e galeria em {FIGURES}",flush=True)
