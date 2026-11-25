# =============================================================================
# Reajuste do modelo hierarquico Gamma com JAGS (pacote rjags), para comparar
# com o amostrador Metropolis-dentro-de-Gibbs escrito a mao em mcmc_completo.R.
# O modelo e identico; so muda quem escolhe/implementa os passos de amostragem.
#
# Uso:  cd r_mcmc && Rscript mcmc_jags.R [N_ITER BURN_IN]
# Saidas: trace_jags.png, resumo_jags.md (na pasta do script)
# =============================================================================
suppressPackageStartupMessages({ library(rjags); library(coda) })

# ---------------- configuracao ----------------
N_ITER   <- 20000
BURN_IN  <- 5000
THIN     <- 10
N_ADAPT  <- 1000                  # adaptacao do JAGS (dentro do burn-in)
SEMENTE  <- 20240601
# inicios das cadeias: (a_k, b_k, a_rho, b_rho)
INICIOS <- rbind(c(9, 6, 1, 40), c(4, 3, 2, 100), c(20, 2, 0.7, 30), c(1, 0.5, 5, 10))
NOMES   <- c("a_k", "b_k", "a_rho", "b_rho")
# referencia: media e dp do amostrador manual (resumo_r.md)
REF <- rbind(a_k = c(1.463, 0.106), b_k = c(0.319, 0.030),
             a_rho = c(0.290, 0.020), b_rho = c(9.495, 1.100))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) { N_ITER <- as.integer(args[1]); BURN_IN <- as.integer(args[2]) }
arg_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
PASTA <- if (length(arg_file)) dirname(normalizePath(sub("^--file=", "", arg_file[1]))) else getwd()
N_ADAPT <- min(N_ADAPT, BURN_IN %/% 2)   # teste rapido: adaptacao cabe no burn-in
set.seed(SEMENTE)

# ---------------- modelo em linguagem BUGS ----------------
MODELO <- "
model {
  for (i in 1:n_pontos) {
    mu[i] <- 1 + (N[i] - 1) * rho[seq[i]]       # media: mu_st = 1 + (N_st - 1) rho_s
    Z[i] ~ dgamma(k[seq[i]], k[seq[i]] / mu[i]) # Z_st ~ Gamma(forma k_s, taxa k_s/mu_st)
  }
  for (s in 1:S) {
    k[s]   ~ dgamma(a_k, b_k)                   # k_s   ~ Gamma(a_k,   taxa b_k)
    rho[s] ~ dgamma(a_rho, b_rho)               # rho_s ~ Gamma(a_rho, taxa b_rho)
  }
  log_ak ~ dunif(-5, 11);  a_k   <- exp(log_ak)   # log a_k   ~ U(-5, 11)
  log_bk ~ dunif(-5, 11);  b_k   <- exp(log_bk)   # log b_k   ~ U(-5, 11)
  log_ar ~ dunif(-5, 11);  a_rho <- exp(log_ar)   # log a_rho ~ U(-5, 11)
  log_br ~ dunif(-5, 11);  b_rho <- exp(log_br)   # log b_rho ~ U(-5, 11)
}"

# ---------------- dados ----------------
preparar_dados <- function() {
  d <- read.csv(file.path(PASTA, "calibracao.csv"))
  d <- d[order(d$seq), ]
  list(Z = pmax(d$Z, 1e-12), N = d$N, seq = as.integer(factor(d$seq)),
       n_pontos = nrow(d), S = length(unique(d$seq)))
}

# inicios de uma cadeia (hiperparametros em log + latentes + RNG fixo)
inicio_cadeia <- function(c, S) {
  list(log_ak = log(INICIOS[c, 1]), log_bk = log(INICIOS[c, 2]),
       log_ar = log(INICIOS[c, 3]), log_br = log(INICIOS[c, 4]),
       k = exp(rnorm(S, 0, 1)), rho = exp(rnorm(S, log(0.01), 1.5)),
       .RNG.name = "base::Mersenne-Twister", .RNG.seed = SEMENTE + c)
}

# ---------------- ajuste ----------------
ajustar <- function(dados) {
  inits <- lapply(1:nrow(INICIOS), inicio_cadeia, S = dados$S)
  modelo <- jags.model(textConnection(MODELO), data = dados, inits = inits,
                       n.chains = nrow(INICIOS), n.adapt = N_ADAPT)
  update(modelo, n.iter = BURN_IN - N_ADAPT)          # restante do burn-in
  amostras <- coda.samples(modelo, c(NOMES, "k", "rho"),
                           n.iter = N_ITER - BURN_IN, thin = THIN)
  list(modelo = modelo, amostras = amostras)
}

# ---------------- diagnosticos ----------------
# mcmc.list -> mcmc.list so com as colunas escolhidas, em log
em_log <- function(am, cols) {
  as.mcmc.list(lapply(am, function(m) mcmc(log(m[, cols, drop = FALSE]),
                                           start = start(m), thin = thin(m))))
}
rhat <- function(am_log) gelman.diag(am_log, autoburnin = FALSE, multivariate = FALSE)$psrf[, 1]

diag_hiper <- function(am) {
  mat <- as.matrix(am[, NOMES])
  lg  <- em_log(am, NOMES)
  data.frame(media = colMeans(mat), dp = apply(mat, 2, sd),
             rhat = rhat(lg)[NOMES], ess = effectiveSize(lg)[NOMES])
}

diag_latentes <- function(am, S) {
  tab <- NULL
  for (v in c("k", "rho")) {
    lg <- em_log(am, paste0(v, "[", 1:S, "]"))
    r <- rhat(lg); e <- effectiveSize(lg)
    tab <- rbind(tab, data.frame(latente = paste0("log ", v, "_s"), rhat_max = max(r),
      rhat_med = median(r), ess_min = min(e), ess_med = median(e)))
  }
  tab
}

# amostradores escolhidos pelo JAGS, por tipo de no
resumo_samplers <- function(modelo) {
  ls <- list.samplers(modelo)
  nomes <- names(ls)
  nos <- vapply(seq_along(ls), function(i) {
    n <- sub("\\[.*", "", ls[[i]][1]); n }, character(1))
  df <- data.frame(no = nos, amostrador = nomes)
  df$grupo <- ifelse(df$no %in% c("k", "rho"), df$no, "hiper (log_*)")
  as.data.frame(table(no = df$grupo, amostrador = df$amostrador), stringsAsFactors = FALSE)
}

# ---------------- grafico ----------------
grafico_trace <- function(am, arquivo) {
  png(arquivo, width = 1100, height = 800, res = 110)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  cores <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
  for (p in NOMES) {
    ys <- lapply(am, function(m) m[, p])
    plot(NA, xlim = c(1, length(ys[[1]])), ylim = range(unlist(ys)), log = "y",
         xlab = "iteracao (apos thin)", ylab = p, main = p)
    for (c in seq_along(ys)) lines(as.numeric(ys[[c]]), col = cores[c], lwd = 0.7)
  }
  dev.off()
}

# ---------------- fluxo principal ----------------
t0 <- Sys.time()
dados <- preparar_dados()
aj <- ajustar(dados)
tempo <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

th <- diag_hiper(aj$amostras)
tl <- diag_latentes(aj$amostras, dados$S)
ts <- resumo_samplers(aj$modelo)
grafico_trace(aj$amostras, file.path(PASTA, "trace_jags.png"))

# ---------------- resumo em markdown ----------------
tab_md <- function(df) {
  c(paste0("| ", paste(names(df), collapse = " | "), " |"),
    paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|"),
    apply(df, 1, function(r) paste0("| ", paste(trimws(r), collapse = " | "), " |")))
}
th_md <- data.frame(Parametro = NOMES,
  media = sprintf("%.3f", th$media), dp = sprintf("%.3f", th$dp),
  `R-hat` = sprintf("%.4f", th$rhat), ESS = sprintf("%.0f", th$ess),
  `ref media` = sprintf("%.3f", REF[NOMES, 1]), `ref dp` = sprintf("%.3f", REF[NOMES, 2]),
  `dif/dp ref` = sprintf("%.2f", (th$media - REF[NOMES, 1]) / REF[NOMES, 2]),
  check.names = FALSE)
tl_md <- data.frame(Latente = tl$latente, `R-hat max` = sprintf("%.4f", tl$rhat_max),
  `R-hat mediano` = sprintf("%.4f", tl$rhat_med), `ESS min` = sprintf("%.0f", tl$ess_min),
  `ESS mediano` = sprintf("%.0f", tl$ess_med), check.names = FALSE)
ts_md <- data.frame(`No` = ts$no, Amostrador = ts$amostrador, Quantidade = ts$Freq,
                    check.names = FALSE)
ts_md <- ts_md[ts_md$Quantidade > 0, ]

linhas <- c("# Modelo hierarquico Gamma com JAGS (rjags)", "",
  sprintf("Config: %d iteracoes, burn-in %d (adaptacao %d), thin %d, %d cadeias, %d sequencias; tempo %.0f s; JAGS %s.",
          N_ITER, BURN_IN, N_ADAPT, THIN, nrow(INICIOS), dados$S, tempo,
          as.character(packageVersion("rjags"))), "",
  "## Hiperparametros (escala natural; R-hat e ESS calculados em log)", "",
  "Colunas `ref` = amostrador manual (resumo_r.md).", "", tab_md(th_md), "",
  sprintf("## Latentes (%d sequencias, em log)", dados$S), "", tab_md(tl_md), "",
  "## Amostradores escolhidos pelo JAGS", "", tab_md(ts_md), "")
writeLines(linhas, file.path(PASTA, "resumo_jags.md"))
cat(linhas, sep = "\n")
cat(sprintf("Tempo total: %.0f s\n", tempo))
