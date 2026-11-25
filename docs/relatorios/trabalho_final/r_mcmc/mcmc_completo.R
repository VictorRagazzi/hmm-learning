# =============================================================================
# MCMC "completo" (Metropolis-dentro-de-Gibbs) para o modelo hierarquico Gamma
# Reimplementacao didatica, em R base, de
#   src/rayleigh_hmm/monte_carlo/mcmc_completo.py   (funcao mcmc_completo)
#   src/rayleigh_hmm/monte_carlo/mc_comum.py        (split_rhat, ess, pontos_mcmc)
#
# Modelo (s = 1..440 sequencias; t = pontos da sequencia s):
#   Z_st | k_s, rho_s ~ Gamma(forma k_s, media mu_st = 1 + (N_st - 1) rho_s)
#                       (taxa = k_s / mu_st)
#   k_s   ~ Gamma(a_k,   taxa b_k)
#   rho_s ~ Gamma(a_rho, taxa b_rho)
#   Hiperpriori: log a, log b ~ Uniforme(-5, 11)  (ou seja, p(a) propto 1/a)
#
# Uso:  cd r_mcmc && Rscript mcmc_completo.R [N_ITER BURN_IN]
# Saidas: trace_hiper.png, preditiva.png, resumo_r.md (na pasta do script)
# =============================================================================

# ---------------- configuracao ----------------
N_ITER            <- 20000
BURN_IN           <- 5000
THIN              <- 10
JANELA_ADAPTACAO  <- 50          # so durante o burn-in
ACEITACAO_ALVO    <- 0.44        # taxa alvo de passos univariados
PASSO_INICIAL     <- 0.5         # dp inicial do passeio em log (latentes)
PASSO_INICIAL_HIPER <- 0.1       # idem, para log a_k e log a_rho
LIMITE_PASSO      <- c(1e-3, 20)
LIMITES_LOG       <- c(-5, 11)   # suporte da hiperpriori em log
M_AMOSTRAS        <- 2000        # tamanho da amostra preditiva
SEMENTE           <- 20240601
# inicios das cadeias: (a_k, b_k, a_rho, b_rho)
INICIOS <- rbind(c(9, 6, 1, 40), c(4, 3, 2, 100), c(20, 2, 0.7, 30), c(1, 0.5, 5, 10))
NOMES   <- c("a_k", "b_k", "a_rho", "b_rho")

# argumentos opcionais para teste rapido
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) { N_ITER <- as.integer(args[1]); BURN_IN <- as.integer(args[2]) }

# pasta do script (para rodar de qualquer lugar)
arg_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
PASTA <- if (length(arg_file)) dirname(normalizePath(sub("^--file=", "", arg_file[1]))) else getwd()

set.seed(SEMENTE)

# ---------------- dados e estatisticas suficientes ----------------
dados <- read.csv(file.path(PASTA, "calibracao.csv"))
dados <- dados[order(dados$seq), ]
ids   <- dados$seq
z     <- pmax(dados$Z, 1e-12)
n_m1  <- dados$N - 1
S     <- length(unique(ids))                       # 440 sequencias
T_s   <- as.numeric(table(ids))                    # n de pontos por sequencia
soma_log_z <- as.numeric(rowsum(log(z), ids, reorder = FALSE))  # soma_t log z_st

# S_s(rho) = sum_t [ log mu_st + z_st / mu_st ]   (vetor de tamanho S)
soma_rho <- function(rho) {
  mu <- 1 + n_m1 * rho[ids]                        # mu_st para todos os pontos
  as.numeric(rowsum(log(mu) + z / mu, ids, reorder = FALSE))
}

# log verossimilhanca do exame s (vetorizada em s):
#   T_s (k log k - lgamma k) + (k - 1) soma_log_z_s - k S_s(rho)       [log_lik_k, py l.69]
log_lik <- function(k, s_rho) {
  T_s * (k * log(k) - lgamma(k)) + (k - 1) * soma_log_z - k * s_rho
}

# Alvo (em log) de um latente x (k ou rho), na coordenada log x:   [log_alvo_latente, py l.64]
#   log L_s + a log x - b x
# A priori Gamma(a, b) tem densidade propto x^(a-1) e^(-b x); o jacobiano da
# mudanca x = exp(u) multiplica por x, e o x^(-1) cancela: sobra a log x - b x.
log_alvo <- function(x, loglik, a, b) loglik + a * log(x) - b * x

# ---------------- passos do amostrador ----------------
# b | a, {x_s} ~ Gamma(forma S a, taxa sum x_s)   [gibbs_taxa, py l.85]
# Condicional completa: prod_s Gamma(x_s | a, b) * p(b), com p(b) propto 1/b (uniforme em log b):
#   propto b^(S a) exp(-b sum x_s) * b^(-1) = b^(S a - 1) exp(-b sum x_s),
#   que e o nucleo de uma Gamma(forma S a, taxa sum x_s) (o 1/b da priori e o jacobiano
#   da escala log se compensam; aqui, fiel ao Python, a forma e S a). Se log b sair de
#   LIMITES_LOG, reamostra (ate 100 vezes) e, por fim, trunca.
gibbs_taxa <- function(a, x) {
  forma <- length(x) * a; taxa <- sum(x)
  b <- rgamma(1, shape = forma, rate = taxa)
  for (i in 1:100) {
    if (log(b) >= LIMITES_LOG[1] && log(b) <= LIMITES_LOG[2]) break
    b <- rgamma(1, shape = forma, rate = taxa)
  }
  min(max(b, exp(LIMITES_LOG[1])), exp(LIMITES_LOG[2]))
}

# Metropolis em log a (passeio aleatorio)            [passo_hiper, py l.73]
# Condicional completa de a (dado b e x_s):
#   p(a | .) propto (1/a) * prod_s Gamma(x_s | a, b) ; em log a o 1/a some:
#   alvo(log a) = S (a log b - lgamma a) + a sum_s log x_s
# Retorna c(novo_log_a, aceitou)
passo_a <- function(log_a, log_b, soma_log_x, passo) {
  alvo <- function(la) { a <- exp(la); S * (a * log_b - lgamma(a)) + a * soma_log_x }
  prop <- log_a + passo * rnorm(1)
  if (prop < LIMITES_LOG[1] || prop > LIMITES_LOG[2]) return(c(log_a, 0))  # fora da priori
  if (log(runif(1)) < alvo(prop) - alvo(log_a)) c(prop, 1) else c(log_a, 0)
}

# ---------------- uma cadeia completa ----------------   [mcmc_completo, py l.97]
roda_cadeia <- function(inicio, n_iter, burn_in) {
  ak <- inicio[1]; bk <- inicio[2]; ar <- inicio[3]; br <- inicio[4]
  lk <- rnorm(S, 0, 1)                         # log k_s inicial
  lr <- rnorm(S, log(0.01), 1.5)               # log rho_s inicial
  k <- exp(lk); rho <- exp(lr)
  s_rho <- soma_rho(rho)                       # guardada para nao recalcular
  passo   <- matrix(PASSO_INICIAL, S, 2)       # passo por sequencia: [, 1] k, [, 2] rho
  passo_h <- c(PASSO_INICIAL_HIPER, PASSO_INICIAL_HIPER)
  acc_lat <- matrix(0, S, 2); acc_h <- c(0, 0) # contadores da janela de adaptacao
  hiper   <- matrix(NA_real_, n_iter, 4)       # log (a_k, b_k, a_rho, b_rho)
  acc_h_tot <- matrix(0, n_iter, 2)
  soma_acc_lat <- c(0, 0)                      # aceitacao media dos latentes pos burn-in
  idx_fino <- seq(burn_in + 1, n_iter, by = THIN)
  fino <- array(NA_real_, c(length(idx_fino), S, 2))
  i_fino <- 0

  for (t in 1:n_iter) {
    # a) Metropolis em log k_s (todas as sequencias de uma vez; aceitacao independente).
    #    Condicional completa de k_s: propto L_s(k_s, rho_s) * Gamma(k_s | a_k, b_k)
    prop <- lk + passo[, 1] * rnorm(S)
    kp   <- exp(prop)
    razao <- log_alvo(kp, log_lik(kp, s_rho), ak, bk) - log_alvo(k, log_lik(k, s_rho), ak, bk)
    ok_k  <- log(runif(S)) < razao
    lk[ok_k] <- prop[ok_k]; k[ok_k] <- kp[ok_k]

    # a') Metropolis em log rho_s.  Condicional completa: L_s(k_s, rho_s) * Gamma(rho_s | a_rho, b_rho)
    prop <- lr + passo[, 2] * rnorm(S)
    rp   <- exp(prop)
    s_p  <- soma_rho(rp)
    razao <- log_alvo(rp, log_lik(k, s_p), ar, br) - log_alvo(rho, log_lik(k, s_rho), ar, br)
    ok_r  <- log(runif(S)) < razao
    lr[ok_r] <- prop[ok_r]; rho[ok_r] <- rp[ok_r]; s_rho[ok_r] <- s_p[ok_r]

    # b) Gibbs: a condicional completa de b_k e Gamma (conjugacao Gamma-Gamma na taxa)
    bk <- gibbs_taxa(ak, k)
    br <- gibbs_taxa(ar, rho)

    # c) Metropolis em log a_k e log a_rho (a condicional de a nao e padrao)
    r1 <- passo_a(log(ak), log(bk), sum(lk), passo_h[1]); ak <- exp(r1[1])
    r2 <- passo_a(log(ar), log(br), sum(lr), passo_h[2]); ar <- exp(r2[1])

    hiper[t, ] <- log(c(ak, bk, ar, br))
    acc_h_tot[t, ] <- c(r1[2], r2[2])
    acc_lat[, 1] <- acc_lat[, 1] + ok_k;  acc_lat[, 2] <- acc_lat[, 2] + ok_r
    acc_h <- acc_h + c(r1[2], r2[2])
    if (t > burn_in) {
      soma_acc_lat <- soma_acc_lat + c(mean(ok_k), mean(ok_r))
      if ((t - burn_in - 1) %% THIN == 0) { i_fino <- i_fino + 1; fino[i_fino, , 1] <- lk; fino[i_fino, , 2] <- lr }
    }

    # adaptacao so no burn-in: passo <- clip(passo * exp(taxa_janela - alvo))
    if (t <= burn_in && t %% JANELA_ADAPTACAO == 0) {
      passo   <- pmin(pmax(passo   * exp(acc_lat / JANELA_ADAPTACAO - ACEITACAO_ALVO), LIMITE_PASSO[1]), LIMITE_PASSO[2])
      passo_h <- pmin(pmax(passo_h * exp(acc_h   / JANELA_ADAPTACAO - ACEITACAO_ALVO), LIMITE_PASSO[1]), LIMITE_PASSO[2])
      acc_lat[] <- 0; acc_h[] <- 0
    }
  }
  pos <- (burn_in + 1):n_iter
  list(hiper = hiper, fino = fino,
       acc_hiper = colMeans(acc_h_tot[pos, , drop = FALSE]),
       acc_lat = soma_acc_lat / (n_iter - burn_in))
}

# ---------------- diagnosticos ----------------
# R-hat dividido (Gelman-Rubin) de x: matriz (cadeias x n)      [split_rhat, mc_comum l.184]
split_rhat <- function(x) {
  n <- ncol(x) %/% 2
  metades <- rbind(x[, 1:n, drop = FALSE], x[, (n + 1):(2 * n), drop = FALSE])
  w <- mean(apply(metades, 1, var))                 # variancia dentro
  b <- n * var(rowMeans(metades))                   # variancia entre
  sqrt(((n - 1) / n * w + b / n) / w)
}

# ESS multicadeia (autocorrelacao media, soma de pares positivos)   [ess, mc_comum l.202]
ess <- function(x) {
  m <- nrow(x); n <- ncol(x)
  max_lag <- min(n - 1, 1000)
  rho <- rowMeans(sapply(1:m, function(i) acf(x[i, ], lag.max = max_lag - 1, plot = FALSE)$acf[, 1, 1]))
  soma <- 0
  for (t in seq(1, length(rho) - 2, by = 2)) {      # lags t, t+1 (rho[1] e o lag 0)
    par <- rho[t + 1] + rho[t + 2]
    if (par < 0) break
    soma <- soma + par
  }
  m * n / (1 + 2 * soma)
}

# ---------------- execucao ----------------
t0 <- Sys.time()
cadeias <- vector("list", nrow(INICIOS))
for (c in seq_len(nrow(INICIOS))) {
  cat(sprintf("cadeia %d/%d ...\n", c, nrow(INICIOS)))
  cadeias[[c]] <- roda_cadeia(INICIOS[c, ], N_ITER, BURN_IN)
}
segundos <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
nc <- length(cadeias)

# array (iteracao x parametro x cadeia), em log
hiper <- array(NA_real_, c(N_ITER, 4, nc))
for (c in 1:nc) hiper[, , c] <- cadeias[[c]]$hiper
pos_fino <- seq(BURN_IN + 1, N_ITER, by = THIN)       # indices pos burn-in com thin

tab <- data.frame(parametro = NOMES, media = NA, dp = NA, rhat = NA, ess = NA)
for (j in 1:4) {
  x <- t(hiper[pos_fino, j, ])                         # cadeias x n_fino (em log)
  v <- exp(x)
  tab$media[j] <- mean(v); tab$dp[j] <- sd(v)          # (python usa dp populacional; diferenca desprezivel)
  tab$rhat[j] <- split_rhat(x); tab$ess[j] <- ess(x)
}
acc_hiper <- mean(sapply(cadeias, function(ch) mean(ch$acc_hiper)))
acc_k <- mean(sapply(cadeias, function(ch) ch$acc_lat[1]))
acc_r <- mean(sapply(cadeias, function(ch) ch$acc_lat[2]))

# R-hat e ESS dos latentes sobre as 440 sequencias
lat <- list()
for (j in 1:2) {
  rh <- numeric(S); es <- numeric(S)
  for (s in 1:S) {
    x <- t(sapply(cadeias, function(ch) ch$fino[, s, j]))
    rh[s] <- split_rhat(x); es[s] <- ess(x)
  }
  lat[[j]] <- c(rhat_max = max(rh), rhat_med = median(rh), ess_min = min(es), ess_med = median(es))
}
names(lat) <- c("log k_s", "log rho_s")

# ---------------- resumo ----------------
fmt <- function(x, d = 3) formatC(x, format = "f", digits = d)
linhas <- c("# MCMC completo em R (Metropolis-dentro-de-Gibbs)", "",
  sprintf("Config: %d iteracoes, burn-in %d, thin %d, %d cadeias, %d sequencias; tempo %.0f s.",
          N_ITER, BURN_IN, THIN, nc, S, segundos), "",
  "## Hiperparametros (escala natural, pos burn-in com thin)", "",
  "| Parametro | media | dp | R-hat | ESS |", "|---|---|---|---|---|",
  sprintf("| %s | %s | %s | %s | %.0f |", tab$parametro, fmt(tab$media), fmt(tab$dp), fmt(tab$rhat, 4), tab$ess), "",
  "## Aceitacao media (pos burn-in)", "",
  sprintf("- hiperparametros (a_k, a_rho): %s", fmt(acc_hiper)),
  sprintf("- latentes k_s: %s; rho_s: %s", fmt(acc_k), fmt(acc_r)), "",
  "## Latentes (440 sequencias)", "",
  "| Latente | R-hat max | R-hat mediano | ESS min | ESS mediano |", "|---|---|---|---|---|",
  sprintf("| %s | %s | %s | %.0f | %.0f |", names(lat),
          sapply(lat, function(l) fmt(l["rhat_max"], 4)), sapply(lat, function(l) fmt(l["rhat_med"], 4)),
          sapply(lat, function(l) l["ess_min"]), sapply(lat, function(l) l["ess_med"])))
writeLines(linhas, file.path(PASTA, "resumo_r.md"))
cat(paste(linhas, collapse = "\n"), "\n")

# ---------------- figura: trace dos hiperparametros ----------------
png(file.path(PASTA, "trace_hiper.png"), width = 1000, height = 900, res = 110)
par(mfrow = c(4, 1), mar = c(3, 4.5, 1, 1))
cores <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
for (j in 1:4) {
  plot(NA, xlim = c(1, N_ITER), ylim = range(exp(hiper[, j, ])), log = "y",
       xlab = "", ylab = NOMES[j])
  for (c in 1:nc) lines(exp(hiper[, j, c]), col = cores[c], lwd = 0.5)
  abline(v = BURN_IN, lty = 2)
  if (j == 1) legend("topright", paste("cadeia", 1:nc), col = cores, lty = 1, horiz = TRUE, cex = 0.7)
}
mtext("iteracao", side = 1, line = 2)
dev.off()

# ---------------- preditiva ----------------   [pontos_mcmc, mc_comum l.234]
amostras <- exp(do.call(rbind, lapply(1:nc, function(c) hiper[pos_fino, , c])))
h  <- amostras[sample(nrow(amostras), M_AMOSTRAS, replace = TRUE), ]
kk <- rgamma(M_AMOSTRAS, shape = h[, 1], rate = h[, 2])   # k   ~ Gamma(a_k,   b_k)
rr <- rgamma(M_AMOSTRAS, shape = h[, 3], rate = h[, 4])   # rho ~ Gamma(a_rho, b_rho)

# p(Z | N) = media das M densidades Gamma(forma k, media 1 + (N-1) rho)
dens_pred <- function(zg, N) {
  mu <- 1 + (N - 1) * rr
  rowMeans(sapply(seq_len(M_AMOSTRAS), function(m) dgamma(zg, shape = kk[m], rate = kk[m] / mu[m])))
}
zg <- seq(0.01, 8, length.out = 400)
Ns <- c(10, 60, 180)
dp_ <- lapply(Ns, function(N) dens_pred(zg, N))
png(file.path(PASTA, "preditiva.png"), width = 900, height = 600, res = 110)
plot(zg, dexp(zg, 1), type = "l", lty = 2, lwd = 2, col = "grey40", ylim = c(0, max(1, unlist(dp_))),
     xlab = "Z", ylab = "densidade", main = "Preditiva p(Z | N) sob o MCMC completo")
cl <- c("#1b9e77", "#d95f02", "#7570b3")
for (i in seq_along(Ns)) lines(zg, dp_[[i]], col = cl[i], lwd = 2)
legend("topright", c("ausente: Exp(1)", paste("N =", Ns)), col = c("grey40", cl),
       lty = c(2, 1, 1, 1), lwd = 2)
dev.off()
cat(sprintf("Tempo total: %.0f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
