#!/usr/bin/env Rscript
# =============================================================================
# BDgraph baseline runner. Called from Python via subprocess (per brief:
# "call it from a script, no need for a fancy bridge").
#
# Usage: Rscript run_bdgraph.R <X_csv_path> <output_dir> <iter> <burnin>
#
# X_csv_path: CSV of X, shape (T, N) -- rows are samples, columns are nodes
#             (note: this is X.T relative to the Python pipeline's (N,T)
#             convention -- BDgraph expects (n_samples, n_variables))
# output_dir: where to write posterior_edge_probs.csv and timing.csv
# iter, burnin: BDgraph MCMC settings
# =============================================================================

suppressMessages(library(BDgraph))

args <- commandArgs(trailingOnly = TRUE)
X_csv_path <- args[1]
output_dir <- args[2]
iter <- as.integer(args[3])
burnin <- as.integer(args[4])

X <- as.matrix(read.csv(X_csv_path, header = FALSE))

t0 <- Sys.time()
sample_bd <- bdgraph(data = X, method = "ggm", algorithm = "bdmcmc",
                      iter = iter, burnin = burnin, save = TRUE, verbose = FALSE)
t1 <- Sys.time()
elapsed_sec <- as.numeric(difftime(t1, t0, units = "secs"))

# Posterior edge inclusion probabilities (N x N, symmetric)
edge_probs <- BDgraph::plinks(sample_bd, round = 6)
edge_probs <- edge_probs + t(edge_probs)  # plinks returns upper-triangular only

# Posterior mean precision matrix (point estimate for RelFrob comparison)
K_est <- BDgraph::precision(sample_bd, round = 8)

write.csv(edge_probs, file.path(output_dir, "bdgraph_edge_probs.csv"),
          row.names = FALSE)
write.csv(K_est, file.path(output_dir, "bdgraph_K_est.csv"), row.names = FALSE)

n_effective_samples <- iter - burnin
timing <- data.frame(
  elapsed_sec = elapsed_sec,
  iter = iter,
  burnin = burnin,
  n_effective_samples = n_effective_samples,
  samples_per_sec = n_effective_samples / elapsed_sec
)
write.csv(timing, file.path(output_dir, "bdgraph_timing.csv"), row.names = FALSE)

cat(sprintf("Done. elapsed=%.2fs, samples/sec=%.2f\n",
            elapsed_sec, n_effective_samples / elapsed_sec))
