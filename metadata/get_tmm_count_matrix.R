library(edgeR)
library(dplyr)

#' Perform TMM Normalization on Gene Counts using Length Factors
#' @description This function takes raw counts and length matrices, filters genes based on CPM 
#' and sample frequency, and returns TMM-normalized CPMs.
#' @param counts_mat A matrix of raw gene counts (rows = genes, cols = samples).
#' @param length_mat A matrix of gene lengths (same dimensions as counts_mat).
#' @param min_cpm The minimum CPM value for a gene to be considered "expressed". Default is 1.
#' @param filter_threshold A numeric value (0 to 1) representing the minimum proportion of samples 
#' where a gene must have CPM >= min_cpm. Default is 0.3 (30%).
#' @param remove_spikeins Logical; whether to remove ERCC spike-ins from the final table. Default is TRUE.
#' @return A matrix of TMM-normalized CPM values.

get_tmm_count_matrix <- function(counts_mat,
                                length_mat,
                                min_cpm = 1,
                                filter_threshold = 0.3,
                                remove_spikeins = TRUE) {

  # Validation: Ensure matrices match
  if (!all(colnames(counts_mat) == colnames(length_mat))) {
    stop("Error: Column names for counts_mat and length_mat do not match.")
  }

  message("Starting TMM normalization...")

  # 1. Scaling factors for length
  message("-> Calculating length-based scaling factors")
  normMat <- length_mat / exp(rowMeans(log(length_mat)))
  normCts <- counts_mat / normMat

  # 2. Effective library sizes (Composition bias)
  message("-> Computing effective library sizes")
  eff.lib <- edgeR::calcNormFactors(normCts) * colSums(normCts)

  # 3. Create DGEList and set offsets
  # Applying the logic from edgeR's recommended tximport workflow
  normMat_offset <- sweep(normMat, 2, eff.lib, "*")
  y <- edgeR::DGEList(counts_mat)
  y <- edgeR::scaleOffset(y, log(normMat_offset))

  # 4. Filtering based on user inputs
  message(paste0("-> Filtering: Keeping genes with CPM >= ", min_cpm, 
                 " in at least ", 100 * filter_threshold, "% of samples"))

  min_samples <- round(filter_threshold * ncol(counts_mat))
  keep <- rowSums(edgeR::cpm(y) >= min_cpm) >= min_samples
  y <- y[keep, , keep.lib.sizes = FALSE]

  # 5. Calculate TMM CPMs
  tmm_cpm <- edgeR::cpm(y, offset = y$offset, log = FALSE)

  # 6. Spike-in Removal
  if (remove_spikeins) {
    message("-> Removing ERCC spike-ins")
    tmm_cpm <- tmm_cpm[!grepl("ERCC-", rownames(tmm_cpm)), ]
  }

  message(paste("Normalization complete. Final gene count:", nrow(tmm_cpm)))
  return(tmm_cpm)
}
