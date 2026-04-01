#!/usr/bin/env Rscript

# This script runs GSEA analysis using different ontologies.
# Genes are ranked by: -log10(FDR) * sign(logFC)

# GSEA-related constants
MIN_GENEGROUP_SIZE = 10
MAX_GENEGROUP_SIZE = 500
N_PERM = 10000

# Valid ontologies
VALID_OPTIONS <- c("All", "D", "R", "BP", "CC", "MF", "H", "C1", "C2","C3","C4","C5", "C6","C7","C8")

# GMT files
GMT_FORMAT = "/mnt/efs/home/erusu/gsea/GMTs/%s.all.v2024.1.Hs.entrez.gmt"

################################################################################
#                          FUNCTION DEFINITION SITE                            #
################################################################################

# Function to print help
print_help <- function() {
  cat("
Description: This script runs GSEA analysis using different ontologies.
             It accepts a DE table with Ensembl IDs, logFC, and FDR columns.
             Genes are ranked by: -log10(FDR) * sign(logFC)

Usage:
    Rscript run_GSEA.R [table_path] [option_string] [fdr_cutoff] [out_folder_path] [target_terms_path]

Arguments:
    table_path        : Full path to the DE results table (required).
                        Must contain columns for gene IDs, logFC, and adjusted p-value.
                        Recognized logFC columns: 'logFC', 'log2FoldChange'
                        Recognized FDR columns:   'adj.P.Val', 'padj', 'FDR', 'qvalue'
    option_string     : String for ontology selection (required).
                        Options: 'All', 'D', 'R', 'BP', 'CC', 'MF', 'H', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'
    fdr_cutoff        : FDR threshold for reporting significant results (optional). Default: 0.05
    out_folder_path   : Path to the output directory (optional). Default: GSEA_results
    target_terms_path : Path to a file with specific term IDs (one per line) to filter results (optional).

Example:
    Rscript run_GSEA.R de_results.tsv 'H,R' 0.05 './gsea_out'
    Rscript run_GSEA.R de_results.tsv 'All' 0.25 './gsea_out' targets.txt
\n")
}

# --- GO GSEA with manual filtering ---
call_gsea_GO_filtered <- function(ranked_list, ontology, fdr) {
  goannot <- suppressMessages(AnnotationDbi::select(org.Hs.eg.db, keys = keys(org.Hs.eg.db, keytype = "GOALL"),
                                                    columns = c("ENTREZID"), keytype = "GOALL"))
  go_terms <- suppressMessages(AnnotationDbi::select(GO.db, keys = unique(goannot$GOALL),
                                                     columns = c("TERM", "ONTOLOGY"), keytype = "GOID"))
  
  goannot_full <- merge(goannot, go_terms, by.x = "GOALL", by.y = "GOID", all.x = TRUE)
  goannot_ont <- subset(goannot_full, ONTOLOGY == ontology)
  
  gene_counts <- goannot_ont %>% count(GOALL, name = "geneset_size")
  filtered_ids <- gene_counts %>%
    filter(geneset_size >= MIN_GENEGROUP_SIZE & geneset_size <= MAX_GENEGROUP_SIZE) %>%
    pull(GOALL)
  
  goannot_filtered <- goannot_ont %>% filter(GOALL %in% filtered_ids)
  term2gene <- goannot_filtered[, c("GOALL", "ENTREZID")]
  term2name <- unique(goannot_filtered[, c("GOALL", "TERM")])
  
  set.seed(123)
  res <- GSEA(ranked_list, TERM2GENE = term2gene, TERM2NAME = term2name,
              minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE,
              pvalueCutoff = fdr, pAdjustMethod = "fdr", nPermSimple = N_PERM,
              eps = 0)
  
  report_results(res, ontology, fdr)
}

# --- Reactome GSEA ---
call_gsea_reactome <- function(ranked_list, fdr) {
  set.seed(123)
  res <- gsePathway(ranked_list,
                    minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE,
                    pvalueCutoff = fdr, pAdjustMethod = "fdr", nPermSimple = N_PERM,
                    eps = 0)
  report_results(res, "R", fdr)
}

# --- Disease GSEA ---
call_gsea_disease <- function(ranked_list, fdr) {
  set.seed(123)
  res <- gseDGN(ranked_list,
                minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE,
                pvalueCutoff = fdr, pAdjustMethod = "fdr", nPermSimple = N_PERM,
                eps = 0)
  report_results(res, "D", fdr)
}

# --- GMT/MSigDB GSEA ---
call_gsea_gmt <- function(ranked_list, ontology, fdr) {
  msigdb_gmt <- read.gmt(sprintf(GMT_FORMAT, ontology))
  set.seed(123)
  res <- GSEA(ranked_list, TERM2GENE = msigdb_gmt,
              minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE,
              pvalueCutoff = fdr, pAdjustMethod = "fdr", nPermSimple = N_PERM,
              eps = 0)
  report_results(res, ontology, fdr)
}

# --- GSEA Barplot (NES-based) ---
custom_gsea_barplot <- function(res, ontology, fdr_val, N_bar = 20) {
  df <- res@result[!is.na(res@result$p.adjust) & res@result$p.adjust < fdr_val, ]
  if (nrow(df) == 0) return(NULL)
  
  df$Clean_description <- gsub("HALLMARK_", "", df$Description)
  df$Clean_description <- gsub("_", " ", df$Clean_description)
  
  df <- df[order(-abs(df$NES)), ]
  df_top <- head(df, N_bar)
  
  ggplot(df_top, aes(x = reorder(Clean_description, NES), y = NES, fill = p.adjust)) +
    geom_bar(stat = "identity", colour = "black", alpha = 0.7) +
    coord_flip() +
    labs(x = "", y = "Normalized Enrichment Score (NES)", title = paste("Top GSEA", ontology, "Terms")) +
    theme_minimal(base_size = 9) +
    scale_x_discrete(labels = label_wrap(40)) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "black", linewidth = 1) +
    scale_fill_gradient(low = "blue", high = "black", name = "FDR")
}

save_plot <- function(p, file_path_out) {
  tryCatch({
    png(file_path_out, units = "in", width = 7, height = 6, res = 600)
    print(p)
  }, error = function(e) {
    message(paste("Error generating plot for", file_path_out, ":", e$message))
  }, finally = {
    if (dev.cur() > 1) dev.off()
  })
}

# --- Result Reporting ---
report_results <- function(res, ont, fdr_val) {
  
  out_dir <- PATH
  
  # --- TARGETED FILTERING LOGIC ---
  if (!is.null(TARGET_LIST)) {
    res@result <- res@result[res@result$ID %in% TARGET_LIST, ]
    out_dir <- file.path(out_dir, "targeted_results")
  }
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  if (is.null(res) || nrow(res@result) == 0) {
    message(paste("No results for", ont))
  } else {
    sig_hits <- res@result[!is.na(res@result$p.adjust) & res@result$p.adjust < fdr_val, ]
    
    if (nrow(sig_hits) > 0) {
      message(paste("Found", nrow(sig_hits), "significant terms for", ont))
      write.table(sig_hits, file.path(out_dir, paste0("GSEA_table_", ont, ".tsv")),
                  quote = FALSE, sep = "\t", row.names = FALSE)
      
      try({
        p1 <- custom_gsea_barplot(res, ont, fdr_val)
        if (!is.null(p1)) save_plot(p1, file.path(out_dir, paste0("GSEA_Bar_", ont, ".png")))
      }, silent = TRUE)
      
      try({
        res_clean <- res
        res_clean@result <- sig_hits
        p3 <- dotplot(res_clean, showCategory = 15, font.size = 9) + ggtitle(paste("Dotplot", ont))
        save_plot(p3, file.path(out_dir, paste0("GSEA_Dot_", ont, ".png")))
      }, silent = TRUE)
      
      try({
        res_read <- res
        try({ res_read <- setReadable(res, OrgDb = org.Hs.eg.db, keyType = "ENTREZID") }, silent = TRUE)
        p4 <- cnetplot(res_read, foldChange = RANKED_LIST, cex_label_category = 0.7, node_label = "category") +
          scale_color_gradient2(name = "Rank metric", low = "blue", mid = "white", high = "red") +
          ggtitle(paste("Cnetplot", ont))
        save_plot(p4, file.path(out_dir, paste0("GSEA_Cnet_", ont, ".png")))
      }, silent = TRUE)
      
      try({
        p8 <- upsetplot(res) + theme(axis.text.x = element_text(size = 4), axis.text.y = element_text(size = 4)) +
          ggtitle(paste("Upsetplot", ont))
        save_plot(p8, file.path(out_dir, paste0("GSEA_Upset_", ont, ".png")))
      }, silent = TRUE)
      
      try({
        res_sim <- pairwise_termsim(res)
        p6 <- treeplot(res_sim, showCategory = 20, nWords = 4, fontsize_cladelab = 2.5, cladelab_offset = 12) +
          hexpand(0.8) + ggtitle(paste("Treeplot", ont))
        ggsave(file.path(out_dir, paste0("GSEA_Tree_", ont, ".png")), plot = p6, width = 12, height = 8, dpi = 600)
        
        p7 <- emapplot(res_sim, cex_label_category = 0.4) + ggtitle(paste("Enrichment Map", ont))
        save_plot(p7, file.path(out_dir, paste0("GSEA_Emap_", ont, ".png")))
      }, silent = TRUE)
      
      # GSEA-specific: enrichment score plots for top terms
      try({
        top_ids <- head(sig_hits$ID[order(sig_hits$p.adjust)], 5)
        for (tid in top_ids) {
          p_gsea <- gseaplot2(res, geneSetID = tid, title = tid, pvalue_table = TRUE)
          safe_name <- gsub("[:/\\\\]", "_", tid)
          save_plot(p_gsea, file.path(out_dir, paste0("GSEA_ES_", ont, "_", safe_name, ".png")))
        }
      }, silent = TRUE)
      
      # Ridge plot
      try({
        res_clean <- res
        res_clean@result <- sig_hits
        p_ridge <- ridgeplot(res_clean, showCategory = 15) + ggtitle(paste("Ridge plot", ont))
        save_plot(p_ridge, file.path(out_dir, paste0("GSEA_Ridge_", ont, ".png")))
      }, silent = TRUE)
      
    } else {
      message(paste("No enrichment found for", ont, "at FDR <", fdr_val))
      write.table(res@result, file.path(out_dir, paste0("GSEA_table_", ont, "_notSig.tsv")),
                  quote = FALSE, sep = "\t", row.names = FALSE)
    }
  }
}

################################################################################
#                                   RUN SCRIPT                                 #
################################################################################

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args || length(args) < 2) { print_help(); quit(status = 0) }

suppressMessages({
  library(clusterProfiler); library(ReactomePA); library(org.Hs.eg.db)
  library(DOSE); library(enrichplot); library(ggplot2); library(AnnotationDbi)
  library(scales); library(dplyr); library(GO.db); library(ggtree)
})

file_path     <- args[1]
option_string <- args[2]
PADJ_CUTOFF   <- if (!is.na(args[3])) as.numeric(args[3]) else 0.05
RAW_OUT_PATH  <- if (!is.na(args[4])) args[4] else "GSEA_results"

# --- LOAD TARGET TERMS ---
TARGET_LIST <- NULL
if (!is.na(args[5]) && file.exists(args[5])) {
  TARGET_LIST <- unique(readLines(args[5]))
  message(paste("Filtering for", length(TARGET_LIST), "targeted terms."))
}

if (grepl("^~|^/|/", RAW_OUT_PATH)) { PATH = path.expand(RAW_OUT_PATH) } else { PATH = file.path(dirname(file_path), RAW_OUT_PATH) }

# --- READ INPUT TABLE ---
if (!file.exists(file_path)) stop(paste("Input file not found:", file_path))
df_input <- read.table(file_path, header = TRUE, fill = TRUE, stringsAsFactors = FALSE, sep = "\t")
if (ncol(df_input) < 3) df_input <- read.table(file_path, header = TRUE, fill = TRUE, stringsAsFactors = FALSE, sep = ",")

# --- IDENTIFY COLUMNS ---
id_col <- intersect(colnames(df_input), c("gene_id", "ensembl_gene_id", "GeneID"))[1]
if (is.na(id_col)) id_col <- colnames(df_input)[1]

fc_col <- intersect(colnames(df_input), c("logFC", "log2FoldChange"))[1]
if (is.na(fc_col)) stop("No logFC column found. Expected 'logFC' or 'log2FoldChange'.")

fdr_col <- intersect(colnames(df_input), c("adj.P.Val", "padj", "FDR", "qvalue"))[1]
if (is.na(fdr_col)) stop("No FDR column found. Expected 'adj.P.Val', 'padj', 'FDR', or 'qvalue'.")

message(paste("Using columns:", id_col, "(ID),", fc_col, "(logFC),", fdr_col, "(FDR)"))

# --- CLEAN IDs ---
df_input$ensembl_flat <- gsub("\\..*$", "", df_input[[id_col]])

# Remove rows with NA in key columns
df_input <- df_input[!is.na(df_input[[fc_col]]) & !is.na(df_input[[fdr_col]]), ]

# Replace FDR == 0 with smallest non-zero value (to avoid Inf in -log10)
min_nonzero_fdr <- min(df_input[[fdr_col]][df_input[[fdr_col]] > 0], na.rm = TRUE)
df_input[[fdr_col]][df_input[[fdr_col]] == 0] <- min_nonzero_fdr

# --- COMPUTE RANKING METRIC: -log10(FDR) * sign(logFC) ---
df_input$rank_metric <- -log10(df_input[[fdr_col]]) * sign(df_input[[fc_col]])

# --- MAP TO ENTREZ ---
entrez_map <- suppressMessages(mapIds(org.Hs.eg.db, keys = df_input$ensembl_flat,
                                      column = "ENTREZID", keytype = "ENSEMBL", multiVals = "first"))
df_input$entrez <- entrez_map[df_input$ensembl_flat]
df_input <- df_input[!is.na(df_input$entrez), ]

# Handle duplicate Entrez IDs: keep the one with highest absolute rank metric
df_input <- df_input[order(-abs(df_input$rank_metric)), ]
df_input <- df_input[!duplicated(df_input$entrez), ]

# --- BUILD RANKED LIST ---
ranked_list <- df_input$rank_metric
names(ranked_list) <- df_input$entrez
ranked_list <- sort(ranked_list, decreasing = TRUE)

# Store globally for cnetplot coloring
RANKED_LIST <- ranked_list

message(paste("Ranked gene list:", length(ranked_list), "genes"))
message(paste("Range:", round(min(ranked_list), 2), "to", round(max(ranked_list), 2)))

# --- RUN GSEA ---
dir.create(PATH, showWarnings = FALSE, recursive = TRUE)
opts_to_run <- if (option_string == "All") VALID_OPTIONS[-1] else unlist(strsplit(option_string, ","))

for (opt in opts_to_run) {
  message(paste("Running GSEA for:", opt))
  if (opt %in% c("BP", "CC", "MF")) {
    call_gsea_GO_filtered(ranked_list, opt, PADJ_CUTOFF)
  } else if (opt %in% c("H", "C1", "C2", "C4", "C6", "C8")) {
    call_gsea_gmt(ranked_list, opt, PADJ_CUTOFF)
  } else if (opt == "D") {
    call_gsea_disease(ranked_list, PADJ_CUTOFF)
  } else if (opt == "R") {
    call_gsea_reactome(ranked_list, PADJ_CUTOFF)
  }
}

message(paste("GSEA analyses completed. Results in:", PATH))
