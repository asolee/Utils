#!/usr/bin/env Rscript

# This script runs ORA analysis using different ontologies
# UPDATED: Added support for targeted term filtering via a 7th argument.

# ORA-related constants
MIN_GENEGROUP_SIZE = 10
MAX_GENEGROUP_SIZE = 500

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
Description: This script runs ORA analysis using different ontologies.
             It accepts a gene list (Ensembl) and a universe source.

Usage:
    Rscript run_ORAs.R [table_path] [option_string] [universe_path] [logfc_cutoff] [fdr_cutoff] [out_folder_path] [target_terms_path]

Arguments:
    table_path        : Full path to the gene list file (required)
    option_string     : String for filtering (required). 
                        Options: 'All', 'D', 'R', 'BP', 'CC', 'MF', 'H', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'
    universe_path     : Full path to the universe file (required). Accepts:
                        - .rds file containing a txi object (with $counts)
                        - .tsv/.csv count matrix (genes as rows, samples as columns)
                        - .txt file with one Ensembl gene ID per line
    logfc_cutoff      : ABS(LogFC) threshold to filter input genes (optional). Default: 0
    fdr_cutoff        : FDR threshold for significance (optional). Default: 0.5
    out_folder_path   : Path to the output directory (optional). Default: ORA_results
    target_terms_path : Path to a file with specific IDs (one per line) to filter results (optional).

Example:
    Rscript run_ORAs.R genes.txt 'H,R' universe.rds 1.0 0.05 './results' targets.txt
    Rscript run_ORAs.R genes.txt 'BP' raw_counts.tsv
    Rscript run_ORAs.R genes.txt 'H' gene_list.txt
\n")
}

# GO Enrichment with manual filtering
call_enrichment_std_GO_filtered <- function(diffexp_list, ontology, universe, fdr, fc_vector = NULL, suffix = "") {
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
  
  gene_map <- suppressMessages(AnnotationDbi::select(org.Hs.eg.db, keys = unique(c(diffexp_list, universe)),
                                                     columns = "ENTREZID", keytype = "ENSEMBL"))
  gene_map <- gene_map[!is.na(gene_map$ENTREZID), ]
  entrez_diffexp <- gene_map$ENTREZID[match(diffexp_list, gene_map$ENSEMBL)]
  entrez_universe <- gene_map$ENTREZID[match(universe, gene_map$ENSEMBL)]
  
  res <- enricher(gene = entrez_diffexp, universe = entrez_universe,
                  TERM2GENE = term2gene, TERM2NAME = term2name,
                  pvalueCutoff = fdr, pAdjustMethod = "fdr")
  
  report_results(res, ontology, fdr, fc_vector, suffix)
}

# Reactome Enrichment
call_enrichment_reactome <- function(diffexp_list, universe, fdr, fc_vector = NULL, suffix = ""){
  set.seed(123)
  res <- enrichPathway(diffexp_list, universe = universe,
                       minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE,
                       pvalueCutoff = fdr, pAdjustMethod = "fdr") 
  report_results(res, "R", fdr, fc_vector, suffix)
}

# Disease Enrichment
call_enrichment_disease <- function(diffexp_list, universe, fdr, fc_vector = NULL, suffix = "") {
  res <- enrichDGN(diffexp_list, universe = universe,
                   minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE,
                   pvalueCutoff = fdr, pAdjustMethod = "fdr")
  report_results(res, "D", fdr, fc_vector, suffix)
}

# GMT/MSigDB Enrichment
call_enrichment_gmt <- function(diffexp_list, ontology, universe, fdr, fc_vector = NULL, suffix = "") {
  msigdb_gmt <- read.gmt(sprintf(GMT_FORMAT, ontology))
  res <- enricher(gene = diffexp_list, universe = universe,
                  minGSSize = MIN_GENEGROUP_SIZE, maxGSSize = MAX_GENEGROUP_SIZE, 
                  pvalueCutoff = fdr, pAdjustMethod = "fdr", 
                  TERM2GENE = msigdb_gmt)
  report_results(res, ontology, fdr, fc_vector, suffix)
}

# Plotting Function
custom_barplot <- function(res, ontology, fdr_val, N_bar = 20) {
  df_sorted <- res@result[!is.na(res@result$p.adjust) & res@result$p.adjust < fdr_val, ]
  if(nrow(df_sorted) == 0) return(NULL)
  
  df_sorted <- df_sorted %>%
    mutate(BgRatio_calc = sapply(strsplit(as.character(BgRatio), "/"), 
                                 function(x) as.numeric(x[1]) / as.numeric(x[2])),
           GeneRatio_calc = sapply(strsplit(as.character(GeneRatio), "/"), 
                                   function(x) as.numeric(x[1]) / as.numeric(x[2])))
  
  df_sorted$max_diff <- df_sorted$GeneRatio_calc / df_sorted$BgRatio_calc
  df_sorted <- df_sorted[order(-df_sorted$max_diff), ]
  
  df_sorted$Clean_description <- gsub("HALLMARK_", "", df_sorted$Description)
  df_sorted$Clean_description <- gsub("_", " ", df_sorted$Clean_description)
  
  df_top <- head(df_sorted, N_bar)
  
  ggplot(df_top, aes(x = reorder(Clean_description, max_diff), y = max_diff, fill = p.adjust)) +
    geom_bar(stat = "identity", colour = "black", alpha = 0.7) +
    coord_flip() + 
    labs(x = "", y = "Ratio of enrichment", title = paste("Top", ontology, "Terms")) +
    theme_minimal(base_size = 9) +
    scale_x_discrete(labels = label_wrap(40)) + 
    geom_hline(yintercept = 1, linetype="dotted", color = "black", linewidth=1) +
    scale_fill_gradient(low = "blue", high = "black", name = "FDR")
}

save_plot <- function(p, file_path_out) {
  tryCatch({
    png(file_path_out, units="in", width=7, height=6, res=600)
    print(p)
  }, error = function(e) {
    message(paste("Error generating plot for", file_path_out, ":", e$message))
  }, finally = {
    if (dev.cur() > 1) dev.off()
  })
}

# Result Reporting Function (Updated for Filtering)
report_results <- function(res, ont, fdr_val, fc_vector = NULL, suffix = ""){
  
  out_dir <- PATH
  if (suffix == "_upreg") out_dir <- file.path(PATH, "upreg")
  if (suffix == "_downreg") out_dir <- file.path(PATH, "downreg")
  
  # --- TARGETED FILTERING LOGIC ---
  if (!is.null(TARGET_LIST)) {
    res@result <- res@result[res@result$ID %in% TARGET_LIST, ]
    out_dir <- file.path(out_dir, "targeted_results")
  }
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  if (is.null(res) || nrow(res@result) == 0){
    message(paste("No results for", ont, suffix))
  } else {
    sig_hits <- res@result[!is.na(res@result$p.adjust) & res@result$p.adjust < fdr_val, ]
    
    if (nrow(sig_hits) > 0) {
      message(paste("Found", nrow(sig_hits), "terms for", ont, suffix))
      write.table(sig_hits, file.path(out_dir, paste0("ORA_table_", ont, suffix, ".tsv")), 
                  quote = FALSE, sep = "\t", row.names = FALSE)
      
      try({
        p1 <- custom_barplot(res, ont, fdr_val)
        if(!is.null(p1)) save_plot(p1, file.path(out_dir, paste0("ORA_Bar_", ont, suffix, ".png")))
      }, silent=TRUE)
      
      try({
        res_clean <- res
        res_clean@result <- sig_hits
        p3 <- dotplot(res_clean, showCategory = 15, font.size = 9) + ggtitle(paste("Dotplot", ont))
        save_plot(p3, file.path(out_dir, paste0("ORA_Dot_", ont, suffix, ".png")))
      }, silent=TRUE)
      
      try({
        res_read <- res
        try({ res_read <- setReadable(res, OrgDb = org.Hs.eg.db, keyType="ENTREZID") }, silent=TRUE)
        p4 <- cnetplot(res_read, foldChange = fc_vector, cex_label_category = 0.7, node_label = "category") + 
          scale_color_gradient2(name = "logFC", low = "blue", mid = "white", high = "red") +
          ggtitle(paste("Cnetplot", ont))
        save_plot(p4, file.path(out_dir, paste0("ORA_Cnet_", ont, suffix, ".png")))
      }, silent=TRUE)
      
      try({
        p8 <- upsetplot(res) + theme(axis.text.x = element_text(size = 4), axis.text.y = element_text(size = 4)) + 
          ggtitle(paste("Upsetplot", ont))
        save_plot(p8, file.path(out_dir, paste0("ORA_Upset_", ont, suffix, ".png")))
      }, silent=TRUE)
      
      try({
        res_sim <- pairwise_termsim(res)
        p6 <- treeplot(res_sim, showCategory = 20, nWords = 4, fontsize_cladelab = 2.5, cladelab_offset = 12) + 
          hexpand(0.8) + ggtitle(paste("Treeplot", ont))
        ggsave(file.path(out_dir, paste0("ORA_Tree_", ont, suffix, ".png")), plot = p6, width = 12, height = 8, dpi = 600)
        
        p7 <- emapplot(res_sim, cex_label_category = 0.4) + ggtitle(paste("Enrichment Map", ont))
        save_plot(p7, file.path(out_dir, paste0("ORA_Emap_", ont, suffix, ".png")))
      }, silent=TRUE)
      
    } else {
      message(paste("No enrichment found under", ont, suffix, "at FDR <", fdr_val))
      write.table(res@result, file.path(out_dir, paste0("ORA_table_", ont, suffix, "_notSig.tsv")), 
                  quote = FALSE, sep = "\t", row.names = FALSE)    
    }
  }
}

################################################################################
#                                   RUN SCRIPT                                 #
################################################################################

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args || length(args) < 3) { print_help(); quit(status = 0) }

suppressMessages({
  library(clusterProfiler); library(ReactomePA); library(org.Hs.eg.db)
  library(DOSE); library(enrichplot); library(ggplot2); library(AnnotationDbi)
  library(scales); library(dplyr); library(GO.db); library(ggtree)
})

file_path     <- args[1]
option_string <- args[2]
txi_path      <- args[3]
LOGFC_CUTOFF  <- if(!is.na(args[4])) as.numeric(args[4]) else 0
PADJ_CUTOFF   <- if(!is.na(args[5])) as.numeric(args[5]) else 0.5
RAW_OUT_PATH  <- if(!is.na(args[6])) args[6] else "ORA_results"

# --- LOAD TARGET TERMS ---
TARGET_LIST <- NULL
if (!is.na(args[7]) && file.exists(args[7])) {
  TARGET_LIST <- unique(readLines(args[7]))
  message(paste("Filtering for", length(TARGET_LIST), "targeted terms."))
}

if (grepl("^~|^/|/", RAW_OUT_PATH)) { PATH = path.expand(RAW_OUT_PATH) } else { PATH = file.path(dirname(file_path), RAW_OUT_PATH) }

if (!file.exists(file_path)) stop(paste("Gene list file not found:", file_path))
df_input <- read.table(file_path, header = TRUE, fill = TRUE, stringsAsFactors = FALSE)
known_cols <- c("logFC", "AveExpr", "P.Value", "adj.P.Val", "gene_id", "gene_name", "baseMean", "log2FoldChange", "padj")
if (!any(colnames(df_input) %in% known_cols)) {
  df_input <- read.table(file_path, header = FALSE, fill = TRUE, stringsAsFactors = FALSE)
  df_input <- df_input[!grepl("ERCC", df_input[,1]), , drop=FALSE]
  colnames(df_input)[1] <- "gene_id"
  if (ncol(df_input) >= 2 && is.numeric(df_input[,2])) colnames(df_input)[2] <- "logFC"
}

id_col <- if("gene_id" %in% colnames(df_input)) "gene_id" else colnames(df_input)[1]
df_input$ensembl_flat <- gsub("\\..*$", "", df_input[[id_col]])

fc_col <- intersect(colnames(df_input), c("logFC", "log2FoldChange", "FoldChange"))[1]
fc_vector <- NULL
if (!is.na(fc_col)) {
  if (LOGFC_CUTOFF > 0) df_input <- df_input[abs(df_input[[fc_col]]) > LOGFC_CUTOFF, ]
  sym_map <- suppressMessages(mapIds(org.Hs.eg.db, keys = df_input$ensembl_flat, column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first"))
  fc_vector <- df_input[[fc_col]]
  names(fc_vector) <- sym_map
  fc_vector <- fc_vector[!is.na(names(fc_vector))]
}

entrez_ids <- suppressMessages(mapIds(org.Hs.eg.db, keys = df_input$ensembl_flat, column = "ENTREZID", keytype = "ENSEMBL", multiVals = "first"))
geneList_entrez <- na.omit(unname(entrez_ids))

if (!file.exists(txi_path)) stop(paste("Universe file not found:", txi_path))
universe_ext <- tolower(tools::file_ext(txi_path))

if (universe_ext == "rds") {
  rds_obj <- readRDS(txi_path)
  if (is.list(rds_obj) && "counts" %in% names(rds_obj)) {
    universe_flat <- gsub("\\..*$", "", rownames(rds_obj$counts))
  } else if (is.matrix(rds_obj) || is.data.frame(rds_obj)) {
    universe_flat <- gsub("\\..*$", "", rownames(rds_obj))
  } else {
    stop("RDS object must be a txi list (with $counts), a matrix, or a data.frame.")
  }
  message(paste("Universe loaded from RDS:", length(universe_flat), "genes"))
} else if (universe_ext %in% c("tsv", "csv")) {
  sep_char <- if (universe_ext == "csv") "," else "\t"
  count_mat <- read.table(txi_path, header = TRUE, sep = sep_char, row.names = 1,
                          check.names = FALSE, nrows = 5)
  # Only need rownames, re-read just first column for efficiency on large files
  count_mat <- read.table(txi_path, header = TRUE, sep = sep_char,
                          check.names = FALSE, stringsAsFactors = FALSE)
  universe_flat <- gsub("\\..*$", "", count_mat[, 1])
  message(paste("Universe loaded from count matrix:", length(universe_flat), "genes"))
} else {
  # Treat as plain gene list (one Ensembl ID per line)
  universe_flat <- gsub("\\..*$", "", readLines(txi_path))
  universe_flat <- universe_flat[nchar(universe_flat) > 0]
  message(paste("Universe loaded from gene list:", length(universe_flat), "genes"))
}

universe_flat <- unique(universe_flat)
universe_entrez <- na.omit(unname(suppressMessages(mapIds(org.Hs.eg.db, keys = universe_flat, column = "ENTREZID", keytype = "ENSEMBL", multiVals = "first"))))

dir.create(PATH, showWarnings = FALSE, recursive = TRUE)
opts_to_run <- if(option_string == "All") VALID_OPTIONS[-1] else unlist(strsplit(option_string, ","))

gene_sets_to_test <- list(list(name = "", ids = df_input$ensembl_flat, ent = geneList_entrez))
if (!is.null(fc_vector)) {
  up_idx <- which(df_input[[fc_col]] > 0); down_idx <- which(df_input[[fc_col]] < 0)
  if(length(up_idx) > 0) {
    up_ent <- na.omit(unname(suppressMessages(mapIds(org.Hs.eg.db, keys = df_input$ensembl_flat[up_idx], column = "ENTREZID", keytype = "ENSEMBL", multiVals="first"))))
    gene_sets_to_test[[length(gene_sets_to_test)+1]] <- list(name="_upreg", ids=df_input$ensembl_flat[up_idx], ent=up_ent)
  }
  if(length(down_idx) > 0) {
    down_ent <- na.omit(unname(suppressMessages(mapIds(org.Hs.eg.db, keys = df_input$ensembl_flat[down_idx], column = "ENTREZID", keytype = "ENSEMBL", multiVals="first"))))
    gene_sets_to_test[[length(gene_sets_to_test)+1]] <- list(name="_downreg", ids=df_input$ensembl_flat[down_idx], ent=down_ent)
  }
}

for (gset in gene_sets_to_test) {
  for (opt in opts_to_run) {
    if (opt %in% c("BP", "CC", "MF")) {
      call_enrichment_std_GO_filtered(gset$ids, opt, universe_flat, PADJ_CUTOFF, fc_vector, gset$name)
    } else if (opt %in% c("H", "C1", "C2", "C4", "C6", "C8")) {
      call_enrichment_gmt(gset$ent, opt, universe_entrez, PADJ_CUTOFF, fc_vector, gset$name)
    } else if (opt == "D") {
      call_enrichment_disease(gset$ent, universe_entrez, PADJ_CUTOFF, fc_vector, gset$name)
    } else if (opt == "R") {
      call_enrichment_reactome(gset$ent, universe_entrez, PADJ_CUTOFF, fc_vector, gset$name)
    }
  }
}
message(paste("Analyses completed. Results in:", PATH))