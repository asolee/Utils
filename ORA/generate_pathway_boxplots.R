#!/usr/bin/env Rscript

# Libraries
suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

#' Helper to convert p-value to asterisks
get_asterisks <- function(p) {
  if (is.na(p) || is.null(p)) return("")
  if (p <= 0.001) return("***")
  if (p <= 0.01)  return("**")
  if (p <= 0.05)  return("*")
  return("ns")
}

#' Generate Boxplots and TSV for Genes
#' @param summary_name Optional: Custom name for the output folder and plot titles
generate_pathway_boxplots <- function(exp_matrix, samples1, name1, samples2, name2, 
                                      ontology = NULL, specific_set, out_base_folder,
                                      summary_name = NULL) {
  
  # 1. Setup Output Directory and Naming
  # Priority: summary_name > ontology/pathway > "Manual_Gene_List"
  if (!is.null(summary_name)) {
    display_folder_name <- summary_name
  } else {
    display_folder_name <- if(length(specific_set) > 1) "Manual_Gene_List" else specific_set
  }
  
  sanitized_name <- gsub(" ", "_", display_folder_name)
  out_dir <- file.path(out_base_folder, sanitized_name)
  
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # 2. Identify and Map Genes
  target_gene_map <- data.frame()
  if (!is.null(ontology)) {
    GMT_FORMAT <- "/mnt/efs/home/erusu/gsea/GMTs/%s.all.v2024.1.Hs.entrez.gmt"
    gmt_file_path <- sprintf(GMT_FORMAT, ontology)
    gmt_data <- read.gmt(gmt_file_path)
    pathway_genes_entrez <- gmt_data %>% filter(term == gsub(" ", "_", specific_set)) %>% pull(gene)
    target_gene_map <- suppressMessages(AnnotationDbi::select(org.Hs.eg.db, keys = pathway_genes_entrez,
                                                              columns = c("ENSEMBL", "SYMBOL"), keytype = "ENTREZID"))
  } else {
    is_ensembl <- grepl("^ENSG", specific_set)
    if (any(is_ensembl)) {
      map_ens <- suppressMessages(AnnotationDbi::select(org.Hs.eg.db, keys = gsub("\\..*$", "", specific_set[is_ensembl]),
                                                        columns = "SYMBOL", keytype = "ENSEMBL"))
      target_gene_map <- rbind(target_gene_map, map_ens)
    }
    if (any(!is_ensembl)) {
      map_sym <- suppressMessages(AnnotationDbi::select(org.Hs.eg.db, keys = specific_set[!is_ensembl],
                                                        columns = "ENSEMBL", keytype = "SYMBOL"))
      target_gene_map <- rbind(target_gene_map, map_sym[, c("ENSEMBL", "SYMBOL")])
    }
  }
  
  # 3. Match against Matrix
  raw_rownames <- rownames(exp_matrix)
  clean_rownames <- gsub("\\..*$", "", raw_rownames)
  matches <- unique(target_gene_map[target_gene_map$ENSEMBL %in% clean_rownames, ])
  if (nrow(matches) == 0) stop("No genes matched in the expression matrix.")
  
  all_genes_df <- list()
  stats_list <- list()
  valid_ensembl_ids <- c() 
  
  # 4. Iterate and Plot Individuals
  for (row_idx in 1:nrow(matches)) {
    ens_id <- matches$ENSEMBL[row_idx]
    gene_symbol <- matches$SYMBOL[row_idx]
    display_name <- if(is.na(gene_symbol) || gene_symbol == "") ens_id else gene_symbol
    
    matrix_idx <- which(clean_rownames == ens_id)[1]
    val1 <- as.numeric(exp_matrix[matrix_idx, samples1])
    val2 <- as.numeric(exp_matrix[matrix_idx, samples2])
    
    if (sum(abs(c(val1, val2)), na.rm = TRUE) == 0) {
      warning(paste("Skipping gene:", display_name, "(", ens_id, ") - Zero expression in all samples."))
      next
    }
    
    valid_ensembl_ids <- c(valid_ensembl_ids, ens_id)
    
    plot_df <- data.frame(
      Expression = c(val1, val2),
      Group = factor(c(rep(name1, length(val1)), rep(name2, length(val2))), levels = c(name1, name2)),
      Gene = display_name
    ) %>% drop_na(Expression)
    
    p_val <- tryCatch({
      t_res <- t.test(Expression ~ Group, data = plot_df)
      t_res$p.value
    }, error = function(e) return(NA))
    
    p_label <- if(is.na(p_val)) "p = NA" else paste0("p = ", format.pval(p_val, digits = 3))
    stars <- get_asterisks(p_val)
    
    y_max <- max(plot_df$Expression, na.rm = TRUE)
    y_range <- if(diff(range(plot_df$Expression)) == 0) 1 else diff(range(plot_df$Expression))
    
    all_genes_df[[display_name]] <- plot_df
    stats_list[[display_name]] <- data.frame(
      Gene = display_name,
      p_label = p_label,
      stars = stars,
      y_bracket = y_max + y_range * 0.05,
      y_text = y_max + y_range * 0.15,
      y_stars = y_max + y_range * 0.25
    )
    
    p <- ggplot(plot_df, aes(x = Group, y = Expression, fill = Group)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.7) +
      geom_jitter(width = 0.2, size = 1.5, alpha = 0.6) +
      annotate("segment", x = 1, xend = 2, y = y_max + y_range*0.05, yend = y_max + y_range*0.05) +
      annotate("text", x = 1.5, y = y_max + y_range*0.12, label = p_label, size = 3) +
      annotate("text", x = 1.5, y = y_max + y_range*0.18, label = stars, size = 4, fontface = "bold") +
      theme_classic() +
      labs(title = paste("Gene:", display_name), 
           subtitle = if(!is.null(summary_name)) summary_name else "",
           y = "log2 CPM", x = "") +
      scale_fill_manual(values = setNames(c("#69b3a2", "#404080"), c(name1, name2))) +
      theme(legend.position = "none")
    
    ggsave(file.path(out_dir, paste0(display_name, "_boxplot.png")), plot = p, width = 4, height = 4)
  }
  
  if (length(all_genes_df) == 0) {
    stop("No genes passed the zero-expression filter. No plots generated.")
  }
  
  # 5. GENERATE SUMMARY FACETED PLOT
  message("Generating summary image with stats...")
  summary_df <- bind_rows(all_genes_df)
  summary_stats <- bind_rows(stats_list)
  
  ps <- ggplot(summary_df, aes(x = Group, y = Expression, fill = Group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.2, size = 0.8, alpha = 0.4) +
    geom_segment(data = summary_stats, aes(x = 1, xend = 2, y = y_bracket, yend = y_bracket), inherit.aes = FALSE) +
    geom_text(data = summary_stats, aes(x = 1.5, y = y_text, label = p_label), inherit.aes = FALSE, size = 2.5) +
    geom_text(data = summary_stats, aes(x = 1.5, y = y_stars, label = stars), inherit.aes = FALSE, size = 3.5, fontface = "bold") +
    facet_wrap(~Gene, scales = "free_y") + 
    theme_bw() +
    scale_fill_manual(values = setNames(c("#69b3a2", "#404080"), c(name1, name2))) +
    labs(title = paste("Gene Set Summary:", display_folder_name),
         subtitle = paste(name1, "vs", name2),
         y = "log2 CPM", x = "") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "bottom",
          strip.background = element_rect(fill = "grey90"),
          strip.text = element_text(face = "bold"))
  
  n_genes <- length(unique(summary_df$Gene))
  grid_cols <- ceiling(sqrt(n_genes))
  grid_rows <- ceiling(n_genes / grid_cols)
  
  ggsave(file.path(out_dir, paste0("AAA_summary", sanitized_name, ".png")), 
         plot = ps, width = grid_cols * 3, height = grid_rows * 3.5, dpi = 300)
  
  # 6. SAVE EXPRESSION DATA TSV
  message("Saving expression data to TSV...")
  selected_samples <- c(samples1, samples2)
  matrix_subset <- exp_matrix[clean_rownames %in% valid_ensembl_ids, selected_samples, drop = FALSE]
  
  export_df <- as.data.frame(matrix_subset)
  export_df$ENSEMBL <- gsub("\\..*$", "", rownames(export_df))
  
  export_df <- export_df %>%
    left_join(matches, by = "ENSEMBL") %>%
    dplyr::select(ENSEMBL, SYMBOL, everything())
  
  write.table(export_df, file.path(out_dir, paste0("expression_data_", sanitized_name, ".tsv")), 
              sep = "\t", quote = FALSE, row.names = FALSE)
  
  message(paste("Process complete. Files saved in:", out_dir))
}