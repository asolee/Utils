library(ComplexHeatmap)
library(circlize)
library(grid)

plot_design <- function(design_mat, output_folder, filename = "design_heatmap.png") {
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }

  # 1. Prepare Data
  # Convert to data frame to preserve column types/names for rowAnnotation
  df <- as.data.frame(design_mat)
  
  # Identify binary vs continuous
  is_binary <- sapply(df, function(x) all(x %in% c(0, 1), na.rm = TRUE))
  df_bin <- df[, is_binary, drop = FALSE]
  df_cont <- df[, !is_binary, drop = FALSE]

  # 2. Define Color Mappings (As per Chapter 3.2 & 5.4)
  # Binary mapping (Shared)
  col_bin <- colorRamp2(c(0, 1), c("white", "dodgerblue4"))
  
  # Continuous mappings (Unique for each column)
  col_list <- list()
  if (ncol(df_cont) > 0) {
    for (col_nm in colnames(df_cont)) {
      vals <- df_cont[[col_nm]]
      
      if (is.numeric(vals)) {
        v_min <- min(vals, na.rm = TRUE)
        v_max <- max(vals, na.rm = TRUE)
        
        # Handle case where all values are NA or infinite
        if (!is.finite(v_min) || !is.finite(v_max)) {
          v_min <- 0
          v_max <- 1
        }
        
        # Ensure range for colorRamp2
        if(v_min == v_max) { v_min <- v_min - 0.5; v_max <- v_max + 0.5 }
        
        # Each column gets its own scale function
        col_list[[col_nm]] <- colorRamp2(
          seq(v_min, v_max, length.out = 3), 
          c("white", "gold", "firebrick3")
        )
      } else {
        # Handle non-numeric columns as discrete
        uniq_vals <- unique(vals[!is.na(vals)])
        if (length(uniq_vals) > 0) {
          col_list[[col_nm]] <- structure(
            circlize::rand_color(length(uniq_vals)), 
            names = as.character(uniq_vals)
          )
        }
      }
    }
  }

  # 3. Create Row Annotation (The "ComplexHeatmap" way)
  # Use do.call argument unpacking to ensure separate annotations with individual scales
  if (ncol(df_cont) > 0) {
      anno_args <- as.list(df_cont)
      anno_args$col <- col_list
      anno_args$show_annotation_name <- TRUE
      anno_args$annotation_name_rot <- 90
      anno_args$simple_anno_size <- unit(8, "mm")
      
      right_annos <- do.call(rowAnnotation, anno_args)
  } else {
      right_annos <- rowAnnotation()
  }

  # 4. Main Heatmap (Binary part)
  h_main <- Heatmap(
    as.matrix(df_bin),
    name = "Binary", 
    col = col_bin,
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    rect_gp = gpar(col = "white", lwd = 1),
    show_row_names = TRUE,
    row_names_side = "left",
    column_title = "Design Matrix Structure",
    right_annotation = right_annos,
    heatmap_legend_param = list(
      at = c(0, 1), 
      labels = c("0 (Absence)", "1 (Presence)")
    )
  )

  # 5. Save to File
  if(!grepl("\\.png$", filename)) filename <- paste0(filename, ".png")
  full_path <- file.path(output_folder, filename)
  
  # Calculation based on Chapter 15: Setting height based on rows
  n_rows <- nrow(df)
  # Standard: ~0.2 inches per row + 2 inches for titles/margins
  img_height_in <- max(5, (n_rows * 0.2) + 2) 

  png(full_path, width = 12, height = img_height_in, units = "in", res = 300)
  
  # draw() handles the legend list automatically when using 'df' in rowAnnotation
  draw(h_main, merge_legend = FALSE)
  
  dev.off()

  message(paste("Design heatmap saved to:", full_path))
}