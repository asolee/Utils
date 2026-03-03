library(ggplot2)
library(dplyr)
library(scales)

#' Generate basic plots for metadata columns
#' @description Generates basic plots (pie charts for categorical data and bar plots for numeric data) for each column in the dataset.
#' @param dataset A dataframe or tibble.
#' @param output_folder The folder where the plots will be saved.
#' @return None. Plots are saved as PNG files in the specified output folder.

save_metadata_plots <- function(dataset, output_folder) {
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }

  sample_names <- rownames(dataset)

  for (col_name in colnames(dataset)) {
    data_col <- dataset[[col_name]]
    p <- NULL

    # 1. Categorical Data (Pie Chart)
    if (is.character(data_col) || is.factor(data_col)) {
      df_pie <- as.data.frame(table(data_col))
      colnames(df_pie) <- c("Category", "Count")

      # Simplified label and position calculation
      df_pie <- df_pie %>%
        mutate(
          perc = Count / sum(Count),
          # Count (Percentage) label
          label_text = paste0(Count, " (", percent(perc, accuracy = 1), ")")
        )

      p <- ggplot(df_pie, aes(x = "", y = Count, fill = Category)) +
        geom_bar(stat = "identity", width = 1, color = "white") +
        coord_polar("y", start = 0) +
        # Automatically finds the center of the slice
        geom_text(aes(label = label_text), 
                  position = position_stack(vjust = 0.5),
                  size = 3) +
        theme_void() +
        theme(plot.background = element_rect(fill = "white", color = NA)) +
        labs(title = col_name)

    }
    # 2. Numeric Data (Bar Plot)
    else if (is.numeric(data_col)) {
      df_bar <- data.frame(
        Sample = factor(sample_names, levels = sample_names),
        Value = data_col
      )

      p <- ggplot(df_bar, aes(x = Sample, y = Value)) +
        geom_col(fill = "steelblue") +
        theme_minimal() +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1),
          plot.background = element_rect(fill = "white", color = NA)
        ) +
        labs(title = col_name, x = "Samples", y = col_name)
    }

    # 3. Save the plot if it was created
    if (!is.null(p)) {
      col_name <- gsub("/", "_", col_name)
      ggsave(file.path(output_folder, paste0(col_name, ".png")),
             plot = p, width = 7, height = 7, bg = "white")
      message(paste("Saved:", col_name))
    }
  }
}
