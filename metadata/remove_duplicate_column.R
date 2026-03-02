#' Global Deduplicate Columns
#' @description Cleans up all suffixed duplicates (.x, .y, .1, .2) if they are identical to the original.
#' @param data A dataframe or tibble.
#' @return A cleaned dataframe.
deduplicate_all <- function(data) {
  
  # 1. Find all columns that have a suffix (e.g., column.x, column.1)
  # This regex looks for a dot followed by x, y, or numbers at the end of a string
  all_names <- names(data)
  suffixed_cols <- all_names[grepl("\\.(x|y|[0-9])$", all_names)]
  
  if (length(suffixed_cols) == 0) {
    message("No suffixed duplicates found.")
    return(data)
  }
  
  # 2. Identify the "Base Names" (e.g., "Date" is the base for "Date.x" and "Date.1")
  base_names <- unique(gsub("\\.(x|y|[0-9])$", "", suffixed_cols))
  
  output_df <- data
  
  for (base in base_names) {
    # Find all siblings: e.g., "Date", "Date.x", "Date.1"
    siblings <- all_names[gsub("\\.(x|y|[0-9])$", "", all_names) == base]
    
    if (length(siblings) > 1) {
      # Use the first sibling as the "Source of Truth"
      primary <- siblings[1]
      others <- siblings[-1]
      
      for (duplicate in others) {
        # Check if identical to the primary
        are_identical <- isTRUE(all.equal(output_df[[primary]], 
                                          output_df[[duplicate]], 
                                          check.attributes = FALSE))
        
        if (are_identical) {
          message(paste0("[", duplicate, "] matches [", primary, "]. Removing duplicate."))
          output_df <- output_df %>% select(-all_of(duplicate))
        } else {
          warning(paste0("[", duplicate, "] DIFFERS from [", primary, "]! Keeping both."))
        }
      }
      
      # Final Rename: If the primary still has a suffix (like 'Date.x'), clean it up
      clean_name <- gsub("\\.(x|y|[0-9])$", "", primary)
      if (primary != clean_name && !(clean_name %in% names(output_df))) {
        output_df <- output_df %>% rename(!!clean_name := !!sym(primary))
      }
    }
  }
  
  return(output_df)
}