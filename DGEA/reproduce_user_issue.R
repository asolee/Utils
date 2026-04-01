source("/mnt/efs/home/gasole/repos/personal/Utils/DGEA/plot_design.R")

# Recreate the user's data structure
# Note: I'm only creating a subset to reproduce the structure
design2 <- data.frame(
  "Timepoint..simplified.Pre-treatment" = sample(0:1, 20, replace=TRUE),
  "Timepoint..simplified.Progression" = sample(0:1, 20, replace=TRUE),
  # ... skipping many binary columns, adding a couple ...
  "Subject.IDOV_PL10" = sample(0:1, 20, replace=TRUE),
  "Subject.IDOV_PL11" = sample(0:1, 20, replace=TRUE),
  
  # Continuous columns
  "GNATENKO_PLATELET_SIGNATURE_raw" = runif(20, 9.5, 10.5), # Approx range from user
  "Genes_contributing_to_80._of_reads" = sample(70:530, 20, replace=TRUE) # Approx range
)

# Ensure names are accurate including dots if any
# The user output shows "Genes_contributing_to_80._of_reads"
# R data.frame might sanitize "80%" to "80.", so this name is realistic.

print("Colnames of design2:")
print(colnames(design2))

# Run plot_design
output_dir <- "/mnt/efs/home/gasole/repos/personal/Utils/DGEA/reproduce_user_test/"
if (!dir.exists(output_dir)) dir.create(output_dir)

# We want to check the internal state of plot_design annotations.
# But since we can't easily hook into it without editing, I'll rely on generating the plot 
# and maybe inspecting the code behavior by modifying plot_design first to debug.

print("Running plot_design...")
tryCatch({
    plot_design(design2, output_dir, "reproduce_user_heatmap.png")
    print("Execution finished.")
}, error = function(e) {
    print(paste("Error:", e$message))
})
