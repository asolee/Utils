####MAIN####
#generate heat-maps automatically

####Libraries####
invisible(library("optparse"))
options(warn=-1)

####option parsing####
option_list <- list(
  make_option(c("-i", "--input_matrix"), help="Numerical matrix\n\t\t[default=%default]", default=""),
  make_option(c("-t", "--input_title"), help="Title for correlation plot\n\t\t[default=%default]", default=""),
  make_option(c("-m", "--method"), help="distance method, in case of correlation method --> d = 1 - corr(x,y,method)\n\t\t[default=%default]", default="euclidean"),
  make_option(c("-d", "--directory"), help="output folder name (full path)\n\t\t[default=%default]", default=""),
  make_option(c("-o", "--output"), help="output file name (without extension)\n\t\t[default=%default]", default="")
  )

parser <- OptionParser(
  usage = "%prog [options] file", 
  option_list=option_list,
  description = "
  Produce correlation plot.\n
  requirements:
  -> ComplexHeatmap
	"
)
arguments <- parse_args(parser, positional_arguments = TRUE)
opt <- arguments$options

#libraries
cat("\nLoading libraries... ")
x <- c("ComplexHeatmap","viridis")
invisible(capture.output(lapply(x, library, character.only = TRUE)))
cat("\nDONE!\n\n")

####check input parameters####
#define possible parameters and read input files
methods <- strsplit(opt$method, ",")[[1]]
coeff = c("euclidean","maximum","manhattan","canberra","binary","minkowski","pearson", "spearman","kendall")
if(sum(!sum(methods %in% coeff) == length(methods))){
  stop("distance or correlation method not supported, please select a method from the following list:\n",paste0(coeff,collapse = ', '))
}

input_matrix <- as.matrix(read.table(opt$input_matrix,header = T))

####print parameters####
cat(paste0("input matrix --> ",opt$input_matrix,"\n"))
cat(paste0("method --> ",paste(methods,collapse = ", "),"\n"))
cat(paste0("title --> ",opt$input_title),"\n")
cat(paste0("output directory --> ",opt$directory,"\n"))
cat(paste0("output file --> ",opt$output,"\n"))

#Heat-map creation
for(i in 1:length(methods)){
if(methods[i] %in% c("euclidean","maximum","manhattan","canberra","binary","minkowski")){
distance <- as.matrix(dist(t(input_matrix),method = methods[i]))
} else {
distance <- as.matrix(1 - cor(input_matrix,method = methods[i]))
}
#perform heatmap
plot <- Heatmap(distance,
                row_names_max_width = max_text_width(
                  rownames(distance), 
                  gp = gpar(fontsize = 12)),
                column_names_max_height = max_text_width(
                  rownames(distance), 
                  gp = gpar(fontsize = 12)),
        row_dend_width = unit(4, "cm"),
        column_dend_height = unit(4, "cm"),
        clustering_distance_columns = "euclidean",
        clustering_method_columns = "average",
        clustering_distance_rows = "euclidean",
        clustering_method_rows = "average",
        heatmap_legend_param = list(
        title = "distance"),
        col=viridis(100,direction = -1))
png(filename = paste0(opt$directory,"/",opt$output,"_",methods[i],".png"),width = 900,height = 900)
draw(plot)
dev.off()
#perform dendogram
distance <- dist(t(input_matrix),method = methods[i])
hc <- hclust(distance, method = "average")
png(filename = paste0(opt$directory,"/",opt$output,"_dendogram.png"),width = 900,height = 900)
plot(hc, hang = -1)
dev.off()
}
