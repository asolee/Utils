####MAIN####
#generate heat-maps automatically

####Libraries####
invisible(library("optparse"))
options(warn=-1)

####option parsing####
option_list <- list(
  make_option(c("-i", "--input_matrix"), help="Numerical matrix\n\t\t[default=%default]", default=""),
  make_option(c("-t", "--input_title"), help="Title for correlation plot\n\t\t[default=%default]", default=""),
  make_option(c("-m", "--methods"), help="correlation methods\n\t\t[default=%default]", default="pearson"),
  make_option(c("-d", "--directory"), help="output folder name (full path)\n\t\t[default=%default]", default=""),
  make_option(c("-o", "--output"), help="output file name (without extension)\n\t\t[default=%default]", default=""),
  make_option(c("-l", "--limits"), help="limits values for correlation
                                         \n\t\t-> tight = min-max in a specific correlation matrix
                                         \n\t\t-> default = limits from -1 to 1
                                         \n\t\t-> focus = select focus color scale using --min_corr and --max_corr", default="default"),
  make_option(c("--min_focus"), help="min correlation if you use focus limits\n\t\t[default=%default]", default="0.9", type= "double"),
  make_option(c("--max_focus"), help="max correlation if you use focus limits\n\t\t[default=%default]", default="1", type="double")
  )

parser <- OptionParser(
  usage = "%prog [options] file", 
  option_list=option_list,
  description = "
  Produce correlation plot.\n\n
  requirements:
  -> ggcorrplot
	"
)
arguments <- parse_args(parser, positional_arguments = TRUE)
opt <- arguments$options

#libraries
cat("\nLoading libraries... ")
x <- c("ggcorrplot")
invisible(capture.output(lapply(x, library, character.only = TRUE)))
cat("\nDONE!\n\n")

####check input parameters####
#define possible parameters and read input files
methods <- strsplit(opt$methods, ",")[[1]]
coeff = c("pearson", "spearman","kendall")
if(sum(!sum(methods %in% coeff) == length(methods))){
  stop("distance or correlation methods not supported, please select a method from the following list:\n",paste0(coeff,collapse = ', '))
}

#check min and max correlation
if(!opt$limits %in% c("tight","default","focus")){
  stop("only tight,default and focus options are available for limits flag")
}
if(opt$limits == "focus"){
  min_value <- as.numeric(opt$min_focus)
  max_value <- as.numeric(opt$max_focus)
}

#load matrix
input_matrix <- as.matrix(read.table(opt$input_matrix,header = T))

####print parameters####
cat(paste0("input matrix --> ",opt$input_matrix,"\n"))
cat(paste0("methods --> ",paste(methods,collapse = ", "),"\n"))
cat(paste0("title --> ",opt$input_title),"\n")
cat(paste0("output directory --> ",opt$directory,"\n"))
cat(paste0("output file --> ",opt$output,"\n"))
cat(paste0("limits --> ",opt$limits,"\n"))
if(opt$limits == "focus"){
  cat(paste0("min focus --> ",opt$min_focus,"\n"))
  cat(paste0("max focus --> ",opt$max_focus,"\n"))
}
  

#correlation creation
cat("\nGenerating correlation plots... \n")
for(i in 1:length(methods)){
  
  if(opt$limits == "tight"){
corr_table <- cor(input_matrix,method = methods[i])
min_corr <- min(corr_table)
max_corr <- max(corr_table)
corr_plot <- ggcorrplot(corr_table)+
             ggtitle(paste0(opt$input_title," (",methods[i],")"))+
             scale_fill_gradient2(low="white", high="red", limits = c(min_corr,max_corr)) +
             theme(text = element_text(size = 20,face = "bold"),
             axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(filename = paste0(opt$directory,"/",opt$output,"_",methods[i],".png"),
       plot = corr_plot,width = 15,height = 15)
  }
  if(opt$limits == "default"){
    corr_plot <- ggcorrplot(cor(input_matrix,method = methods[i]))+
      ggtitle(paste0(opt$input_title," (",methods[i],")"))+
      theme(text = element_text(size = 20,face = "bold"),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    ggsave(filename = paste0(opt$directory,"/",opt$output,"_",methods[i],".png"),
           plot = corr_plot,width = 15,height = 15)
  }
  if(opt$limits == "focus"){
    corr_table <- cor(input_matrix,method = methods[i])
    min_corr <- min(corr_table)
    print(min_corr)
    med = min_value
    max_corr <- max_value
    corr_plot <- ggcorrplot(corr_table)+
      ggtitle(paste0(opt$input_title," (",methods[i],")"))+
      scale_fill_gradientn(
        colours = c("white", "white", "red"),
        limits = c(min_corr, max_corr),
        values = c(min_corr, med, max_corr)) +
      theme(text = element_text(size = 20,face = "bold"),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
  
    ggsave(filename = paste0(opt$directory,"/",opt$output,"_",methods[i],".png"),
           plot = corr_plot,width = 15,height = 15)
  }
  
}
cat("\nDONE!\n\n")
